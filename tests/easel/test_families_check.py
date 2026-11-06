"""`easel families check`: the shipped families pass; planted families break each rule and fail, one line per problem."""

import shutil
from pathlib import Path

import pytest
import yaml

from easel import cli, families_check

PKG = "easel"
OWN_BLOCK = "inference"
OTHER_BLOCK = "training"
REPO = Path(__file__).resolve().parents[2]
FAMILIES = REPO / "families"
DROP = object()


def check(families_dir: Path, *, strict: bool = False) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = families_check.run(families_dir=families_dir, strict=strict, out=lines.append)
    return code, lines


def family(fid: str, inherits: str | None = None, **fields: object) -> dict:
    """A minimal valid family mapping; a field set to DROP is left out."""
    data: dict = {
        "schema": f"{PKG}/family/1",
        "id": fid,
        "name": "Planted Demo Family",
        "status": "listed",
        "inherits": inherits,
        "description": "a family planted by a test",
        "licence_summary": {"status": "unverified", "name": None, "text": "unverified: read the model card", "url": "https://example.invalid/demo-card"},
        "host_ram_gb": {"train": None, "eval": None, "eval_includes_comfyui_server": None, "basis": "not measured"},
    }
    data.update(fields)
    return {k: v for k, v in data.items() if v is not DROP}


def plant(folder: Path, data: dict, name: str | None = None) -> Path:
    path = folder / f"{name or data['id']}.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def planted(tmp_path: Path) -> Path:
    """A copy of the shipped families in tmp_path, ready for one planted file."""
    folder = tmp_path / "families"
    shutil.copytree(FAMILIES, folder)
    return folder


# --- the shipped data ---------------------------------------------------------------------------------------------

def test_shipped_families_pass():
    code, lines = check(FAMILIES)
    assert code == 0, lines
    count = len(list(FAMILIES.glob("*.yaml")))
    assert lines[-1].startswith(f"ok: {count} family file(s) checked"), lines
    for line in lines[:-1]:
        assert line.startswith("warn: ") and "licence_summary.url is null" in line, line


def test_cli_families_check_exits_0(capsys):
    assert cli.main(["families", "check"]) == 0
    assert "family file(s) checked" in capsys.readouterr().out


def test_cli_help_has_no_parentheses(capsys):
    for argv in (["families", "--help"], ["families", "check", "--help"]):
        with pytest.raises(SystemExit) as exited:
            cli.main(argv)
        assert exited.value.code == 0
        out = capsys.readouterr().out
        assert "(" not in out and ")" not in out, out


def test_every_shipped_family_carries_its_own_licence_summary_with_a_url_key():
    for path in FAMILIES.glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data.get("licence_summary"), dict), path.name
        assert "url" in data["licence_summary"], path.name
        assert isinstance(data.get("host_ram_gb"), dict), path.name


# --- licence summary ----------------------------------------------------------------------------------------------

def test_planted_child_without_licence_summary_fails(planted):
    plant(planted, family("planted", inherits="sdxl", licence_summary=DROP, host_ram_gb=DROP))
    code, lines = check(planted)
    assert code == 1
    assert any(line.startswith("families/planted.yaml: no own licence_summary; a licence is never inherited") for line in lines), lines
    assert lines[-1].startswith("FAIL: 1 problem(s)")


def test_planted_root_without_licence_summary_fails(planted):
    plant(planted, family("planted", licence_summary=DROP))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: licence_summary is missing" in lines


def test_licence_summary_without_url_key_fails(planted):
    plant(planted, family("planted", licence_summary={"status": "unverified", "text": "read the card"}))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: licence_summary.url is missing" in lines
    assert "families/planted.yaml: licence_summary.name is missing" in lines


def test_licence_status_outside_the_three_fails(planted):
    plant(planted, family("planted", licence_summary={"status": "fine", "name": None, "text": "read the card", "url": None}))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: licence_summary.status must be one of verified, unverified, unknown" in lines


def test_licence_url_that_is_not_http_fails(planted):
    plant(planted, family("planted", licence_summary={"status": "unverified", "name": None, "text": "read the card", "url": "see the card"}))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: licence_summary.url must be an http(s) address or null" in lines


def test_null_licence_url_warns_and_fails_only_when_strict(tmp_path):
    folder = tmp_path / "families"
    folder.mkdir()
    plant(folder, family("planted", licence_summary={"status": "unverified", "name": None, "text": "read the card", "url": None}))
    code, lines = check(folder)
    assert code == 0, lines
    assert lines == ["warn: families/planted.yaml: licence_summary.url is null: no public model card recorded yet",
                     "ok: 1 family file(s) checked, 1 warning(s)"]
    code, lines = check(folder, strict=True)
    assert code == 1
    assert lines[0] == "families/planted.yaml: licence_summary.url is null: no public model card recorded yet"


# --- inheritance --------------------------------------------------------------------------------------------------

def test_inherited_host_ram_counts(planted):
    plant(planted, family("planted", inherits="sdxl", host_ram_gb=DROP))
    code, lines = check(planted)
    assert code == 0, lines


def test_host_ram_missing_everywhere_fails(planted):
    plant(planted, family("planted", host_ram_gb=DROP))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: host_ram_gb is missing, also after inheritance" in lines


def test_missing_parent_fails(planted):
    plant(planted, family("planted", inherits="no_such_family"))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: inherits 'no_such_family', but there is no no_such_family.yaml" in lines


def test_inheritance_cycle_fails(tmp_path):
    folder = tmp_path / "families"
    folder.mkdir()
    plant(folder, family("alpha", inherits="beta"))
    plant(folder, family("beta", inherits="alpha"))
    code, lines = check(folder)
    assert code == 1
    assert "families/alpha.yaml: inheritance cycle: alpha -> beta -> alpha" in lines
    assert "families/beta.yaml: inheritance cycle: beta -> alpha -> beta" in lines


def test_deep_merge_child_wins_and_nested_mappings_merge():
    parent = {"a": 1, "block": {"x": 1, "y": 2}, "list": [1, 2]}
    child = {"a": 2, "block": {"y": 3}, "list": [9]}
    assert families_check.deep_merge(parent, child) == {"a": 2, "block": {"x": 1, "y": 3}, "list": [9]}
    assert parent == {"a": 1, "block": {"x": 1, "y": 2}, "list": [1, 2]}


# --- host RAM and the tool's own block -------------------------------------------------------------------------------

def test_eval_number_needs_the_comfyui_flag(planted):
    ram = {"train": 5.5, "eval": 8.5, "eval_includes_comfyui_server": None, "basis": "measured"}
    plant(planted, family("planted", host_ram_gb=ram))
    code, lines = check(planted)
    assert code == 1
    assert any("eval_includes_comfyui_server must say true or false" in line for line in lines), lines


def test_negative_ram_fails(planted):
    ram = {"train": -1, "eval": None, "eval_includes_comfyui_server": None, "basis": "measured"}
    plant(planted, family("planted", host_ram_gb=ram))
    code, lines = check(planted)
    assert code == 1
    assert "families/planted.yaml: host_ram_gb.train must be a positive number of GB or null" in lines


def test_supported_training_needs_a_train_ram_number(planted):
    plant(planted, family("planted", training={"supported": True}))
    code, lines = check(planted)
    if OWN_BLOCK == "training":
        assert code == 1
        assert "families/planted.yaml: training.supported is true, so host_ram_gb.train must be a number" in lines
    else:
        assert code == 0, lines


# --- shape and unknown keys -----------------------------------------------------------------------------------------

def test_unknown_keys_warn_and_fail_only_when_strict(tmp_path):
    folder = tmp_path / "families"
    folder.mkdir()
    plant(folder, family("planted", colour="blue", **{OTHER_BLOCK: {}}))
    code, lines = check(folder)
    assert code == 0, lines
    assert "warn: families/planted.yaml: unknown key colour" in lines
    assert f"warn: families/planted.yaml: unknown key {OTHER_BLOCK}; it belongs in the other tool's copy of planted.yaml" in lines
    code, lines = check(folder, strict=True)
    assert code == 1
    assert "families/planted.yaml: unknown key colour" in lines


def test_wrong_schema_id_and_status_fail(planted):
    plant(planted, family("planted", schema="other/family/1", status="great"), name="renamed")
    code, lines = check(planted)
    assert code == 1
    assert f"families/renamed.yaml: schema must be '{PKG}/family/1'" in lines
    assert "families/renamed.yaml: id must equal the file name 'renamed'" in lines
    assert any(line.startswith("families/renamed.yaml: status must be one of") for line in lines), lines


def test_yaml_that_does_not_parse_fails(planted):
    (planted / "broken.yaml").write_text("id: [unclosed\n", encoding="utf-8")
    code, lines = check(planted)
    assert code == 1
    assert any(line.startswith("families/broken.yaml: does not parse as YAML") for line in lines), lines


def test_no_families_folder_exits_2(tmp_path):
    code, lines = check(tmp_path / "absent")
    assert code == 2
    assert len(lines) == 1
