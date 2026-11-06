"""`easel registry list|show|explain|index` on the demo instance, and the verbs' refusals."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli

ANIME_XL, BASE_XL, ROBOT, INK, VAE, UPSCALER = ("bbbabeef2c0d57fe", "85306657134cbde8", "cf56db17414bcb55",
                                                "60d12866cf6377b0", "b4d12aa2b3121a5e", "99134ef8d78fb1bb")


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def test_list_prints_one_line_per_resource(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "list"], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines == [
        f"{ANIME_XL} | demo-anime-xl.safetensors | checkpoint | illustrious | sfw | active",
        f"{BASE_XL} | demo-base-xl.safetensors | checkpoint | sdxl | sfw | active",
        f"{INK} | demo-style-ink.safetensors | lora | sdxl | sfw | active",
        f"{ROBOT} | demo_robot-v1.safetensors | lora | illustrious | sfw | active",
        f"{VAE} | demo-xl-vae.safetensors | vae | sdxl | sfw | active",
        f"{UPSCALER} | demo-4x.safetensors | upscaler | - | sfw | active",
    ]


def test_list_filters_and_json(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "list", "--type", "lora", "--family", "sdxl"], capsys)
    assert code == 0 and out.splitlines() == [f"{INK} | demo-style-ink.safetensors | lora | sdxl | sfw | active"]
    code, out, _err = run_cli(["registry", "list", "--family", "illustrious", "--json"], capsys)
    rows = json.loads(out)
    assert code == 0 and [r["id"] for r in rows] == [ANIME_XL, ROBOT]
    assert set(rows[0]) >= {"id", "file_name", "type", "family", "rating", "status", "licence", "path"}
    code, out, err = run_cli(["registry", "list", "--family", "flux1"], capsys)
    assert code == 0 and out == "" and "no resource matches" in err


def test_show_names_the_loader_the_licence_and_the_prompt_policy(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "show", ROBOT], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == f"{ROBOT} demo_robot-v1.safetensors"
    assert lines[1] == "path: <models_root>/loras/demo_robot-v1.safetensors (status active)"
    assert lines[2].startswith("type: lora; loader: LoraLoader")
    assert "prompt text: allowed under policy.prompt_authoring = sfw-only (class sfw)" in lines
    assert any(line.startswith("licence: verified - Demo LoRA Licence; commercial use images") for line in lines)
    assert any("trained by kiln project demo_robot run run-20300102-001 step 800" in line for line in lines)
    assert lines[-1].startswith("overrides: provenance.source_class, semantics.category (registry/overrides.yaml")
    code, out, _err = run_cli(["registry", "show", UPSCALER], capsys)
    assert "type: upscaler; loader: UpscaleModelLoader, then ImageUpscaleWithModel" in out.splitlines()


def test_show_withholds_descriptive_text_of_a_non_sfw_resource(home: Path, capsys) -> None:
    path = home / "registry" / "overrides.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["resources"][ROBOT]["set"]["semantics.content.class"] = "unknown"
    data["resources"][ROBOT]["set"]["semantics.content.rating"] = "unknown"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    code, out, _err = run_cli(["registry", "show", ROBOT], capsys)
    assert code == 0
    assert "Demo Robot v1" not in out and "dmrbt" not in out and "visor" not in out
    assert "title: withheld (class unknown under policy.prompt_authoring = sfw-only)" in out
    assert ("prompt text: refused under policy.prompt_authoring = sfw-only (unknown counts as nsfw); name this "
            "resource by its id only") in out


def test_show_json_is_the_effective_record(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "show", INK, "--json"], capsys)
    record = json.loads(out)
    assert code == 0 and record["id"] == INK and record["schema"] == "easel/resource/1"


def test_explain_says_where_each_value_came_from(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "explain", ROBOT], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[1] == "header: safetensors header read"
    assert lines[2].startswith("type: lora <- folder loras/ (data/folders.yaml")
    assert "family: illustrious <- the record (an import, the person or an earlier scan; a scan never replaces a " \
           "family)" in lines
    assert "  the header suggests sdxl (ss_base_model_version sdxl_base_v1-0)" in lines
    assert any(line.startswith("compat source family: illustrious (lora.trained_on_family)") for line in lines)
    assert "rating: sfw, class sfw <- basis manual: the person" in lines
    assert "source class: self-trained <- registry/overrides.yaml (trained with kiln, project demo_robot)" in lines
    code, out, _err = run_cli(["registry", "explain", INK], capsys)
    assert "family: sdxl <- ss_base_model_version sdxl_base_v1-0" in out.splitlines()


def test_explain_of_a_file_that_is_gone(home: Path, capsys) -> None:
    (home / "models" / "vae" / "demo-xl-vae.safetensors").unlink()
    code, out, _err = run_cli(["registry", "explain", VAE], capsys)
    assert code == 0 and "header: the file is not under <models_root> any more" in out


def test_index_rebuilds_byte_identically(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "index"], capsys)
    assert code == 0 and out.strip() == "wrote registry/INDEX.md and registry/index.jsonl (6 resource(s))"
    first = ((home / "registry" / "INDEX.md").read_bytes(), (home / "registry" / "index.jsonl").read_bytes())
    assert run_cli(["registry", "index"], capsys)[0] == 0
    assert first == ((home / "registry" / "INDEX.md").read_bytes(), (home / "registry" / "index.jsonl").read_bytes())
    assert b"generated by `easel index`; do not edit" in first[0]


@pytest.mark.parametrize(("arg", "message"), [
    ("ffffffffffffffff", "no resource ffffffffffffffff in the registry"),
    ("xyz", "not a resource id"),
    ("cf56", "not a resource id"),
])
def test_an_unknown_or_malformed_id_exits_2(home: Path, capsys, arg: str, message: str) -> None:
    code, _out, err = run_cli(["registry", "show", arg], capsys)
    assert code == 2 and message in err


def test_an_ambiguous_prefix_exits_2(home: Path, capsys) -> None:
    src = home / "registry" / "resources" / f"{INK}.yaml"
    twin = "60d128660000000a"
    text = src.read_text(encoding="utf-8").replace(f"id: {INK}", f"id: {twin}").replace(
        "sha256: 60d12866cf6377b0", "sha256: 60d128660000000a")
    (home / "registry" / "resources" / f"{twin}.yaml").write_text(text, encoding="utf-8")
    code, _out, err = run_cli(["registry", "show", "60d12866"], capsys)
    assert code == 2 and "matches 2 resources" in err


def test_a_broken_record_is_a_warning_for_the_read_verbs(home: Path, capsys) -> None:
    (home / "registry" / "resources" / f"{VAE}.yaml").write_text("schema: easel/resource/1\nid: [\n",
                                                                  encoding="utf-8")
    code, out, err = run_cli(["registry", "list"], capsys)
    assert code == 0 and len(out.splitlines()) == 5
    assert f"warning: registry/resources/{VAE}.yaml: not readable" in err


def test_without_an_instance_the_verbs_exit_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    code, _out, err = run_cli(["registry", "list", "--workspace", str(tmp_path / "nowhere")], capsys)
    assert code == 2 and "no easel.toml" in err


def test_rehash_needs_hash(home: Path, capsys) -> None:
    code, _out, err = run_cli(["registry", "scan", "--rehash"], capsys)
    assert code == 2 and "--rehash goes with --hash" in err


@pytest.mark.parametrize("action", ["add", "download", "audit", "dups", "register-upstream"])
def test_write_actions_are_built(capsys, action: str) -> None:
    code, out, _err = run_cli(["registry", action, "--help"], capsys)
    assert code == 0 and "planned:" not in out
