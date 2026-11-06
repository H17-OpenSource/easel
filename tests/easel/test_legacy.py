"""The adopted-workspace bridges: `recipe to-legacy` prints the host's own lines from config, `runs import-legacy`
reads a host run folder (and, with --with-reviews, its review rows read-only) into easel's run records.

Everything host-shaped here is synthetic and built by the test: the command templates, the host folder with its
manifest and generated 8x8 PNGs, and a review table in a temporary SQLite file. Host ids are assembled at run time.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from easel import legacy
from easel._common import hashing
from easel.registry.shipped import schema_errors

REPO = Path(__file__).resolve().parents[2]
RCP = "rcp-20300101-001"
HOST_RUN = "".join(("G", "-", "20300102", "-", "004"))   # a host id of the dated shape, built at run time
HOST_RECIPE = "".join(("R", "-", "20300102", "-", "002"))
TEMPLATES = {
    "propose_command": 'python "{tools}/host_recipe.py" propose --from-json {recipe_json} --save',
    "dryrun_command": 'python "{tools}/host_studio.py" run {recipe_id} --dry-run',
    "submit_command": 'python "{tools}/host_jobs.py" submit --by <agent> --title {recipe_id} '
                      '--cmd "{tools}/host_studio.py run {recipe_id} --count {count}" --minutes 15 --gb 3.7',
    "status_command": 'python "{tools}/host_jobs.py" status {job_id}',
}


def _fixtures():
    spec = importlib.util.spec_from_file_location("make_fixtures", REPO / "tests" / "fixtures" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_adopt(home: Path, values: dict[str, str]) -> None:
    """Fill the demo's empty [adopt] table (literal TOML strings, forward slashes)."""
    toml = home / "easel.toml"
    lines = "".join(f"{k} = '{v}'\n" for k, v in values.items())
    text = toml.read_text(encoding="utf-8")
    assert "[adopt]\n" in text
    toml.write_text(text.replace("[adopt]\n", "[adopt]\n" + lines, 1), encoding="utf-8")


def _templates(tools: str) -> dict[str, str]:
    # `{tools}` is the test's own stand-in for a host folder; the code never sees it as a placeholder.
    return {k: v.replace("{tools}", tools) for k, v in TEMPLATES.items()}


def _to_legacy(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="easel recipe to-legacy")
    legacy.add_to_legacy_args(parser)
    args = parser.parse_args(list(argv))
    args.tool = "easel"
    return args


def _import(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="easel runs import-legacy")
    legacy.add_import_legacy_args(parser)
    args = parser.parse_args(list(argv))
    args.tool = "easel"
    return args


# --- to-legacy -------------------------------------------------------------------------------------------------------

def test_to_legacy_prints_the_four_config_templates_in_order(demo_workspace, frozen_clock, capsys, tmp_path):
    tools = (tmp_path / "host tools").as_posix()
    templates = _templates(tools)
    _set_adopt(demo_workspace, templates)
    code = legacy.run_to_legacy(_to_legacy(RCP, "--host-recipe", HOST_RECIPE, "--job-id", "J-7", "--count", "3"))
    out = capsys.readouterr()
    assert code == 0, out.err
    lines = out.out.splitlines()
    request = (demo_workspace / "legacy" / "requests" / f"{RCP}.json").as_posix()
    values = {"{recipe_json}": request, "{recipe_id}": HOST_RECIPE, "{count}": "3", "{job_id}": "J-7"}
    expected = []
    for key in ("propose_command", "dryrun_command", "submit_command", "status_command"):
        line = templates[key]
        for placeholder, value in values.items():
            line = line.replace(placeholder, value)
        expected.append(line)
    assert lines == expected  # every printed character comes from the config or from the values given


def test_to_legacy_writes_the_request_the_propose_line_reads(demo_workspace, frozen_clock, capsys, tmp_path):
    _set_adopt(demo_workspace, _templates(tmp_path.as_posix()))
    assert legacy.run_to_legacy(_to_legacy(RCP)) == 0
    capsys.readouterr()
    request = json.loads((demo_workspace / "legacy" / "requests" / f"{RCP}.json").read_text(encoding="utf-8"))
    recipe = json.loads((demo_workspace / "recipes" / f"{RCP}.json").read_text(encoding="utf-8"))
    assert request["text"] == recipe["prompt"]
    assert request["count"] == recipe["params"]["count"]
    assert request["size"] == f"{recipe['params']['width']}x{recipe['params']['height']}"
    assert request["hires"] is False and request["rating"] == "sfw"
    assert request["loras"] == [f"{_stem(demo_workspace, 'cf56db17414bcb55')}:0.8"]
    assert request["checkpoint"] == _stem(demo_workspace, recipe["checkpoint"])


def _stem(home: Path, sha16: str) -> str:
    import yaml

    record = yaml.safe_load((home / "registry" / "resources" / f"{sha16}.yaml").read_text(encoding="utf-8"))
    return Path(record["identity"]["file_name"]).stem


def test_to_legacy_leaves_unknown_ids_as_placeholders(demo_workspace, frozen_clock, capsys, tmp_path):
    templates = _templates(tmp_path.as_posix())
    _set_adopt(demo_workspace, templates)
    assert legacy.run_to_legacy(_to_legacy(RCP)) == 0
    out = capsys.readouterr()
    lines = out.out.splitlines()
    assert lines[1] == templates["dryrun_command"].replace("{recipe_id}", legacy.HOST_RECIPE_PLACEHOLDER)
    assert lines[3] == templates["status_command"].replace("{job_id}", legacy.JOB_PLACEHOLDER)
    assert "--host-recipe" in out.err and "<agent>" in out.err


def test_to_legacy_names_every_missing_adopt_key(demo_workspace, capsys, tmp_path):
    templates = _templates(tmp_path.as_posix())
    del templates["submit_command"], templates["status_command"]
    _set_adopt(demo_workspace, templates)
    assert legacy.run_to_legacy(_to_legacy(RCP)) == 2
    err = capsys.readouterr().err
    assert "[adopt] submit_command is not set" in err and "[adopt] status_command is not set" in err
    assert "propose_command" not in err


def test_to_legacy_on_the_plain_demo_exits_2(demo_workspace, capsys):
    assert legacy.run_to_legacy(_to_legacy(RCP)) == 2
    assert "[adopt] propose_command is not set" in capsys.readouterr().err


def test_to_legacy_refuses_an_unknown_placeholder_and_an_unapproved_recipe(demo_workspace, capsys, tmp_path):
    templates = _templates(tmp_path.as_posix())
    templates["status_command"] += " {nope}"
    _set_adopt(demo_workspace, templates)
    assert legacy.run_to_legacy(_to_legacy(RCP)) == 2
    assert "{nope}" in capsys.readouterr().err
    path = demo_workspace / "recipes" / f"{RCP}.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps({**recipe, "status": "proposed"}), encoding="utf-8")
    assert legacy.run_to_legacy(_to_legacy(RCP)) == 2
    assert "approve" in capsys.readouterr().err


def test_to_legacy_refuses_an_id_that_would_change_the_command(demo_workspace, capsys, tmp_path):
    _set_adopt(demo_workspace, _templates(tmp_path.as_posix()))
    assert legacy.run_to_legacy(_to_legacy(RCP, "--job-id", "J-1; echo")) == 2
    assert "command line" in capsys.readouterr().err


def test_recipe_verb_registers_to_legacy_when_built(demo_workspace, capsys, tmp_path):
    from easel import cli

    if not cli.module_exists("easel.commands.recipe"):
        pytest.skip("easel recipe is not built yet (T3)")
    templates = _templates(tmp_path.as_posix())
    _set_adopt(demo_workspace, templates)
    code = cli.main(["recipe", "to-legacy", RCP, "--host-recipe", HOST_RECIPE, "--job-id", "J-7"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert len(out.out.splitlines()) == 4


# --- import-legacy ---------------------------------------------------------------------------------------------------

@pytest.fixture
def host(demo_workspace, tmp_path):
    """A synthetic host: one finished run folder with three generated PNGs, and its catalog with reviews."""
    fx = _fixtures()
    out_root = tmp_path / "host" / "output"
    folder = out_root / "shelf-a" / "box-1" / HOST_RUN  # any layout: import-legacy finds the folder by the id
    folder.mkdir(parents=True)
    outputs = []
    for i, n in enumerate((5, 6, 7), 1):
        data = fx.png_bytes(n)
        (folder / f"{i:05d}.png").write_bytes(data)
        outputs.append({"file": f"{i:05d}.png", "sha256": hashing.sha256_bytes(data), "width": 8, "height": 8,
                        "bytes": len(data)})
    (folder / "graph.json").write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {}}}),
                                       encoding="utf-8")
    ckpt = "bbbabeef2c0d57fe"
    manifest = {"run_id": HOST_RUN, "recipe_id": HOST_RECIPE, "created_at": "2030-01-02T10:00:00", "task": "txt2img",
                "comfyui": {"version": "0.43.1", "prompt_id": "00000000-0000-4000-8000-000000000009"},
                "workflow": {"compiler": "sdxl_txt2img", "graph_file": "graph.json"},
                "prompt": "dmrbt, robot, waving in a tidy workshop", "negative": "lowres", "seed": 2002, "batch": 3,
                "width": 832, "height": 1216, "steps": 28, "cfg": 5.5, "sampler": "euler_ancestral",
                "scheduler": "normal", "denoise": 1.0, "duration_s": 61.5,
                "artifacts": [{"artifact_id": "x1", "role": "base", "slot": "", "weight": None,
                               "filename": "illustrious/demo.safetensors", "sha256": ckpt + "0" * 48},
                              {"artifact_id": "x2", "role": "lora", "slot": "lora.1", "weight": 0.8,
                               "filename": "loras/demo_robot.safetensors", "sha256": None}],
                "outputs": outputs}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    db = tmp_path / "host" / "catalog.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE run_output (run_output_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, rel_path TEXT, "
        "bytes INTEGER, sha256 TEXT, width INTEGER, height INTEGER, metadata_stripped INTEGER DEFAULT 0);"
        "CREATE TABLE run_review (review_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
        "run_output_id INTEGER, verdict TEXT, checks TEXT, note TEXT, source TEXT DEFAULT 'chat', created_at TEXT);")
    other = "".join(("G", "-", "20300102", "-", "003"))
    conn.execute("INSERT INTO run_output (run_id, rel_path, sha256) VALUES (?, ?, ?)", (other, "x.png", "0" * 64))
    for o in outputs:
        conn.execute("INSERT INTO run_output (run_id, rel_path, bytes, sha256, width, height) VALUES (?,?,?,?,?,?)",
                     (HOST_RUN, o["file"], o["bytes"], o["sha256"], 8, 8))
    rows = [(HOST_RUN, 2, "keep", json.dumps({"face": "pass", "hands": "na"}), None, "app", "2030-01-02 11:00:00"),
            (HOST_RUN, 3, "reject", json.dumps({"face": "fail"}), "the visor split", "chat", "2030-01-02 11:01:00"),
            (HOST_RUN, 4, "maybe", None, None, "app", "2030-01-02 11:02:00"),
            (HOST_RUN, 4, "keep", None, "on a second look it is fine", "app", "2030-01-02 11:05:00"),
            (other, 1, "reject", None, None, "app", "2030-01-02 11:06:00")]
    conn.executemany("INSERT INTO run_review (run_id, run_output_id, verdict, checks, note, source, created_at) "
                     "VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()
    _set_adopt(demo_workspace, {**_templates(tmp_path.as_posix()), "output_root": out_root.as_posix(),
                                "run_registry": db.as_posix()})
    return {"home": demo_workspace, "folder": folder, "db": db, "outputs": outputs, "out_root": out_root}


def test_import_writes_a_schema_valid_run_with_copied_outputs(host, frozen_clock, capsys):
    assert legacy.run_import_legacy(_import(HOST_RUN)) == 0
    out = capsys.readouterr().out
    run_dir = host["home"] / "runs" / "run-20300102-004"
    assert out.splitlines()[0] == f"run: run-20300102-004 (legacy {HOST_RUN})"
    assert out.splitlines()[1] == f"imported {HOST_RUN} as run-20300102-004: 3 of 3 outputs copied to runs/run-20300102-004/outputs/"
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert schema_errors("run", run) == []
    assert run["legacy_id"] == HOST_RUN and run["status"] == "ok" and run["count"] == 3
    assert run["imported_from"] == {"kind": "legacy", "manifest": f"shelf-a/box-1/{HOST_RUN}/manifest.json"}
    for item, src in zip(run["outputs"], host["outputs"], strict=True):
        assert item["sha256"] == src["sha256"] and (item["width"], item["height"]) == (8, 8)
        assert hashing.sha256_file(run_dir / item["file"]) == src["sha256"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert schema_errors("run-manifest", manifest) == []
    assert [r["role"] for r in manifest["resources"]] == ["checkpoint", "lora"]
    assert manifest["resources"][0]["resource"] == "bbbabeef2c0d57fe"
    assert (run_dir / "graph.json").is_file() and manifest["graph_sha256"]


def test_import_with_reviews_imports_each_row_once_and_leaves_the_host_db_alone(host, frozen_clock, capsys):
    before = hashing.sha256_file(host["db"])
    assert legacy.run_import_legacy(_import(HOST_RUN, "--with-reviews")) == 0
    assert "4 reviews imported" in capsys.readouterr().out.splitlines()
    path = host["home"] / "runs" / "run-20300102-004" / "reviews.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 4
    assert all(schema_errors("run-review-event", line) == [] for line in lines)
    assert [(x["n"], x["verdict"]) for x in lines] == [(1, "keep"), (2, "reject"), (3, "maybe"), (3, "keep")]
    assert lines[0]["checks"] == {"face": "pass", "hands": "na"} and lines[1]["via"] == "chat"
    assert {x["source"] for x in lines} == {"operator"} and all(x["legacy_id"] for x in lines)
    assert hashing.sha256_file(host["db"]) == before  # opened mode=ro: not one byte changed

    assert legacy.run_import_legacy(_import(HOST_RUN, "--with-reviews")) == 0
    out = capsys.readouterr()
    assert "0 reviews imported" in out.out.splitlines() and "already holds" in out.out
    assert "stop:" in out.err
    assert len(path.read_text(encoding="utf-8").splitlines()) == 4

    conn = sqlite3.connect(host["db"])
    conn.execute("INSERT INTO run_review (run_id, run_output_id, verdict, source, created_at) VALUES (?,?,?,?,?)",
                 (HOST_RUN, None, "keep", "cli", "2030-01-02 12:00:00"))
    conn.commit()
    conn.close()
    assert legacy.run_import_legacy(_import(HOST_RUN, "--with-reviews")) == 0
    assert "1 reviews imported" in capsys.readouterr().out.splitlines()
    last = json.loads(path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["n"] is None and last["via"] == "cli"


def test_zero_reviews_is_the_stop_line(host, capsys):
    conn = sqlite3.connect(host["db"])
    conn.execute("DELETE FROM run_review WHERE run_id = ?", (HOST_RUN,))
    conn.commit()
    conn.close()
    assert legacy.run_import_legacy(_import(HOST_RUN, "--with-reviews")) == 0
    out = capsys.readouterr()
    assert "0 reviews imported" in out.out.splitlines()
    assert "not reviewed" in out.err


def test_reimport_restores_a_missing_output_copy(host, capsys):
    assert legacy.run_import_legacy(_import(HOST_RUN)) == 0
    copy = host["home"] / "runs" / "run-20300102-004" / "outputs" / "002.png"
    copy.unlink()
    assert legacy.run_import_legacy(_import(HOST_RUN)) == 0
    assert "1 missing output copy added" in capsys.readouterr().out
    assert hashing.sha256_file(copy) == host["outputs"][1]["sha256"]


def test_import_refuses_a_missing_manifest_and_a_collision(host, capsys):
    absent = "".join(("G", "-", "20300102", "-", "099"))
    assert legacy.run_import_legacy(_import(absent)) == 2
    assert f"no folder {absent}/ with a manifest.json under <adopt.output_root>" in capsys.readouterr().err
    taken = host["home"] / "runs" / "run-20300102-004"
    taken.mkdir(parents=True)
    (taken / "run.json").write_text(json.dumps({"legacy_id": "something else"}), encoding="utf-8")
    assert legacy.run_import_legacy(_import(HOST_RUN)) == 2
    assert "holds another run" in capsys.readouterr().err


@pytest.mark.parametrize("parts", [(), ("a",), ("a", "b"), ("a", "b", "c")])
def test_find_manifest_takes_any_layout_up_to_three_levels(tmp_path, parts):
    """No host folder scheme is built in: the run folder is found by its id at any depth from 0 to 3."""
    folder = tmp_path.joinpath(*parts, HOST_RUN)
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text("{}", encoding="utf-8")
    assert legacy.find_manifest(tmp_path, HOST_RUN) == folder / "manifest.json"


def test_find_manifest_refuses_too_deep_and_ambiguous(tmp_path):
    deep = tmp_path.joinpath("a", "b", "c", "d", HOST_RUN)
    deep.mkdir(parents=True)
    (deep / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(legacy.LegacyError, match="searched 3 levels down"):
        legacy.find_manifest(tmp_path, HOST_RUN)
    for parts in (("x",), ("y", "z")):
        twin = tmp_path.joinpath(*parts, HOST_RUN)
        twin.mkdir(parents=True)
        (twin / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(legacy.LegacyError, match="has 2 manifests"):
        legacy.find_manifest(tmp_path, HOST_RUN)


def test_with_reviews_needs_a_catalog_with_run_review(host, tmp_path, capsys):
    other = tmp_path / "not-a-catalog.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE training_run (run_id TEXT)")
    conn.commit()
    conn.close()
    toml = host["home"] / "easel.toml"
    text = toml.read_text(encoding="utf-8").replace(host["db"].as_posix(), other.as_posix())
    toml.write_text(text, encoding="utf-8")
    assert legacy.run_import_legacy(_import(HOST_RUN, "--with-reviews")) == 2
    assert "no run_review table" in capsys.readouterr().err


def test_import_needs_adopt_output_root(demo_workspace, capsys):
    assert legacy.run_import_legacy(_import(HOST_RUN)) == 2
    assert "[adopt] output_root is not set" in capsys.readouterr().err


def test_runs_verb_registers_import_legacy_when_built(host, frozen_clock, capsys):
    from easel import cli

    if not cli.module_exists("easel.commands.runs"):
        pytest.skip("easel runs is not built yet (T3)")
    code = cli.main(["runs", "import-legacy", HOST_RUN, "--with-reviews"])
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "4 reviews imported" in out.out.splitlines()


def test_run_ids_follow_the_export_mapping():
    assert legacy.run_id_for(HOST_RUN) == "run-20300102-004"
    assert legacy.run_id_for("Host Run 7", "2030-01-03T09:00:00") == "run-20300103-host-run-7"
