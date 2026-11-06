"""Workflow templates: the three shipped ones pass `workflows check` and fill cleanly; planted faults are found;
register, validate (mock ComfyUI) and approve work on an instance."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from easel import workflows
from easel._common import comfy_graph
from easel.cli import main

SHIPPED = workflows.SHIPPED
RUN = "run-20300101-001"


def test_the_shipped_templates_pass_check_without_an_instance(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["workflows", "check"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert sorted(line.split(" | ")[0] for line in out if line.startswith("ok ")) == \
        ["ok detailer", "ok txt2img", "ok txt2img-lora-hires"]
    assert out[-1] == "workflows: 3 checked, 0 with problems"


def test_the_detailer_names_its_packs_and_their_licences():
    meta = yaml.safe_load((SHIPPED / "detailer" / "meta.yaml").read_text(encoding="utf-8"))
    packs = {p["name"]: p for p in meta["node_packs"]}
    assert packs["ComfyUI-Impact-Pack"]["licence"] == "GPL-3.0"
    assert packs["ComfyUI-Impact-Subpack"]["licence"] == "AGPL-3.0"
    assert packs["ComfyUI-Impact-Subpack"]["installs"] == [{"name": "ultralytics", "licence": "AGPL-3.0"}]
    assert meta["notes"].lower().startswith("unavailable without both node packs")
    readme = (SHIPPED / "detailer" / "README.md").read_text(encoding="utf-8")
    assert "unavailable without two custom node packs" in readme.lower()
    for wid in ("txt2img", "txt2img-lora-hires"):
        assert yaml.safe_load((SHIPPED / wid / "meta.yaml").read_text(encoding="utf-8"))["node_packs"] == []


@pytest.mark.parametrize("wid", ["txt2img", "txt2img-lora-hires", "detailer"])
def test_each_shipped_template_fills_without_a_placeholder_left(wid):
    t = workflows.load(SHIPPED / wid, "shipped")
    required = {s["name"]: f"demo-{s['name']}" for s in t.slots["slots"] if s["required"]}
    graph = comfy_graph.fill_template(t.graph, t.slots, required)
    text = json.dumps(graph)
    assert "<" not in text, f"{wid}: a placeholder survived filling"
    for name, value in required.items():
        assert value in text, f"{wid}: slot {name} was not filled"


def _instance_copy(home: Path, wid: str, new: str) -> Path:
    target = home / "workflows" / new
    shutil.copytree(SHIPPED / wid, target)
    meta = yaml.safe_load((target / "meta.yaml").read_text(encoding="utf-8"))
    slots = json.loads((target / "slots.json").read_text(encoding="utf-8"))
    meta["id"], slots["workflow"] = new, new
    (target / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    (target / "slots.json").write_text(json.dumps(slots), encoding="utf-8")
    return target


def test_planted_faults_are_found(demo_workspace, capsys):
    bad = _instance_copy(demo_workspace, "detailer", "bad-detailer")
    meta = yaml.safe_load((bad / "meta.yaml").read_text(encoding="utf-8"))
    meta["node_packs"] = [p for p in meta["node_packs"] if p["name"] != "ComfyUI-Impact-Subpack"]
    meta["node_packs"][0]["licence"] = "MIT"
    (bad / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    graph = json.loads((bad / "graph.json").read_text(encoding="utf-8"))
    graph["6"]["inputs"]["samples"] = ["99", 0]
    graph["10"] = {"class_type": "VAELoader", "inputs": {"vae_name": "hard-coded.safetensors"}}
    (bad / "graph.json").write_text(json.dumps(graph), encoding="utf-8")
    assert main(["workflows", "check", "bad-detailer"]) == 1
    out = capsys.readouterr().out
    assert "comes from ComfyUI-Impact-Subpack" in out
    assert "licence 'MIT' differs from the lock's 'GPL-3.0'" in out
    assert "links to node 99" in out
    assert "has no slot; a template never hard-codes a model file" in out


def test_register_validate_and_approve_on_an_instance(demo_workspace, frozen_clock, mock_comfy, capsys, tmp_path):
    src = workflows.load(SHIPPED / "txt2img", "shipped")
    (tmp_path / "mine.json").write_text(json.dumps(src.graph), encoding="utf-8")
    (tmp_path / "mine.slots.json").write_text(json.dumps({"slots": src.slots["slots"]}), encoding="utf-8")
    assert main(["workflows", "register", str(tmp_path / "mine.json"), "--slots", str(tmp_path / "mine.slots.json"),
                 "--task", "txt2img", "--family", "sdxl"]) == 0
    assert "registered mine as draft in workflows/mine" in capsys.readouterr().out
    meta_file = demo_workspace / "workflows" / "mine" / "meta.yaml"
    assert yaml.safe_load(meta_file.read_text(encoding="utf-8"))["status"] == "draft"

    info = {node["class_type"]: {"input": {"required": {k: ["STRING", {}] for k in node["inputs"]}}}
            for node in src.graph.values()}
    mock_comfy.routes[("GET", "/object_info/")] = (200, info)
    toml = demo_workspace / "easel.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace("http://127.0.0.1:8188", mock_comfy.url),
                    encoding="utf-8")
    assert main(["workflows", "validate", "mine"]) == 0
    assert "valid on the server (ComfyUI 0.43.1" in capsys.readouterr().out
    meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert meta["validated_at"] == "2030-01-01T00:00:00+00:00"

    assert main(["workflows", "approve", "mine", "--run", RUN, "--note", "looks right"]) == 2
    assert "did not use the template" in capsys.readouterr().err
    run_json = demo_workspace / "runs" / RUN / "run.json"
    run = json.loads(run_json.read_text(encoding="utf-8"))
    run_json.write_text(json.dumps({**run, "workflow": "mine"}), encoding="utf-8")
    assert main(["workflows", "approve", "mine", "--run", RUN, "--note", "looks right"]) == 0
    meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
    assert meta["status"] == "approved" and meta["validator_note"] == f"{RUN}: looks right"
    assert main(["workflows", "check", "mine"]) == 0


def test_validate_names_a_missing_node_class(demo_workspace, mock_comfy, capsys):
    mock_comfy.routes[("GET", "/object_info/")] = (200, {})
    toml = demo_workspace / "easel.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace("http://127.0.0.1:8188", mock_comfy.url),
                    encoding="utf-8")
    assert main(["workflows", "validate", "detailer"]) == 1
    assert "the server has no node class FaceDetailer" in capsys.readouterr().out


def test_approving_a_shipped_template_copies_it_into_the_instance(demo_workspace, frozen_clock, capsys):
    run_json = demo_workspace / "runs" / RUN / "run.json"
    run = json.loads(run_json.read_text(encoding="utf-8"))
    run_json.write_text(json.dumps({**run, "workflow": "txt2img"}), encoding="utf-8")
    before = (SHIPPED / "txt2img" / "meta.yaml").read_bytes()
    assert main(["workflows", "approve", "txt2img", "--run", RUN, "--note", "the operator says it works"]) == 0
    copy = yaml.safe_load((demo_workspace / "workflows" / "txt2img" / "meta.yaml").read_text(encoding="utf-8"))
    assert copy["status"] == "approved" and copy["approved_at"] == "2030-01-01T00:00:00+00:00"
    assert (SHIPPED / "txt2img" / "meta.yaml").read_bytes() == before
    assert main(["workflows", "list"]) == 0
    assert "txt2img | approved | txt2img | sdxl | instance" in capsys.readouterr().out
