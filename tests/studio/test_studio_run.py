"""`easel run` on the demo: the dry-run graph offline, and a whole render against the mock ComfyUI."""

from __future__ import annotations

import json

import pytest
from studio_support import (
    PROMPT_ID,
    RECIPE,
    RUN,
    demo_graph,
    history_entry,
    native_env,
    png_bytes,
    prompts_sent,
    run_cli,
    schema_problems,
    serve_render,
)


@pytest.fixture
def home(demo_workspace, frozen_clock):
    return demo_workspace


def _graph_and_summary(out: str) -> tuple[dict, str]:
    graph, end = json.JSONDecoder().raw_decode(out)
    return graph, out[end:].strip()


def _approved(home, capsys, *extra: str) -> str:
    code, _out, err = run_cli(["recipe", "propose", "--text", "robot, workshop", "--family", "illustrious", "--lora",
                               "cf56db17414bcb55", "--hires", "none", "--count", "2", "--seed", "7", *extra], capsys)
    assert code == 0, err
    code, _out, err = run_cli(["recipe", "approve", "rcp-20300101-002"], capsys)
    assert code == 0, err
    return "rcp-20300101-002"


def test_dry_run_prints_the_graph_on_the_demo(home, capsys):
    code, out, err = run_cli(["run", RECIPE, "--dry-run"], capsys)
    assert code == 0, err
    graph, summary = _graph_and_summary(out)
    classes = [node["class_type"] for node in graph.values()]
    assert classes[0] == "CheckpointLoaderSimple" and "KSampler" in classes and classes[-1] == "SaveImage"
    assert graph["1"]["inputs"]["ckpt_name"] == "demo-anime-xl.safetensors"
    lora = next(n for n in graph.values() if n["class_type"] == "LoraLoader")
    assert lora["inputs"]["lora_name"] == "demo_robot-v1.safetensors" and lora["inputs"]["strength_model"] == 0.8
    sampler = next(n for n in graph.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["seed"] == 1001 and sampler["inputs"]["steps"] == 28
    assert summary.startswith(f"(dry run: {len(graph)} nodes, 4 image(s), seeds 1001..1004, hires none")
    assert "nothing queued" in summary
    assert not any((home / "runs").glob("run-20300101-002"))


def test_dry_run_draft_drops_the_hires_pass(home, capsys):
    run_cli(["recipe", "propose", "--text", "robot", "--family", "illustrious", "--hires", "latent"], capsys)
    code, out, _ = run_cli(["run", "rcp-20300101-002", "--dry-run"], capsys)
    assert code == 0 and "LatentUpscaleBy" in out
    code, out, _ = run_cli(["run", "rcp-20300101-002", "--dry-run", "--draft"], capsys)
    graph, summary = _graph_and_summary(out)
    assert code == 0 and "LatentUpscaleBy" not in out and "draft" in summary
    assert sum(1 for n in graph.values() if n["class_type"] == "KSampler") == 1


def test_render_on_the_mock_server(home, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    rcp = _approved(home, capsys)
    serve_render(mock_comfy, demo_graph(rcp), image=png_bytes(3))
    code, out, err = run_cli(["run", rcp], capsys)
    assert code == 0, err + out
    folder = home / "runs" / "run-20300101-002"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert schema_problems("run", run) == [] and schema_problems("run-manifest", manifest) == []
    assert run["status"] == "ok" and run["recipe"] == rcp and run["prompt_id"] == PROMPT_ID
    assert [(o["n"], o["file"], o["seed"], o["width"]) for o in run["outputs"]] == \
        [(1, "outputs/001.png", 7, 8), (2, "outputs/002.png", 8, 8)]
    assert (folder / "outputs" / "001.png").read_bytes() == png_bytes(3)
    assert manifest["seeds"] == [7, 8] and manifest["comfyui"]["version"] == "0.43.1"
    assert [r["resource"] for r in manifest["resources"]] == ["bbbabeef2c0d57fe", "cf56db17414bcb55"]
    seeds = [next(n for n in g.values() if n["class_type"] == "KSampler")["inputs"]["seed"]
             for g in prompts_sent(mock_comfy)]
    assert seeds == [7, 8]
    assert json.loads((home / "recipes" / f"{rcp}.json").read_text(encoding="utf-8"))["status"] == "run"
    assert "next: ask the operator for a verdict per output" in out


def test_a_rerun_reproduces_and_compare_says_so(home, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    rcp = _approved(home, capsys)
    serve_render(mock_comfy, demo_graph(rcp), image=png_bytes(5))
    assert run_cli(["run", rcp], capsys)[0] == 0
    assert run_cli(["run", rcp], capsys)[0] == 0
    code, out, _ = run_cli(["runs", "compare", "run-20300101-002", "run-20300101-003"], capsys)
    assert code == 0 and out.splitlines()[-1] == "reproduced exactly"
    assert "output 1: identical pixels" in out
    code, out, _ = run_cli(["runs", "compare", RUN, "run-20300101-002"], capsys)
    assert code == 1 and "settings differ in" in out


def test_out_of_memory_is_recorded(home, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    rcp = _approved(home, capsys)
    serve_render(mock_comfy, demo_graph(rcp), entry=history_entry(error="CUDA out of memory. Tried to allocate"))
    code, _out, err = run_cli(["run", rcp], capsys)
    assert code == 1
    assert "run-20300101-002: oom" in err
    run = json.loads((home / "runs" / "run-20300101-002" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "oom" and run["error"] and schema_problems("run", run) == []


def test_server_refusing_a_model_name_fails_before_queueing(home, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    rcp = _approved(home, capsys)
    graph = demo_graph(rcp)
    graph["1"]["inputs"]["ckpt_name"] = "another.safetensors"
    serve_render(mock_comfy, graph)
    info = mock_comfy.routes[("GET", "/object_info/")][1]
    info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"] = [["another.safetensors"], {}]
    code, _out, err = run_cli(["run", rcp], capsys)
    assert code == 1 and "the server cannot run this graph" in err
    assert prompts_sent(mock_comfy) == []
    run = json.loads((home / "runs" / "run-20300101-002" / "run.json").read_text(encoding="utf-8"))
    assert run["status"] == "failed"


def test_an_unapproved_recipe_does_not_render(home, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    run_cli(["recipe", "propose", "--text", "robot", "--hires", "none"], capsys)
    code, _out, err = run_cli(["run", "rcp-20300101-002"], capsys)
    assert code == 2 and "it renders once the operator approved it" in err
    assert prompts_sent(mock_comfy) == []


def test_no_room_waits_and_no_wait_exits_3(home, mock_comfy, monkeypatch, capsys):
    from easel._common import sysinfo

    native_env(monkeypatch, mock_comfy.url)
    monkeypatch.setenv("EASEL_LIMITS__SKIP_RAM_CHECK", "false")
    monkeypatch.setattr(sysinfo, "free_ram_gb", lambda *a, **k: 1.0)
    rcp = _approved(home, capsys)
    code, out, err = run_cli(["run", rcp, "--no-wait"], capsys)
    assert code == 3
    assert "waiting: free 1.0 GB < 8.5 GB needed by family illustrious" in out + err
    assert not (home / "runs" / "run-20300101-002").exists()
    assert prompts_sent(mock_comfy) == []


def test_no_server_on_a_native_instance(home, monkeypatch, capsys):
    from studio_support import closed_url

    native_env(monkeypatch, closed_url())
    rcp = _approved(home, capsys)
    code, _out, err = run_cli(["run", rcp], capsys)
    assert code == 1 and "ComfyUI is not reachable" in err
    assert "the tool's default port is not necessarily the server's" in err
    assert not (home / "runs" / "run-20300101-002").exists()
