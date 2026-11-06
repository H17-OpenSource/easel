"""The GPU verbs refuse on an adopted workspace (policy.render = legacy-runner) with the one to-legacy line."""

from __future__ import annotations

import pytest
from studio_support import RECIPE, closed_url, make_adopted, native_env, prompts_sent, run_cli, set_values

LINE = "adopted workspace: run the lines `easel recipe to-legacy <rcp>` prints (runbook 90)"


@pytest.fixture
def adopted(demo_workspace, frozen_clock, monkeypatch):
    make_adopted(demo_workspace)
    monkeypatch.setenv("EASEL_COMFYUI__URL", closed_url())
    return demo_workspace


@pytest.mark.parametrize("argv", [
    ["run", RECIPE],
    ["run", RECIPE, "--count", "1", "--no-wait"],
    ["jobs", "submit", RECIPE, "--count", "4", "--no-runner"],
    ["jobs", "run", "--once"],
    ["comfy", "start"],
    ["comfy", "stop"],
], ids=lambda argv: " ".join(argv[:2]))
def test_legacy_runner_refuses_gpu_verbs(adopted, argv, capsys):
    code, out, err = run_cli(argv, capsys)
    assert code == 2, out + err
    assert err.splitlines()[0] == LINE
    assert not (adopted / "runs" / "run-20300101-002").exists()
    assert not list((adopted / "jobs").glob("job-20300101-002*"))


def test_the_refusal_names_the_recipe(adopted, capsys):
    _code, _out, err = run_cli(["run", RECIPE], capsys)
    assert f"next: easel recipe to-legacy {RECIPE}" in err


def test_a_dry_run_stays_allowed_on_an_adopted_workspace(adopted, capsys):
    code, out, _ = run_cli(["run", RECIPE, "--dry-run"], capsys)
    assert code == 0 and "CheckpointLoaderSimple" in out and "nothing queued" in out


def test_jobs_allowed_must_list_the_verb(demo_workspace, frozen_clock, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    set_values(demo_workspace, {"jobs.allowed": "[]"})
    run_cli(["recipe", "approve", RECIPE], capsys)
    code, _out, err = run_cli(["jobs", "submit", RECIPE, "--no-runner"], capsys)
    assert code == 2
    assert err.startswith("REFUSED: easel run is not in jobs.allowed")
    assert prompts_sent(mock_comfy) == []


def test_comfy_start_needs_a_managed_lifecycle(demo_workspace, frozen_clock, monkeypatch, capsys):
    native_env(monkeypatch, closed_url())
    code, _out, err = run_cli(["comfy", "start"], capsys)
    assert code == 2 and "comfyui.manage_lifecycle = false" in err
    set_values(demo_workspace, {"comfyui.manage_lifecycle": "true"})
    code, _out, err = run_cli(["comfy", "start"], capsys)
    assert code == 2 and "comfyui.dir and comfyui.python must be set" in err
