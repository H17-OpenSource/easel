"""`easel next`: the stage from the instance's state, three lines, and `--stage` alone."""

from __future__ import annotations

import json
import shutil

import pytest
from studio_support import RUN, closed_url, make_adopted, native_env, run_cli


@pytest.fixture
def home(demo_workspace, frozen_clock, mock_comfy, monkeypatch):
    native_env(monkeypatch, mock_comfy.url)
    return demo_workspace


def _next(capsys) -> dict:
    code, out, _ = run_cli(["next", "--json"], capsys)
    assert code == 0
    return json.loads(out)


def test_three_lines_and_stage_alone(home, capsys):
    code, out, _ = run_cli(["next"], capsys)
    lines = out.splitlines()
    assert code == 0 and len(lines) == 3
    assert lines == ["stage: 03-render", "read: docs/runbooks/03-render.md",
                     'next: easel recipe propose --text "<the scene>" --family <family> --hires '
                     '<none|latent|model> --explain']
    code, out, _ = run_cli(["next", "--stage"], capsys)
    assert code == 0 and out == "03-render\n"


def test_an_approved_recipe_without_a_run(home, capsys):
    run_cli(["recipe", "propose", "--text", "robot", "--hires", "none"], capsys)
    run_cli(["recipe", "approve", "rcp-20300101-002"], capsys)
    assert _next(capsys)["next"] == "easel run rcp-20300101-002 --dry-run"


def test_an_open_job(home, capsys):
    run_cli(["jobs", "submit", "rcp-20300101-001", "--no-runner"], capsys)
    answer = _next(capsys)
    assert (answer["stage"], answer["next"]) == ("03-render", "easel jobs status job-20300101-002")


def test_unreviewed_outputs_route_to_review(home, capsys):
    (home / "runs" / RUN / "reviews.jsonl").unlink()
    answer = _next(capsys)
    assert (answer["stage"], answer["next"]) == ("04-review-and-learn", f"easel runs show {RUN}")


def test_an_empty_registry_routes_to_resources(home, capsys):
    shutil.rmtree(home / "registry" / "resources")
    answer = _next(capsys)
    assert (answer["stage"], answer["next"]) == ("02-resources", "easel registry scan --hash")


def test_a_native_instance_without_a_server_is_setup(home, monkeypatch, capsys):
    monkeypatch.setenv("EASEL_COMFYUI__URL", closed_url())
    assert _next(capsys)["stage"] == "01-setup"


def test_an_adopted_instance_skips_the_server_and_routes_to_90(home, monkeypatch, capsys):
    make_adopted(home)
    monkeypatch.setenv("EASEL_COMFYUI__URL", closed_url())
    code, out, _ = run_cli(["next", "--stage"], capsys)
    assert code == 0 and out == "90-adopt-existing\n"
    (home / "runs" / RUN / "reviews.jsonl").unlink()
    assert _next(capsys)["stage"] == "90-adopt-existing"
    run_cli(["recipe", "propose", "--text", "robot", "--hires", "none"], capsys)
    run_cli(["recipe", "approve", "rcp-20300101-002"], capsys)
    shutil.rmtree(home / "runs" / RUN)  # no run waits for a verdict any more
    assert _next(capsys)["next"] == "easel recipe to-legacy rcp-20300101-002"
