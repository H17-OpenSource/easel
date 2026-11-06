"""`easel jobs submit|status|cancel|run` on the demo, with the runner driven in-process against the mock server."""

from __future__ import annotations

import json

import pytest
from studio_support import (
    RECIPE,
    demo_graph,
    native_env,
    png_bytes,
    prompts_sent,
    run_cli,
    schema_problems,
    serve_render,
)


@pytest.fixture
def home(demo_workspace, frozen_clock, mock_comfy, monkeypatch):
    native_env(monkeypatch, mock_comfy.url)
    return demo_workspace


def _job(home, job_id):
    return json.loads((home / "jobs" / f"{job_id}.json").read_text(encoding="utf-8"))


def test_submit_writes_a_waiting_job(home, mock_comfy, capsys):
    code, out, err = run_cli(["jobs", "submit", RECIPE, "--count", "2", "--by", "demo-agent", "--no-runner"], capsys)
    assert code == 0, err
    assert out.splitlines()[0].startswith("submitted job-20300101-002: easel run rcp-20300101-001 --count 2; "
                                          "waits for 8.5 GB free (family illustrious)")
    job = _job(home, "job-20300101-002")
    assert schema_problems("job", job) == []
    assert (job["status"], job["verb"], job["args"], job["by"]) == ("waiting", "run", [RECIPE, "--count", "2"],
                                                                     "demo-agent")
    assert job["scheduler"] == {"kind": "none", "ref": None} and job["target"] == {"recipe": RECIPE, "run": None}
    assert prompts_sent(mock_comfy) == []  # submitting renders nothing


def test_submit_starts_the_runner_hidden(home, monkeypatch, capsys):
    from easel._common import background

    started = []

    def fake_start(cmd, log, pid_file, cwd=None, env=None, **_):
        started.append(cmd)
        return background.Started(pid=4242, method="wmi", detail="test", pid_file=pid_file, log=log)

    monkeypatch.setattr(background, "start", fake_start)
    code, out, _ = run_cli(["jobs", "submit", RECIPE], capsys)
    assert code == 0
    assert started and started[0][1:5] == ["-m", "easel", "jobs", "run"] and "--workspace" in started[0]
    assert "runner: started (pid 4242, started via wmi)" in out


def test_status_and_cancel(home, capsys):
    run_cli(["jobs", "submit", RECIPE, "--no-runner"], capsys)
    code, out, _ = run_cli(["jobs", "status"], capsys)
    assert code == 0 and out.splitlines()[0].startswith("runner: not running")
    assert any(line.startswith("job-20300101-002  waiting") for line in out.splitlines())
    code, out, _ = run_cli(["jobs", "status", "job-20300101-002"], capsys)
    assert code == 0 and "  command: easel run rcp-20300101-001 --count 4" in out
    code, out, _ = run_cli(["jobs", "cancel", "job-20300101-002"], capsys)
    assert code == 0 and out.startswith("job-20300101-002 cancelled")
    assert _job(home, "job-20300101-002")["status"] == "cancelled"
    code, _out, err = run_cli(["jobs", "cancel", "job-20300101-002"], capsys)
    assert code == 1 and "nothing to cancel" in err
    code, _out, err = run_cli(["jobs", "cancel", "job-20300101-001"], capsys)
    assert code == 1 and "is done" in err


def test_the_runner_renders_a_job(home, mock_comfy, capsys):
    serve_render(mock_comfy, demo_graph(RECIPE), image=png_bytes(9))
    run_cli(["jobs", "submit", RECIPE, "--count", "1", "--no-runner"], capsys)
    code, out, err = run_cli(["jobs", "run"], capsys)
    assert code == 0, out + err
    assert out.splitlines()[-1] == "queue empty: the runner exits"
    job = _job(home, "job-20300101-002")
    assert schema_problems("job", job) == []
    assert (job["status"], job["exit_code"], job["target"]["run"]) == ("done", 0, "run-20300101-002")
    assert [h["status"] for h in job["history"]] == ["waiting", "running", "done"]
    run = json.loads((home / "runs" / "run-20300101-002" / "run.json").read_text(encoding="utf-8"))
    assert run["job"] == "job-20300101-002" and run["status"] == "ok" and len(run["outputs"]) == 1
    assert (home / "jobs" / "job-20300101-002.log").read_text(encoding="utf-8").strip()
    code, out, _ = run_cli(["jobs", "status", "job-20300101-002"], capsys)
    assert "  log tail (jobs/job-20300101-002.log):" in out


def test_the_runner_records_a_failed_render(home, mock_comfy, capsys):
    run_cli(["jobs", "submit", RECIPE, "--count", "1", "--no-runner"], capsys)
    code, _out, _err = run_cli(["jobs", "run", "--once"], capsys)  # the mock lists no model names: refused graph
    assert code == 1
    job = _job(home, "job-20300101-002")
    assert job["status"] == "failed" and job["exit_code"] == 1
    assert schema_problems("job", job) == []


def test_a_job_waits_without_room(home, monkeypatch, capsys):
    from easel._common import sysinfo

    monkeypatch.setenv("EASEL_LIMITS__SKIP_RAM_CHECK", "false")
    monkeypatch.setattr(sysinfo, "free_ram_gb", lambda *a, **k: 1.0)
    run_cli(["jobs", "submit", RECIPE, "--no-runner"], capsys)
    code, out, err = run_cli(["jobs", "run", "--no-wait"], capsys)
    assert code == 3
    assert "waiting: free 1.0 GB < 8.5 GB needed by family illustrious" in out + err
    assert _job(home, "job-20300101-002")["status"] == "waiting"


def test_submit_refuses_an_unapproved_recipe(home, capsys):
    run_cli(["recipe", "propose", "--text", "robot", "--hires", "none"], capsys)
    code, _out, err = run_cli(["jobs", "submit", "rcp-20300101-002", "--no-runner"], capsys)
    assert code == 2 and "it renders once the operator approved it" in err
    assert not (home / "jobs" / "job-20300101-002.json").exists()


def _command(code: int, printed: str = "") -> str:
    """A scheduler command template: this interpreter printing `printed` and exiting with `code` (no shell)."""
    import sys

    body = (f"print('{printed}'); " if printed else "") + f"raise SystemExit({code})"
    return f'"{sys.executable}" -c "{body}"'


def test_a_command_scheduler_plans_and_grants(home, mock_comfy, capsys):
    from studio_support import set_values

    set_values(home, {"scheduler.kind": "command", "scheduler.plan_command": _command(0, "demo-ref-1"),
                      "scheduler.wait_command": _command(0, "granted"), "scheduler.start_command": _command(0),
                      "scheduler.done_command": _command(0)})
    serve_render(mock_comfy, demo_graph(RECIPE))
    code, _out, err = run_cli(["jobs", "submit", RECIPE, "--count", "1", "--no-runner"], capsys)
    assert code == 0, err
    assert _job(home, "job-20300101-002")["scheduler"] == {"kind": "command", "ref": "demo-ref-1"}
    code, out, err = run_cli(["jobs", "run", "--once"], capsys)
    assert code == 0, out + err
    assert _job(home, "job-20300101-002")["status"] == "done"


def test_a_command_scheduler_that_ends_the_job(home, capsys):
    from studio_support import set_values

    set_values(home, {"scheduler.kind": "command", "scheduler.plan_command": _command(0, "demo-ref-2"),
                      "scheduler.wait_command": _command(3, "gone"), "scheduler.start_command": _command(0),
                      "scheduler.done_command": _command(0)})
    assert run_cli(["jobs", "submit", RECIPE, "--no-runner"], capsys)[0] == 0
    code, _out, _err = run_cli(["jobs", "run", "--once"], capsys)
    assert code == 2
    assert _job(home, "job-20300101-002")["status"] == "cancelled"


def test_an_unreachable_scheduler_fails_closed(home, capsys):
    from studio_support import set_values

    set_values(home, {"scheduler.kind": "command", "scheduler.plan_command": "no-such-program-" + "zz",
                      "scheduler.wait_command": _command(0), "scheduler.start_command": _command(0),
                      "scheduler.done_command": _command(0)})
    code, _out, err = run_cli(["jobs", "submit", RECIPE, "--no-runner"], capsys)
    assert code == 3
    assert "scheduler unreachable" in err and "failing closed" in err
    assert not (home / "jobs" / "job-20300101-002.json").exists()
