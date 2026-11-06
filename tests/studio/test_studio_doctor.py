"""`easel doctor`: adopted with the server down is exit 0 with the two literal lines; native states."""

from __future__ import annotations

import os

from studio_support import closed_url, make_adopted, native_env, run_cli, set_values


def test_adopted_with_the_server_down_is_ok(demo_workspace, frozen_clock, monkeypatch, capsys):
    make_adopted(demo_workspace)
    url = closed_url()
    monkeypatch.setenv("EASEL_COMFYUI__URL", url)
    code, out, err = run_cli(["doctor"], capsys)
    lines = out.splitlines()
    assert code == 0, out + err
    assert f"comfyui: not running at {url} (the host runner starts it per job) ... ok" in lines
    assert "comfyui version: unknown until a job has run (lock says 0.43.0) ... warn" in lines
    assert not any(line.startswith("FAIL") for line in lines)
    assert "instance: adopted (policy.render = legacy-runner: the host runner renders) ... ok" in lines
    assert "adopt: the four host command templates are set ... ok" in lines
    assert lines[-1] == "next: easel next"


def test_native_with_the_server_up(demo_workspace, frozen_clock, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    code, out, _ = run_cli(["doctor"], capsys)
    lines = out.splitlines()
    assert code == 0, out
    assert f"comfyui: up at {mock_comfy.url} (version 0.43.1) ... ok" in lines
    assert "comfyui version: 0.43.1 (lock says 0.43.1) ... ok" in lines
    assert "registry: 6 resources ... ok" in lines
    assert "jobs: easel run allowed (jobs.allowed) ... ok" in lines


def test_native_with_no_server_is_a_fail(demo_workspace, frozen_clock, monkeypatch, capsys):
    url = closed_url()
    native_env(monkeypatch, url)
    code, out, _ = run_cli(["doctor"], capsys)
    lines = out.splitlines()
    assert code == 1
    assert f"FAIL: comfyui: nothing answers at {url}" in lines
    assert any("the tool's default port is not necessarily the server's" in line for line in lines)
    assert lines[-1].startswith("next: report to the operator (the server is theirs to start)")
    assert not any("restart" in line or "start the server" in line for line in lines)  # agents never run the server


def test_an_older_server_is_a_fail(demo_workspace, frozen_clock, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    mock_comfy.routes[("GET", "/system_stats")] = (200, {"system": {"comfyui_version": "0.40.0"}, "devices": []})
    code, out, _ = run_cli(["doctor"], capsys)
    assert code == 1
    assert "FAIL: comfyui version: 0.40.0 is older than comfyui.min_version 0.43.0" in out.splitlines()


def test_an_unset_variable_is_a_fail_line(demo_workspace, frozen_clock, mock_comfy, monkeypatch, capsys):
    name = "EASEL_TEST_" + "UNSET_" + str(os.getpid())
    monkeypatch.delenv(name, raising=False)
    native_env(monkeypatch, mock_comfy.url)
    set_values(demo_workspace, {"paths.models_root": "${" + name + "}"})
    code, out, _ = run_cli(["doctor"], capsys)
    assert code == 1
    assert f"FAIL: ${{{name}}} unset (paths.models_root)" in out.splitlines()


def test_no_instance_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    code, out, _ = run_cli(["doctor"], capsys)
    assert code == 1
    assert any(line.startswith("FAIL: config: no easel.toml found") for line in out.splitlines())


def test_no_instance_next_line_names_only_a_built_path(tmp_path, monkeypatch, capsys):
    """While `easel init` is planned, `next:` names the operator's copy of the example file, never the stub."""
    from easel import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    _code, out, _ = run_cli(["doctor"], capsys)
    assert "FAIL: config: no easel.toml found (looked at --workspace, $EASEL_HOME, " in out
    assert str(tmp_path) not in out and str(tmp_path.resolve()) not in out  # where it looked, in words only
    last = out.splitlines()[-1]
    if cli.planned_task("init") is None:
        assert last.startswith("next: create one with `easel init <folder>`")
    else:
        assert last.startswith("next: the operator copies easel.example.toml to <home>/easel.toml")
        assert f"(`easel init` is planned: {cli.planned_task('init')})" in last
    monkeypatch.setattr(cli, "module_exists", lambda name: name != "easel.commands.init")
    assert cli.new_instance_hint().startswith("the operator copies easel.example.toml")
    monkeypatch.setattr(cli, "module_exists", lambda name: True)
    assert cli.new_instance_hint().startswith("create one with `easel init <folder>`")
