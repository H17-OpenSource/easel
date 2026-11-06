"""`easel comfy status|start|stop` without ever starting a process."""

from __future__ import annotations

from pathlib import Path

from studio_support import closed_url, make_adopted, native_env, run_cli, set_values


def test_status_up_and_down(demo_workspace, frozen_clock, mock_comfy, monkeypatch, capsys):
    native_env(monkeypatch, mock_comfy.url)
    code, out, _ = run_cli(["comfy", "status"], capsys)
    assert code == 0
    assert out.splitlines()[0] == f"comfyui: up at {mock_comfy.url}, version 0.43.1, python 3.12.6"
    assert "  queue: 0 running, 0 pending" in out.splitlines()
    url = closed_url()
    monkeypatch.setenv("EASEL_COMFYUI__URL", url)
    code, out, _ = run_cli(["comfy", "status"], capsys)
    assert code == 1 and out.splitlines()[0] == f"comfyui: not running at {url}"


def test_status_on_an_adopted_instance(demo_workspace, frozen_clock, monkeypatch, capsys):
    make_adopted(demo_workspace)
    url = closed_url()
    monkeypatch.setenv("EASEL_COMFYUI__URL", url)
    code, out, _ = run_cli(["comfy", "status"], capsys)
    assert code == 0
    assert out.splitlines()[0] == f"comfyui: not running at {url} (the host runner starts it per job)"


def test_stop_without_a_pid_file(demo_workspace, frozen_clock, monkeypatch, capsys):
    native_env(monkeypatch, closed_url())
    code, out, _ = run_cli(["comfy", "stop"], capsys)
    assert code == 1 and out.startswith("no pid file: easel did not start the running server")


def test_start_command_points_outputs_into_the_instance(demo_workspace, frozen_clock, monkeypatch, tmp_path):
    from easel.studio import common, lifecycle

    native_env(monkeypatch, "http://127.0.0.1:8188")
    comfy = tmp_path / "comfy"
    comfy.mkdir()
    (comfy / "main.py").write_text("", encoding="utf-8")
    python = tmp_path / "python.bin"
    python.write_text("", encoding="utf-8")
    set_values(demo_workspace, {"comfyui.manage_lifecycle": "true", "comfyui.dir": str(comfy),
                                "comfyui.python": str(python)})
    studio = common.open_studio("easel")
    folder, interpreter = lifecycle.check_can_manage(studio)
    cmd = lifecycle.start_command(studio, folder, interpreter)
    assert cmd[:4] == [str(python), "-u", "main.py", "--listen"]
    assert cmd[4:7] == ["127.0.0.1", "--port", "8188"]
    out_dir = Path(cmd[cmd.index("--output-directory") + 1])
    assert out_dir == demo_workspace / "runtime" / "comfyui" / "output"
    assert cmd[-2] == "--input-directory"  # the shipped default start_args = [] adds nothing
    set_values(demo_workspace, {"comfyui.start_args": '["--lowvram", "--preview-method", "none"]'})
    cmd = lifecycle.start_command(common.open_studio("easel"), folder, interpreter)
    assert cmd[-3:] == ["--lowvram", "--preview-method", "none"]  # the instance's extra flags come last
