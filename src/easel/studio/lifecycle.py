"""The ComfyUI server's lifecycle: is it there, start it hidden, stop what easel started.

ComfyUI is a separate program (GPL-3.0) that easel talks to over HTTP only. Whether easel may start and stop it
is the instance's choice: `comfyui.manage_lifecycle = true` with `comfyui.dir` (the folder holding `main.py`)
and `comfyui.python` (its interpreter) set. Otherwise easel only asks whether a server answers at `comfyui.url`
and leaves starting it to the operator or an external scheduler ([scheduler]). On an adopted workspace
(`policy.render = "legacy-runner"`) the host runner starts the server per job, so `start` and `stop` refuse.

A start goes through `_common.background` (WMI first on Windows, so the server outlives the shell that started
it), writes `<runtime>/comfyui.pid` and its sidecar (`started via wmi|detached`), and sends the server's own
outputs and inputs to `<runtime>/comfyui/{output,input}` instead of its install folder.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable
from pathlib import Path

from easel._common import background, ids
from easel._common.comfy_client import Client

from .common import EXIT_PROBLEMS, EXIT_USAGE, REFUSED_LEGACY, Studio, StudioError

START_WAIT_S = 300
POLL_S = 2.0
PORT_LESSON = ("the tool's default port is not necessarily the server's: when the operator names another address, "
               "easel config set comfyui.url <that address>")

Say = Callable[[str], None]


def client(studio: Studio) -> Client:
    """A client for the instance's server (`comfyui.url`, `comfyui.timeout_s`)."""
    timeout = float(studio.cfg.get("comfyui.timeout_s", 30.0) or 30.0)
    return Client.from_config(studio.cfg, timeout=timeout)


def url(studio: Studio) -> str:
    return client(studio).url


def pid_file(studio: Studio) -> Path:
    return studio.runtime / "comfyui.pid"


def managed(studio: Studio) -> bool:
    return bool(studio.cfg.get("comfyui.manage_lifecycle", False))


def check_can_manage(studio: Studio) -> tuple[Path, Path]:
    """(comfyui dir, python) when easel may start the server here; StudioError(2) saying why not otherwise."""
    if studio.legacy_runner:
        raise StudioError(EXIT_USAGE, [REFUSED_LEGACY, "the host runner starts the server per job; easel never starts "
                                                       "one on an adopted workspace"])
    if not managed(studio):
        raise StudioError(EXIT_USAGE, ["comfyui.manage_lifecycle = false: easel does not start or stop the server on "
                                       "this instance",
                                       "next: start it the way this machine does, or ask the operator to set "
                                       "comfyui.manage_lifecycle, comfyui.dir and comfyui.python"])
    folder = str(studio.cfg.get("comfyui.dir", "") or "")
    python = str(studio.cfg.get("comfyui.python", "") or "")
    if not folder or not python:
        raise StudioError(EXIT_USAGE, ["comfyui.dir and comfyui.python must be set for easel to start the server",
                                       "next: easel config set comfyui.dir <comfyui_dir>; easel config set "
                                       "comfyui.python <comfyui_python>"])
    folder_path, python_path = Path(folder).expanduser(), Path(python).expanduser()
    if not (folder_path / "main.py").is_file():
        raise StudioError(EXIT_USAGE, ["comfyui.dir holds no main.py: <comfyui_dir> is not a ComfyUI checkout"])
    if not python_path.is_file():
        raise StudioError(EXIT_USAGE, ["comfyui.python is not a file: <comfyui_python>"])
    return folder_path, python_path


def start_command(studio: Studio, folder: Path, python: Path) -> list[str]:
    """The server's command line: listen where `comfyui.url` points, no browser, outputs under `<runtime>`."""
    parts = urllib.parse.urlsplit(url(studio))
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if parts.scheme == "https" else 80)
    out_dir, in_dir = studio.runtime / "comfyui" / "output", studio.runtime / "comfyui" / "input"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_dir.mkdir(parents=True, exist_ok=True)
    extra = [str(a) for a in (studio.cfg.get("comfyui.start_args", []) or [])]
    return [str(python), "-u", "main.py", "--listen", host, "--port", str(port), "--disable-auto-launch",
            "--output-directory", str(out_dir), "--input-directory", str(in_dir), *extra]


def start(studio: Studio, say: Say, wait_s: float = START_WAIT_S) -> background.Started:
    """Start the server hidden and wait until it answers; StudioError(1) when it exits or stays silent."""
    folder, python = check_can_manage(studio)
    cmd = start_command(studio, folder, python)
    log = studio.runtime / "logs" / f"comfyui-{ids.now(studio.tool):%Y%m%d-%H%M%S}.log"
    started = background.start(cmd, log, pid_file(studio), cwd=folder)
    say(f"starting ComfyUI (pid {started.pid}, {started.status_line}); log {studio.shown(log)}")
    the_client = client(studio)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if the_client.ping() is not None:
            say(f"comfyui: up at {the_client.url}")
            return started
        if started.poll() is not None:
            raise StudioError(EXIT_PROBLEMS, [f"ComfyUI exited before it answered; read {studio.shown(log)}"])
        time.sleep(POLL_S)
    raise StudioError(EXIT_PROBLEMS, [f"ComfyUI did not answer within {wait_s:g} s (still loading?); read "
                                      f"{studio.shown(log)}"])


def stop(studio: Studio) -> int | None:
    """Stop the server easel started (its pid file); None when easel started none."""
    return background.stop(pid_file(studio))


def ensure_server(studio: Studio, say: Say) -> tuple[Client, bool]:
    """(client, started here): a server that answers, started now when this instance manages it."""
    the_client = client(studio)
    if the_client.ping() is not None:
        return the_client, False
    if managed(studio) and not studio.legacy_runner:
        start(studio, say)
        return the_client, True
    raise StudioError(EXIT_PROBLEMS, [f"ComfyUI is not reachable at {the_client.url}",
                                      "fix: report to the operator (the server is theirs to start); "
                                      + PORT_LESSON])


def status_lines(studio: Studio) -> tuple[bool, list[str]]:
    """(server answers, lines) for `easel comfy status`: version, devices, queue, and how easel started it."""
    the_client = client(studio)
    stats = the_client.ping()
    lines: list[str] = []
    if stats is None:
        if studio.adopted:
            lines.append(f"comfyui: not running at {the_client.url} (the host runner starts it per job)")
        else:
            lines.append(f"comfyui: not running at {the_client.url}")
            lines.append("  " + PORT_LESSON)
    else:
        system = stats.get("system") or {}
        python = str(system.get("python_version") or "?").split()[0]
        lines.append(f"comfyui: up at {the_client.url}, version {system.get('comfyui_version', '?')}, python {python}")
        for device in stats.get("devices") or []:
            free, total = device.get("vram_free"), device.get("vram_total")
            if isinstance(free, int | float) and isinstance(total, int | float):
                lines.append(f"  {device.get('name', 'device')}: VRAM {free / 2**30:.1f} free of {total / 2**30:.1f} GB")
        try:
            queue = the_client.queue_state()
            lines.append(f"  queue: {queue['running']} running, {queue['pending']} pending")
        except Exception:  # noqa: BLE001  (the queue count is extra; a status line must still print)
            pass
    if pid_file(studio).exists():
        lines.append(f"  started by easel: {background.status_line(pid_file(studio))}")
    return stats is not None, lines
