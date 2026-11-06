"""Helpers shared by the studio tests (imported by name: pytest puts this folder on sys.path).

The tests run the CLI in-process on a private copy of the demo instance (`demo_workspace`) against the mock
ComfyUI of `tests/conftest.py`. Nothing here starts a real process or a real server: the RAM check is switched
off through the instance's own key (`limits.skip_ram_check`), and the mock answers `/object_info`, `/prompt`,
`/history` and `/view` with values built from the graph the studio compiles.
"""

from __future__ import annotations

import socket
import struct
import zlib
from pathlib import Path

import pytest

PROMPT_ID = "00000000-0000-4000-8000-000000000001"
RECIPE = "rcp-20300101-001"
RUN = "run-20300101-001"
LORA = "cf56db17414bcb55"
CHECKPOINT = "bbbabeef2c0d57fe"
# The model files the demo's records name, as the server lists them (relative to their type folder).
SERVER_NAMES = {
    "CheckpointLoaderSimple": ("ckpt_name", ["demo-anime-xl.safetensors", "demo-base-xl.safetensors"]),
    "LoraLoader": ("lora_name", ["demo_robot-v1.safetensors", "demo-style-ink.safetensors"]),
    "VAELoader": ("vae_name", ["demo-xl-vae.safetensors"]),
    "UpscaleModelLoader": ("model_name", ["demo-4x.safetensors"]),
}


def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """easel's CLI in-process: (exit code, stdout, stderr)."""
    from easel import cli

    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def png_bytes(value: int = 0) -> bytes:
    """A valid 8x8 RGB PNG of one flat value (a stand-in for a rendered image; never looked at)."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([value % 256]) * 24 for _ in range(8))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def closed_url() -> str:
    """The URL of a loopback port nothing listens on (bound, then released)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


def native_env(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    """Point the demo at `url`, and let jobs start without measuring this machine's RAM or disk."""
    monkeypatch.setenv("EASEL_COMFYUI__URL", url)
    monkeypatch.setenv("EASEL_LIMITS__SKIP_RAM_CHECK", "true")
    monkeypatch.setenv("EASEL_LIMITS__DISK_FLOOR_GB", "0")


def set_values(home: Path, values: dict[str, str]) -> None:
    """Write values into the instance's easel.toml through `config set` (validated, atomic)."""
    from easel._common import config

    for key, raw in values.items():
        config.set_value("easel", key, raw, workspace=home)


def make_adopted(home: Path) -> None:
    """Turn the demo copy into an adopted, disarmed instance, the shape of a host workspace's instance file."""
    set_values(home, {
        "instance.adopted": "true", "policy.render": "legacy-runner", "jobs.allowed": "[]",
        "scheduler.kind": "none", "comfyui.manage_lifecycle": "false",
        "adopt.propose_command": "host-propose --from-json {recipe_json}",
        "adopt.dryrun_command": "host-run {recipe_id} --dry-run",
        "adopt.submit_command": "host-submit --by <agent> --title {recipe_id} --count {count}",
        "adopt.status_command": "host-status {job_id}",
    })


def object_info_for(graph: dict) -> dict:
    """A /object_info answer that accepts `graph` exactly: each class's inputs as required, model names listed."""
    info: dict = {}
    for node in graph.values():
        cls = node["class_type"]
        entry = info.setdefault(cls, {"input": {"required": {}}, "input_order": {"required": []}})
        for name in node.get("inputs", {}):
            if name in entry["input"]["required"]:
                continue
            field, names = SERVER_NAMES.get(cls, (None, None))
            entry["input"]["required"][name] = [list(names), {}] if name == field else ["*", {}]
            entry["input_order"]["required"].append(name)
    return info


def history_entry(*, images: int = 1, error: str | None = None) -> dict:
    """A finished /history entry: `images` saved files, or an execution error with `error` as its message."""
    if error is not None:
        return {"status": {"status_str": "error", "completed": False, "messages": [
            ["execution_error", {"node_id": "6", "node_type": "KSampler", "exception_type": "RuntimeError",
                                 "exception_message": error}]]}, "outputs": {}}
    files = [{"filename": f"easel_{i:05d}_.png", "subfolder": "easel", "type": "output"} for i in range(1, images + 1)]
    return {"status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"9": {"images": files}}}


def serve_render(mock, graph: dict, *, entry: dict | None = None, image: bytes | None = None) -> None:
    """Make the mock answer a whole render of graphs shaped like `graph`."""
    mock.routes[("GET", "/object_info/")] = (200, object_info_for(graph))
    mock.routes[("GET", "/history/")] = (200, {PROMPT_ID: entry or history_entry()})
    mock.routes[("GET", "/view")] = (200, image if image is not None else png_bytes())


def demo_graph(recipe_id: str = RECIPE) -> dict:
    """The graph the studio compiles for a recipe of the instance found through EASEL_HOME (offline)."""
    from easel.studio import common, render
    from easel.studio import recipe as recipes

    studio = common.open_studio("easel")
    graph, _summary = render.dry_run(studio, recipes.load(studio, recipe_id))
    return graph


def prompts_sent(mock) -> list[dict]:
    """The graphs POSTed to /prompt, in order."""
    return [body["prompt"] for method, path, body in mock.requests if method == "POST" and path == "/prompt"]


def schema_problems(name: str, data: object) -> list[str]:
    from easel.registry.shipped import schema_errors

    return schema_errors(name, data)
