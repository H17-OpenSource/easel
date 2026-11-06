"""Shared pytest setup: the fixtures of the test plan.

The text is the same in both repos except PKG. No fixture reaches the network, a GPU or a real tool: the demo
instance is a private copy, the scheduler and the image source are scripted, and ComfyUI is a small local mock.
"""

import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import socket
import threading
from collections.abc import Iterator
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

PKG = "easel"
REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "examples" / "demo"
FROZEN_NOW = "2030-01-01T00:00:00+00:00"


def _make_fixtures():
    """tests/fixtures/make_fixtures.py as a module (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location("make_fixtures", REPO / "tests" / "fixtures" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- fixtures (test plan, section "Tests") ---------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_tool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean room: no KILN_* or EASEL_* variable from the caller's shell reaches a test; a test sets what it needs."""
    for name in list(os.environ):
        if name.upper().startswith(("KILN_", "EASEL_")):
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def frozen_clock(monkeypatch: pytest.MonkeyPatch) -> datetime:
    """Freeze both tools' clocks at 2030-01-01T00:00:00+00:00 (ids, journal lines and `at` fields follow it)."""
    monkeypatch.setenv("KILN_NOW", FROZEN_NOW)
    monkeypatch.setenv("EASEL_NOW", FROZEN_NOW)
    return datetime.fromisoformat(FROZEN_NOW)


@pytest.fixture
def demo_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of examples/demo/ (the synthetic demo_robot instance), found through the tool's home variable.

    The demo's media (images, fake model files) are never committed: tests/fixtures/make_fixtures.py generates them
    into tmp_path and copies them where the demo's records expect them, byte for byte matching their sha256.
    """
    if not DEMO.is_dir():
        pytest.skip("examples/demo/ not built yet (T6)")
    home = tmp_path / "demo"
    shutil.copytree(DEMO, home)
    _make_fixtures().place(home, PKG, out=tmp_path / "_gen")
    monkeypatch.setenv(f"{PKG.upper()}_HOME", str(home))
    return home


def _png(n: int) -> bytes:
    """An 8x8 RGB PNG whose pixels come from n (every n gives other bytes), written the fixture generator's way:
    hand-made chunks and stored deflate, so the bytes are the same on every machine."""
    fx = _make_fixtures()
    rows = b"".join(b"\x00" + hashlib.sha256(f"fake source {n} row {y}".encode()).digest()[:24] for y in range(8))
    ihdr = (8).to_bytes(4, "big") * 2 + bytes((8, 2, 0, 0, 0))
    return (fx.PNG_SIGNATURE + fx._chunk(b"IHDR", ihdr) + fx._chunk(b"IDAT", fx._stored_zlib(rows))
            + fx._chunk(b"IEND", b""))


@pytest.fixture
def fake_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A scripted image source on `kiln.sources.base.Source` (kiln only), so `kiln source run` runs no real tool.

    While the test runs, a jobs file whose `source` is `fake_source.kind` (default "codex") is served by the fake
    instead of the real source; every other kind stays real. Per job it makes `want` new 8x8 PNGs (other bytes
    every time) in a scratch folder and hands them back; the shared code then copies and registers them as usual.
    Knobs: `outcomes[<job id>]` sets a job's status (ok, empty, blocked, refused, quota, failed; anything but ok
    makes no file); `problems[<job id>]` makes `check` refuse the job. Record: `calls` lists (verb, job id) for
    check, describe and run, in order; `made` lists every file handed back.
    """
    if PKG != "kiln":
        pytest.skip("fake_source stands in for kiln's image sources (kiln.sources.base); easel has no image source")
    base = importlib.import_module(f"{PKG}.sources.base")
    jobs = importlib.import_module(f"{PKG}.sources.jobs")
    out = tmp_path / "_fake_source"

    class FakeSource(base.Source):
        kind = "codex"
        makes_images = True

        def __init__(self) -> None:
            self.outcomes: dict[str, str] = {}
            self.problems: dict[str, list[str]] = {}
            self.calls: list[tuple[str, str]] = []
            self.made: list[Path] = []

        def check(self, job, ctx) -> list[str]:
            self.calls.append(("check", job.id))
            return list(self.problems.get(job.id, []))

        def describe(self, job, ctx) -> list[str]:
            self.calls.append(("describe", job.id))
            return [f"fake source: would make {job.want} image(s)"]

        def run(self, job, ctx):
            self.calls.append(("run", job.id))
            status = self.outcomes.get(job.id, "ok")
            files: list[Path] = []
            if status == "ok":
                out.mkdir(parents=True, exist_ok=True)
                for k in range(1, job.want + 1):
                    path = out / f"{job.id}-{k}.png"
                    path.write_bytes(_png(len(self.made) + 1))
                    self.made.append(path)
                    files.append(path)
            return base.Outcome(status=status, exit_code=0 if status == "ok" else 1, files=files,
                                lines=[f"fake source: {status}, {len(files)} file(s)"],
                                manifest={"tool_version": "fake-source 0", "model": "fake"})

    fake = FakeSource()
    real_get = jobs.get
    monkeypatch.setattr(jobs, "get", lambda kind: fake if kind == fake.kind else real_get(kind))
    return fake


@pytest.fixture
def fake_trainer() -> None:
    """A stand-in for the trainer process; needs the training launcher."""
    pytest.skip("fake_trainer needs the training launcher (session-2)")


class ScriptedMachine:
    """What a fake scheduler measures and does: scripted free RAM and disk, a clock that moves only when the
    scheduler sleeps, and a record of every line it says and every sleep.

    `ram` and `disk` are readings in GB, taken in order; the last one repeats. A reading that is an exception
    instance is raised instead, for example `sysinfo.SysinfoUnavailable("no counter")` for "cannot measure".
    """

    def __init__(self) -> None:
        self.ram: list = [64.0]
        self.disk: list = [500.0]
        self.now = 0.0
        self.said: list[str] = []
        self.slept: list[float] = []
        self.disk_paths: list[Path] = []

    @staticmethod
    def _next(readings: list) -> float:
        value = readings.pop(0) if len(readings) > 1 else readings[0]
        if isinstance(value, BaseException):
            raise value
        return float(value)

    def free_ram(self) -> float:
        return self._next(self.ram)

    def free_disk(self, path: Path) -> float:
        self.disk_paths.append(Path(path))
        return self._next(self.disk)

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def say(self, line: str) -> None:
        self.said.append(line)


@pytest.fixture
def fake_scheduler(tmp_path: Path):
    """A `_common.scheduler.NullScheduler` (scheduler.kind = "none") on a ScriptedMachine, so a wait never sleeps
    and nothing measures the real machine. `fake_scheduler.machine` holds the knobs and the record.

    Defaults: 64 GB free RAM and 500 GB free disk; floors 2 GB RAM and 20 GB disk; a 5 s poll; the stop file
    tmp_path/stop; disk measured at tmp_path; the family files of this repository's families/.
    """
    scheduler = importlib.import_module(f"{PKG}._common.scheduler")
    machine = ScriptedMachine()
    fake = scheduler.NullScheduler(
        families_dir=REPO / "families", host_ram_floor_gb=2.0, disk_floor_gb=20.0, poll_s=5.0,
        stop_file=tmp_path / "stop", default_disk_path=tmp_path, say=machine.say, free_ram=machine.free_ram,
        free_disk=machine.free_disk, clock=machine.clock, sleep=machine.sleep)
    fake.machine = machine
    return fake


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every outbound connection except loopback (the mock ComfyUI), so no test reaches the internet."""
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex

    def _check(address: object) -> None:
        if not isinstance(address, tuple):  # AF_UNIX paths and the like stay on the machine
            return
        host = str(address[0])
        if host in ("localhost", "::1") or host.startswith("127."):
            return
        raise OSError(f"network disabled in tests (no_network fixture): {host}")

    def connect(self: socket.socket, address: object) -> None:
        _check(address)
        return real_connect(self, address)

    def connect_ex(self: socket.socket, address: object) -> int:
        _check(address)
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


class MockComfy:
    """A small stand-in for ComfyUI's HTTP API on 127.0.0.1 and a free port.

    `routes` maps (method, path) to (status, payload); a key ending in "/" also matches longer paths, so
    ("GET", "/history/") answers /history/<prompt id>. A bytes payload is sent as-is, anything else as JSON.
    Every request is appended to `requests` as (method, path, body) with a JSON body decoded when possible, and
    to `queries` as (method, path, query string), so a test can assert a /view query end to end.
    """

    def __init__(self) -> None:
        self.routes: dict[tuple[str, str], tuple[int, object]] = {
            ("GET", "/system_stats"): (200, {"system": {"comfyui_version": "0.43.1", "python_version": "3.12.6", "os": "demo"}, "devices": []}),
            ("GET", "/object_info"): (200, {}),
            ("GET", "/object_info/"): (200, {}),
            ("GET", "/queue"): (200, {"queue_running": [], "queue_pending": []}),
            ("GET", "/history"): (200, {}),
            ("GET", "/history/"): (200, {}),
            ("POST", "/prompt"): (200, {"prompt_id": "00000000-0000-4000-8000-000000000001", "number": 1, "node_errors": {}}),
            ("POST", "/interrupt"): (200, {}),
        }
        self.requests: list[tuple[str, str, object]] = []
        self.queries: list[tuple[str, str, str]] = []
        self.url = ""

    def lookup(self, method: str, path: str) -> tuple[int, object]:
        if (method, path) in self.routes:
            return self.routes[(method, path)]
        prefixes = [p for (m, p) in self.routes if m == method and p.endswith("/") and path.startswith(p)]
        if prefixes:
            return self.routes[(method, max(prefixes, key=len))]
        return 404, {"error": f"not mocked: {method} {path}"}


@pytest.fixture
def mock_comfy() -> Iterator[MockComfy]:
    """Serve a MockComfy for one test; `mock_comfy.url` is its base URL."""
    mock = MockComfy()

    class Handler(BaseHTTPRequestHandler):
        def _answer(self, method: str) -> None:
            parts = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body: object = json.loads(raw) if raw else None
            except ValueError:
                body = raw
            mock.requests.append((method, parts.path, body))
            mock.queries.append((method, parts.path, parts.query))
            status, payload = mock.lookup(method, parts.path)
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream" if isinstance(payload, bytes) else "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._answer("GET")

        def do_POST(self) -> None:
            self._answer("POST")

        def log_message(self, format: str, *args: object) -> None:  # keep test output quiet
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    mock.url = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield mock
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
