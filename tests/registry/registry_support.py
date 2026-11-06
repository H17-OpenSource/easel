"""Helpers shared by the registry tests (imported by name: pytest puts this folder on sys.path).

`header_reader(monkeypatch)` makes sure a safetensors header reader is importable as
`easel._common.embedded_meta.safetensors_header`. When the real module is installed the tests use it; until then
(it is built in another task) a minimal stand-in that reads the 8-byte length and the JSON header is put in
`sys.modules` for the one test, so the registry's own logic is still exercised.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import shutil
import struct
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPECIAL = "easel._common.embedded_meta"


def _stand_in_header(path: Path) -> dict:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError("file too small for a safetensors header")
    (size,) = struct.unpack("<Q", data[:8])
    if size <= 0 or 8 + size > len(data):
        raise ValueError(f"implausible header length {size}")
    try:
        header = json.loads(data[8:8 + size].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError("header is not a JSON object")
    return header


def header_reader(monkeypatch: pytest.MonkeyPatch) -> str:
    """'real' when the T1b reader is installed, else 'stand-in' (installed into sys.modules for this test)."""
    try:
        module = importlib.import_module(SPECIAL)
        if callable(getattr(module, "safetensors_header", None)):
            return "real"
    except ImportError:
        pass
    stand_in = types.ModuleType(SPECIAL)
    stand_in.safetensors_header = _stand_in_header
    monkeypatch.setitem(sys.modules, SPECIAL, stand_in)
    return "stand-in"


def fixtures_module():
    """tests/fixtures/make_fixtures.py, imported by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("make_fixtures", REPO / "tests" / "fixtures" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_sha16() -> dict[str, str]:
    """{relative path under models/: sha16} of the six generated fake models."""
    fx = fixtures_module()
    return {fx.model_relpath(i): fx.sha256_of(f"models/{fx.model_relpath(i)}")[:16] for i in range(len(fx.MODELS))}


def empty_registry(home: Path) -> None:
    """Drop the demo's resource records and changelog, keeping overrides.yaml: a registry that was never scanned."""
    shutil.rmtree(home / "registry" / "resources")
    (home / "registry" / "changelog.jsonl").unlink()


def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str, str]:
    """Call easel's CLI in-process; returns (exit code, stdout, stderr)."""
    from easel import cli

    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def safetensors_bytes(tensors: dict[str, str], metadata: dict[str, str] | None = None) -> bytes:
    """A tiny header-only safetensors file: tensor name -> dtype, each one value of shape [1]."""
    header: dict[str, object] = {"__metadata__": dict(metadata or {})}
    width = {"F16": 2, "BF16": 2, "F32": 4}
    offset = 0
    for name, dtype in tensors.items():
        header[name] = {"dtype": dtype, "shape": [1], "data_offsets": [offset, offset + width.get(dtype, 2)]}
        offset += width.get(dtype, 2)
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw + b"\x00" * offset
