"""The CI licence gate, proven by sabotage: it passes a permissive package and refuses a GPL-classified one.

The command is read from the `piplicenses` step of .github/workflows/ci.yml, so the test and CI cannot drift:
the same `--from=mixed`, the same `--ignore-packages easel kiln pip setuptools` and the same `--allow-only` list.
It runs against a scratch virtual environment in tmp_path (pip-licenses `--python` points at it), never the
environment running the tests: two hand-built wheels, one MIT-classified and one GPL-classified, are installed
there with `pip install --no-index` from local files only.
"""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CI = REPO / ".github" / "workflows" / "ci.yml"
IGNORED = ["easel", "kiln", "pip", "setuptools"]
GPL = "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
MIT = "License :: OSI Approved :: MIT License"


def gate_command() -> list[str]:
    """The licence-gate step of ci.yml as an argument list, POSIX-quoted as CI's bash reads it."""
    import yaml

    workflow = yaml.safe_load(CI.read_text(encoding="utf-8"))
    runs = [step.get("run", "") for job in workflow["jobs"].values() for step in job["steps"]]
    gates = [run for run in runs if "-m piplicenses" in run]
    assert len(gates) == 1, "ci.yml needs exactly one pip-licenses step"
    return shlex.split(gates[0])


def test_the_ci_step_has_the_reviewed_shape() -> None:
    argv = gate_command()
    assert argv[:3] == ["python", "-m", "piplicenses"]
    assert "--from=mixed" in argv
    start = argv.index("--ignore-packages") + 1
    assert argv[start:start + len(IGNORED)] == IGNORED
    assert "--allow-only" in argv
    assert "--partial-match" not in argv, "partial matching would let an unreviewed licence through"
    assert not any(a.startswith("--fail-on") for a in argv), "--fail-on matches whole names only; use --allow-only"
    allowed = [name.strip() for name in argv[argv.index("--allow-only") + 1].split(";")]
    assert all(allowed) and len(allowed) == len(set(allowed))
    assert not [name for name in allowed if "GPL" in name.upper()]


def wheel(folder: Path, name: str, classifier: str) -> Path:
    """A minimal pure-Python wheel with one licence classifier, written by hand (no build tool needed)."""
    dist = name.replace("-", "_")
    files = {
        f"{dist}/__init__.py": b'"""A licence-gate fixture; it does nothing."""\n',
        f"{dist}-0.0.1.dist-info/METADATA": (f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.1\n"
                                             f"Summary: licence gate fixture\nClassifier: {classifier}\n").encode(),
        f"{dist}-0.0.1.dist-info/WHEEL": b"Wheel-Version: 1.0\nGenerator: hand\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = f"{dist}-0.0.1.dist-info/RECORD"
    rows = []
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        rows.append(f"{path},sha256={digest},{len(data)}")
    rows.append(f"{record},,")
    target = folder / f"{dist}-0.0.1-py3-none-any.whl"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, data in files.items():
            archive.writestr(path, data)
        archive.writestr(record, "\n".join(rows) + "\n")
    return target


@pytest.fixture
def scratch_python(tmp_path: Path) -> Path:
    """A fresh virtual environment in tmp_path; only pip is in it (pip-licenses skips pip as a system package)."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True, capture_output=True, timeout=300)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    assert python.is_file()
    return python


def pip_install(python: Path, wheel_file: Path) -> None:
    env = {**os.environ, "PIP_CONFIG_FILE": os.devnull, "PIP_NO_INPUT": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    done = subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel_file)],
                          capture_output=True, text=True, env=env, timeout=300)
    assert done.returncode == 0, done.stdout + done.stderr


def run_gate(python: Path) -> subprocess.CompletedProcess[str]:
    argv = gate_command()
    command = [sys.executable, *argv[1:3], "--python", str(python), *argv[3:]]
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


def test_the_gate_passes_mit_and_refuses_gpl(tmp_path: Path, scratch_python: Path) -> None:
    pytest.importorskip("piplicenses", reason="pip-licenses is part of the [dev] extra")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    pip_install(scratch_python, wheel(wheels, "gate-fixture-mit", MIT))
    allowed = run_gate(scratch_python)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert "gate-fixture-mit" in allowed.stdout  # the scratch environment was really the one read

    pip_install(scratch_python, wheel(wheels, "gate-fixture-gpl", GPL))
    refused = run_gate(scratch_python)
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "not in allow-only licenses" in refused.stderr
    assert "gate-fixture-gpl" in refused.stderr
