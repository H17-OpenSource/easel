"""_common.ids: the frozen clock (and its one stderr line), content ids and the dated sequence ids."""

import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from easel._common import hashing, ids

PKG = "easel"
# The warning built from the variable name: the literal line is a scanner hazard (it must never reach a doc).
WARNING = f"clock frozen ({ids.now_var(PKG)} set)"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def fresh_warning(monkeypatch):
    """Forget that this process already printed the frozen-clock line."""
    monkeypatch.setattr(ids, "_warned", set())


def test_frozen_clock_and_one_stderr_line(frozen_clock, fresh_warning, capsys):
    assert ids.now(PKG) == datetime(2030, 1, 1, tzinfo=UTC)
    assert ids.now_iso(PKG) == "2030-01-01T00:00:00+00:00"
    assert ids.today(PKG) == "2030-01-01" and ids.stamp(PKG) == "20300101"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == WARNING + "\n"


def test_the_tool_name_picks_the_variable(monkeypatch, fresh_warning, capsys):
    monkeypatch.setenv("OTHERTOOL_NOW", "2030-01-01T00:00:00+00:00")  # any tool name works: none is built in
    assert ids.now("othertool").year == 2030
    assert abs(ids.now(PKG) - datetime.now().astimezone()) < timedelta(minutes=1)  # easel's clock is not frozen
    assert capsys.readouterr().err == f"clock frozen ({ids.now_var('othertool')} set)\n"


def test_unfrozen_clock_is_aware_local_time():
    value = ids.now(PKG)
    assert value.tzinfo is not None
    assert abs(value - datetime.now().astimezone()) < timedelta(minutes=1)


def test_bad_or_naive_frozen_value(monkeypatch, fresh_warning, capsys):
    monkeypatch.setenv("EASEL_NOW", "tomorrow")
    with pytest.raises(ids.IdError, match="EASEL_NOW"):
        ids.now(PKG)
    monkeypatch.setenv("EASEL_NOW", "2030-01-01T12:30:00")
    assert ids.now(PKG) == datetime(2030, 1, 1, 12, 30, tzinfo=UTC)
    capsys.readouterr()


def test_warning_once_per_process_and_stdout_clean():
    """A real process: three clock reads, one stderr line, stdout holds only what the program printed."""
    code = f"from {PKG}._common import ids; print(ids.now_iso('{PKG}')); ids.now('{PKG}'); ids.today('{PKG}')"
    env = {**{k: v for k, v in os.environ.items() if not k.upper().startswith(("EASEL_", "KILN_"))},
           "EASEL_NOW": "2030-01-01T00:00:00+00:00",
           "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout == "2030-01-01T00:00:00+00:00\n"
    assert done.stderr.splitlines() == [WARNING]


def test_content_ids():
    sha = hashing.sha256_bytes(b"demo_robot")
    assert ids.image_id(sha) == "img-" + sha[:12]
    assert ids.resource_name(sha.upper()) == sha[:16]
    with pytest.raises(ids.IdError):
        ids.image_id("abc")


def test_next_seq_id_counts_only_todays_ids(frozen_clock, fresh_warning, capsys):
    existing = ["rcp-20300101-001", "rcp-20300101-007.json", "rcp-20301231-009", "job-20300101-050", "README.md"]
    assert ids.next_seq_id("rcp", existing, PKG) == "rcp-20300101-008"
    assert ids.next_seq_id("job", [], PKG) == "job-20300101-001"
    with pytest.raises(ids.IdError):
        ids.next_seq_id("Bad Prefix", [], PKG)
    capsys.readouterr()


def test_claim_seq_dir_never_hands_out_an_id_twice(tmp_path, frozen_clock, fresh_warning, capsys):
    first = ids.claim_seq_dir(tmp_path, "rcp", PKG)
    second = ids.claim_seq_dir(tmp_path, "rcp", PKG)
    assert [first[0], second[0]] == ["rcp-20300101-001", "rcp-20300101-002"]
    assert first[1].is_dir() and second[1].is_dir()
    capsys.readouterr()


def test_claim_run_dir_is_exclusive(tmp_path, frozen_clock, fresh_warning, capsys):
    run_id, folder = ids.claim_run_dir(tmp_path, "001", PKG)
    assert run_id == "run-20300101-001" and folder.is_dir()
    with pytest.raises(FileExistsError):
        ids.claim_run_dir(tmp_path, "001", PKG)
    with pytest.raises(ids.IdError):
        ids.claim_run_dir(tmp_path, "Demo Robot", PKG)
    capsys.readouterr()


def test_next_decision_id():
    # Low decision numbers look like a real project's ids, so the literals here stay in the 9000s.
    assert ids.next_decision_id([]) == "dec-" + format(1, "04d")
    assert ids.next_decision_id(["dec-9001.md", "dec-9041", "notes.md"]) == "dec-9042"
