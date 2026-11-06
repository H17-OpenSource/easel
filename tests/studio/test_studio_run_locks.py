"""The run folder's store lock: `runs review` appends under it, and the renderer's run.json updates keep notes.

`easel note <run-id>` appends to run.json's `notes` under the run folder's lock. The renderer rewrites run.json
several times while a job runs (queued, running, outputs, ok), from the record it holds in memory, so each of its
writes takes the same lock and copies the notes on disk first: a note written mid-render survives.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from studio_support import RUN, run_cli


@pytest.fixture
def home(demo_workspace, frozen_clock):
    return demo_workspace


def _record_lock_roots(monkeypatch: pytest.MonkeyPatch) -> list:
    """Wrap store.lock so every root it is asked for is recorded; the real lock still runs for each one."""
    from easel._common import store

    real = store.lock
    roots: list = []

    @contextmanager
    def recording(root, tool, timeout=None):
        roots.append(Path(root).resolve())
        with real(root, tool, timeout) as held:
            yield held

    monkeypatch.setattr(store, "lock", recording)
    return roots


def test_runs_review_appends_under_the_run_folders_lock(home, monkeypatch, capsys):
    roots = _record_lock_roots(monkeypatch)
    code, _out, err = run_cli(["runs", "review", RUN, "--verdict", "keep", "--output", "2", "--note", "fine"], capsys)
    assert code == 0, err
    assert (home / "runs" / RUN).resolve() in roots


def test_a_note_written_during_a_render_survives_the_next_update(home, monkeypatch, capsys):
    from easel.studio import render

    folder = home / "runs" / RUN
    stale = json.loads((folder / "run.json").read_text(encoding="utf-8"))  # the renderer's copy in memory
    code, _out, err = run_cli(["note", RUN, "the operator says: keep the workshop light"], capsys)
    assert code == 0, err
    roots = _record_lock_roots(monkeypatch)
    stale["status"] = "ok"
    render.write_run(folder, stale, "easel")
    assert folder.resolve() in roots
    on_disk = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert "keep the workshop light" in (on_disk["notes"] or "")
    assert stale["notes"] == on_disk["notes"]  # later writes from the same record carry the note too


def test_the_first_write_keeps_the_renderers_own_note(tmp_path):
    from easel.studio import render

    folder = tmp_path / "runs" / "run-20300101-009"
    folder.mkdir(parents=True)
    run = json.loads((_demo_run_json()).read_text(encoding="utf-8"))
    run.update(id="run-20300101-009", notes="draft: rendered without the hires pass")
    render.write_run(folder, run, "easel")
    assert json.loads((folder / "run.json").read_text(encoding="utf-8"))["notes"] == run["notes"]


def _demo_run_json() -> Path:
    return Path(__file__).resolve().parents[2] / "examples" / "demo" / "runs" / RUN / "run.json"
