"""_common.journal: note, stuck, report, the one STUCK template, and the frozen-clock line on stderr only."""

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from easel import cli
from easel._common import config, ids, journal

PKG = "easel"
SRC = Path(__file__).resolve().parents[2] / "src"
# The warning built from the variable name: the literal line is a scanner hazard (it must never reach a doc).
WARNING = f"clock frozen ({ids.now_var(PKG)} set)"

# The template of the plan's section "The AI guide for weaker models", retyped here on purpose: a change to the
# shape must be made in two places, and the docs generated from it change with it.
EXPECTED_TEMPLATE = """## STUCK <date from the clock> <stage>
command: <exact>
exit code: <n>
output (last 20 lines): ...
expected: <runbook file:line> says ...
tried: 1) ... 2) ...
hypothesis: ...
need: <operator yes | image judgement | maintainer>
"""


@pytest.fixture
def fresh_warning(monkeypatch):
    monkeypatch.setattr(ids, "_warned", set())


@pytest.fixture(autouse=True)
def no_repo_schema(monkeypatch, tmp_path):
    """These tests are about the journal: the instance TOMLs here are minimal and skip the repo's config schema."""
    monkeypatch.setattr(config, "default_schema_path", lambda: tmp_path / "no-schema.json")


def run(argv, capsys):
    code = cli.main(argv)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_one_stuck_template():
    assert journal.STUCK_TEMPLATE == EXPECTED_TEMPLATE
    assert journal.STUCK_KEYS == ("command", "exit_code", "output", "expected", "tried", "hypothesis", "need")
    # rendering with nothing filled in gives the template back, so every rendering shares its shape
    assert journal.render_stuck("<date from the clock>") == journal.STUCK_TEMPLATE


def test_render_stuck_fills_fields_and_keeps_the_last_20_output_lines():
    output = "\n".join(f"line {i}" for i in range(1, 31)) + "\n"
    block = journal.render_stuck("2030-01-01 00:00", "05-captions", "captions apply refused", command="easel x",
                                 exit_code=2, output=output, tried=["read --help", "ran doctor"])
    lines = block.splitlines()
    assert lines[:4] == ["## STUCK 2030-01-01 00:00 05-captions", "summary: captions apply refused",
                         "command: easel x", "exit code: 2"]
    assert lines[4] == "output (last 20 lines):"
    assert lines[5:25] == [f"    line {i}" for i in range(11, 31)]
    assert lines[25] == "expected: <runbook file:line> says ..."
    assert lines[26] == "tried: 1) read --help 2) ran doctor"
    assert lines[-1] == "need: <operator yes | image judgement | maintainer>"
    with pytest.raises(ValueError):
        journal.render_stuck("x", nosuch="y")


def test_note_and_pitfall_lines(tmp_path, frozen_clock, fresh_warning, capsys):
    path = journal.note("seeded the demo", tool=PKG, journal_dir=tmp_path)
    journal.note("two lines\nsecond", tool=PKG, journal_dir=tmp_path, pitfall="the lock was held.",
                 rule="wait, do not delete .lock.")
    assert path == tmp_path / "2030-01-01.md"
    assert path.read_bytes().decode("utf-8") == (
        "# Journal 2030-01-01\n\n- 00:00 seeded the demo\n- 00:00 two lines\n  second\n"
        "  Pitfall: the lock was held. Rule: wait, do not delete .lock.\n")
    with pytest.raises(ValueError):
        journal.note("x", tool=PKG, journal_dir=tmp_path, pitfall="no rule")
    with pytest.raises(ValueError):
        journal.note("   ", tool=PKG, journal_dir=tmp_path)
    capsys.readouterr()


def test_report_collects_stuck_blocks_and_bullets(tmp_path, frozen_clock, fresh_warning, capsys):
    journal.note("first", tool=PKG, journal_dir=tmp_path)
    journal.stuck("lock held", tool=PKG, journal_dir=tmp_path, stage="03-render", exit_code=4)
    journal.note("second", tool=PKG, journal_dir=tmp_path)
    text = journal.report(tool=PKG, journal_dir=tmp_path)
    assert text.startswith("# Journal report 2030-01-01\n")
    assert "## STUCK notes: 1" in text and "### STUCK 2030-01-01 00:00 03-render\nsummary: lock held\n" in text
    assert "exit code: 4" in text
    assert "## Bullets: 2\n\n- 00:00 first\n- 00:00 second\n" in text
    empty = journal.report(tool=PKG, journal_dir=tmp_path, day="2030-01-02")
    assert "no journal file for 2030-01-02" in empty and "## STUCK notes: 0" in empty
    with pytest.raises(ValueError):
        journal.report(tool=PKG, journal_dir=tmp_path, day="01/02/2030")
    capsys.readouterr()


def test_report_keeps_a_stuck_block_whole_across_blank_output_lines(tmp_path, frozen_clock, fresh_warning, capsys):
    """pytest output and chained tracebacks hold blank lines; the block must not end at the first one."""
    journal.stuck("tests fail", tool=PKG, journal_dir=tmp_path, stage="03-render", output="line1\n\nline3\n",
                  hypothesis="a stale fixture", need="maintainer")
    journal.note("a note\n\nits third line", tool=PKG, journal_dir=tmp_path)
    blocks, bullets = journal.split_day((tmp_path / "2030-01-01.md").read_text(encoding="utf-8"))
    assert len(blocks) == 1 and blocks[0].splitlines()[-1] == "need: maintainer"
    assert "    line3" in blocks[0] and "hypothesis: a stale fixture" in blocks[0]
    assert bullets == ["- 00:00 a note\n  \n  its third line"]
    text = journal.report(tool=PKG, journal_dir=tmp_path)
    assert "need: maintainer" in text and "its third line" in text
    capsys.readouterr()


# --- the verb -------------------------------------------------------------------------------------------------

def test_cli_stuck_template_prints_the_template_and_writes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    code, out, err = run(["journal", "stuck", "--template"], capsys)
    assert (code, out, err) == (0, journal.STUCK_TEMPLATE, "")
    assert list(tmp_path.iterdir()) == []
    code, _, err = run(["journal", "stuck", "--dir", str(tmp_path)], capsys)
    assert code == 2 and "needs a line" in err and list(tmp_path.iterdir()) == []


def test_cli_note_warns_on_stderr_and_keeps_stdout_clean(tmp_path, frozen_clock, fresh_warning, capsys):
    code, out, err = run(["journal", "note", "hello", "--dir", str(tmp_path)], capsys)
    assert code == 0
    assert out == f"noted in {tmp_path / '2030-01-01.md'}\n"
    assert "clock frozen" not in out
    assert err == WARNING + "\n"


def test_cli_uses_the_instance_journal_folder(tmp_path, frozen_clock, fresh_warning, capsys):
    (tmp_path / "easel.toml").write_bytes(b'[paths]\njournal = "notes"\n')
    code, out, _ = run(["journal", "note", "from the instance", "--workspace", str(tmp_path)], capsys)
    assert (code, out) == (0, "noted in notes/2030-01-01.md\n")
    assert (tmp_path / "notes" / "2030-01-01.md").exists()


def test_cli_stuck_reads_output_from_stdin_and_prints_the_block(tmp_path, frozen_clock, fresh_warning, monkeypatch,
                                                               capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Traceback\nboom\n"))
    code, out, _ = run(["journal", "stuck", "doctor fails", "--dir", str(tmp_path), "--stage", "01-setup",
                        "--command", "easel doctor", "--exit-code", "1", "--output", "-", "--tried", "once",
                        "--need", "maintainer"], capsys)
    assert code == 0
    assert out.startswith("## STUCK 2030-01-01 00:00 01-setup\nsummary: doctor fails\ncommand: easel doctor\n")
    assert "output (last 20 lines):\n    Traceback\n    boom\n" in out
    assert out.rstrip().endswith(f"recorded in {tmp_path / '2030-01-01.md'}")
    assert "need: maintainer" in (tmp_path / "2030-01-01.md").read_text(encoding="utf-8")


def test_cli_report_out_is_relative_to_the_instance(tmp_path, frozen_clock, fresh_warning, capsys):
    (tmp_path / "easel.toml").write_bytes(b"[paths]\n")
    run(["journal", "note", "x", "--workspace", str(tmp_path)], capsys)
    code, out, _ = run(["journal", "report", "--out", "journal/trial-2030-01-01.md", "--workspace", str(tmp_path)],
                       capsys)
    assert (code, out) == (0, "report written to journal/trial-2030-01-01.md\n")
    assert "- 00:00 x" in (tmp_path / "journal" / "trial-2030-01-01.md").read_text(encoding="utf-8")


def test_cli_without_an_instance_or_dir_is_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "nohome"))
    code, _, err = run(["journal", "note", "x"], capsys)
    assert code == 2 and "no easel.toml found" in err
    code, _, err = run(["journal", "note", "x", "--pitfall", "p", "--dir", str(tmp_path)], capsys)
    assert code == 2 and err.startswith("refused: a pitfall needs its rule")


def test_frozen_clock_line_once_per_process_on_stderr_only(tmp_path):
    """A real `python -m easel journal ...` process: stdout is the doc-safe part; stderr has the one warning."""
    env = {**{k: v for k, v in os.environ.items() if not k.upper().startswith(("EASEL_", "KILN_"))},
           "EASEL_NOW": "2030-01-01T00:00:00+00:00",
           "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run([sys.executable, "-m", PKG, "journal", "stuck", "demo", "--dir", str(tmp_path)],
                          capture_output=True, text=True, env=env, timeout=60)
    assert done.returncode == 0, done.stderr
    assert "clock frozen" not in done.stdout
    assert done.stdout.startswith("## STUCK 2030-01-01 00:00 <stage>\n")
    assert done.stderr.splitlines() == [WARNING]
