"""The current verdict per output (`easel runs reviews --current`), and `runs review|show|list`."""

from __future__ import annotations

import json

import pytest
from studio_support import RUN, run_cli, schema_problems


@pytest.fixture
def home(demo_workspace, frozen_clock):
    return demo_workspace


def _current(home) -> dict:
    from easel.studio import common, ledger

    return ledger.current_reviews(common.open_studio("easel"))


def test_current_verdicts_on_the_demo(home):
    current = _current(home)
    assert {key: row["verdict"] for key, row in current.items()} == {
        (RUN, 1): "keep", (RUN, 2): "keep", (RUN, 3): "reject", (RUN, 4): "keep"}
    assert current[(RUN, 4)]["note"] == "on a second look the vest is orange"  # the later line replaced "maybe"


def test_reviews_current_cli(home, capsys):
    code, out, _ = run_cli(["runs", "reviews", "--current"], capsys)
    assert code == 0
    lines = out.splitlines()
    assert len(lines) == 4
    assert lines[3].startswith(f"{RUN}  output 4   keep    operator")
    code, out, _ = run_cli(["runs", "reviews", RUN], capsys)
    assert code == 0 and len(out.splitlines()) == 5  # the history keeps every line
    code, out, _ = run_cli(["runs", "reviews", RUN, "--current", "--json"], capsys)
    rows = [json.loads(line) for line in out.splitlines()]
    assert [r["n"] for r in rows] == [1, 2, 3, 4]


def test_a_changed_mind_replaces_the_verdict(home, capsys):
    code, out, err = run_cli(["runs", "review", RUN, "--verdict", "reject", "--output", "1", "--note",
                              "the visor is too dark", "--checks", "face=fail,hands=na"], capsys)
    assert code == 0, err
    assert out.splitlines()[0] == f"recorded reject on {RUN} output 1 (operator)"
    current = _current(home)
    assert current[(RUN, 1)]["verdict"] == "reject" and len(current) == 4
    last = (home / "runs" / RUN / "reviews.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    event = json.loads(last)
    assert schema_problems("run-review-event", event) == []
    assert (event["at"], event["via"], event["checks"]) == ("2030-01-01T00:00:00+00:00", "cli",
                                                             {"face": "fail", "hands": "na"})


def test_a_whole_run_verdict_is_its_own_row(home, capsys):
    assert run_cli(["runs", "review", RUN, "--verdict", "maybe", "--note", "as a set"], capsys)[0] == 0
    current = _current(home)
    assert current[(RUN, None)]["verdict"] == "maybe" and current[(RUN, 1)]["verdict"] == "keep"
    code, out, _ = run_cli(["runs", "show", RUN], capsys)
    assert code == 0 and "  whole run: maybe (operator) - as a set" in out.splitlines()


def test_a_torn_last_line_is_skipped_and_the_next_append_is_clean(home, capsys):
    path = home / "runs" / RUN / "reviews.jsonl"
    with open(path, "ab") as handle:
        handle.write(b'{"schema": "easel/run-review-event/1", "run": "' + RUN.encode() + b'", "n": 2, "verd')
    assert _current(home)[(RUN, 2)]["verdict"] == "keep"
    assert run_cli(["runs", "review", RUN, "--verdict", "reject", "--output", "2"], capsys)[0] == 0
    assert _current(home)[(RUN, 2)]["verdict"] == "reject"


def test_verdicts_are_the_operators(home, capsys):
    code, _out, err = run_cli(["runs", "review", RUN, "--verdict", "keep", "--source", "assistant"], capsys)
    assert code == 2 and "policy.reviewer = human-only" in err
    code, _out, err = run_cli(["runs", "review", RUN, "--verdict", "keep", "--output", "9"], capsys)
    assert code == 2 and "has outputs 1, 2, 3, 4" in err
    code, _out, err = run_cli(["runs", "review", RUN, "--verdict", "keep", "--checks", "face=ok"], capsys)
    assert code == 2 and "--checks" in err


def test_runs_list_and_show(home, capsys):
    code, out, _ = run_cli(["runs", "list"], capsys)
    assert code == 0
    assert out.splitlines()[0].startswith(f"{RUN}  ok        recipe rcp-20300101-001")
    assert "3 keep, 1 reject" in out
    code, out, _ = run_cli(["runs", "show", RUN], capsys)
    lines = out.splitlines()
    assert code == 0 and lines[0].startswith(f"{RUN}  ok  recipe rcp-20300101-001  job job-20300101-001")
    assert "  output 3: outputs/003.png  8x8  seed 1003  reject (operator) - the visor split into two eyes" in lines
    code, _out, err = run_cli(["runs", "show", "run-20300101-099"], capsys)
    assert code == 2 and "no run run-20300101-099" in err
