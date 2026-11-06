"""`easel note <id> "<text>"` on the demo: a resource, a recipe and a workflow note go to the registry changelog,
a run note into run.json; notes are appended, never replaced."""

from __future__ import annotations

import json
from pathlib import Path

from easel.cli import main
from easel.registry.shipped import schema_errors

LORA = "cf56db17414bcb55"
RUN = "run-20300101-001"
RCP = "rcp-20300101-001"


def _changelog(home: Path) -> list[dict]:
    text = (home / "registry" / "changelog.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def test_a_resource_note_is_a_changelog_line(demo_workspace, frozen_clock, capsys):
    assert main(["note", LORA[:8], "too saturated above 0.9"]) == 0
    assert capsys.readouterr().out.strip() == f"note added to resource {LORA} (registry/changelog.jsonl, by operator)"
    line = _changelog(demo_workspace)[-1]
    assert schema_errors("changelog-event", line) == []
    assert line["target"] == {"kind": "resource", "id": LORA} and line["note"] == "too saturated above 0.9"
    assert line["changes"] == {} and line["by"] == "easel note"
    assert main(["note", LORA]) == 0
    assert capsys.readouterr().out.splitlines() == ["2030-01-01T00:00:00+00:00 operator: too saturated above 0.9",
                                                    f"1 note on resource {LORA}"]


def test_run_notes_are_appended_to_run_json(demo_workspace, frozen_clock, capsys):
    assert main(["note", RUN, "the operator liked output 2"]) == 0
    assert main(["note", RUN, "redo with warmer light", "--assistant"]) == 0
    run = json.loads((demo_workspace / "runs" / RUN / "run.json").read_text(encoding="utf-8"))
    assert schema_errors("run", run) == []
    assert run["notes"].splitlines() == ["2030-01-01T00:00:00+00:00 operator: the operator liked output 2",
                                         "2030-01-01T00:00:00+00:00 assistant: redo with warmer light"]


def test_a_recipe_note_lands_in_the_recipe_and_the_changelog(demo_workspace, frozen_clock, capsys):
    assert main(["note", RCP, "try a warmer light"]) == 0
    recipe = json.loads((demo_workspace / "recipes" / f"{RCP}.json").read_text(encoding="utf-8"))
    assert schema_errors("recipe", recipe) == [] and recipe["notes"].endswith("operator: try a warmer light")
    line = _changelog(demo_workspace)[-1]
    assert line["target"] == {"kind": "recipe", "id": RCP} and line["changes"]["notes"]["from"] is None


def test_a_workflow_note(demo_workspace, capsys):
    assert main(["note", "detailer", "needs a face detector file"]) == 0
    assert _changelog(demo_workspace)[-1]["target"] == {"kind": "workflow", "id": "detailer"}


def test_unknown_ids_and_empty_notes_are_refused(demo_workspace, capsys):
    before = len(_changelog(demo_workspace))
    assert main(["note", "run-20300101-999", "x"]) == 2
    assert main(["note", "nothing-here", "x"]) == 2
    assert main(["note", LORA, "   "]) == 2
    assert "empty" in capsys.readouterr().err
    assert len(_changelog(demo_workspace)) == before


def test_changelog_readers_take_note_lines_as_notes_not_changes(demo_workspace, frozen_clock, capsys):
    """A note line (action update, changes {}) must never count as a field change for audit or standards readers."""
    from easel._common import config
    from easel.registry import resources as res

    reg = res.open_registry(config.load("easel", demo_workspace, strict=True), "easel")
    before = res.read_changelog(reg)
    assert before.notes == [] and before.changes  # the demo's scan lines are changes
    assert main(["note", LORA, "too saturated above 0.9"]) == 0
    assert main(["note", RCP, "try a warmer light"]) == 0
    after = res.read_changelog(reg)
    assert after.changes == before.changes  # notes add no change
    assert [(n["target"]["kind"], n["note"]) for n in after.notes] == [("resource", "too saturated above 0.9"),
                                                                      ("recipe", "try a warmer light")]
    assert all(res.is_note(n) for n in after.notes)
    validated = {"action": "update", "note": "checked on the mock server",
                 "changes": {"validated_at": {"from": None, "to": "2030-01-01T00:00:00+00:00"}}}
    assert not res.is_note(validated)  # a remark beside a real field change stays a change
    assert not res.is_note({"action": "scan", "note": None, "changes": {}})
