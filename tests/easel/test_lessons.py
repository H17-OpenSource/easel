"""`easel lessons add|list|render` on the demo instance under the frozen clock."""

from __future__ import annotations

import json
from pathlib import Path

from easel import lessons
from easel.cli import main
from easel.registry.shipped import schema_errors

ADD = ["lessons", "add", "--family", "illustrious", "--finding", "The ink LoRA bleeds above 0.7.",
       "--evidence", "run-20300101-001 output 3 rejected", "--rule", "Keep the ink LoRA at 0.6 or less."]


def _lines(home: Path) -> list[dict]:
    text = (home / "lessons" / "entries.jsonl").read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def test_add_appends_a_schema_valid_line_with_the_next_id(demo_workspace, frozen_clock, capsys):
    assert main([*ADD, "--source", "review", "--topic", "lora weight"]) == 0
    out = capsys.readouterr().out
    line = _lines(demo_workspace)[-1]
    assert schema_errors("lessons-entry", line) == []
    assert line["id"] == "les-20300101-002" and line["at"] == "2030-01-01T00:00:00+00:00"
    assert (line["source"], line["origin"], line["family"]) == ("operator", "review", "illustrious")
    assert out.splitlines() == ["les-20300101-002 added to lessons/entries.jsonl (family illustrious, source review, "
                                "recorded by operator)", "next: easel lessons render"]


def test_civitai_feedback_and_the_assistants_own_finding(demo_workspace, frozen_clock, capsys):
    assert main([*ADD, "--source", "civitai"]) == 0
    assert main([*ADD, "--assistant"]) == 0
    first, second = _lines(demo_workspace)[-2:]
    assert first["origin"] == "civitai" and first["source"] == "operator"
    assert second["source"] == "assistant" and second["origin"] == "manual"
    assert second["id"] == "les-20300101-003"


def test_unknown_family_and_empty_text_are_refused(demo_workspace, capsys):
    before = len(_lines(demo_workspace))
    assert main(["lessons", "add", "--family", "nosuch", "--finding", "a", "--evidence", "b", "--rule", "c"]) == 2
    assert "unknown family 'nosuch'" in capsys.readouterr().err
    assert main(["lessons", "add", "--finding", "  ", "--evidence", "b", "--rule", "c"]) == 2
    assert len(_lines(demo_workspace)) == before


def test_render_writes_the_four_column_page(demo_workspace, frozen_clock, capsys):
    assert main(ADD) == 0
    assert main(["lessons", "add", "--finding", "Hires above 1.5 blurs fine lines.", "--evidence", "two runs",
                 "--rule", "Keep hires.scale at 1.5 or less."]) == 0
    assert main(["lessons", "render"]) == 0
    out = capsys.readouterr().out
    assert "2 pages: 2 written, 0 unchanged, 0 refused" in out
    page = (demo_workspace / "lessons" / "illustrious.md").read_text(encoding="utf-8")
    assert lessons.MARKER in page and "| Finding | Evidence | Rule | Recorded in |" in page
    assert "| lora weight: The demo_robot LoRA at 0.8" in page and "les-20300101-002 (2030-01-01, manual)" in page
    assert (demo_workspace / "lessons" / "general.md").is_file()
    assert main(["lessons", "render"]) == 0
    assert "0 written, 2 unchanged" in capsys.readouterr().out


def test_render_never_overwrites_a_page_it_did_not_generate(demo_workspace, tmp_path, capsys):
    out = tmp_path / "docs-lessons"
    out.mkdir()
    (out / "illustrious.md").write_text("# hand-written\n", encoding="utf-8")
    assert main(["lessons", "render", "--out", str(out)]) == 1
    assert "refused" in capsys.readouterr().err
    assert (out / "illustrious.md").read_text(encoding="utf-8") == "# hand-written\n"


def test_list_filters_by_family(demo_workspace, capsys):
    assert main(["lessons", "add", "--finding", "f", "--evidence", "e", "--rule", "r"]) == 0
    capsys.readouterr()
    assert main(["lessons", "list", "--family", "illustrious"]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "1 lesson"
