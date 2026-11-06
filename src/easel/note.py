"""`easel note <id> "<text>"`: attach a short note, in the person's words, to a resource, a recipe, a workflow or a run.

Why: a remark like "too saturated above 0.9" belongs next to the thing it is about, dated and attributed, so the
next session finds it with the thing instead of in a chat log.

Where a note goes (no new file format; each kind keeps its notes where its own records already have room):
- a resource (`<sha16>`, or a unique prefix of at least 6 hex digits), a recipe (`rcp-YYYYMMDD-NNN`) or a
  workflow template (its id): one line in `<home>/registry/changelog.jsonl` (easel/changelog-event/1) with
  `action: update`, no field changes and the text as `note`; a recipe's own `notes` field gets the line too, so
  `recipe show` prints it. Readers of the changelog take such a line as a note, never as a field change
  (`easel.registry.resources.is_note`, `read_changelog`);
- a run (`run-YYYYMMDD-<slug>`): the run's `notes` field in `runs/<run-id>/run.json` (a run is not a registry
  record, so it has no changelog line). Verdicts are not notes: they go through `easel runs review`.
Notes are appended, never replaced: each one is `<time> <who>: <text>` on its own line. `easel note <id>` without
text prints the notes an id has. Who: the operator by default; `--assistant` stores the line under the second
source (`policy.second_source_name`), which never stands for the operator's words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from easel._common import ids, store
from easel.cli import instance_not_found, new_instance_hint
from easel.registry.shipped import schema_errors, shown

RUN_ID = re.compile(r"^run-\d{8}-[a-z0-9][a-z0-9_-]*$")
RECIPE_ID = re.compile(r"^rcp-\d{8}-\d{3}$")
HEX = re.compile(r"^[0-9a-fA-F]{6,64}$")
WORKFLOW_ID = re.compile(r"^[a-z][a-z0-9_-]*$")
BY = "easel note"
MAX_CHARS = 2000


class NoteError(Exception):
    """A note that cannot be attached: `.lines` to print, exit 2."""

    def __init__(self, *lines: str) -> None:
        super().__init__("\n".join(lines))
        self.lines = list(lines)


@dataclass(frozen=True)
class Target:
    """What a note is about: `kind` (resource, recipe, workflow, run), its id and the file its note lands in."""

    kind: str
    id: str
    file: Path


def _instance_dir(cfg, key: str, default: str) -> Path:
    if cfg.get(key, None) is not None:
        return cfg.path(key)
    if cfg.root is None:
        raise NoteError(instance_not_found(), new_instance_hint())
    return cfg.root / default


def find_target(cfg, text: str) -> Target:
    """The record `text` names, checked to exist; NoteError when it names nothing (or more than one thing)."""
    from easel import workflows
    from easel.registry import resources as res

    word = str(text).strip()
    if RUN_ID.match(word):
        run_json = _instance_dir(cfg, "paths.runs", "runs") / word / "run.json"
        if not run_json.is_file():
            raise NoteError(f"no run {word}: {shown(run_json, cfg.root)} does not exist")
        return Target("run", word, run_json)
    if RECIPE_ID.match(word):
        recipe = _instance_dir(cfg, "paths.recipes", "recipes") / f"{word}.json"
        if not recipe.is_file():
            raise NoteError(f"no recipe {word}: {shown(recipe, cfg.root)} does not exist")
        return Target("recipe", word, recipe)
    reg = res.open_registry(cfg, "easel")
    if HEX.match(word):
        records = res.load_records(reg).records
        try:
            sha16 = res.resolve_id(records, word)
        except LookupError as exc:
            if not WORKFLOW_ID.match(word):
                raise NoteError(str(exc)) from None
        else:
            return Target("resource", sha16, reg.changelog_file)
    if WORKFLOW_ID.match(word) and word in workflows.discover(cfg):
        return Target("workflow", word, reg.changelog_file)
    raise NoteError(f"{word!r} names no run, recipe, resource or workflow template here",
                    "ids look like run-20300101-001, rcp-20300101-001, a resource's sha16 (or 6+ hex digits of it) "
                    "or a workflow id from `easel workflows list`")


def note_line(text: str, source: str, tool: str) -> str:
    """One stored note: `<time> <who>: <text>`, whitespace folded to single spaces."""
    clean = " ".join(str(text).split())
    if not clean:
        raise NoteError("the note is empty")
    if len(clean) > MAX_CHARS:
        raise NoteError(f"a note is at most {MAX_CHARS} characters ({len(clean)} given); write a lesson or a "
                        "journal note for more")
    return f"{ids.now_iso(tool)} {source}: {clean}"


def _append(old: object, line: str) -> str:
    return f"{old}\n{line}" if isinstance(old, str) and old.strip() else line


def _update_json_notes(path: Path, line: str, schema: str, tool: str) -> tuple[str | None, str]:
    """Append `line` to the `notes` field of a JSON record under its folder's lock; (old, new)."""
    import json

    with store.lock(path.parent, tool):
        record = json.loads(path.read_text(encoding="utf-8-sig"))
        old = record.get("notes")
        record["notes"] = _append(old, line)
        problems = schema_errors(schema, record)
        if problems:
            raise NoteError(f"note not written: {path.name} would fail its schema:", *[f"  {p}" for p in problems])
        store.write_json(path, record)
    return old, record["notes"]


def add(cfg, target: Target, text: str, *, source: str = "operator", tool: str = "easel") -> str:
    """Attach the note; returns the stored line."""
    from easel.registry import resources as res

    line = note_line(text, source, tool)
    if target.kind == "run":
        _update_json_notes(target.file, line, "run", tool)
        return line
    changes: dict = {}
    if target.kind == "recipe":
        old, new = _update_json_notes(target.file, line, "recipe", tool)
        changes = {"notes": {"from": old, "to": new}}
    reg = res.open_registry(cfg, tool)
    event = res.changelog_line(tool=tool, source=source, by=BY, action="update", kind=target.kind, target=target.id,
                               changes=changes, note=" ".join(str(text).split()))
    problems = schema_errors("changelog-event", event)
    if problems:
        raise NoteError("note not written: the changelog line fails its schema:", *[f"  {p}" for p in problems])
    res.append_changelog(reg, [event])
    return line


def notes_of(cfg, target: Target) -> list[str]:
    """The notes an id has, oldest first, as `<time> <who>: <text>` lines."""
    import json

    if target.kind == "run":
        notes = json.loads(target.file.read_text(encoding="utf-8-sig")).get("notes")
        return [n for n in str(notes).splitlines() if n.strip()] if notes else []
    from easel.registry import resources as res

    out = []
    for rec in store.read_jsonl(target.file).records:
        tgt = rec.get("target") or {}
        # only note lines: a validator's remark on a field change (workflows validate) is not the person's note
        if tgt.get("kind") == target.kind and tgt.get("id") == target.id and res.is_note(rec):
            out.append(f"{rec.get('at')} {rec.get('source')}: {rec.get('note')}")
    return out
