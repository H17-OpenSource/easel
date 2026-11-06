"""The resource records: read, override, validate and write them, keep the changelog and the generated index.

Files (all under `<home>/registry/`, named by `paths.registry`):
- `resources/<sha16>.yaml`: one tool-owned record per model file (schema easel/resource/1); the stem is the first
  16 hex digits of the file's sha256. `<sha16>.usage.yaml` beside it is the author's usage text (another schema).
- `overrides.yaml`: the person's corrections (schema easel/overrides/1), for example "this LoRA is self-trained".
  A scan writes them into the records; every read applies them too, so a fresh override shows at once.
- `changelog.jsonl`: one line per change (schema easel/changelog-event/1), appended under the store lock. A line
  with action `update`, a `note` and no field changes is a note (`easel note`), not a change: readers that count
  or replay changes take `read_changelog(reg).changes`, which leaves the notes out (`is_note`).
- `index.jsonl` + `INDEX.md`: the generated view, rebuilt from the records by every writing command and by
  `easel registry index`; no timestamp inside, so an unchanged registry rebuilds byte-identically.

Every record is checked against `schemas/resource.schema.json` before it is written, so a bad record never lands.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from easel._common import store

from .shipped import DataError, load_yaml, reason, schema_errors, shown

RESOURCE_SCHEMA = "easel/resource/1"
CHANGELOG_SCHEMA = "easel/changelog-event/1"
SHA16 = re.compile(r"^[0-9a-f]{16}$")
INDEX_COLUMNS = ("id", "file_name", "type", "family", "rating", "source_class", "status", "licence")
INDEX_TITLE = "Registry: resources"
TYPE_ORDER = ("checkpoint", "unet", "lora", "lycoris", "embedding", "vae", "text_encoder", "clip", "clip_vision",
              "controlnet", "ipadapter", "upscaler", "detector", "sam", "style_model", "model_patch")


@dataclass
class Registry:
    """Where one instance keeps its registry, and the tool name every write is made under."""

    tool: str
    root: Path | None
    registry: Path
    models_root: Path
    standards: Path
    extra_roots: tuple[Path, ...] = ()  # paths.extra_model_roots: more folders laid out like models_root

    @property
    def model_roots(self) -> tuple[Path, ...]:
        """models_root first, then the extra roots: the order a scan walks them and ComfyUI searches them."""
        return (self.models_root, *self.extra_roots)

    def model_file(self, rel: str) -> Path:
        """The file a record's `identity.path` names: the first root that holds it (models_root when none does).

        `identity.path` is relative to the root the file was found under; like ComfyUI with several base paths,
        the first root that has the name wins, so a scan reports a later root's same name as shadowed.
        """
        for base in self.model_roots:
            candidate = base / rel
            if candidate.is_file():
                return candidate
        return self.models_root / rel

    def root_label(self, index: int) -> str:
        """How output names a model root: `<models_root>`, then `<extra_model_roots[0]>`, ..."""
        return "<models_root>" if index == 0 else f"<extra_model_roots[{index - 1}]>"

    @property
    def pending_dir(self) -> Path:
        return self.registry / "pending"

    @property
    def resources_dir(self) -> Path:
        return self.registry / "resources"

    @property
    def overrides_file(self) -> Path:
        return self.registry / "overrides.yaml"

    @property
    def changelog_file(self) -> Path:
        return self.registry / "changelog.jsonl"

    def record_path(self, sha16: str) -> Path:
        return self.resources_dir / f"{sha16}.yaml"

    def shown(self, path: Path) -> str:
        """A path as output shows it: relative to the instance root, never absolute."""
        return shown(path, self.root)


def open_registry(cfg, tool: str = "easel") -> Registry:
    """The registry of a loaded configuration (`_common.config.load`)."""
    standards = cfg.get("paths.standards", "standards")
    root = cfg.root
    extra: list[Path] = []
    for value in cfg.get("paths.extra_model_roots", None) or []:
        path = Path(str(value)).expanduser()
        extra.append(path if path.is_absolute() or root is None else root / path)
    return Registry(tool=tool, root=root, registry=cfg.path("paths.registry"),
                    models_root=cfg.path("paths.models_root"),
                    standards=(root / standards) if root is not None else Path(standards),
                    extra_roots=tuple(extra))


# --- dotted fields ----------------------------------------------------------------------------------------------------

_MISSING = object()


def get_field(record: Mapping, dotted: str, default: object = None) -> object:
    """The value at a dotted key (`provenance.source_class`), or `default` when a part is missing or null."""
    node: object = record
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def set_field(record: dict, dotted: str, value: object) -> None:
    """Set a dotted key, creating (or replacing a null with) the mappings on the way."""
    parts = dotted.split(".")
    node = record
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = value


def diff(old: Mapping | None, new: Mapping, prefix: str = "") -> dict[str, dict]:
    """Field-by-field changes {dotted: {from, to}} between two records (leaves and lists compared whole)."""
    changes: dict[str, dict] = {}
    old = old or {}
    for key in dict.fromkeys([*old.keys(), *new.keys()]):
        a, b = old.get(key, None), new.get(key, None)
        dotted = f"{prefix}{key}"
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            changes.update(diff(a, b, dotted + "."))
        elif a != b:
            changes[dotted] = {"from": a, "to": b}
    return changes


# --- records ----------------------------------------------------------------------------------------------------------

@dataclass
class Loaded:
    """The records of one registry: by sha16, plus a line per file that could not be used."""

    records: dict[str, dict] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    unreadable: set[str] = field(default_factory=set)


def load_records(reg: Registry) -> Loaded:
    """Every `resources/<sha16>.yaml`; a file that does not parse or fails the schema is listed, not used."""
    out = Loaded()
    folder = reg.resources_dir
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.yaml")):
        if not SHA16.match(path.stem) or ".tmp-" in path.name:
            continue  # <sha16>.usage.yaml and other companions have their own schemas
        try:
            data = store.read_record(path, "easel/resource/")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            out.problems.append(f"{reg.shown(path)}: not readable: {reason(exc)}")
            out.unreadable.add(path.stem)
            continue
        errors = schema_errors("resource", data)
        if data.get("id") != path.stem:
            errors.insert(0, f"id {data.get('id')!r} differs from the file name")
        if errors:
            out.problems.append(f"{reg.shown(path)}: not a valid {RESOURCE_SCHEMA} record: {errors[0]}")
            out.unreadable.add(path.stem)
            continue
        out.records[path.stem] = data
    return out


def ordered(record: Mapping) -> dict:
    """The record with its keys in the schema's order (stable files, readable diffs)."""
    keys = ("schema", "id", "legacy_id", "identity", "provenance", "integrity", "lora", "semantics", "licence",
            "when_to_use", "usage", "relations", "lifecycle", "legacy")
    out = {k: record[k] for k in keys if k in record}
    out.update({k: v for k, v in record.items() if k not in out})
    return out


def write_record(reg: Registry, record: Mapping) -> Path:
    """Validate and write one record atomically; DataError (nothing written) when it fails the schema."""
    data = ordered({**record, "schema": RESOURCE_SCHEMA})
    errors = schema_errors("resource", data)
    if errors:
        raise DataError([f"resource {data.get('id')}: not written, it fails {RESOURCE_SCHEMA}:"] +
                        [f"  {e}" for e in errors])
    return store.write_record(reg.record_path(str(data["id"])), RESOURCE_SCHEMA, data)


def resolve_id(records: Mapping[str, Mapping], text: str) -> str:
    """The sha16 a command-line argument names: the full 16 hex digits, or a unique prefix of at least 6."""
    want = text.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{6,64}", want):
        raise LookupError(f"not a resource id (6 to 16 hex digits of the file's sha256): {text!r}")
    want = want[:16]
    hits = [k for k in records if k.startswith(want)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise LookupError(f"no resource {want} in the registry; `easel registry list` shows the ids")
    raise LookupError(f"{want} matches {len(hits)} resources ({', '.join(sorted(hits)[:5])}); give more digits")


def sort_key(record: Mapping) -> tuple:
    rtype = get_field(record, "identity.type", "")
    rank = TYPE_ORDER.index(rtype) if rtype in TYPE_ORDER else len(TYPE_ORDER)
    return (rank, str(get_field(record, "identity.file_name", "")).lower(), str(record.get("id")))


# --- overrides --------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Override:
    """One resource's entry in overrides.yaml: dotted field -> value, with the person's note."""

    sha16: str
    values: Mapping[str, object]
    note: str | None
    at: str | None


def load_overrides(reg: Registry) -> dict[str, Override]:
    """overrides.yaml by sha16; an absent file is no overrides, a malformed one is DataError (exit 2)."""
    path = reg.overrides_file
    if not path.is_file():
        return {}
    data = load_yaml(path, reg.shown(path))
    if data is None:
        return {}
    problems = schema_errors("overrides", data)
    if problems or not isinstance(data, dict):
        raise DataError([f"{reg.shown(path)} is not a valid easel/overrides/1 file:"] + [f"  {p}" for p in problems])
    return {sha: Override(sha16=sha, values=dict(entry["set"]), note=entry.get("note"), at=entry.get("at"))
            for sha, entry in (data.get("resources") or {}).items()}


def apply_override(record: Mapping, override: Override | None) -> tuple[dict, dict[str, dict], list[str]]:
    """The record with the override's values set, the changes it made, and a line per value refused.

    Each value is applied on its own and kept only when the record still passes the schema, so one bad value
    (a typo, a wrong type) cannot block the others.
    """
    current = copy.deepcopy(dict(record))
    changes: dict[str, dict] = {}
    refused: list[str] = []
    if override is None:
        return current, changes, refused
    for dotted, value in override.values.items():
        before = get_field(current, dotted, _MISSING)
        if before == value:
            continue
        trial = copy.deepcopy(current)
        set_field(trial, dotted, value)
        errors = schema_errors("resource", trial)
        if errors:
            refused.append(f"override {override.sha16} {dotted}: refused, the record would fail the schema "
                           f"({errors[0]})")
            continue
        current = trial
        changes[dotted] = {"from": None if before is _MISSING else before, "to": value}
    return current, changes, refused


def effective(records: Mapping[str, Mapping], overrides: Mapping[str, Override]) -> dict[str, dict]:
    """Every record with its override applied: the view every read command shows."""
    return {sha: apply_override(rec, overrides.get(sha))[0] for sha, rec in records.items()}


# --- changelog and index ----------------------------------------------------------------------------------------------

def changelog_line(*, tool: str, source: str, by: str, action: str, kind: str, target: str,
                   changes: Mapping[str, Mapping], note: str | None = None) -> dict:
    """One changelog event, `schema` first, the way the demo's lines read."""
    from easel._common import ids

    return {"schema": CHANGELOG_SCHEMA, "at": ids.now_iso(tool), "v": 1, "source": source, "by": by,
            "action": action, "target": {"kind": kind, "id": target}, "changes": dict(changes), "note": note,
            "legacy_id": None}


def append_changelog(reg: Registry, lines: Iterable[Mapping]) -> int:
    """Append the lines to changelog.jsonl in one write under the store lock; returns how many."""
    lines = list(lines)
    if not lines:
        return 0
    data = "".join(json.dumps(line, ensure_ascii=False, default=str) + "\n" for line in lines).encode("utf-8")
    with store.lock(reg.registry, reg.tool):
        store.append_bytes_locked(reg.changelog_file, data)
    return len(lines)


def is_note(event: Mapping) -> bool:
    """True for a note line (`easel note`): action `update`, a `note`, and no field changes.

    A note on a recipe also records the recipe's own `notes` field (`changes: {notes: ...}`); that is still a note.
    """
    if event.get("action") != "update" or not event.get("note"):
        return False
    changes = event.get("changes") or {}
    return not changes or set(changes) == {"notes"}


@dataclass
class Changelog:
    """`changelog.jsonl` split for its readers: the field changes, the notes, and the line numbers not read."""

    changes: list[dict] = field(default_factory=list)
    notes: list[dict] = field(default_factory=list)
    bad_lines: list[int] = field(default_factory=list)


def read_changelog(reg: Registry) -> Changelog:
    """Every changelog line, notes apart from changes; a missing file is an empty changelog."""
    read = store.read_jsonl(reg.changelog_file)
    out = Changelog(bad_lines=list(read.bad_lines))
    for event in read.records:
        if str(event.get("schema", CHANGELOG_SCHEMA)) != CHANGELOG_SCHEMA:
            continue
        (out.notes if is_note(event) else out.changes).append(dict(event))
    return out


def index_row(record: Mapping) -> dict:
    """The generated-view row of one record (also what `registry list --json` prints)."""
    return {"id": record.get("id"),
            "file_name": get_field(record, "identity.file_name"),
            "type": get_field(record, "identity.type"),
            "family": get_field(record, "identity.family"),
            "rating": get_field(record, "semantics.content.rating"),
            "class": get_field(record, "semantics.content.class"),
            "source_class": get_field(record, "provenance.source_class"),
            "status": get_field(record, "lifecycle.install_status"),
            "licence": get_field(record, "licence.status"),
            "path": get_field(record, "identity.path")}


def rebuild_index(reg: Registry, records: Mapping[str, Mapping]) -> tuple[Path, Path]:
    """Rewrite registry/index.jsonl and registry/INDEX.md from `records` (call with the effective view)."""
    rows = [index_row(rec) for rec in sorted(records.values(), key=sort_key)]
    with store.lock(reg.registry, reg.tool):
        return store.write_index(reg.registry, rows, columns=INDEX_COLUMNS, title=INDEX_TITLE, tool=reg.tool)
