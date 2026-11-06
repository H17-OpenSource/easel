"""`easel registry scan [--hash]`: walk `<models_root>`, classify every model file, write the records.

Two passes. First every model file under `<models_root>` is classified (folder + header, `classify.py`) and,
with `--hash`, hashed; then each distinct sha256 becomes or updates `resources/<sha16>.yaml`.

Rules carried over from the scanner the file registry replaces, each learnt the hard way:
- A scan never blanks what only a person or an API can supply. It owns the file's place and bytes (path, file
  name, integrity, install status, scan time); everything else it fills only where the record has nothing.
- A file at a known path keeps its record. Without `--hash` an unchanged file (same size) keeps its id; a new
  file is reported, not registered, because its id is its hash. With `--hash` a file whose size and time match
  its record is not read again (`--rehash` reads everything).
- Records are never deleted: a file no longer under `<models_root>` turns `missing`, and comes back `active`.
- Online-only placeholders of a cloud-sync client are not read (reading would download them): they are
  reported and neither hashed nor marked missing.
- The same bytes at two paths are one resource (one sha256); the second path is reported as a duplicate.
- `registry/overrides.yaml` is applied last, so the person's word wins; each value applied is a changelog line.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from easel._common import hashing, ids

from . import classify as cls
from . import resources as res
from .shipped import DataError, reason

SKIP_DIRS = frozenset({"__pycache__", ".git", "venv", ".venv", "site-packages", "node_modules"})
# Windows file attributes of a cloud-sync placeholder whose bytes are not on this disk.
_PLACEHOLDER_ATTRS = 0x1000 | 0x40000 | 0x400000  # OFFLINE | RECALL_ON_OPEN | RECALL_ON_DATA_ACCESS
_TIMESTAMPS = ("lifecycle.scanned_at", "lifecycle.updated_at", "integrity.hashed_at")
LIVE_STATUSES = ("active", "staged", "pending", "missing")  # statuses a scan may set; the rest are the person's


class ScanRefused(Exception):
    """The scan cannot start (no models folder, no header reader): exit 2 with these lines."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__("\n".join(lines))
        self.lines = lines


@dataclass
class Found:
    """One model file found under models_root in pass one."""

    rel: str
    path: Path
    size: int
    mtime: float
    classified: cls.Classified
    sha256: str | None = None
    hashed_now: bool = False


@dataclass
class ScanReport:
    """What a scan did, as the lines it prints and the counts of its summary."""

    files: int = 0
    lines: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    unchanged: int = 0
    missing: list[str] = field(default_factory=list)
    unhashed: int = 0
    skipped: int = 0
    duplicates: int = 0
    conflicts: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    written: int = 0
    changelog: int = 0
    hashed: int = 0
    index: tuple[Path, Path] | None = None
    records: dict[str, dict] = field(default_factory=dict)

    def say(self, line: str) -> None:
        self.lines.append(line)


def _placeholder(stat: os.stat_result) -> bool:
    return bool(getattr(stat, "st_file_attributes", 0) & _PLACEHOLDER_ATTRS)


def _modified_since(mtime: float, stamp: object) -> bool:
    """True when the file's mtime is later than the record's hashed_at (or hashed_at is unknown)."""
    if not isinstance(stamp, str) or not stamp:
        return True
    try:
        when = datetime.fromisoformat(stamp.replace(" ", "T"))
    except ValueError:
        return True
    return mtime > when.timestamp()


def _walk(root: Path):
    """Model files under `root` in a stable order, skipping caches, VCS folders and hidden folders."""
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith("."))
        for name in sorted(files):
            path = Path(folder) / name
            if path.suffix.lower() in cls.MODEL_SUFFIXES and ".tmp-" not in name:
                yield path


def walk_roots(reg: res.Registry, say: Callable[[str], None], seen: set[str] | None = None):
    """(path, rel) for every model file under models_root, then under each of `paths.extra_model_roots`.

    `rel` is relative to the root the file is under. A name an earlier root already has is shadowed (ComfyUI loads
    the earlier one) and reported, not yielded; a missing extra root is reported and skipped. `seen` collects the
    lower-cased names yielded, so a caller can tell which records lost their file.
    """
    seen = set() if seen is None else seen
    for index, base in enumerate(reg.model_roots):
        if index and not base.is_dir():
            say(f"skipped {reg.root_label(index)} (paths.extra_model_roots): not a folder")
            continue
        for path in _walk(base):
            rel = path.relative_to(base).as_posix()
            if rel.lower() in seen:
                say(f"shadowed {reg.root_label(index)}/{rel}: an earlier model root has the same name, which "
                    "ComfyUI loads instead; not registered")
                continue
            seen.add(rel.lower())
            yield path, rel


def _header(path: Path) -> tuple[cls.Header | None, str]:
    suffix = path.suffix.lower()
    if suffix in cls.PICKLE_SUFFIXES:
        return None, "pickle format: never opened (loading one runs code); only hashed"
    if suffix in cls.GGUF_SUFFIXES:
        return None, "GGUF: the header is not read; type and family come from the folder and the name"
    try:
        return cls.read_header(path), "safetensors header read"
    except ValueError as exc:
        return None, f"safetensors header unreadable: {str(exc).splitlines()[0]}"
    except OSError as exc:
        return None, f"not readable: {reason(exc)}"


def _relative(path: Path | None, root: Path | None) -> str | None:
    if path is None or root is None or not path.is_file():
        return None
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None


def new_record(found: Found, reg: res.Registry, now: str) -> dict:
    """The record of a file the registry has never seen: what the folder and header say, the rest unknown."""
    c, sha256 = found.classified, str(found.sha256)
    rtype = str(c.type.value) if c.type else "checkpoint"
    family = str(c.family.value) if c.family else None
    facts = cls.integrity_facts(c.header) if c.header else {"dtypes": [], "tensor_count": None,
                                                           "param_count": None, "precision": None}
    sha16 = sha256[:16]
    return res.ordered({
        "schema": res.RESOURCE_SCHEMA, "id": sha16, "legacy_id": None,
        "identity": {"type": rtype, "file_name": found.path.name, "path": found.rel, "family": family,
                     "base_model_label": None, "title": cls.title_from_header(c.header)},
        "provenance": {"source_class": "unknown", "site": None, "url": None, "download_url": None, "author": None,
                       "model_id": None, "version_id": None, "upstream_revision": None, "downloaded_at": None,
                       "status": "unverified", "record": None, "trained_by": None},
        "integrity": {"sha256": sha256, "bytes": found.size, "tensor_count": facts["tensor_count"],
                      "param_count": facts["param_count"], "dtypes": facts["dtypes"], "precision": facts["precision"],
                      "quantization": cls.quantization(found.path), "hashed_at": now},
        "lora": cls.lora_facts(c.header, family) if rtype in ("lora", "lycoris") else None,
        "semantics": {"description": None, "category": None, "tags": [], "content": dict(c.content)},
        "licence": {"status": "unknown", "name": None, "spdx": None, "url": None, "commercial_use": None,
                    "credit_required": None, "derivatives_allowed": None, "relicense_allowed": None,
                    "redistributable": None, "gated": None, "source": None, "checked_at": None, "note": None},
        "when_to_use": None,
        "usage": {"card": _relative(reg.standards / "cards" / f"{sha16}.yaml", reg.root),
                  "author_notes": _relative(reg.resources_dir / f"{sha16}.usage.yaml", reg.root)},
        "relations": [],
        "lifecycle": {"install_status": "active", "tested_at": None, "test_verdict": None, "favorite": False,
                      "notes": None, "created_at": now, "updated_at": now, "scanned_at": now},
    })


def _fill(record: dict, dotted: str, value: object) -> None:
    """Set a field only where the record has nothing (None, absent or an empty list)."""
    if value is None:
        return
    if res.get_field(record, dotted) in (None, []):
        res.set_field(record, dotted, value)


def update_record(old: dict, found: Found, reg: res.Registry, now: str, markers: cls.Markers) -> dict:
    """A known record brought up to date: the scan-owned fields set, every other field only filled when empty."""
    rec = res.ordered(copy.deepcopy(old))
    c = found.classified
    rec["identity"]["path"], rec["identity"]["file_name"] = found.rel, found.path.name
    integrity = rec["integrity"]
    integrity["sha256"], integrity["bytes"] = found.sha256, found.size
    if found.hashed_now:
        integrity["hashed_at"] = now
    if c.header is not None:
        integrity.update(cls.integrity_facts(c.header))
    integrity["quantization"] = integrity.get("quantization") or cls.quantization(found.path)
    family = str(c.family.value) if c.family else None
    if rec["identity"].get("family") is None and family:
        rec["identity"]["family"] = family
        if isinstance(rec.get("lora"), dict) and rec["lora"].get("trained_on_family") is None:
            rec["lora"]["trained_on_family"] = family
    _fill(rec, "identity.title", cls.title_from_header(c.header))
    if rec["identity"]["type"] in ("lora", "lycoris"):
        facts = cls.lora_facts(c.header, None)
        if rec.get("lora") is None:
            rec["lora"] = cls.lora_facts(c.header, rec["identity"].get("family"))
        for key in ("network_dim", "network_alpha", "network_module", "lora_type", "clip_skip"):
            _fill(rec, f"lora.{key}", facts[key])
    content = res.get_field(rec, "semantics.content") or {}
    if content.get("basis") in (None, "default"):
        fresh, _why = cls.rating_from_markers(cls.marker_texts(found.path.name, c.header, rec), markers)
        rec["semantics"]["content"] = fresh
    rec.setdefault("usage", {"card": None, "author_notes": None})
    _fill(rec, "usage.card", _relative(reg.standards / "cards" / f"{rec['id']}.yaml", reg.root))
    _fill(rec, "usage.author_notes", _relative(reg.resources_dir / f"{rec['id']}.usage.yaml", reg.root))
    lifecycle = rec["lifecycle"]
    if lifecycle.get("install_status") in LIVE_STATUSES:
        lifecycle["install_status"] = "active"
    lifecycle["scanned_at"] = now
    return rec


def _summary_type(c: cls.Classified) -> str:
    family = c.family.value if c.family else "family unknown"
    return f"{c.type.value if c.type else 'type unknown'}, {family}"


def scan(reg: res.Registry, folders: cls.Folders, markers: cls.Markers, *, do_hash: bool, rehash: bool = False,
         by: str = "easel registry scan", progress: Callable[[str], None] | None = None) -> ScanReport:
    """Scan `reg.models_root` into `reg`; write the records, the changelog lines and the index."""
    report = ScanReport()
    if not reg.models_root.is_dir():
        raise ScanRefused(["models folder not found: <models_root> (paths.models_root) does not exist",
                           "set it with `easel config set paths.models_root <folder>`"])
    try:
        from easel._common import embedded_meta  # noqa: F401  (the header reader, built by T1b)
    except ImportError as exc:
        raise ScanRefused(["scan needs the header reader easel._common.embedded_meta, which is not installed "
                           "in this build yet; report it and do not work around it"]) from exc
    now = ids.now_iso(reg.tool)
    loaded = res.load_records(reg)
    report.problems += loaded.problems
    overrides = res.load_overrides(reg)
    records = dict(loaded.records)
    by_path = {str(res.get_field(r, "identity.path")).lower(): sha for sha, r in records.items()
               if res.get_field(r, "identity.path")}

    # pass one: find, classify, hash
    found: list[Found] = []
    seen: set[str] = set()
    for path, rel in walk_roots(reg, report.say, seen):
        report.files += 1
        try:
            stat = path.stat()
        except OSError as exc:
            report.skipped += 1
            report.say(f"skipped {rel}: not readable ({reason(exc)})")
            continue
        if _placeholder(stat):
            report.skipped += 1
            report.say(f"skipped {rel}: an online-only placeholder of a cloud-sync client (not read, not hashed)")
            continue
        header, note = _header(path)
        c = cls.classify(path, rel, folders, markers, header, note)
        if c.type is None:
            report.skipped += 1
            report.say(f"skipped {rel}: not in a known folder of {folders.source} and the header names no type; "
                       "the operator moves it into the right folder")
            continue
        item = Found(rel=rel, path=path, size=stat.st_size, mtime=stat.st_mtime, classified=c)
        known = records.get(by_path.get(rel.lower(), ""))
        known_sha = res.get_field(known, "integrity.sha256") if known else None
        same_size = known is not None and res.get_field(known, "integrity.bytes") == stat.st_size
        if do_hash:
            if known_sha and same_size and not rehash and not _modified_since(stat.st_mtime,
                                                                               res.get_field(known, "integrity.hashed_at")):
                item.sha256 = str(known_sha)
            else:
                if progress:
                    progress(f"hashing {rel} ({stat.st_size} bytes)")
                item.sha256, item.hashed_now = hashing.sha256_file(path), True
                report.hashed += 1
        elif known_sha and same_size:
            item.sha256 = str(known_sha)
        elif known is not None:
            report.unhashed += 1
            report.say(f"changed {rel}: its size differs from record {known['id']}; `easel registry scan --hash` "
                       "hashes it again")
            continue
        else:
            report.unhashed += 1
            report.say(f"found {rel} ({_summary_type(c)}): not hashed yet; `easel registry scan --hash` registers it")
            continue
        found.append(item)

    # pass two: one record per sha256
    groups: dict[str, list[Found]] = {}
    for item in found:
        groups.setdefault(str(item.sha256)[:16], []).append(item)
    handled: set[str] = set()
    changelog: list[dict] = []
    for sha16, items in groups.items():
        if sha16 in loaded.unreadable:
            report.problems.append(f"{items[0].rel}: its record registry/resources/{sha16}.yaml is unreadable; the "
                                   "operator fixes or moves that file, then scan again")
            continue
        old = records.get(sha16)
        keep = res.get_field(old, "identity.path") if old else None
        items.sort(key=lambda f: (f.rel != keep, f.rel))
        item = items[0]
        for dup in items[1:]:
            report.duplicates += 1
            report.say(f"duplicate {dup.rel}: the same bytes as {item.rel} (one resource per sha256; "
                       "`easel registry dups` lists them later; deleting is the operator's call)")
        handled.add(sha16)
        scanned = new_record(item, reg, now) if old is None else update_record(old, item, reg, now, markers)
        final, ochanges, refused = res.apply_override(scanned, overrides.get(sha16))
        report.problems += refused
        changes = {k: v for k, v in res.diff(old, scanned).items() if k not in _TIMESTAMPS} if old else {}
        if changes or ochanges or old is None:
            final["lifecycle"]["updated_at"] = now
        try:
            if final != old:
                res.write_record(reg, final)
                report.written += 1
        except DataError as exc:
            report.problems += exc.lines
            continue
        records[sha16] = final
        if old is None:
            report.new.append(sha16)
            report.say(f"new {sha16} {item.rel} ({_summary_type(item.classified)})")
            changelog.append(res.changelog_line(tool=reg.tool, source=f"{reg.tool}:registry scan", by=by,
                                                action="scan", kind="resource", target=sha16,
                                                changes={"lifecycle.install_status": {"from": None, "to": "active"}},
                                                note=f"new file {item.rel}"))
        elif changes:
            report.changed.append(sha16)
            report.say(f"changed {sha16} {item.rel}: {', '.join(sorted(changes))}")
            changelog.append(res.changelog_line(tool=reg.tool, source=f"{reg.tool}:registry scan", by=by,
                                                action="scan", kind="resource", target=sha16, changes=changes))
        else:
            report.unchanged += 1
        if ochanges:
            shown = ", ".join(f"{k} -> {v['to']}" for k, v in ochanges.items())
            report.say(f"override {sha16}: {shown} (registry/overrides.yaml)")
            changelog.append(res.changelog_line(tool=reg.tool, source="operator", by=by, action="set",
                                                kind="override", target=sha16, changes=ochanges,
                                                note="applied registry/overrides.yaml"))
        cues = markers.hits(markers.non_sfw, cls.marker_texts(item.path.name, item.classified.header, final))
        if res.get_field(final, "semantics.content.class") == "sfw" and cues:
            basis = res.get_field(final, "semantics.content.basis") or "unknown"
            report.conflicts.append(sha16)
            report.say(f"RATING CONFLICT {sha16} {item.rel}: rated sfw (basis {basis}) but carries the non-sfw cue "
                       f"{', '.join(cues)}; never use it as sfw; ask the operator to rate it")

    # files that are gone: missing, never deleted
    for sha16, rec in sorted(records.items()):
        rel = res.get_field(rec, "identity.path")
        status = res.get_field(rec, "lifecycle.install_status")
        if sha16 in handled or not rel or status not in ("active", "staged") or str(rel).lower() in seen:
            continue
        gone = res.ordered(copy.deepcopy(rec))
        gone["lifecycle"]["install_status"] = "missing"
        gone["lifecycle"]["updated_at"] = gone["lifecycle"]["scanned_at"] = now
        try:
            res.write_record(reg, gone)
            report.written += 1
        except DataError as exc:
            report.problems += exc.lines
            continue
        records[sha16] = gone
        report.missing.append(sha16)
        report.say(f"missing {sha16} {rel}: not found under <models_root> (the record stays; status missing)")
        changelog.append(res.changelog_line(tool=reg.tool, source=f"{reg.tool}:registry scan", by=by, action="scan",
                                            kind="resource", target=sha16,
                                            changes={"lifecycle.install_status": {"from": status, "to": "missing"}},
                                            note=f"{rel} not found under the models folder"))

    for sha16 in sorted(set(overrides) - set(records)):
        report.say(f"override {sha16}: no such resource in the registry yet (registry/overrides.yaml); it applies "
                   "once the file is registered")
    report.changelog = res.append_changelog(reg, changelog)
    report.records = res.effective(records, overrides)
    report.index = res.rebuild_index(reg, report.records)
    return report


def summary_lines(report: ScanReport, reg: res.Registry, *, do_hash: bool) -> list[str]:
    """The closing lines of a scan: counts, unknown licences, conflicts, what was written."""
    unknown = sorted(sha for sha, r in report.records.items() if res.get_field(r, "licence.status") == "unknown")
    lines = [f"scanned {report.files} file(s) under <models_root>: {len(report.new)} new, {len(report.changed)} "
             f"changed, {report.unchanged} unchanged, {len(report.missing)} missing, {report.unhashed} not hashed, "
             f"{report.skipped} skipped, {report.duplicates} duplicate(s)"
             + (f"; {report.hashed} hashed" if do_hash else "")]
    lines.append(f"unknown licence: {len(unknown)}" + (f" ({', '.join(unknown)})" if unknown else ""))
    lines.append(f"rating conflicts: {len(report.conflicts)}"
                 + (f" ({', '.join(report.conflicts)})" if report.conflicts else ""))
    if report.index:
        idx, md = report.index
        lines.append(f"wrote {report.written} record(s) under {reg.shown(reg.resources_dir)}/, {reg.shown(md)}, "
                     f"{reg.shown(idx)}; {report.changelog} changelog line(s)")
    if report.unhashed and not do_hash:
        lines.append("next: easel registry scan --hash")
    return lines
