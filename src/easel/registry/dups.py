"""`easel registry dups`: the same bytes under two names, found cheap-first.

The registry keeps one record per sha256, so a second copy of a file has no record of its own: this verb walks
the model roots (`paths.models_root`, then `paths.extra_model_roots`) and compares the files themselves:
1. only files of the same size can be the same bytes (most files are ruled out without reading them);
2. a file whose record knows its sha256 (same path, same size, not modified since it was hashed) is not read;
3. the others are compared on 8 sampled chunks of 256 KiB (a few MB per file, even for a checkpoint);
4. `--full` hashes every sampled match (sha256) so each group is confirmed, not just probable.
Read-only: nothing is moved or deleted; which copy goes is the operator's call.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import hashing

from . import resources as res
from . import scan as scanner
from .shipped import reason

SAMPLE_AT = (0.02, 0.11, 0.23, 0.37, 0.53, 0.68, 0.82, 0.941)
SAMPLE_BYTES = 256 * 1024


@dataclass
class File:
    """One model file under a root: where it is, how big, and what is known of its bytes."""

    label: str
    rel: str
    path: Path
    size: int
    sha256: str | None = None
    sample: str | None = None

    @property
    def shown(self) -> str:
        return f"{self.label}/{self.rel}"


@dataclass
class Group:
    """Files with the same bytes: confirmed by sha256, or probable (equal samples)."""

    key: str
    size: int
    files: list[File]
    confirmed: bool

    def as_dict(self) -> dict:
        return {"sha256": self.key if self.confirmed else None, "sample": None if self.confirmed else self.key,
                "confirmed": self.confirmed, "bytes": self.size, "files": [f.shown for f in self.files]}


@dataclass
class DupReport:
    groups: list[Group] = field(default_factory=list)
    files: int = 0
    roots: int = 0
    sampled: int = 0
    hashed: int = 0
    notes: list[str] = field(default_factory=list)


def sample_hash(path: Path, size: int) -> str:
    """sha256 over 8 chunks taken at fixed fractions of the file (and their offsets): a cheap fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for fraction in SAMPLE_AT:
            offset = max(0, min(int(size * fraction), max(0, size - SAMPLE_BYTES)))
            handle.seek(offset)
            digest.update(offset.to_bytes(8, "little"))
            digest.update(handle.read(SAMPLE_BYTES))
    return digest.hexdigest()


def _known_sha(file: File, records_by_path: Mapping[str, Mapping], reg: res.Registry) -> str | None:
    rec = records_by_path.get(file.rel.lower())
    if rec is None or res.get_field(rec, "integrity.bytes") != file.size:
        return None
    if reg.model_file(file.rel).resolve() != file.path.resolve():
        return None  # the record's path names the same file name under an earlier root
    if scanner._modified_since(file.path.stat().st_mtime, res.get_field(rec, "integrity.hashed_at")):
        return None
    sha = res.get_field(rec, "integrity.sha256")
    return str(sha) if sha else None


def files_under_roots(reg: res.Registry, say: Callable[[str], None]) -> list[File]:
    """Every model file of every root, shadowed names included (they can be copies too)."""
    out: list[File] = []
    for index, base in enumerate(reg.model_roots):
        if not base.is_dir():
            if index:
                say(f"skipped {reg.root_label(index)} (paths.extra_model_roots): not a folder")
            continue
        for path in scanner._walk(base):
            try:
                size = path.stat().st_size
            except OSError as exc:
                say(f"skipped {reg.root_label(index)}/{path.relative_to(base).as_posix()}: not readable "
                    f"({reason(exc)})")
                continue
            out.append(File(reg.root_label(index), path.relative_to(base).as_posix(), path, size))
    return out


def find(reg: res.Registry, records: Mapping[str, Mapping], *, full: bool = False,
         progress: Callable[[str], None] | None = None) -> DupReport:
    """Groups of files with the same bytes, largest first."""
    report = DupReport(roots=len(reg.model_roots))
    files = files_under_roots(reg, report.notes.append)
    report.files = len(files)
    by_path = {str(res.get_field(r, "identity.path")).lower(): r for r in records.values()
               if res.get_field(r, "identity.path")}
    by_size: dict[int, list[File]] = {}
    for f in files:
        if f.size > 0:
            by_size.setdefault(f.size, []).append(f)
    for size, same in sorted(by_size.items(), reverse=True):
        if len(same) < 2:
            continue
        for f in same:
            f.sha256 = _known_sha(f, by_path, reg)
        if all(f.sha256 for f in same):
            buckets = _bucket(same, lambda f: str(f.sha256))
            report.groups += [Group(k, size, v, True) for k, v in buckets.items() if len(v) > 1]
            continue
        for f in same:
            if progress:
                progress(f"sampling {f.shown} ({size} bytes)")
            f.sample = sample_hash(f.path, size)
            report.sampled += 1
        for key, members in _bucket(same, lambda f: str(f.sample)).items():
            if len(members) < 2:
                continue
            if full:
                for f in members:
                    if not f.sha256:
                        if progress:
                            progress(f"hashing {f.shown} ({size} bytes)")
                        f.sha256 = hashing.sha256_file(f.path)
                        report.hashed += 1
            if all(f.sha256 for f in members):
                report.groups += [Group(k, size, v, True) for k, v in
                                  _bucket(members, lambda f: str(f.sha256)).items() if len(v) > 1]
            else:
                report.groups.append(Group(key, size, members, False))
    return report


def _bucket(files: list[File], key: Callable[[File], str]) -> dict[str, list[File]]:
    out: dict[str, list[File]] = {}
    for f in sorted(files, key=lambda f: (f.label != "<models_root>", f.label, f.rel.lower())):
        out.setdefault(key(f), []).append(f)
    return out


def lines(report: DupReport) -> list[str]:
    """One line per group, then the summary."""
    out = []
    for g in report.groups:
        how = "confirmed (sha256)" if g.confirmed else "probable (8 sampled chunks match; --full confirms)"
        key = g.key[:16] if g.confirmed else f"~{g.key[:15]}"
        out.append(f"{key} | {len(g.files)} files | {g.size} bytes each | " +
                   " | ".join(f.shown for f in g.files) + f" | {how}")
    extra = sum(len(g.files) - 1 for g in report.groups)
    held = sum((len(g.files) - 1) * g.size for g in report.groups)
    confirmed = sum(1 for g in report.groups if g.confirmed)
    if report.groups:
        out.append(f"dups: {len(report.groups)} group(s) ({confirmed} confirmed, {len(report.groups) - confirmed} "
                   f"probable), {extra} extra file(s), {held} bytes held twice; {report.files} file(s) under "
                   f"{report.roots} model root(s); deleting a copy is the operator's call")
    else:
        out.append(f"dups: no two files hold the same bytes among {report.files} file(s) under {report.roots} "
                   "model root(s)")
    return out
