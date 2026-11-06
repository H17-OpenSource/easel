"""`easel registry download <pending>`: fetch a pending entry's file, verify it, register it, scan.

Steps, each one safe to repeat:
1. read `registry/pending/<name>.yaml` (written by `registry add`);
2. ask the site again for the file's checksum (`fresh`: never a cached answer, so a changed upload is noticed);
3. when the registry already holds those bytes, nothing is downloaded: the source is merged into that record;
4. download into `<models_root>/<folder of the type>/` (data/folders.yaml) through `_common.civitai`/`hf`: a
   `.part` file resumed with a Range request, the disk floor (`limits.disk_floor_gb`) kept, a login page refused,
   the sha256 (or a small Hugging Face file's git blob id) checked; a mismatch deletes the part file, so the next
   run starts over (run it once more; a second mismatch: stop and report);
5. write `registry/resources/<sha16>.yaml` from the pending entry (id, path, sha256, size, `downloaded_at`,
   provenance `verified`, install status `active`), append one changelog line, remove the pending entry;
6. run `registry scan` (no hashing) so the header facts (dtypes, LoRA dims, rating cues) are filled in.
An interrupted run leaves the `.part` file and the pending entry; running the same command again resumes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import civitai, hf, ids

from . import add as adding
from . import classify as cls
from . import remote
from . import resources as res
from . import scan as scanner
from .upstream import merge_source


@dataclass
class Downloaded:
    """What `download` did: the resource id, the lines to print, and whether nothing had to be fetched."""

    sha16: str | None = None
    lines: list[str] = field(default_factory=list)
    already: bool = False
    scan_report: scanner.ScanReport | None = None


@dataclass
class _Plan:
    site: str
    expected: str | None
    fetch: Callable[[Path, Callable | None], civitai.Download]


def _civitai_plan(cfg, record: Mapping) -> _Plan:
    client = civitai.Client.from_config(cfg)
    vid = res.get_field(record, "provenance.version_id")
    if not vid:
        raise civitai.Refused("refused: the pending entry names no Civitai version (provenance.version_id)")
    vs = civitai.summarize_version(client.version(vid, fresh=True))
    name = str(res.get_field(record, "identity.file_name") or "")
    file = next((f for f in vs["files"] if str(f.get("name") or "").lower() == name.lower()), None)
    if file is None:
        raise civitai.NotFound(f"civitai: version {vid} no longer lists {name}; add it again "
                               "(`easel registry add <link> --replace`)")
    if not file.get("sha256"):
        raise civitai.Refused(f"refused: civitai no longer publishes a sha256 for {name}; nothing to verify against")
    return _Plan(civitai.SITE, file["sha256"], lambda folder, progress: client.download(file, folder,
                                                                                      progress=progress))


def _hf_plan(cfg, record: Mapping) -> _Plan:
    client = hf.Client.from_config(cfg)
    target = remote.parse_link(str(res.get_field(record, "provenance.download_url") or ""))
    if target.site != hf.SITE or not target.path:
        raise civitai.Refused("refused: the pending entry's download_url is not a Hugging Face file link")
    repo, path, revision = str(target.model_id), target.path, target.revision
    info = client.info(repo, revision, fresh=True)
    entry = next((f for f in hf.files(info) if f["path"] == path), None)
    if entry is None:
        raise civitai.NotFound(f"huggingface: {repo} no longer has {path} at {revision}")
    return _Plan(hf.SITE, entry["sha256"], lambda folder, progress: client.download(
        repo, path, folder, revision=revision, progress=progress))


def _progress(name: str, say: Callable[[str], None] | None) -> Callable[[int, int | None], None] | None:
    """A progress callback that says one line per tenth of the file (or per GiB when the size is unknown)."""
    if say is None:
        return None
    state = {"next": 0.1, "next_bytes": 1 << 30}

    def report(got: int, total: int | None) -> None:
        if total:
            while got / total >= state["next"] and state["next"] <= 1.0:
                say(f"downloading {name}: {round(state['next'] * 100)}% of {total} bytes")
                state["next"] += 0.1
        elif got >= state["next_bytes"]:
            say(f"downloading {name}: {got} bytes")
            state["next_bytes"] += 1 << 30

    return report


def _merge_into_existing(reg: res.Registry, record: Mapping, existing: Mapping, pending: Path, by: str,
                         note: str) -> list[str]:
    merged = merge_source(existing, record)
    changes = res.diff(existing, merged)
    sha16 = str(existing["id"])
    if changes:
        merged["lifecycle"]["updated_at"] = ids.now_iso(reg.tool)
        res.write_record(reg, merged)
        res.append_changelog(reg, [res.changelog_line(tool=reg.tool, source=f"{reg.tool}:registry download", by=by,
                                                      action="update", kind="resource", target=sha16,
                                                      changes=changes, note=note)])
    pending.unlink(missing_ok=True)
    loaded = res.load_records(reg)
    res.rebuild_index(reg, res.effective(loaded.records, res.load_overrides(reg)))
    return [f"already registered: {sha16} {res.get_field(existing, 'identity.file_name')} has these bytes; "
            "nothing downloaded",
            f"merged the source into it: {', '.join(sorted(changes)) or 'nothing new'}; removed "
            f"{reg.shown(pending)}"]


def finished_record(pending_record: Mapping, sha256: str, rel: str, size: int, reg: res.Registry, now: str) -> dict:
    """The resource record of a downloaded file, from its pending entry."""
    rec = res.ordered(copy.deepcopy(dict(pending_record)))
    sha16 = sha256[:16]
    rec["id"] = sha16
    rec["identity"]["path"] = rel
    rec["identity"]["file_name"] = Path(rel).name
    rec["integrity"].update({"sha256": sha256, "bytes": size, "hashed_at": now})
    rec["provenance"].update({"downloaded_at": now, "status": "verified", "source_class": "downloaded"})
    rec["lifecycle"].update({"install_status": "active", "updated_at": now})
    rec["usage"] = {"card": scanner._relative(reg.standards / "cards" / f"{sha16}.yaml", reg.root),
                    "author_notes": scanner._relative(reg.resources_dir / f"{sha16}.usage.yaml", reg.root)}
    return rec


def download(reg: res.Registry, cfg, pending_text: str, *, by: str = "easel registry download",
             say: Callable[[str], None] | None = None) -> Downloaded:
    """Download, verify and register one pending entry; see the module docstring."""
    pending = adding.resolve_pending(reg, pending_text)
    record = adding.load_pending(reg, pending)
    site = res.get_field(record, "provenance.site")
    if site == civitai.SITE:
        plan = _civitai_plan(cfg, record)
    elif site == hf.SITE:
        plan = _hf_plan(cfg, record)
    else:
        raise civitai.Refused(f"refused: {pending.stem}: easel downloads from civitai and huggingface, not {site!r}")
    folders, markers = cls.load_folders(cfg.root), cls.load_markers(cfg.root)
    rtype = str(res.get_field(record, "identity.type"))
    folder = folders.folder_for(rtype)
    if not folder:
        raise civitai.Refused(f"refused: {folders.source} names no folder for the type {rtype}")
    out = Downloaded()
    note = f"from {reg.shown(pending)}"
    loaded = res.load_records(reg)
    if plan.expected and plan.expected[:16] in loaded.records:
        out.sha16, out.already = plan.expected[:16], True
        out.lines = _merge_into_existing(reg, record, loaded.records[out.sha16], pending, by, note)
        out.lines.append(f"next: easel registry show {out.sha16}")
        return out
    target_dir = reg.models_root / folder
    done = plan.fetch(target_dir, _progress(str(res.get_field(record, "identity.file_name")), say))
    sha16 = done.sha256[:16]
    out.sha16 = sha16
    rel = done.path.relative_to(reg.models_root).as_posix()
    verified = f"{done.checked} verified" + (f", resumed at byte {done.resumed_from}" if done.resumed_from else "")
    out.lines.append(f"{'present' if done.already_present else 'downloaded'} <models_root>/{rel} "
                     f"({done.bytes} bytes, {verified})")
    out.lines += [f"note: {n}" for n in done.notes]
    loaded = res.load_records(reg)
    if sha16 in loaded.records:  # the site gave no sha256 up front (a small Hugging Face file) and we had it
        out.already = True
        out.lines += _merge_into_existing(reg, record, loaded.records[sha16], pending, by, note)
        out.lines.append(f"duplicate: <models_root>/{rel} holds the same bytes as "
                         f"{res.get_field(loaded.records[sha16], 'identity.path')}; deleting one is the operator's "
                         "call (`easel registry dups`)")
        return out
    now = ids.now_iso(reg.tool)
    final = finished_record(record, done.sha256, rel, done.bytes, reg, now)
    res.write_record(reg, final)
    res.append_changelog(reg, [res.changelog_line(
        tool=reg.tool, source=f"{reg.tool}:registry download", by=by, action="download", kind="resource",
        target=sha16, changes={"lifecycle.install_status": {"from": "pending", "to": "active"},
                               "identity.path": {"from": None, "to": rel}}, note=note)])
    pending.unlink(missing_ok=True)
    report = scanner.scan(reg, folders, markers, do_hash=False, by=by)
    out.scan_report = report
    out.lines.append(f"registered {sha16} {final['identity']['file_name']} ({rtype}, family "
                     f"{final['identity']['family'] or 'unknown'}); removed {reg.shown(pending)}")
    out.lines += [line for line in report.lines if sha16 in line or line.startswith("RATING CONFLICT")]
    out.lines += [line for line in scanner.summary_lines(report, reg, do_hash=False) if not line.startswith("next:")]
    out.lines.append(f"next: easel registry show {sha16}")
    return out
