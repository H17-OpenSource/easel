"""`easel registry register-upstream <sha16>`: record where a resource came from when easel did not download it.

Three ways, each checked before anything is written:
- `--civitai`: ask Civitai's public API which model version has this file's sha256 (`/api/v1/model-versions/
  by-hash/<sha256>`); the answer is the source, so the hash is verified by construction;
- `--hf <repo> --path <file> [--revision <rev>]`: read the repo's file list and compare its checksum (the LFS
  sha256, or the git blob id of a small file) with the registered file; a mismatch is refused (exit 1) unless
  `--force`, because a wrong repo or path must never be recorded as the source;
- `--url <url> --site <site>`: a source with no API to check against; recorded as `provenance.status: partial`.

What is written: the `provenance` block (site, url, download url, model and version ids, revision, status), and,
only where the record knows nothing yet, the licence block, the rating (when its basis is the default), the
family and the trigger words. `--force` also replaces a provenance or licence that is already there. A
self-trained resource is refused: its source is its training run, not a site. One changelog line per change.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import civitai, hf, ids

from . import remote
from . import resources as res

SITE_PATTERN = r"^[a-z][a-z0-9.-]*$"
PROVENANCE_KEYS = ("site", "url", "download_url", "author", "model_id", "version_id", "upstream_revision")


@dataclass
class Registered:
    """What register-upstream did: the record as written, the changes, the lines to print."""

    sha16: str
    record: dict
    changes: dict[str, dict] = field(default_factory=dict)
    lines: list[str] = field(default_factory=list)


def merge_source(record: Mapping, incoming: Mapping, *, force: bool = False) -> dict:
    """`record` with what a source says filled in: provenance, and the licence, rating, family, title, triggers
    only where the record has nothing (`force` replaces the provenance and the licence too)."""
    rec = res.ordered(copy.deepcopy(dict(record)))
    prov, new_prov = rec["provenance"], incoming.get("provenance") or {}
    if force or prov.get("source_class") == "unknown" or not prov.get("site"):
        for key in (*PROVENANCE_KEYS, "status", "source_class", "downloaded_at"):
            if key in new_prov and (force or new_prov[key] is not None or prov.get(key) is None):
                prov[key] = new_prov[key]
    lic = incoming.get("licence")
    if lic and (force or res.get_field(rec, "licence.status") in (None, "unknown")):
        rec["licence"] = copy.deepcopy(dict(lic))
    content = res.get_field(incoming, "semantics.content")
    if content and content.get("basis") == "source" and res.get_field(rec, "semantics.content.basis") in (None,
                                                                                                         "default"):
        rec["semantics"]["content"] = dict(content)
    for dotted in ("identity.family", "identity.base_model_label", "identity.title"):
        value = res.get_field(incoming, dotted)
        if value is not None and res.get_field(rec, dotted) is None:
            res.set_field(rec, dotted, value)
    triggers = res.get_field(incoming, "lora.trigger_words") or []
    if triggers and isinstance(rec.get("lora"), dict) and not rec["lora"].get("trigger_words"):
        rec["lora"]["trigger_words"] = list(triggers)
    if isinstance(rec.get("lora"), dict) and rec["lora"].get("trained_on_family") is None:
        rec["lora"]["trained_on_family"] = res.get_field(rec, "identity.family")
    return rec


def _from_civitai(cfg, sha256: str, tool: str) -> tuple[dict, str]:
    client = civitai.Client.from_config(cfg)
    version_raw = client.version_by_hash(sha256)
    vs = civitai.summarize_version(version_raw)
    if not vs["model_id"]:
        raise civitai.NetError(f"civitai: the version for {sha256[:16]} names no model")
    model_raw = client.model(vs["model_id"])
    match = next((f for f in vs["files"] if f.get("sha256") == sha256), civitai.primary_file(vs)) or {}
    rating = remote.stricter([civitai.rating_from(model_raw), vs["rating"]])
    incoming = {
        "provenance": {"source_class": "downloaded", "site": civitai.SITE, "url": vs["api_url"],
                       "download_url": match.get("download_url"), "model_id": vs["model_id"],
                       "version_id": vs["version_id"], "upstream_revision": vs["version_id"], "status": "verified"},
        "licence": civitai.licence_from(model_raw, version_raw, checked_at=ids.today(tool)),
        "semantics": {"content": remote.content_from_rating(rating)},
        "identity": {"family": remote.family_for(vs.get("base_model"), remote.family_ids(cfg.root)),
                     "base_model_label": vs.get("base_model"),
                     "title": " ".join(str(p) for p in (model_raw.get("name"), vs.get("name")) if p) or None},
        "lora": {"trigger_words": vs["trigger_words"]},
    }
    return incoming, f"civitai model {vs['model_id']} version {vs['version_id']} (found by its sha256)"


def _from_hf(cfg, record: Mapping, file: Path, repo: str, path: str, revision: str | None, force: bool,
             tool: str) -> tuple[dict, str]:
    client = hf.Client.from_config(cfg)
    info = client.info(repo, revision)
    entry = next((f for f in hf.files(info) if f["path"] == path), None)
    if entry is None:
        raise civitai.NotFound(f"huggingface: {repo} has no file {path!r}")
    sha256 = str(res.get_field(record, "integrity.sha256") or "")
    if entry["sha256"]:
        same, kind, remote_sum, local_sum = entry["sha256"] == sha256, "sha256", entry["sha256"], sha256
    elif entry["git_sha1"]:
        if not file.is_file():
            raise civitai.Refused(f"refused: the file is not under the model roots, so its git blob id cannot be "
                                  f"compared with {repo}/{path}")
        local_sum = hf.git_blob_sha1(file)
        same, kind, remote_sum = local_sum == entry["git_sha1"], "git blob sha1", entry["git_sha1"]
    else:
        raise civitai.Refused(f"refused: {repo}/{path}: the API lists no checksum for it; nothing to compare with")
    if not same and not force:
        raise civitai.ShaMismatch(f"HASH MISMATCH: {repo}/{path} is not this file ({kind}: the repo has "
                                  f"{remote_sum}, the registered file {local_sum}); nothing recorded",
                                  "check the repo and the path; --force records it anyway (only on the operator's "
                                  "word)")
    summary = hf.summarize(info, checked_at=ids.today(tool))
    pinned = summary.get("revision") or revision or "main"
    base = summary.get("base_model")
    base_label = ", ".join(str(b) for b in base) if isinstance(base, list) else (str(base) if base else None)
    incoming = {
        "provenance": {"source_class": "downloaded", "site": hf.SITE, "url": f"{hf.HF_ROOT}{repo}",
                       "download_url": hf.resolve_url(repo, path, pinned), "model_id": repo, "version_id": None,
                       "upstream_revision": pinned, "status": "verified" if same else "partial"},
        "licence": summary["licence"],
        "semantics": {"content": remote.content_from_rating(summary["rating"])},
        "identity": {"family": remote.family_for(base_label, remote.family_ids(cfg.root)),
                     "base_model_label": base_label},
    }
    return incoming, f"huggingface {repo} {path} at {pinned[:12]} ({kind} {'matches' if same else 'FORCED'})"


def register(reg: res.Registry, cfg, sha16_text: str, *, civitai_hash: bool = False, hf_repo: str | None = None,
             hf_path: str | None = None, revision: str | None = None, url: str | None = None,
             site: str | None = None, licence_name: str | None = None, author: str | None = None,
             title: str | None = None, force: bool = False, by: str = "easel registry register-upstream"
             ) -> Registered:
    """Look the source up (or take it as given), check it against the file, merge it in and write the record."""
    loaded = res.load_records(reg)
    sha16 = res.resolve_id(loaded.records, sha16_text)
    record = loaded.records[sha16]
    if res.get_field(record, "provenance.source_class") == "self-trained" or \
            res.get_field(record, "provenance.trained_by"):
        raise civitai.Refused(f"refused: {sha16} is self-trained; its source is its training run, not a download "
                              "site")
    ways = sum(bool(w) for w in (civitai_hash, hf_repo, url))
    if ways != 1:
        raise civitai.Refused("refused: give exactly one source: --civitai, --hf <repo> --path <file>, or --url "
                              "<url> --site <site>")
    sha256 = str(res.get_field(record, "integrity.sha256") or "")
    if civitai_hash:
        incoming, how = _from_civitai(cfg, sha256, reg.tool)
    elif hf_repo:
        if not hf_path:
            raise civitai.Refused("refused: --hf needs --path <file inside the repo>")
        file = reg.model_file(str(res.get_field(record, "identity.path") or ""))
        incoming, how = _from_hf(cfg, record, file, hf.check_repo(hf_repo), hf_path, revision, force, reg.tool)
    else:
        if not site or not re.match(SITE_PATTERN, site):
            raise civitai.Refused("refused: --url needs --site <lower-case site name, for example github>")
        if not re.match(r"^https?://", str(url)):
            raise civitai.Refused(f"refused: not an http(s) URL: {url!r}")
        incoming = {"provenance": {"source_class": "downloaded", "site": site, "url": url, "download_url": None,
                                   "model_id": None, "version_id": None, "upstream_revision": None,
                                   "status": "partial"}}
        how = f"{site} {url} (no API to check the file against: status partial)"
    if author:
        incoming["provenance"]["author"] = author
    if title:
        incoming.setdefault("identity", {})["title"] = title
    if licence_name:
        incoming["licence"] = {"status": "unverified", "name": licence_name, "spdx": None, "url": None,
                               "commercial_use": None, "credit_required": None, "derivatives_allowed": None,
                               "relicense_allowed": None, "redistributable": None, "gated": None, "source": "manual",
                               "checked_at": ids.today(reg.tool), "note": "named on the command line"}
    merged = merge_source(record, incoming, force=force)
    changes = res.diff(record, merged)
    out = Registered(sha16, merged, changes)
    if not changes:
        out.lines = [f"{sha16}: nothing to change ({how}); --force replaces a recorded source or licence"]
        return out
    merged["lifecycle"]["updated_at"] = ids.now_iso(reg.tool)
    res.write_record(reg, merged)  # DataError (exit 2, nothing written) when the result fails the schema
    res.append_changelog(reg, [res.changelog_line(tool=reg.tool, source=f"{reg.tool}:registry register-upstream",
                                                  by=by, action="update", kind="resource", target=sha16,
                                                  changes=changes, note=how)])
    effective = res.effective({**loaded.records, sha16: merged}, res.load_overrides(reg))
    res.rebuild_index(reg, effective)
    out.lines = [f"registered the source of {sha16} {res.get_field(merged, 'identity.file_name')}: {how}",
                 f"changed: {', '.join(sorted(changes))}",
                 f"next: easel registry show {sha16}"]
    return out
