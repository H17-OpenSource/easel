"""`easel registry add <link>`: record a candidate model as a pending entry, `registry/pending/<name>.yaml`.

Nothing is downloaded here. The entry is an `easel/resource/1` record with `id: null` and
`lifecycle.install_status: pending` (the resource schema's pending form): what the site's public API says about
the file (type, base model, size, trigger words), its licence block and its rating. `registry download <name>`
then fetches and verifies the file. The site's sha256 is read again at download time, because the pending form
carries no hash (`integrity.sha256` is null while `id` is null).

Refusals (exit 2, nothing written): a link of an unknown form; a version whose file has no published checksum
(easel downloads verified files only); a type easel has no folder for (give `--type`); a version with several
model files and no `--file`; an entry that is already pending (`--replace` rewrites it from the API). A file whose
sha256 the registry already holds is not added again: the command names the record (exit 0).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import yaml

from easel._common import civitai, hf, ids, store

from . import classify as cls
from . import remote
from . import resources as res
from .shipped import DataError, reason, schema_errors

TYPES_WITH_FOLDERS = ("checkpoint", "lora", "lycoris", "vae", "text_encoder", "clip", "clip_vision", "controlnet",
                      "ipadapter", "upscaler", "embedding", "detector", "sam", "style_model", "unet", "model_patch")


@dataclass
class Added:
    """What `add` did: the entry's name and file, the record, the lines to print, and an already-registered id."""

    name: str
    path: Path | None
    record: dict
    lines: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    existing: str | None = None


# --- pending entries --------------------------------------------------------------------------------------------

def pending_files(reg: res.Registry) -> list[Path]:
    """Every pending entry, sorted by name."""
    folder = reg.pending_dir
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.yaml") if p.is_file() and ".tmp-" not in p.name)


def resolve_pending(reg: res.Registry, text: str) -> Path:
    """The pending entry a command-line argument names (its name, its file name, or its path); LookupError else."""
    stem = PurePosixPath(str(text).replace("\\", "/")).name
    stem = stem[:-5] if stem.endswith(".yaml") else stem
    path = reg.pending_dir / f"{stem}.yaml"
    if path.is_file():
        return path
    names = [p.stem for p in pending_files(reg)]
    known = f"pending now: {', '.join(names[:10])}" if names else "nothing is pending (`easel registry add <link>`)"
    raise LookupError(f"no pending entry {stem!r} in {reg.shown(reg.pending_dir)}/; {known}")


def load_pending(reg: res.Registry, path: Path) -> dict:
    """A pending entry, checked: a valid resource record with id null and install status pending (DataError else)."""
    try:
        data = store.read_record(path, "easel/resource/")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise DataError([f"{reg.shown(path)}: not readable: {reason(exc)}"]) from exc
    errors = schema_errors("resource", data)
    if data.get("id") is not None or res.get_field(data, "lifecycle.install_status") != "pending":
        errors.insert(0, "a pending entry has id null and lifecycle.install_status pending")
    if errors:
        raise DataError([f"{reg.shown(path)}: not a valid pending entry: {errors[0]}"])
    return data


def write_pending(reg: res.Registry, name: str, record: Mapping, *, replace: bool) -> Path:
    """Validate and write `registry/pending/<name>.yaml` under the registry lock; refuse an existing entry."""
    data = res.ordered({**record, "schema": res.RESOURCE_SCHEMA})
    errors = schema_errors("resource", data)
    if errors:
        raise DataError([f"pending entry {name}: not written, it fails {res.RESOURCE_SCHEMA}:"] +
                        [f"  {e}" for e in errors])
    path = reg.pending_dir / f"{name}.yaml"
    with store.lock(reg.registry, reg.tool):
        if path.exists() and not replace:
            raise civitai.Refused(f"refused: already pending: {reg.shown(path)}",
                                  f"next: easel registry download {name}  (or add it again with --replace to "
                                  "read it from the site once more)")
        store.write_record(path, res.RESOURCE_SCHEMA, data)
    return path


def _blank_record(now: str) -> dict:
    """The pending form of a resource record: everything unknown, no hash, no path, install status pending."""
    return {
        "schema": res.RESOURCE_SCHEMA, "id": None, "legacy_id": None,
        "identity": {"type": "checkpoint", "file_name": "", "path": None, "family": None, "base_model_label": None,
                     "title": None},
        "provenance": {"source_class": "downloaded", "site": None, "url": None, "download_url": None, "author": None,
                       "model_id": None, "version_id": None, "upstream_revision": None, "downloaded_at": None,
                       "status": "unverified", "record": None, "trained_by": None},
        "integrity": {"sha256": None, "bytes": None, "tensor_count": None, "param_count": None, "dtypes": [],
                      "precision": None, "quantization": None, "hashed_at": None},
        "lora": None,
        "semantics": {"description": None, "category": None, "tags": [],
                      "content": remote.content_from_rating("unknown")},
        "licence": {"status": "unknown"},
        "when_to_use": None,
        "usage": {"card": None, "author_notes": None},
        "relations": [],
        "lifecycle": {"install_status": "pending", "tested_at": None, "test_verdict": None, "favorite": False,
                      "notes": None, "created_at": now, "updated_at": now, "scanned_at": None},
    }


def _lora_block(rtype: str, family: str | None, triggers: list[str]) -> dict | None:
    if rtype not in ("lora", "lycoris"):
        return None
    return {"network_dim": None, "network_alpha": None, "network_module": None, "lora_type": None,
            "trained_on_family": family, "trigger_words": list(triggers), "recommended_weight": None,
            "clip_skip": None}


def _check_type(rtype: str, site_type: object) -> str:
    if rtype not in TYPES_WITH_FOLDERS:
        raise civitai.Refused(f"refused: the site calls this model {site_type!r}, which is no easel resource type; "
                              "give --type <type> if you know what it is")
    return rtype


# --- the two sites ----------------------------------------------------------------------------------------------

@dataclass
class Candidate:
    """One file a site offers, reduced to what a pending entry needs."""

    name: str
    record: dict
    sha256: str | None
    rtype: str
    rating: str
    texts: list[str]


def from_civitai(cfg, target: remote.Target, *, file: str | None, rtype: str | None, family: str | None,
                 tool: str, now: str) -> Candidate:
    """A pending record from Civitai's model and version records (the uploader, description and images dropped)."""
    client = civitai.Client.from_config(cfg)
    version_raw: Mapping | None = None
    model_id = target.model_id
    if target.version_id:
        version_raw = client.version(target.version_id)
        model_id = model_id or (None if version_raw.get("modelId") is None else str(version_raw["modelId"]))
        if not model_id:
            raise civitai.NetError(f"civitai: version {target.version_id} names no model")
    model_raw = client.model(model_id)
    if version_raw is None:
        versions = [v for v in model_raw.get("modelVersions") or [] if isinstance(v, Mapping)]
        if not versions:
            raise civitai.NotFound(f"civitai: model {model_id} lists no version")
        version_raw = versions[0]
    vs = civitai.summarize_version(version_raw, model_raw)
    main = civitai.primary_file(vs)
    if file:
        chosen = next((f for f in vs["files"] if str(f.get("name") or "").lower() == file.lower()), None)
        if chosen is None:
            names = ", ".join(str(f.get("name")) for f in vs["files"]) or "none"
            raise civitai.NotFound(f"civitai: version {vs['version_id']} has no file {file!r}; its files: {names}")
    else:
        chosen = main
    if chosen is None:
        raise civitai.NotFound(f"civitai: version {vs['version_id']} lists no file")
    file_name = civitai.safe_file_name(chosen.get("name"))
    if not chosen.get("sha256"):
        raise civitai.Refused(f"refused: civitai publishes no sha256 for {file_name}; easel downloads verified files "
                              "only: download it by hand into <models_root>, then `easel registry scan --hash`")
    if not chosen.get("download_url"):
        raise civitai.Refused(f"refused: {file_name} has no download URL under {civitai.DOWNLOAD_ROOT}")
    site_type = model_raw.get("type")
    rtype = _check_type(rtype or civitai.resource_type(site_type), site_type)
    family = family or remote.family_for(vs.get("base_model"), remote.family_ids(cfg.root))
    rating = remote.stricter([civitai.rating_from(model_raw), vs["rating"]])
    rec = _blank_record(now)
    title = " ".join(str(p) for p in (model_raw.get("name"), vs.get("name")) if p) or None
    rec["identity"].update({"type": rtype, "file_name": file_name, "family": family,
                            "base_model_label": vs.get("base_model"), "title": title})
    rec["provenance"].update({"site": civitai.SITE, "url": vs["api_url"], "download_url": chosen["download_url"],
                              "model_id": str(model_id), "version_id": vs["version_id"],
                              "upstream_revision": vs["version_id"]})
    rec["integrity"]["bytes"] = chosen.get("bytes")
    rec["lora"] = _lora_block(rtype, family, vs["trigger_words"])
    rec["semantics"]["content"] = remote.content_from_rating(rating)
    rec["licence"] = civitai.licence_from(model_raw, version_raw, checked_at=ids.today(tool))
    stem = None if chosen is main else PurePosixPath(file_name).stem
    name = remote.pending_name(civitai.SITE, str(model_id), str(vs["version_id"]), stem)
    texts = [file_name, title or "", *vs["trigger_words"]]
    return Candidate(name, rec, chosen["sha256"], rtype, rating, texts)


def from_hf(cfg, target: remote.Target, *, file: str | None, rtype: str | None, family: str | None,
            tool: str, now: str) -> Candidate:
    """A pending record from a Hugging Face repo's info (files with checksums, the card's licence, gating)."""
    client = hf.Client.from_config(cfg)
    repo = str(target.model_id)
    info = client.info(repo, target.revision)
    summary = hf.summarize(info, checked_at=ids.today(tool))
    entries = summary["files"]
    models = [f for f in entries if PurePosixPath(f["path"]).suffix.lower() in cls.MODEL_SUFFIXES]
    want = file or target.path
    if want:
        chosen = next((f for f in entries if f["path"] == want), None)
        if chosen is None:
            named = [f for f in entries if PurePosixPath(f["path"]).name.lower() == want.lower()]
            chosen = named[0] if len(named) == 1 else None
        if chosen is None:
            listed = ", ".join(f["path"] for f in models[:10]) or "none"
            raise civitai.NotFound(f"huggingface: {repo} has no file {want!r}; its model files: {listed}")
    elif len(models) == 1:
        chosen = models[0]
    else:
        listed = ", ".join(f["path"] for f in models[:10]) or "none"
        raise civitai.Refused(f"refused: {repo} holds {len(models)} model files ({listed}); give --file <path>")
    if not (chosen.get("sha256") or chosen.get("git_sha1")):
        raise civitai.Refused(f"refused: {repo}/{chosen['path']}: the API lists no checksum for it; nothing to "
                              "verify a download against")
    file_name = civitai.safe_file_name(PurePosixPath(chosen["path"]).name)
    rtype = _check_type(rtype or summary["type"], summary["type"])
    base = summary.get("base_model")
    base_label = ", ".join(str(b) for b in base) if isinstance(base, list) else (str(base) if base else None)
    family = family or remote.family_for(base_label, remote.family_ids(cfg.root))
    revision = summary.get("revision") or target.revision or "main"
    rec = _blank_record(now)
    rec["identity"].update({"type": rtype, "file_name": file_name, "family": family, "base_model_label": base_label,
                            "title": f"{summary.get('name') or repo} {chosen['path']}"})
    rec["provenance"].update({"site": hf.SITE, "url": f"{hf.HF_ROOT}{repo}",
                              "download_url": hf.resolve_url(repo, chosen["path"], revision), "model_id": repo,
                              "version_id": None, "upstream_revision": revision})
    rec["integrity"]["bytes"] = chosen.get("bytes")
    rec["lora"] = _lora_block(rtype, family, [])
    rec["semantics"]["content"] = remote.content_from_rating(summary["rating"])
    rec["licence"] = summary["licence"]
    stem = None if len(models) == 1 and chosen is models[0] else PurePosixPath(chosen["path"]).stem
    name = remote.pending_name(hf.SITE, repo, revision[:12], stem)
    return Candidate(name, rec, chosen.get("sha256"), rtype, summary["rating"], [file_name, repo])


# --- the verb ---------------------------------------------------------------------------------------------------

def _yes(value: object) -> str:
    return "-" if value is None else ("yes" if value else "no")


def describe(record: Mapping, *, authoring: str) -> list[str]:
    """The lines that say what a pending entry is (the title withheld for a class that is not sfw under sfw-only)."""
    cls_ = res.get_field(record, "semantics.content.class")
    title = res.get_field(record, "identity.title")
    if authoring == "sfw-only" and cls_ != "sfw":
        title = f"(title withheld: class {cls_})"
    lic = record.get("licence") or {}
    size = res.get_field(record, "integrity.bytes")
    return [
        f"  {title or '-'} | {res.get_field(record, 'identity.type')} | family "
        f"{res.get_field(record, 'identity.family') or 'unknown'} | {size if size is not None else '?'} bytes | "
        f"{res.get_field(record, 'identity.file_name')}",
        f"  licence: {lic.get('status')} {lic.get('name') or '-'}; commercial use {lic.get('commercial_use') or '-'}; "
        f"credit required {_yes(lic.get('credit_required'))}; derivatives {_yes(lic.get('derivatives_allowed'))}; "
        f"gated {_yes(lic.get('gated'))}",
        f"  rating: {res.get_field(record, 'semantics.content.rating')} "
        f"(class {cls_}, basis {res.get_field(record, 'semantics.content.basis')})",
    ]


def add(reg: res.Registry, cfg, link: str, *, file: str | None = None, rtype: str | None = None,
        family: str | None = None, replace: bool = False, by: str = "easel registry add") -> Added:
    """Read the link's model from the site's public API and write its pending entry; see the module docstring."""
    target = remote.parse_link(link)
    now = ids.now_iso(reg.tool)
    maker = from_civitai if target.site == civitai.SITE else from_hf
    cand = maker(cfg, target, file=file, rtype=rtype, family=family, tool=reg.tool, now=now)
    authoring = str(cfg.get("policy.prompt_authoring", "sfw-only"))
    if cand.sha256:
        loaded = res.load_records(reg)
        sha16 = cand.sha256[:16]
        if sha16 in loaded.records:
            known = loaded.records[sha16]
            return Added(cand.name, None, cand.record, existing=sha16, lines=[
                f"already registered: {sha16} {res.get_field(known, 'identity.file_name')} has these bytes; "
                "nothing added",
                f"next: easel registry register-upstream {sha16} --{'civitai' if target.site == 'civitai' else 'hf'}"
                + ("" if target.site == "civitai" else f" {target.model_id} --path <file>")
                + "  (records where it came from)"])
    record = copy.deepcopy(cand.record)
    path = write_pending(reg, cand.name, record, replace=replace)
    res.append_changelog(reg, [res.changelog_line(
        tool=reg.tool, source=f"{reg.tool}:registry add", by=by, action="add", kind="resource",
        target=f"pending/{cand.name}", changes={"lifecycle.install_status": {"from": None, "to": "pending"}},
        note=f"{target.site} model {res.get_field(record, 'provenance.model_id')}"
             + (f" version {res.get_field(record, 'provenance.version_id')}"
                if res.get_field(record, "provenance.version_id") else ""))])
    out = Added(cand.name, path, record)
    out.lines = [f"pending {reg.shown(path)}", *describe(record, authoring=authoring)]
    markers = cls.load_markers(cfg.root)
    cues = markers.hits(markers.non_sfw, cand.texts)
    if cand.rating == "sfw" and cues:
        out.warnings.append(f"RATING CONFLICT {cand.name}: the site says sfw but the name carries the non-sfw cue "
                            f"{', '.join(cues)}; never use it as sfw; ask the operator to rate it")
    if res.get_field(record, "licence.gated"):
        out.warnings.append(f"gated: {cand.name} needs a login or an accepted licence; easel never logs in, so the "
                            "download may stop at a login page (then the operator downloads it by hand)")
    if res.get_field(record, "licence.status") == "unknown":
        out.warnings.append(f"licence unknown: the site gives no licence terms for {cand.name}; the operator reads "
                            "the model page and decides")
    out.lines.append(f"next: easel registry download {cand.name}")
    return out
