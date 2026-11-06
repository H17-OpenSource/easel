"""Find candidate models on Civitai and Hugging Face, and vet one before anyone downloads it.

Only the public APIs are used, through `_common.civitai` and `_common.hf` (one request per second, backoff on HTTP
429, answers cached for 24 h under `<home>/cache/`): Civitai's `/api/v1/` and the Hugging Face model API. Nothing
here logs in, reads a web page or reads comments.

- `easel search civitai --query "<words>" --type <type>` and `easel search hf --query "<words>" --type <type>`
  print one line per hit: the API URL to vet, the name, type, base model, rating and licence as the site states
  them. They write nothing.
- `easel search vet <url> [<url> ...]` reads one model version (Civitai: an `/api/v1/` model, version or by-hash
  URL, a download URL, or a model page link; Hugging Face: a repo's API or page URL), sorts it into exactly one
  class and appends one easel/candidates-entry/1 line per URL to `<home>/candidates/<date>.jsonl`. The classes,
  the first that applies wins (every reason is listed in `reasons`):
  1. `rating-conflict`: the site's ratings disagree with each other or with the content markers
     (`data/content_markers.yaml`), or the model touches the subject policy (an age-down or real-person cue, or
     the site's own real-person flag). Never treated as sfw; the operator decides.
  2. `gated`: the file needs a login, early access or an accepted form, or there is no public download URL.
  3. `too-big`: the file is larger than `search.max_file_gb`, or downloading it would leave less free disk than
     `limits.disk_floor_gb` on the models folder's disk.
  4. `licence-unclear`: the licence is unknown, or a field the decision needs (commercial use, derivatives) is
     not stated, or a Hugging Face card names no usable licence.
  5. `ok`: none of the above. `ok` is not a yes: the operator still decides whether to add it.
What the API does not say stays unknown (null); nothing is guessed.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from easel._common import civitai, hf, ids, store
from easel.cli import instance_not_found, new_instance_hint
from easel.registry.shipped import REPO, schema_errors

SCHEMA = "easel/candidates-entry/1"
CLASSES = ("rating-conflict", "gated", "too-big", "licence-unclear", "ok")
GIB = 1 << 30
# easel types -> Hugging Face tag filters (a type without one searches untyped).
HF_FILTERS = {"lora": ("lora",), "checkpoint": ("text-to-image",), "embedding": ("textual_inversion",),
              "controlnet": ("controlnet",)}
# Civitai base model names -> family ids, matched on lower-cased words, first match wins.
BASE_MODEL_FAMILIES = (("illustrious", "illustrious"), ("noobai", "noobai"), ("pony", "pony"),
                       ("sdxl", "sdxl"), ("sd 1.", "sd15"), ("sd 2.", "sd21"), ("sd 3.5", "sd35"),
                       ("flux.1", "flux1"), ("wan video 2.2", "wan22"), ("wan video", "wan21"),
                       ("hunyuan video", "hunyuan_video"), ("ltxv", "ltxv"), ("qwen", "qwen_image"))
CANDIDATE_TYPES = ("checkpoint", "lora", "lycoris", "vae", "text_encoder", "clip", "clip_vision", "controlnet",
                   "ipadapter", "upscaler", "embedding", "detector", "sam", "style_model", "unet", "model_patch",
                   "other")


class SearchError(Exception):
    """A URL or option this module does not take: `.lines` to print, exit 2."""

    def __init__(self, *lines: str) -> None:
        super().__init__("\n".join(lines))
        self.lines = list(lines)


@dataclass(frozen=True)
class Target:
    """What a vet URL names: the site, the kind of record and its id (and a version id when the URL gives one)."""

    site: str  # civitai | huggingface
    kind: str  # model | version | hash | repo
    id: str
    version: str | None = None


def family_for(base_model: object, known: set[str] | None = None) -> str | None:
    """The family id a site's base-model name means, when it is one of the shipped families."""
    text = str(base_model or "").lower()
    known = known if known is not None else {p.stem for p in (REPO / "families").glob("*.yaml")}
    for needle, family in BASE_MODEL_FAMILIES:
        if needle in text and family in known:
            return family
    return None


# --- URLs ----------------------------------------------------------------------------------------------------------

def parse_url(url: str) -> Target:
    """The record a vet URL names; SearchError for anything that is not a Civitai or Hugging Face model URL."""
    parts = urllib.parse.urlsplit(str(url).strip())
    host = (parts.hostname or "").lower()
    path = [p for p in parts.path.split("/") if p]
    query = urllib.parse.parse_qs(parts.query)
    if parts.scheme not in ("http", "https"):
        raise SearchError(f"not a web URL: {url!r}")
    if host in ("civitai.com", "www.civitai.com"):
        if path[:2] == ["api", "v1"] and len(path) >= 4 and path[2] == "model-versions":
            if path[3] == "by-hash" and len(path) == 5:
                return Target("civitai", "hash", path[4])
            return Target("civitai", "version", path[3])
        if path[:2] == ["api", "v1"] and len(path) == 4 and path[2] == "models":
            return Target("civitai", "model", path[3], (query.get("modelVersionId") or [None])[0])
        if path[:3] == ["api", "download", "models"] and len(path) == 4:
            return Target("civitai", "version", path[3])
        if len(path) >= 2 and path[0] == "models" and path[1].isdigit():  # a model page link the person pasted
            return Target("civitai", "model", path[1], (query.get("modelVersionId") or [None])[0])
        raise SearchError(f"not a Civitai model, version or download URL: {url!r}")
    if host in ("huggingface.co", "www.huggingface.co"):
        if path[:2] == ["api", "models"] and len(path) >= 3:
            path = path[2:]
        if len(path) >= 2 and path[0] not in ("datasets", "spaces", "api"):
            return Target("huggingface", "repo", f"{path[0]}/{path[1]}")
        raise SearchError(f"not a Hugging Face model URL (owner/name): {url!r}")
    raise SearchError(f"vet takes Civitai and Hugging Face model URLs only, not {host or url!r}")


# --- searching -----------------------------------------------------------------------------------------------------

def _licence_word(licence: Mapping) -> str:
    use = licence.get("commercial_use")
    name = licence.get("name") or licence.get("status")
    return f"licence {name}" + (f" (commercial use: {use})" if use else "")


def civitai_lines(page: civitai.SearchPage) -> list[str]:
    """One line per hit: the newest version's API URL first (what `search vet` takes)."""
    lines = []
    for item in page.items:
        versions = item.get("versions") or []
        newest = versions[0] if versions else {}
        url = newest.get("api_url") or item.get("api_url")
        lines.append(f"{url} | {item.get('name')} | {item.get('type')} | {newest.get('base_model') or '-'} | "
                     f"rating {newest.get('rating') or item.get('rating')} | {_licence_word(item.get('licence') or {})}")
    return lines


def hf_lines(hits: list[dict]) -> list[str]:
    return [f"{h.get('api_url')} | {h.get('model_id')} | {h.get('type')} | {h.get('base_model') or '-'} | "
            f"rating {h.get('rating')} | {_licence_word(h.get('licence') or {})}" for h in hits]


# --- vetting -------------------------------------------------------------------------------------------------------

@dataclass
class Limits:
    """What `too-big` is measured against: a size cap, the disk floor and how free disk is read."""

    folder: Path | None
    floor_gb: float
    max_file_gb: float | None
    disk_free: Callable[[Path], int]


def _size_reasons(size: int | None, limits: Limits) -> list[str]:
    if not size:
        return []
    out = []
    if limits.max_file_gb is not None and size > limits.max_file_gb * GIB:
        out.append(f"too big: {size / GIB:.1f} GB is more than search.max_file_gb {limits.max_file_gb:g}")
    if limits.folder is not None:
        try:
            free = limits.disk_free(limits.folder)
        except OSError:
            free = None
        if free is not None and free - size < limits.floor_gb * GIB:
            out.append(f"too big: {size / GIB:.1f} GB would leave {(free - size) / GIB:.1f} GB free, under the "
                       f"{limits.floor_gb:g} GB floor (limits.disk_floor_gb)")
    return out


def _marker_reasons(texts: list[str], rating: str, markers) -> tuple[list[str], list[str]]:
    """(conflict reasons, plain notes) from the content markers over the model's names, tags and trigger words."""
    conflicts, notes = [], []
    bad = markers.hits(markers.non_sfw, texts)
    if bad and rating == "sfw":
        conflicts.append(f"rating conflict: the site says sfw, the markers find {', '.join(bad)}")
    elif bad:
        notes.append(f"non-sfw cue {', '.join(bad)} (rating {rating})")
    subject = markers.hits(markers.subjects, texts)
    if subject:
        conflicts.append(f"subject policy: {', '.join(subject)} (age-down or real-person cue; policy.subjects allows "
                         "fictional adults only)")
    return conflicts, notes


def _licence_reasons_civitai(licence: Mapping) -> list[str]:
    if licence.get("status") == "unknown":
        return ["licence unclear: no licence fields in the API answer"]
    missing = [k for k in ("commercial_use", "derivatives_allowed") if licence.get(k) is None]
    if missing:
        return [f"licence unclear: {', '.join(missing)} not stated"]
    return []


def _licence_reasons_hf(licence: Mapping) -> list[str]:
    name = licence.get("name")
    if licence.get("status") == "unknown" or not name:
        return ["licence unclear: the card names no licence"]
    if "several licences" in str(licence.get("note") or ""):
        return [f"licence unclear: several licences listed ({name})"]
    if str(name).lower() == "other" and not licence.get("url"):
        return ["licence unclear: licence 'other' without a link to its text"]
    return []


def pick_class(conflicts: list[str], gated: list[str], big: list[str], unclear: list[str]) -> str:
    for cls, reasons in zip(CLASSES, (conflicts, gated, big, unclear), strict=False):
        if reasons:
            return cls
    return "ok"


def _entry(*, site: str, url: str, query: str | None, model_id: object, version_id: object, name: object, rtype: str,
           family: str | None, base_model: object, file: Mapping | None, rating: str, licence: Mapping,
           cls: str, reasons: list[str], tool: str) -> dict:
    return {"schema": SCHEMA, "at": ids.now_iso(tool), "v": 1, "source": f"{tool}:search vet", "by": "easel search vet",
            "site": site, "url": url, "query": query, "model_id": str(model_id), "version_id":
            None if version_id is None else str(version_id), "name": str(name or model_id),
            "type": rtype if rtype in CANDIDATE_TYPES else "other", "family": family,
            "base_model": None if base_model is None else str(base_model),
            "file_name": (file or {}).get("name") or (file or {}).get("path"),
            "bytes": (file or {}).get("bytes"), "sha256": (file or {}).get("sha256"), "rating": rating,
            "licence": dict(licence), "class": cls, "reasons": reasons}


def _strictest(*ratings: str) -> str:
    order = ("explicit", "suggestive", "unknown", "sfw")
    return next((r for r in order if r in ratings), "unknown")


def vet_civitai(client: civitai.Client, target: Target, *, markers, limits: Limits, query: str | None,
                tool: str = "easel") -> dict:
    """The candidates entry of one Civitai model version."""
    if target.kind == "hash":
        version = client.version_by_hash(target.id)
        model = client.model(version.get("modelId"))
    elif target.kind == "version":
        version = client.version(target.id)
        model = client.model(version.get("modelId"))
    else:
        model = client.model(target.id)
        versions = [v for v in model.get("modelVersions") or [] if isinstance(v, Mapping)]
        if target.version is not None:
            versions = [v for v in versions if str(v.get("id")) == str(target.version)] or [client.version(target.version)]
        if not versions:
            raise civitai.NotFound(f"civitai: model {target.id} lists no version")
        version = versions[0]
    today = ids.today(tool)
    vsum = civitai.summarize_version(version, model)
    licence = civitai.licence_from(model, version, checked_at=today)
    file = civitai.primary_file(vsum)
    model_rating, version_rating = civitai.rating_from(model), vsum["rating"]
    rating = _strictest(model_rating, version_rating)
    texts = [str(model.get("name") or ""), str(version.get("name") or ""),
             *[str(t) for t in model.get("tags") or [] if isinstance(t, str)], *vsum["trigger_words"]]
    conflicts, notes = _marker_reasons(texts, rating, markers)
    if model_rating != version_rating and "sfw" in (model_rating, version_rating):
        conflicts.insert(0, f"rating conflict: the model says {model_rating}, the version says {version_rating}")
    if model.get("poi") is True:
        conflicts.append("subject policy: the site flags this model as depicting a real person")
    gated = []
    if licence.get("gated"):
        gated.append(f"gated: availability {version.get('availability') or model.get('availability')}")
    if file is None or not file.get("download_url"):
        gated.append("gated: no public download URL for the primary file")
    big = _size_reasons((file or {}).get("bytes"), limits)
    unclear = _licence_reasons_civitai(licence)
    cls = pick_class(conflicts, gated, big, unclear)
    reasons = conflicts + gated + big + unclear + notes
    if cls == "ok":
        reasons = [f"{rating} by the API rating", "licence fields present"] + notes
    return _entry(site="civitai", url=vsum["api_url"] or civitai.version_url(target.id), query=query,
                  model_id=vsum["model_id"] or model.get("id"), version_id=vsum["version_id"], name=model.get("name"),
                  rtype=civitai.resource_type(model.get("type")), family=family_for(vsum["base_model"]),
                  base_model=vsum["base_model"], file=file, rating=rating, licence=licence, cls=cls, reasons=reasons,
                  tool=tool)


def _hf_file(files: list[dict]) -> dict | None:
    weights = [f for f in files if str(f.get("path", "")).endswith((".safetensors", ".ckpt", ".pt", ".bin", ".gguf"))]
    return max(weights, key=lambda f: f.get("bytes") or 0) if weights else None


def vet_hf(client: hf.Client, target: Target, *, markers, limits: Limits, query: str | None,
           tool: str = "easel") -> dict:
    """The candidates entry of one Hugging Face repo (its largest weights file)."""
    info = client.info(target.id)
    summary = hf.summarize(info, checked_at=ids.today(tool))
    licence = summary["licence"]
    rating = summary["rating"]
    file = _hf_file(summary["files"])
    texts = [str(summary.get("model_id") or ""), *[str(t) for t in info.get("tags") or [] if isinstance(t, str)]]
    conflicts, notes = _marker_reasons(texts, rating, markers)
    gated = [f"gated: the repo is gated ({info.get('gated')})"] if licence.get("gated") else []
    if file is None:
        gated.append("gated: no weights file listed in the repo")
    big = _size_reasons((file or {}).get("bytes"), limits)
    unclear = _licence_reasons_hf(licence)
    cls = pick_class(conflicts, gated, big, unclear)
    reasons = conflicts + gated + big + unclear + notes
    if cls == "ok":
        reasons = [f"licence named: {licence.get('name')} (read its text)", f"rating {rating} (the site states none)"]
    if file is not None:
        file = {**file, "name": Path(str(file.get("path"))).name}
    return _entry(site="huggingface", url=summary["api_url"], query=query, model_id=summary["model_id"],
                  version_id=summary.get("revision"), name=summary.get("name"), rtype=summary.get("type") or "other",
                  family=family_for(summary.get("base_model")), base_model=summary.get("base_model"), file=file,
                  rating=rating, licence=licence, cls=cls, reasons=reasons, tool=tool)


def vet(url: str, *, civitai_client: civitai.Client, hf_client: hf.Client, markers, limits: Limits,
        query: str | None = None, tool: str = "easel") -> dict:
    """The candidates entry for one URL, checked against the schema before anyone writes it."""
    target = parse_url(url)
    if target.site == "civitai":
        entry = vet_civitai(civitai_client, target, markers=markers, limits=limits, query=query, tool=tool)
    else:
        entry = vet_hf(hf_client, target, markers=markers, limits=limits, query=query, tool=tool)
    problems = schema_errors("candidates-entry", entry)
    if problems:
        raise SearchError(f"the vet row for {url} does not fit {SCHEMA}; nothing written:", *[f"  {p}" for p in problems])
    return entry


def append(folder: Path, entries: list[dict], *, tool: str = "easel") -> Path:
    """Append the rows to `<folder>/<today>.jsonl` in one write under the folder's lock."""
    import json

    path = Path(folder) / f"{ids.today(tool)}.jsonl"
    data = "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries).encode("utf-8")
    with store.lock(Path(folder), tool):
        store.append_bytes_locked(path, data)
    return path


def vet_line(entry: Mapping) -> str:
    """`<class> | <name> (<type>, <family>, <size>) | licence ... | rating ...` for the terminal."""
    size = f"{entry['bytes'] / GIB:.2f} GB" if entry.get("bytes") else "size unknown"
    return (f"{entry['class']} | {entry['name']} ({entry['type']}, {entry.get('family') or '-'}, {size}) | "
            f"{_licence_word(entry['licence'])} | rating {entry['rating']} | {entry['url']}")


def make_clients(cfg) -> tuple[civitai.Client, hf.Client]:
    """The two site clients of an instance (tests replace this function)."""
    return civitai.Client.from_config(cfg), hf.Client.from_config(cfg)


def limits_for(cfg, client: civitai.Client) -> Limits:
    """The too-big limits: `search.max_file_gb`, `limits.disk_floor_gb`, measured on the models folder's disk."""
    from easel._common.config import ConfigError

    try:
        folder = cfg.path("paths.models_root")
    except ConfigError:
        folder = cfg.root
    cap = cfg.get("search.max_file_gb", None)
    return Limits(folder=folder, floor_gb=client.disk_floor_gb, max_file_gb=None if cap is None else float(cap),
                  disk_free=client.disk_free)


def candidates_dir(cfg) -> Path:
    if cfg.get("paths.candidates", None) is not None:
        return cfg.path("paths.candidates")
    if cfg.root is None:
        raise SearchError(instance_not_found(), new_instance_hint())
    return cfg.root / "candidates"


def check_type(type_: str | None, site: str) -> str | None:
    if type_ is None:
        return None
    if site == "civitai":
        civitai.site_type(type_)  # Refused (exit 2) for an unknown type
    elif not re.fullmatch(r"[a-z_]+", type_):
        raise SearchError(f"not a type: {type_!r}")
    return type_
