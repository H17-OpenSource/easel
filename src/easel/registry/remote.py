"""What the public APIs say, turned into registry fields: the link parser, the content block, the family guess.

Shared by `registry add`, `registry download` and `registry register-upstream`. Only the public APIs of
`_common.civitai` and `_common.hf` are used; a model page link is read for its ids only and is never fetched.

Rules kept here so the three verbs agree:
- a link names a site and ids; anything else is refused (exit 2) with the forms that work;
- the rating is the stricter of what the model and the version say; sfw only when every source says sfw, and
  unknown counts as not sfw (default-deny), so `semantics.content.basis` is `source` only for a real rating;
- the family is guessed from the site's base-model label only when a family file of that id exists; otherwise
  it stays null for the operator (`--family` on the command line wins);
- a pending entry is named `<source>-<model>-<version>`, lower case, `/` written as `--`.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from easel._common import civitai, hf

from .shipped import REPO

CIVITAI_HOSTS = ("civitai.com", "www.civitai.com")
HF_HOSTS = ("huggingface.co", "www.huggingface.co", "hf.co")
HF_FILE_VIEWS = ("blob", "resolve", "tree")
RATING_ORDER = ("sfw", "unknown", "suggestive", "explicit")  # later = stricter
# Base-model labels whose words differ from every family id (matched as squashed lower-case substrings).
FAMILY_ALIASES = (("stablediffusionxl", "sdxl"), ("stablediffusionv15", "sd15"), ("stablediffusion15", "sd15"),
                  ("stablediffusion21", "sd21"), ("stablediffusion35", "sd35"))
LINK_FORMS = ("a Civitai model link (the page link with or without ?modelVersionId=<id>, or an /api/v1/models/<id>, "
              "/api/v1/model-versions/<id> or /api/download/models/<id> URL)",
              "a Hugging Face repo link (https://huggingface.co/<owner>/<name>, optionally /blob/<revision>/<file>)")


class LinkError(civitai.Refused):
    """A link this module cannot read as a model of a known site (exit 2)."""


@dataclass(frozen=True)
class Target:
    """What a link names: the site, the model (a Civitai id or a Hugging Face repo), a version, a file."""

    site: str
    model_id: str | None
    version_id: str | None = None
    revision: str | None = None
    path: str | None = None


def _digits(value: str | None, what: str, text: str) -> str:
    if value is None or not value.isdigit() or int(value) <= 0:
        raise LinkError(f"refused: {what} in {text!r} is not a positive number")
    return value


def parse_link(text: str) -> Target:
    """The site and ids of a model link; `LinkError` (exit 2) for anything else."""
    raw = str(text or "").strip()
    parts = urllib.parse.urlsplit(raw)
    host = (parts.hostname or "").lower()
    segments = [urllib.parse.unquote(s) for s in parts.path.split("/") if s]
    if parts.scheme not in ("http", "https") or not host:
        raise LinkError(f"refused: not a link: {raw!r}", *[f"  give {form}" for form in LINK_FORMS])
    if host in CIVITAI_HOSTS:
        return _civitai_target(raw, segments, urllib.parse.parse_qs(parts.query))
    if host in HF_HOSTS:
        return _hf_target(raw, segments)
    raise LinkError(f"refused: {host} is not a site easel downloads from", *[f"  give {form}" for form in LINK_FORMS])


def _civitai_target(raw: str, seg: list[str], query: dict[str, list[str]]) -> Target:
    version = (query.get("modelVersionId") or query.get("modelversionid") or [None])[0]
    if seg[:2] == ["api", "v1"] and len(seg) >= 4 and seg[2] == "models":
        return Target("civitai", _digits(seg[3], "the model id", raw),
                      _digits(version, "the version id", raw) if version else None)
    if seg[:2] == ["api", "v1"] and len(seg) >= 4 and seg[2] == "model-versions":
        return Target("civitai", None, _digits(seg[3], "the version id", raw))
    if seg[:2] == ["api", "download"] and len(seg) >= 4 and seg[2] == "models":
        return Target("civitai", None, _digits(seg[3], "the version id", raw))
    if len(seg) >= 2 and seg[0] == "models":
        return Target("civitai", _digits(seg[1], "the model id", raw),
                      _digits(version, "the version id", raw) if version else None)
    raise LinkError(f"refused: {raw!r} names no Civitai model or version", f"  give {LINK_FORMS[0]}")


def _hf_target(raw: str, seg: list[str]) -> Target:
    if seg and seg[0] in ("datasets", "spaces"):
        raise LinkError(f"refused: {raw!r} is a Hugging Face {seg[0][:-1]}, not a model repo")
    revision = path = None
    if seg[:2] == ["api", "models"]:
        rest = seg[2:]
        if "revision" in rest:
            at = rest.index("revision")
            rest, revision = rest[:at], (rest[at + 1] if len(rest) > at + 1 else None)
        repo = "/".join(rest[:2])
    elif len(seg) >= 3 and seg[2] in HF_FILE_VIEWS:
        repo, revision, path = "/".join(seg[:2]), (seg[3] if len(seg) > 3 else None), "/".join(seg[4:]) or None
    elif len(seg) >= 2 and seg[1] in HF_FILE_VIEWS:
        repo, revision, path = seg[0], (seg[2] if len(seg) > 2 else None), "/".join(seg[3:]) or None
    elif 1 <= len(seg) <= 2:
        repo = "/".join(seg)
    else:
        raise LinkError(f"refused: {raw!r} names no Hugging Face model repo", f"  give {LINK_FORMS[1]}")
    try:
        repo = hf.check_repo(repo)
    except civitai.Refused as exc:
        raise LinkError(*exc.lines) from None
    return Target("huggingface", repo, None, revision, path)


# --- ratings, content, families ---------------------------------------------------------------------------------

def stricter(ratings: Iterable[str | None]) -> str:
    """The strictest rating of several sources; `sfw` only when every one says sfw, none at all is `unknown`."""
    seen = [r if r in RATING_ORDER else "unknown" for r in ratings]
    if not seen:
        return "unknown"
    return max(seen, key=RATING_ORDER.index)


def content_from_rating(rating: str) -> dict:
    """The resource schema's `semantics.content` block for a rating a site gave (unknown = default-deny)."""
    if rating == "sfw":
        return {"rating": "sfw", "class": "sfw", "basis": "source"}
    if rating in ("suggestive", "explicit"):
        return {"rating": rating, "class": "nsfw", "basis": "source"}
    return {"rating": "unknown", "class": "unknown", "basis": "default"}


def family_ids(home: Path | None) -> set[str]:
    """The ids of the shipped family files and of the instance's own `<home>/families/*.yaml`."""
    ids = {p.stem for p in (REPO / "families").glob("*.yaml")}
    if home is not None:
        ids |= {p.stem for p in (Path(home) / "families").glob("*.yaml")}
    return ids


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def family_for(label: object, known: Iterable[str]) -> str | None:
    """The family id a site's base-model label names, when exactly that family file exists; else None."""
    if not label:
        return None
    squashed = _squash(str(label))
    known = set(known)
    for alias, fid in FAMILY_ALIASES:
        if alias in squashed and fid in known:
            return fid
    hits = [fid for fid in known if squashed.startswith(_squash(fid)) and _squash(fid)]
    return max(hits, key=len) if hits else None


def pending_name(site: str, model: str, version: str, file_stem: str | None = None) -> str:
    """`<source>-<model>-<version>` (plus `--<file stem>` for a file that is not the version's main file)."""

    def slug(value: str) -> str:
        return re.sub(r"[^a-z0-9._-]+", "_", str(value).lower().replace("/", "--")).strip("._-") or "x"

    name = f"{slug(site)}-{slug(model)}-{slug(version)}"
    return f"{name}--{slug(file_stem)}" if file_stem else name
