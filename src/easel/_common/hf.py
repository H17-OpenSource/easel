"""Hugging Face's public model API: model info, file list, the card's licence, search, verified downloads.

Only these URL shapes are built: `https://huggingface.co/api/models...` (records and search) and
`https://huggingface.co/<repo>/resolve/<revision>/<path>` (files); `https://huggingface.co/<repo>/blob/...`
appears only as a licence link taken from the card. Public endpoints only: no login, no web page is read.

The plumbing is civitai.py's (one request per second, 429/503 backoff with Retry-After, the 24 h file cache
under `<home>/cache/huggingface/`, resumable downloads with the login-page sniff and the disk floor). The token is
optional: the environment variable `search.hf_token_env` names (default HF_TOKEN). It is sent only as an
Authorization header that is not forwarded on a redirect (the file host never sees it) and never appears in a
URL, a cache file, a message or a repr.

Licence (`licence_from`): `cardData.license` (or the repo's single `license:` tag) gives the name and, for the
unambiguous ids in `_SPDX`, the SPDX id; `license_name`/`license_link` fill in an `other` licence. The permission
fields stay null: a licence id names a text, it does not answer the questions, so a person reads it
(`status: unverified`; no licence at all is `unknown`). `gated` comes from the repo's `gated` value.

Checksums: files stored with Git LFS carry a sha256; small files kept in git itself are verified by their git
blob id (sha1 of `blob <size>\\0` + the bytes), because the API publishes no sha256 for them. A download is pinned
to the commit the checksums were read at.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.parse
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath

from . import hashing, ids
from .civitai import (
    DEFAULT_CACHE_HOURS,
    DEFAULT_DISK_FLOOR_GB,
    DEFAULT_REQUESTS_PER_S,
    DEFAULT_TOOL,
    AlreadyExists,
    Check,
    DiskFloor,
    Download,
    FileCache,
    Http,
    LoginRequired,
    NetError,
    NotFound,
    Opener,
    RateLimited,
    Refused,
    ShaMismatch,
    cache_dir_for,
    fetch_file,
    free_bytes,
    safe_file_name,
    sha256_check,
)

__all__ = [
    "AlreadyExists", "Client", "DiskFloor", "Download", "LoginRequired", "NetError", "NotFound", "RateLimited",
    "Refused", "ShaMismatch", "blob_url", "files", "git_blob_sha1", "info_url", "licence_from", "rating_from",
    "resolve_url", "resource_type", "search_url", "summarize",
]

SITE = "huggingface"
HF_ROOT = "https://huggingface.co/"
API_ROOT = f"{HF_ROOT}api/"
PUBLIC_PREFIXES = (HF_ROOT,)
DEFAULT_TOKEN_ENV = "HF_TOKEN"
NOT_FOR_ALL_AUDIENCES = "not-for-all-audiences"

_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?"
_REPO = re.compile(rf"^{_SEGMENT}(?:/{_SEGMENT})?$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# Hugging Face licence ids whose SPDX id is unambiguous; any other id keeps spdx null.
_SPDX = {
    "apache-2.0": "Apache-2.0",
    "mit": "MIT",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "cc0-1.0": "CC0-1.0",
    "cc-by-4.0": "CC-BY-4.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc-by-nc-4.0": "CC-BY-NC-4.0",
    "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
    "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0",
    "creativeml-openrail-m": "CreativeML-OpenRAIL-M",
    "mpl-2.0": "MPL-2.0",
    "unlicense": "Unlicense",
}


# --- URLs --------------------------------------------------------------------------------------------------------

def check_repo(repo_id: object) -> str:
    """`owner/name` (or a bare name) as Hugging Face allows it; anything else is `Refused`."""
    text = str(repo_id or "")
    if not _REPO.match(text) or ".." in text or "--" in text:
        raise Refused(f"refused: not a Hugging Face repo id (owner/name): {text!r}")
    return text


def _quote_path(path: object) -> str:
    text = str(path or "")
    parts = text.split("/")
    if not text or "\\" in text or any(p in ("", ".", "..") for p in parts):
        raise Refused(f"refused: not a file path inside a repo: {text!r}")
    return "/".join(urllib.parse.quote(p, safe="") for p in parts)


def _quote_revision(revision: object) -> str:
    text = str(revision or "")
    if not text or any(c.isspace() for c in text) or ".." in text:
        raise Refused(f"refused: not a revision (branch, tag or commit): {text!r}")
    return urllib.parse.quote(text, safe="")


def info_url(repo_id: object, revision: object = None) -> str:
    """The API URL of a model's info with per-file sizes and LFS sha256s (`blobs=true`)."""
    base = f"{API_ROOT}models/{check_repo(repo_id)}"
    if revision:
        base += f"/revision/{_quote_revision(revision)}"
    return f"{base}?blobs=true"


def search_url(query: str | None = None, *, limit: int = 20, filters: Iterable[str] = (),
               sort: str | None = None) -> str:
    """The API URL of a model search; `filters` are Hugging Face tags (for example `lora`)."""
    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise Refused(f"refused: limit must be 1..100, not {limit!r}")
    params: list[tuple[str, str]] = [("limit", str(limit))]
    if query:
        params.append(("search", query))
    params.extend(("filter", str(f)) for f in filters)
    if sort:
        params.append(("sort", sort))
    return f"{API_ROOT}models?{urllib.parse.urlencode(params)}"


def resolve_url(repo_id: object, path: object, revision: object = "main") -> str:
    """The download URL of one file at one revision."""
    return f"{HF_ROOT}{check_repo(repo_id)}/resolve/{_quote_revision(revision)}/{_quote_path(path)}"


def blob_url(repo_id: object, path: object, revision: object = "main") -> str:
    """The page URL of one file in a repo (used only for a licence link the card gives as a repo path)."""
    return f"{HF_ROOT}{check_repo(repo_id)}/blob/{_quote_revision(revision)}/{_quote_path(path)}"


# --- records -----------------------------------------------------------------------------------------------------

def _tags(info: Mapping) -> list[str]:
    return [t for t in info.get("tags") or [] if isinstance(t, str)]


def _commit(info: Mapping) -> str | None:
    sha = info.get("sha")
    return sha if isinstance(sha, str) and _COMMIT.match(sha) else None


def _git_sha1(value: object) -> str | None:
    return value.lower() if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{40}", value) else None


def files(info: Mapping) -> list[dict]:
    """The repo's files: path, size, sha256 (LFS files) or git blob sha1 (files kept in git), and `lfs`.

    An LFS file's `blobId` is the id of its pointer, not of its bytes, so only files kept in git get `git_sha1`.
    """
    out = []
    for sib in info.get("siblings") or []:
        if not isinstance(sib, Mapping) or not isinstance(sib.get("rfilename"), str):
            continue
        lfs = sib.get("lfs") if isinstance(sib.get("lfs"), Mapping) else None
        sha = (lfs or {}).get("sha256")
        size = sib.get("size", (lfs or {}).get("size"))
        out.append({
            "path": sib["rfilename"],
            "bytes": size if isinstance(size, int) and not isinstance(size, bool) else None,
            "sha256": sha.lower() if isinstance(sha, str) and hashing.is_sha256(sha) else None,
            "git_sha1": None if lfs is not None else _git_sha1(sib.get("blobId")),
            "lfs": lfs is not None,
        })
    return out


def licence_from(info: Mapping, *, checked_at: str | None = None) -> dict:
    """The resource schema's `licence` block from the card's licence and the repo's `gated` value."""
    notes: list[str] = []
    card = info.get("cardData") if isinstance(info.get("cardData"), Mapping) else {}
    raw = card.get("license")
    if raw is None:
        tagged = [t.split(":", 1)[1] for t in _tags(info) if t.startswith("license:")]
        raw = tagged or None
    names = [raw] if isinstance(raw, str) else [n for n in raw if isinstance(n, str)] if isinstance(raw, list) else []
    names = [n.strip() for n in names if n.strip()]
    if len(names) > 1:
        notes.append(f"several licences listed: {', '.join(names)}")
    lic_id = names[0].lower() if len(names) == 1 else None
    if lic_id == "other":
        name = card.get("license_name") if isinstance(card.get("license_name"), str) else "other"
    else:
        name = ", ".join(names) or None
    url = None
    link = card.get("license_link")
    if isinstance(link, str) and link.strip():
        link = link.strip()
        if link.startswith(("https://", "http://")):
            url = link
        else:
            try:
                url = blob_url(info.get("id") or info.get("modelId"), link.removeprefix("./"), _commit(info) or "main")
            except Refused:
                notes.append("the card's licence link is not a usable path")
    gated_raw = info.get("gated")
    gated = None if gated_raw is None else bool(gated_raw)
    if gated_raw == "manual":
        notes.append("access is granted by hand by the repo's authors")
    if names:
        notes.append("permissions are not read from a licence id; read the licence text")
    return {
        "status": "unverified" if names else "unknown",
        "name": name,
        "spdx": _SPDX.get(lic_id) if lic_id else None,
        "url": url,
        "commercial_use": None,
        "credit_required": None,
        "derivatives_allowed": None,
        "relicense_allowed": None,
        "redistributable": None,
        "gated": gated,
        "source": "hf-card",
        "checked_at": checked_at,
        "note": "; ".join(notes) or None,
    }


def rating_from(info: Mapping) -> str:
    """`explicit` for a repo tagged not-for-all-audiences, else `unknown`: the site never declares a repo sfw."""
    return "explicit" if NOT_FOR_ALL_AUDIENCES in _tags(info) else "unknown"


def resource_type(info: Mapping) -> str:
    """A hint at the registry's resource type from tags and pipeline: lora, embedding, controlnet, checkpoint or other."""
    tags = set(_tags(info))
    if "lora" in tags or any(t.startswith("base_model:adapter:") for t in tags):
        return "lora"
    if "textual_inversion" in tags:
        return "embedding"
    if "controlnet" in tags:
        return "controlnet"
    if info.get("pipeline_tag") == "text-to-image":
        return "checkpoint"
    return "other"


def summarize(info: Mapping, *, checked_at: str | None = None) -> dict:
    """A model info (or a search hit) reduced to registry facts; the repo id is the only owner trace kept."""
    repo = info.get("id") or info.get("modelId")
    card = info.get("cardData") if isinstance(info.get("cardData"), Mapping) else {}
    return {
        "site": SITE,
        "model_id": repo,
        "revision": _commit(info),
        "name": str(repo).rsplit("/", 1)[-1] if repo else None,
        "type": resource_type(info),
        "pipeline_tag": info.get("pipeline_tag"),
        "library": info.get("library_name"),
        "base_model": card.get("base_model"),
        "rating": rating_from(info),
        "licence": licence_from(info, checked_at=checked_at),
        "last_modified": info.get("lastModified"),
        "api_url": info_url(repo) if repo else None,
        "files": files(info),
    }


def git_blob_sha1(path: Path) -> str:
    """The git blob id of a file: sha1 over `blob <size>\\0` and the bytes (what the API lists as blobId)."""
    path = Path(path)
    digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(hashing.CHUNK):
            digest.update(block)
    return digest.hexdigest()


# --- the client --------------------------------------------------------------------------------------------------

class Client:
    """Hugging Face's public API with civitai.py's plumbing; `from_config(cfg)` reads the instance's settings."""

    def __init__(self, *, tool: str = DEFAULT_TOOL, token: str | None = None, cache_dir: Path | None = None,
                 cache_hours: float = DEFAULT_CACHE_HOURS, requests_per_s: float = DEFAULT_REQUESTS_PER_S,
                 disk_floor_gb: float = DEFAULT_DISK_FLOOR_GB, opener: Opener | None = None,
                 sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
                 clock: Callable[[], float] = time.time, disk_free: Callable[[Path], int] = free_bytes) -> None:
        self.tool = tool
        self.disk_floor_gb = float(disk_floor_gb)
        self.disk_free = disk_free
        self.http = Http(site=SITE, prefixes=PUBLIC_PREFIXES, tool=tool, token=token, requests_per_s=requests_per_s,
                         cache=FileCache(cache_dir, cache_hours, clock=clock), opener=opener, sleep=sleep,
                         monotonic=monotonic, clock=clock,
                         login_hint=(": accept the terms on the site, then set the token variable that "
                                     "search.hf_token_env names (default HF_TOKEN), or download the file by hand "
                                     "into <models_root>"))

    def __repr__(self) -> str:
        return f"hf.Client({self.http!r})"

    @classmethod
    def from_config(cls, cfg: object, *, env: Mapping[str, str] | None = None, **overrides: object) -> Client:
        """A client from a loaded config: `search.*`, `limits.disk_floor_gb`, the cache under the instance."""
        env = os.environ if env is None else env
        get = cfg.get
        token_env = str(get("search.hf_token_env", DEFAULT_TOKEN_ENV) or DEFAULT_TOKEN_ENV)
        kwargs: dict = {
            "tool": getattr(cfg, "tool", DEFAULT_TOOL),
            "token": env.get(token_env) or None,
            "cache_dir": cache_dir_for(cfg, SITE),
            "cache_hours": float(get("search.cache_hours", DEFAULT_CACHE_HOURS)),
            "requests_per_s": float(get("search.requests_per_s", DEFAULT_REQUESTS_PER_S)),
            "disk_floor_gb": float(get("limits.disk_floor_gb", DEFAULT_DISK_FLOOR_GB)),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def info(self, repo_id: str, revision: str | None = None, *, fresh: bool = False) -> dict:
        """The raw model info with per-file sizes and LFS sha256s (cached)."""
        answer = self.http.get_json(info_url(repo_id, revision), fresh=fresh, what=f"the repo {repo_id}")
        if not isinstance(answer, Mapping):
            raise NetError(f"{SITE}: the info of {repo_id} is not an object")
        return dict(answer)

    def summary(self, repo_id: str, revision: str | None = None, *, fresh: bool = False) -> dict:
        """`summarize` of the model info, the licence dated today."""
        return summarize(self.info(repo_id, revision, fresh=fresh), checked_at=ids.today(self.tool))

    def files(self, repo_id: str, revision: str | None = None, *, fresh: bool = False) -> list[dict]:
        """The repo's files with their checksums."""
        return files(self.info(repo_id, revision, fresh=fresh))

    def search(self, query: str | None = None, *, limit: int = 20, filters: Iterable[str] = (),
               sort: str | None = None, fresh: bool = False) -> list[dict]:
        """Search models; each hit is summarized (licence from its `license:` tag, no file list)."""
        url = search_url(query, limit=limit, filters=filters, sort=sort)
        answer = self.http.get_json(url, fresh=fresh, what=f"the search for {query!r}" if query else "the search")
        if not isinstance(answer, list):
            raise NetError(f"{SITE}: the search answer is not a list")
        checked_at = ids.today(self.tool)
        return [summarize(hit, checked_at=checked_at) for hit in answer if isinstance(hit, Mapping)]

    def download(self, repo_id: str, path: str, folder: Path, *, revision: str | None = None,
                 progress: Callable[[int, int | None], None] | None = None, fresh: bool = False) -> Download:
        """Download one file of a repo into `folder` under its base name, verified, pinned to the info's commit."""
        info = self.info(repo_id, revision, fresh=fresh)
        entry = next((f for f in files(info) if f["path"] == path), None)
        if entry is None:
            raise NotFound(f"{SITE}: {repo_id} has no file {path!r}")
        if entry["sha256"]:
            check = sha256_check(entry["sha256"])
        elif entry["git_sha1"]:
            check = Check("git blob sha1", entry["git_sha1"], git_blob_sha1)
        else:
            raise Refused(f"refused: {repo_id}/{path}: the API lists no checksum for it; nothing to verify against")
        pinned = _commit(info) or revision or "main"
        dest = Path(folder) / safe_file_name(PurePosixPath(path).name)
        return fetch_file(self.http, resolve_url(repo_id, path, pinned), dest, check=check, size=entry["bytes"],
                          disk_floor_gb=self.disk_floor_gb, disk_free=self.disk_free, progress=progress)
