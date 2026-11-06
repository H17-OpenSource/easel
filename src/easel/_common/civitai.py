"""Civitai's public REST API: search, model and version records, licence flags, rating, resumable downloads.

Only two URL families are ever built: `https://civitai.com/api/v1/` (records and search) and
`https://civitai.com/api/download/` (files). Nothing here logs in, drives a browser, scrapes a page or calls the
internal endpoints the website uses for itself: the site's terms allow automated access only through the public
API (docs/reference/civitai-rules.md). Comments are therefore not supported.

What every caller gets (the plumbing in the first half of this file, shared with hf.py):
- a rate limit: at most `search.requests_per_s` requests per second (default 1), retries included;
- HTTP 429 and 503: wait the `Retry-After` the site sends (or 5, 10, 20, 40 s when it sends none), at most
  MAX_RETRIES times and never longer than MAX_WAIT_S at once; then `RateLimited` (exit 3: retry later);
- a file cache of JSON answers for `search.cache_hours` (default 24) under the instance, `<home>/cache/<site>/`,
  so a repeated search does not hit the API again; files are never cached;
- an optional API token, read from the environment variable `search.civitai_token_env` names (default
  CIVITAI_API_TOKEN). It travels only in an Authorization header that is not forwarded on a redirect (the file
  host a download redirects to never sees it), and never appears in a URL, a cache file, a message or a repr;
- downloads into `<file>.part`, resumed with an HTTP Range request, sniffed for a login page (HTML where a file
  should be: `LoginRequired`), checked against the disk floor (`limits.disk_floor_gb`) before the first byte is
  written, verified (sha256), then renamed into place with the store's PermissionError retry (an antivirus often
  holds a fresh multi-GB file; a rename refused for 5 s is exit 4 with the STUCK line). An existing file is never
  overwritten.

Records are reduced to the facts a registry keeps: ids, names, types, base model, trigger words, files, hashes,
licence flags, rating. The uploader's name, the description HTML and the preview images are dropped
(`summarize_model`); a caller never needs them and the registry must not keep them.

Licence flags map to the resource schema's `licence` block (`licence_from`): `allowCommercialUse` ->
`commercial_use` (the highest of Sell > Rent > Image, else none), `allowNoCredit` -> `credit_required`
(inverted), `allowDerivatives` -> `derivatives_allowed`, `allowDifferentLicense` -> `relicense_allowed`,
`availability` -> `gated` (and `redistributable: false` for early-access or private files). A field the API
does not send stays null (not known); nothing is guessed. Site metadata is `status: unverified` until a person
reads the terms; no flag at all is `status: unknown`.

The library prints nothing. Errors are `NetError` subclasses with `.lines` to print and `.code` to exit with
(1 a problem, 2 a refusal or usage error, 3 deferred: retry later), like `config.ConfigError`.
"""

from __future__ import annotations

import email.utils
import http.client
import importlib.metadata
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

from . import hashing, ids, store

# The tool this copy serves (easel or kiln): the package it is loaded under, read, never imported.
DEFAULT_TOOL = (__package__ or "").split(".")[0] or "tool"
SITE = "civitai"
API_ROOT = "https://civitai.com/api/v1/"
DOWNLOAD_ROOT = "https://civitai.com/api/download/"
PUBLIC_PREFIXES = (API_ROOT, DOWNLOAD_ROOT)

DEFAULT_TOKEN_ENV = "CIVITAI_API_TOKEN"
DEFAULT_CACHE_HOURS = 24.0
DEFAULT_REQUESTS_PER_S = 1.0
DEFAULT_DISK_FLOOR_GB = 20.0  # a neutral default; an instance sets limits.disk_floor_gb to what it needs
RETRY_STATUSES = (429, 503)
MAX_RETRIES = 4
MAX_WAIT_S = 300.0
BACKOFF_FIRST_S = 5.0
TIMEOUT_S = 60.0
CHUNK = 1 << 20
GIB = 1 << 30
MAX_JSON_BYTES = 32 << 20
PART_SUFFIX = ".part"
FLOOR_RECHECK_BYTES = 1 << 30  # with an unknown size, re-check the disk floor after every GiB written
LOGIN_HINT = ": the operator downloads it by hand into <models_root>; then rescan the model registry"

Opener = Callable[..., object]  # called as opener(request, timeout=seconds), returns a response


# --- errors -----------------------------------------------------------------------------------------------------

class NetError(Exception):
    """A network or site problem: `.lines` to print (never a token), `.code` to exit with, `.status` (HTTP)."""

    code = 1

    def __init__(self, *lines: str, status: int | None = None, code: int | None = None) -> None:
        super().__init__("\n".join(lines))
        self.lines = list(lines)
        self.status = status
        if code is not None:
            self.code = code


class Refused(NetError):
    """A request this module will not make: a URL outside the public API, a bad id, an unsafe file name."""

    code = 2


class NotFound(NetError):
    """The site has no such record (HTTP 404)."""

    code = 2


class LoginRequired(NetError):
    """The site wants a login or an accepted licence (HTTP 401/403, or a login page where a file should be)."""

    code = 2


class AlreadyExists(NetError):
    """The target file exists and does not match; it is never overwritten."""

    code = 2


class RateLimited(NetError):
    """HTTP 429/503 persisted through the backoff, or the site asked for a longer wait than MAX_WAIT_S."""

    code = 3


class DiskFloor(NetError):
    """Writing the file would leave less free disk than `limits.disk_floor_gb`."""

    code = 3


class ShaMismatch(NetError):
    """The downloaded bytes do not match the checksum the site published."""

    code = 1


# --- the plumbing shared with hf.py ------------------------------------------------------------------------------

def user_agent(tool: str) -> str:
    """`<tool>/<version>`: an honest client name (no browser string)."""
    try:
        version = importlib.metadata.version(tool)
    except importlib.metadata.PackageNotFoundError:
        version = "dev"
    return f"{tool}/{version}"


class RateLimiter:
    """At most `per_s` request starts per second; `sleep` and `monotonic` are injectable so tests never wait."""

    def __init__(self, per_s: float, *, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        if not per_s > 0:
            raise ValueError(f"requests per second must be > 0, not {per_s!r}")
        self.interval = 1.0 / per_s
        self._sleep = sleep
        self._monotonic = monotonic
        self._last: float | None = None

    def wait(self) -> None:
        """Sleep until `interval` has passed since the previous request started, then mark this one."""
        now = self._monotonic()
        if self._last is not None and now < self._last + self.interval:
            delay = self._last + self.interval - now
            self._sleep(delay)
            now += delay
        self._last = now


class FileCache:
    """JSON answers kept one file each for `hours`; a damaged, future-dated or old file is a miss.

    The file name is a hash of the key (the URL plus whether a token was sent; never the token), so a cache folder
    reveals nothing by its names. `hours = 0` or no folder disables the cache.
    """

    def __init__(self, folder: Path | None, hours: float = DEFAULT_CACHE_HOURS, *,
                 clock: Callable[[], float] = time.time) -> None:
        self.folder = Path(folder) if folder is not None else None
        self.seconds = max(0.0, float(hours)) * 3600.0
        self._clock = clock

    @property
    def enabled(self) -> bool:
        return self.folder is not None and self.seconds > 0

    def file_for(self, key: str) -> Path:
        assert self.folder is not None
        return self.folder / f"{hashing.sha256_text(key)[:32]}.json"

    def get(self, key: str) -> tuple[bool, object]:
        """(True, value) for a fresh entry, else (False, None)."""
        if not self.enabled:
            return False, None
        path = self.file_for(key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            age = self._clock() - float(entry["fetched_at"])
            if entry.get("key_sha256") != hashing.sha256_text(key) or not 0 <= age < self.seconds:
                return False, None
            return True, entry["body"]
        except (OSError, ValueError, KeyError, TypeError):
            return False, None

    def put(self, key: str, url: str, value: object) -> None:
        """Store `value` for `key` atomically (through the store's write path)."""
        if not self.enabled:
            return
        entry = {"url": url, "key_sha256": hashing.sha256_text(key), "fetched_at": self._clock(), "body": value}
        store.write_json(self.file_for(key), entry)


def cache_dir_for(cfg: object, site: str) -> Path | None:
    """`<instance root>/cache/<site>/`, or None when no instance is configured (then nothing is cached)."""
    root = getattr(cfg, "root", None)
    return None if root is None else Path(root) / "cache" / site


def retry_after_s(value: str | None, *, clock: Callable[[], float] = time.time) -> float | None:
    """A `Retry-After` header in seconds: an integer, or an HTTP date turned into seconds from now; else None."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, when.timestamp() - clock())


def _status(resp: object) -> int:
    status = getattr(resp, "status", None)
    return int(status if status is not None else resp.getcode())


def _reason(exc: BaseException) -> str:
    reason = getattr(exc, "reason", None)
    return str(reason if reason is not None else exc) or type(exc).__name__


class Http:
    """One site's HTTP client: only URLs under `prefixes`, a rate limit, 429/503 backoff, a JSON cache, a token.

    `opener(request, timeout=...)` defaults to urllib's (system proxies honoured); tests pass their own. The token
    is sent as `Authorization: Bearer ...` through `add_unredirected_header`, so urllib drops it on a redirect.
    """

    def __init__(self, *, site: str, prefixes: Iterable[str], tool: str = DEFAULT_TOOL, token: str | None = None,
                 requests_per_s: float = DEFAULT_REQUESTS_PER_S, cache: FileCache | None = None,
                 opener: Opener | None = None, sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic, clock: Callable[[], float] = time.time,
                 timeout: float = TIMEOUT_S, login_hint: str = "") -> None:
        self.site = site
        self.prefixes = tuple(prefixes)
        self.tool = tool
        self._token = token or None
        self.limiter = RateLimiter(requests_per_s, sleep=sleep, monotonic=monotonic)
        self.cache = cache if cache is not None else FileCache(None)
        self._open = opener if opener is not None else urllib.request.build_opener().open
        self._sleep = sleep
        self._clock = clock
        self.timeout = timeout
        self.user_agent = user_agent(tool)
        self.login_hint = login_hint

    def __repr__(self) -> str:
        return f"Http(site={self.site!r}, token={'set' if self._token else 'none'})"

    @property
    def has_token(self) -> bool:
        return self._token is not None

    def check_url(self, url: str) -> str:
        """`url` unchanged when it starts with one of the public prefixes; otherwise `Refused`."""
        if not isinstance(url, str) or not url.startswith(self.prefixes):
            raise Refused(f"refused: {url!r} is not under {' or '.join(self.prefixes)} (public API only)")
        return url

    def request(self, url: str, headers: Mapping[str, str] | None = None) -> urllib.request.Request:
        """A GET request for a checked URL, with the user agent and (unredirected) the token."""
        req = urllib.request.Request(self.check_url(url), headers={"User-Agent": self.user_agent, **(headers or {})},
                                     method="GET")
        if self._token:
            req.add_unredirected_header("Authorization", f"Bearer {self._token}")
        return req

    def open(self, url: str, *, headers: Mapping[str, str] | None = None, what: str | None = None) -> object:
        """GET `url` under the rate limit, waiting out 429/503; the response is the caller's to close."""
        what = what or url
        waits = 0
        while True:
            req = self.request(url, headers)
            self.limiter.wait()
            try:
                return self._open(req, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                status, retry_header = exc.code, exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                if status not in RETRY_STATUSES:
                    raise self._http_error(status, what) from None
                wait = retry_after_s(retry_header, clock=self._clock)
                wait = BACKOFF_FIRST_S * 2 ** waits if wait is None else wait
                if waits >= MAX_RETRIES or wait > MAX_WAIT_S:
                    raise RateLimited(f"{self.site}: HTTP {status} for {what}: the site asks to slow down; "
                                      f"retry in {max(wait, 1):.0f} s or later, and do not loop",
                                      status=status) from None
                self._sleep(wait)
                waits += 1
            except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
                raise NetError(f"{self.site}: cannot reach {what}: {_reason(exc)}") from None

    def _http_error(self, status: int, what: str) -> NetError:
        if status == 404:
            return NotFound(f"{self.site}: {what}: not found (HTTP 404)", status=status)
        if status in (401, 403):
            return LoginRequired(f"{self.site}: {what}: HTTP {status}, a login or an accepted licence is needed; "
                                 f"this tool never logs in{self.login_hint}", status=status)
        return NetError(f"{self.site}: {what}: HTTP {status}", status=status)

    def get_json(self, url: str, *, fresh: bool = False, what: str | None = None) -> object:
        """The JSON answer for `url`, from the cache when fresh there (unless `fresh`), else fetched and cached."""
        key = url + ("#token" if self._token else "")
        if not fresh:
            hit, value = self.cache.get(key)
            if hit:
                return value
        with self.open(url, headers={"Accept": "application/json"}, what=what) as resp:
            raw = resp.read(MAX_JSON_BYTES + 1)
        if len(raw) > MAX_JSON_BYTES:
            raise NetError(f"{self.site}: {what or url}: the answer is larger than {MAX_JSON_BYTES >> 20} MiB")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise NetError(f"{self.site}: {what or url}: the answer is not JSON (a maintenance or login page?)") \
                from None
        self.cache.put(key, url, value)
        return value


# --- resumable, verified downloads -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Check:
    """How a finished file is verified: `kind` names it in messages, `compute(path)` gives the hex digest."""

    kind: str
    expected: str
    compute: Callable[[Path], str] = hashing.sha256_file

    def matches(self, path: Path) -> bool:
        return self.compute(path).lower() == self.expected.lower()


def sha256_check(expected: str) -> Check:
    """A sha256 check; `expected` must be 64 hex characters (either case)."""
    if not isinstance(expected, str) or not hashing.is_sha256(expected):
        raise Refused(f"refused: not a sha256 to verify against: {expected!r}")
    return Check("sha256", expected.lower())


@dataclass
class Download:
    """A finished download: where it is, its size and sha256, how it was checked, and whether it resumed."""

    path: Path
    bytes: int
    sha256: str
    checked: str
    resumed_from: int = 0
    already_present: bool = False
    notes: list[str] = field(default_factory=list)


def looks_like_html(head: bytes) -> bool:
    """True when the first bytes are an HTML page (a login wall), not a model file."""
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")[:16].lower()
    return text.startswith((b"<!doctype html", b"<html", b"<head", b"<!--"))


def _head(path: Path, n: int = 64) -> bytes:
    with path.open("rb") as handle:
        return handle.read(n)


def free_bytes(folder: Path) -> int:
    """Free bytes on the disk that holds `folder` (its nearest existing ancestor)."""
    folder = Path(folder)
    while not folder.exists() and folder.parent != folder:
        folder = folder.parent
    return shutil.disk_usage(folder).free


def check_disk_floor(folder: Path, need: int | None, floor_gb: float, *, what: str,
                     disk_free: Callable[[Path], int] = free_bytes) -> None:
    """`DiskFloor` when writing `need` bytes (unknown: 0) would leave less than `floor_gb` free."""
    free = disk_free(folder)
    left = free - (need or 0)
    if left < floor_gb * GIB:
        size = f"needs {need / GIB:.1f} GB; " if need else "has an unknown size; "
        raise DiskFloor(f"refused: {what} {size}{free / GIB:.1f} GB free would leave {left / GIB:.1f} GB, under the "
                        f"{floor_gb:g} GB floor (limits.disk_floor_gb); free space first, then run it again")


def _total_size(resp: object, status: int, pos: int) -> int | None:
    """The full file size the server announces: Content-Range's total on 206, Content-Length on 200."""
    headers = resp.headers
    if status == 206:
        match = re.match(r"\s*bytes\s+\d+-\d+/(\d+)\s*$", headers.get("Content-Range") or "")
        if match:
            return int(match.group(1))
        length = headers.get("Content-Length")
        return pos + int(length) if length and length.isdigit() else None
    length = headers.get("Content-Length")
    return int(length) if length and length.isdigit() else None


def _json_message(first: bytes) -> str:
    try:
        obj = json.loads(first.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return first[:200].decode("utf-8", "replace")
    if isinstance(obj, dict):
        for key in ("error", "message", "detail"):
            if isinstance(obj.get(key), str):
                return obj[key][:200]
    return str(obj)[:200]


def fetch_file(http: Http, url: str, dest: Path, *, check: Check | None, size: int | None = None,
               disk_floor_gb: float = DEFAULT_DISK_FLOOR_GB, disk_free: Callable[[Path], int] = free_bytes,
               progress: Callable[[int, int | None], None] | None = None, allow_unverified: bool = False) -> Download:
    """Download `url` to `dest`: resume `<dest>.part`, refuse a login page or a full disk, verify, rename.

    `check` is required unless `allow_unverified`; `size` is the site's size hint, used when the server sends
    none. A `dest` that exists is returned as already present when it passes `check`, else `AlreadyExists`.
    An interrupted transfer keeps the part file (run the same call again to resume); a checksum mismatch
    deletes it, so the next run starts over.
    """
    dest = Path(dest)
    http.check_url(url)
    if check is None and not allow_unverified:
        raise Refused(f"refused: {dest.name}: the site publishes no checksum for it; nothing to verify against")
    if dest.exists():
        if check is not None and check.matches(dest):
            sha = check.expected if check.kind == "sha256" else hashing.sha256_file(dest)
            return Download(dest, dest.stat().st_size, sha, check.kind, already_present=True)
        why = f"its {check.kind} differs from the site's" if check else "there is nothing to verify it against"
        raise AlreadyExists(f"refused: {dest.name} exists and {why}; it is never overwritten: move it aside first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + PART_SUFFIX)
    notes: list[str] = []
    if part.exists() and looks_like_html(_head(part)):
        part.unlink()  # a login page an older run saved where the file should be
        notes.append("discarded a part file that held a web page")

    resumed_from = 0
    for _attempt in range(2):  # a second pass only after a refused range on a part that failed its check
        pos = part.stat().st_size if part.exists() else 0
        headers = {"Accept": "*/*"}
        if pos:
            headers["Range"] = f"bytes={pos}-"
        try:
            resp = http.open(url, headers=headers, what=dest.name)
        except NetError as exc:
            if exc.status != 416 or not pos:
                raise
            if check is None or check.matches(part):  # the part was already complete
                resumed_from = pos
                break
            part.unlink()
            notes.append("the server refused the resume range and the part failed its check; started over")
            continue
        with resp:
            resumed_from = _stream(http, resp, part, pos, dest=dest, size=size, floor_gb=disk_floor_gb,
                                   disk_free=disk_free, progress=progress, notes=notes)
        break

    if check is not None and not check.matches(part):
        part.unlink()
        raise ShaMismatch(f"{http.site}: {dest.name}: {check.kind} mismatch: expected {check.expected}; the part "
                          "file was deleted; run the download once more, and stop and report on a second mismatch")
    sha = check.expected if check is not None and check.kind == "sha256" else hashing.sha256_file(part)
    store.replace(part, dest)  # an antivirus may still hold the fresh file: retry, then exit 4
    return Download(dest, dest.stat().st_size, sha, check.kind if check else "none", resumed_from=resumed_from,
                    notes=notes)


def _stream(http: Http, resp: object, part: Path, pos: int, *, dest: Path, size: int | None, floor_gb: float,
            disk_free: Callable[[Path], int], progress: Callable[[int, int | None], None] | None,
            notes: list[str]) -> int:
    """Write one response's body into the part file; returns the offset it resumed at (0 = from the start)."""
    status = _status(resp)
    if pos and status != 206:
        notes.append(f"the server ignored the resume range (HTTP {status}); started over")
        pos = 0
    ctype = (resp.headers.get("Content-Type") or "").lower()
    suffix = dest.suffix.lower()
    first = resp.read(CHUNK)
    if suffix not in (".html", ".htm") and ("text/html" in ctype or looks_like_html(first)):
        raise LoginRequired(f"{http.site}: {dest.name}: the site sent a web page instead of the file (a login or "
                            f"an accepted licence is needed); this tool never logs in{http.login_hint}")
    if "application/json" in ctype and suffix != ".json":
        raise NetError(f"{http.site}: {dest.name}: the site sent a message instead of the file: {_json_message(first)}")
    total = _total_size(resp, status, pos) or size
    check_disk_floor(dest.parent, (total - pos) if total else None, floor_gb, what=dest.name, disk_free=disk_free)
    got, since_check = pos, 0
    try:
        with part.open("ab" if pos else "wb") as handle:
            chunk = first
            while chunk:
                handle.write(chunk)
                got += len(chunk)
                since_check += len(chunk)
                if progress is not None:
                    progress(got, total)
                if not total and since_check >= FLOOR_RECHECK_BYTES:
                    check_disk_floor(dest.parent, None, floor_gb, what=dest.name, disk_free=disk_free)
                    since_check = 0
                chunk = resp.read(CHUNK)
    except (http.client.HTTPException, OSError) as exc:
        raise NetError(f"{http.site}: {dest.name}: interrupted at {got} of {total or '?'} bytes ({_reason(exc)}); "
                       "run the same download again to resume") from None
    if total and got < total:
        raise NetError(f"{http.site}: {dest.name}: the connection closed at {got} of {total} bytes; run the same "
                       "download again to resume")
    return pos


# --- Civitai records ---------------------------------------------------------------------------------------------

# Civitai's model types -> the registry's resource types (anything else is "other").
SITE_TYPES = {
    "Checkpoint": "checkpoint",
    "LORA": "lora",
    "LoCon": "lycoris",
    "DoRA": "lycoris",
    "TextualInversion": "embedding",
    "VAE": "vae",
    "Controlnet": "controlnet",
    "Upscaler": "upscaler",
}
# The registry's types -> the `types` value a search sends.
QUERY_TYPES = {
    "checkpoint": "Checkpoint",
    "lora": "LORA",
    "lycoris": "LoCon",
    "embedding": "TextualInversion",
    "vae": "VAE",
    "controlnet": "Controlnet",
    "upscaler": "Upscaler",
}
# allowCommercialUse values, most permissive first; RentCivit (renting only on the site itself) grants nothing here.
COMMERCIAL_LADDER = (("Sell", "sell"), ("Rent", "rent"), ("Image", "images"))
_COMMERCIAL_KNOWN = {"None", "Image", "Rent", "RentCivit", "Sell"}
# nsfwLevel bits: 1 PG, 2 PG-13, 4 R, 8 X, 16 XXX, 32 blocked.
_LEVEL_EXPLICIT = 4 | 8 | 16 | 32
_LEVEL_SUGGESTIVE = 2
_LEVEL_SFW = 1


def site_type(type_: str) -> str:
    """The Civitai `types` value for a registry type (`lora` -> `LORA`) or a Civitai type name given as is."""
    if type_ in QUERY_TYPES:
        return QUERY_TYPES[type_]
    if type_ in SITE_TYPES:
        return type_
    known = ", ".join(sorted(QUERY_TYPES))
    raise Refused(f"refused: unknown type {type_!r}; use one of {known} (or a Civitai type name)")


def resource_type(value: object) -> str:
    """The registry type for a Civitai model type; `other` when it has no equivalent there."""
    return SITE_TYPES.get(str(value), "other")


def _positive_id(value: object, what: str) -> int:
    try:
        number = int(str(value).strip())
    except ValueError:
        number = 0
    if number <= 0:
        raise Refused(f"refused: {what} must be a positive integer, not {value!r}")
    return number


def model_url(model_id: object) -> str:
    """The API URL of one model record."""
    return f"{API_ROOT}models/{_positive_id(model_id, 'a model id')}"


def version_url(version_id: object) -> str:
    """The API URL of one model-version record."""
    return f"{API_ROOT}model-versions/{_positive_id(version_id, 'a version id')}"


def by_hash_url(sha256: str) -> str:
    """The API URL that finds the model version a file's sha256 belongs to."""
    if not isinstance(sha256, str) or not hashing.is_sha256(sha256):
        raise Refused(f"refused: not a sha256: {sha256!r}")
    return f"{API_ROOT}model-versions/by-hash/{sha256.lower()}"


def download_url(version_id: object) -> str:
    """The download URL of a version's primary file."""
    return f"{DOWNLOAD_ROOT}models/{_positive_id(version_id, 'a version id')}"


def search_url(query: str | None = None, *, type_: str | None = None, limit: int = 20, cursor: str | None = None,
               sort: str | None = None, nsfw: bool | None = False, base_models: Iterable[str] = ()) -> str:
    """The API URL of a model search. `nsfw=False` (the default) asks the site to hide models without safe images."""
    if not isinstance(limit, int) or not 1 <= limit <= 100:
        raise Refused(f"refused: limit must be 1..100, not {limit!r}")
    params: list[tuple[str, str]] = [("limit", str(limit))]
    if query:
        params.append(("query", query))
    if type_:
        params.append(("types", site_type(type_)))
    if sort:
        params.append(("sort", sort))
    if cursor:
        params.append(("cursor", str(cursor)))
    if nsfw is not None:
        params.append(("nsfw", "true" if nsfw else "false"))
    params.extend(("baseModels", str(b)) for b in base_models)
    return f"{API_ROOT}models?{urllib.parse.urlencode(params)}"


def _flag(value: object, name: str, notes: list[str]) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    notes.append(f"{name} has an unexpected value {value!r}; left unknown")
    return None


def _commercial(value: object, notes: list[str]) -> str | None:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else list(value) if isinstance(value, list | tuple) else None
    if values is None or not all(isinstance(v, str) for v in values):
        notes.append(f"allowCommercialUse has an unexpected value {value!r}; left unknown")
        return None
    unknown = sorted(set(values) - _COMMERCIAL_KNOWN)
    if unknown:
        notes.append(f"allowCommercialUse values not mapped: {', '.join(unknown)}")
    if "RentCivit" in values:
        notes.append("RentCivit: renting the model is allowed only through the site's own generator")
    for site_value, mapped in COMMERCIAL_LADDER:
        if site_value in values:
            return mapped
    return "none"


def licence_from(model: Mapping, version: Mapping | None = None, *, checked_at: str | None = None) -> dict:
    """The resource schema's `licence` block from a model record's flags and the version's availability."""
    notes: list[str] = []
    raw = {key: model.get(key) for key in ("allowCommercialUse", "allowNoCredit", "allowDerivatives",
                                           "allowDifferentLicense")}
    no_credit = _flag(raw["allowNoCredit"], "allowNoCredit", notes)
    availability = (version or {}).get("availability") or model.get("availability")
    gated = redistributable = None
    if availability is not None:
        state = str(availability).lower()
        if state == "public":
            gated = False
        elif state in ("earlyaccess", "private"):
            gated, redistributable = True, False
            notes.append(f"availability {availability}: the file is not public")
        else:
            notes.append(f"availability {availability!r} not mapped")
    licence = {
        "status": "unverified" if any(v is not None for v in raw.values()) else "unknown",
        "name": None,
        "spdx": None,
        "url": None,
        "commercial_use": _commercial(raw["allowCommercialUse"], notes),
        "credit_required": None if no_credit is None else not no_credit,
        "derivatives_allowed": _flag(raw["allowDerivatives"], "allowDerivatives", notes),
        "relicense_allowed": _flag(raw["allowDifferentLicense"], "allowDifferentLicense", notes),
        "redistributable": redistributable,
        "gated": gated,
        "source": "civitai-api",
        "checked_at": checked_at,
        "note": "; ".join(notes) or None,
    }
    return licence


def rating_from(record: Mapping) -> str:
    """sfw | suggestive | explicit | unknown from `nsfwLevel` (bits) or `nsfw`; a contradiction is never sfw."""
    level = record.get("nsfwLevel")
    rating = "unknown"
    if isinstance(level, int) and not isinstance(level, bool) and level > 0:
        if level & _LEVEL_EXPLICIT:
            rating = "explicit"
        elif level & _LEVEL_SUGGESTIVE:
            rating = "suggestive"
        elif level & _LEVEL_SFW:
            rating = "sfw"
    elif record.get("nsfw") is False:
        rating = "sfw"
    if rating == "sfw" and record.get("nsfw") is True:
        rating = "unknown"
    return rating


def _sha256_of(file: Mapping) -> str | None:
    value = (file.get("hashes") or {}).get("SHA256")
    return value.lower() if isinstance(value, str) and hashing.is_sha256(value) else None


def summarize_file(file: Mapping, version_id: object = None) -> dict:
    """One file of a version: name, size, sha256, format, and a download URL under the public download root."""
    meta = file.get("metadata") or {}
    size_kb = file.get("sizeKB")
    url = file.get("downloadUrl")
    if not (isinstance(url, str) and url.startswith(DOWNLOAD_ROOT)):
        url = download_url(version_id) if file.get("primary") and version_id is not None else None
    return {
        "file_id": None if file.get("id") is None else str(file["id"]),
        "name": file.get("name"),
        "bytes": int(round(float(size_kb) * 1024)) if isinstance(size_kb, int | float) else None,
        "sha256": _sha256_of(file),
        "primary": bool(file.get("primary")),
        "kind": file.get("type"),
        "format": meta.get("format"),
        "fp": meta.get("fp"),
        "download_url": url,
    }


def primary_file(version_summary: Mapping) -> dict | None:
    """The version's primary file summary, else its first, else None."""
    files = list(version_summary.get("files") or [])
    return next((f for f in files if f.get("primary")), files[0] if files else None)


def summarize_version(version: Mapping, model: Mapping | None = None) -> dict:
    """A version record reduced to registry facts (no description, no images, no uploader)."""
    vid = version.get("id")
    model_id = version.get("modelId", (model or {}).get("id"))
    words = version.get("trainedWords") or []
    return {
        "version_id": None if vid is None else str(vid),
        "model_id": None if model_id is None else str(model_id),
        "name": version.get("name"),
        "base_model": version.get("baseModel"),
        "trigger_words": [w for w in words if isinstance(w, str)],
        "availability": version.get("availability"),
        "published_at": version.get("publishedAt"),
        "rating": rating_from(version),
        "api_url": version_url(vid) if vid is not None else None,
        "files": [summarize_file(f, vid) for f in version.get("files") or []],
    }


def summarize_model(model: Mapping, *, checked_at: str | None = None) -> dict:
    """A model record reduced to registry facts; the licence block uses the newest (first) version's availability."""
    versions = [v for v in model.get("modelVersions") or [] if isinstance(v, Mapping)]
    mid = model.get("id")
    return {
        "site": SITE,
        "model_id": None if mid is None else str(mid),
        "name": model.get("name"),
        "type": resource_type(model.get("type")),
        "site_type": model.get("type"),
        "rating": rating_from(model),
        "licence": licence_from(model, versions[0] if versions else None, checked_at=checked_at),
        "api_url": model_url(mid) if mid is not None else None,
        "versions": [summarize_version(v, model) for v in versions],
    }


@dataclass
class SearchPage:
    """One page of search results: model summaries, the cursor of the next page (None at the end), the URL asked."""

    items: list[dict]
    next_cursor: str | None
    api_url: str


def safe_file_name(name: object) -> str:
    """A site-given file name usable as one path component; anything that could leave the folder is refused."""
    text = str(name or "")
    if (not text or text in (".", "..") or text.startswith(".") or any(c in text for c in '/\\:*?"<>|\0')
            or text != text.strip()):
        raise Refused(f"refused: unsafe file name from the site: {text!r}")
    return text


class Client:
    """Civitai's public API with the shared plumbing; `from_config(cfg)` reads the instance's settings."""

    def __init__(self, *, tool: str = DEFAULT_TOOL, token: str | None = None, cache_dir: Path | None = None,
                 cache_hours: float = DEFAULT_CACHE_HOURS, requests_per_s: float = DEFAULT_REQUESTS_PER_S,
                 disk_floor_gb: float = DEFAULT_DISK_FLOOR_GB, opener: Opener | None = None,
                 sleep: Callable[[float], None] = time.sleep, monotonic: Callable[[], float] = time.monotonic,
                 clock: Callable[[], float] = time.time, disk_free: Callable[[Path], int] = free_bytes,
                 login_hint: str | None = None) -> None:
        self.tool = tool
        self.disk_floor_gb = float(disk_floor_gb)
        self.disk_free = disk_free
        # The fix line after "this tool never logs in": neutral by default, because this module serves both
        # tools and only the caller knows its own rescan command (it may pass that as `login_hint`).
        hint = LOGIN_HINT if login_hint is None else login_hint
        self.http = Http(site=SITE, prefixes=PUBLIC_PREFIXES, tool=tool, token=token, requests_per_s=requests_per_s,
                         cache=FileCache(cache_dir, cache_hours, clock=clock), opener=opener, sleep=sleep,
                         monotonic=monotonic, clock=clock, login_hint=hint)

    def __repr__(self) -> str:
        return f"civitai.Client({self.http!r})"

    @classmethod
    def from_config(cls, cfg: object, *, env: Mapping[str, str] | None = None, **overrides: object) -> Client:
        """A client from a loaded config: `search.*`, `limits.disk_floor_gb`, the cache under the instance."""
        env = os.environ if env is None else env
        get = cfg.get
        token_env = str(get("search.civitai_token_env", DEFAULT_TOKEN_ENV) or DEFAULT_TOKEN_ENV)
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

    def _checked_at(self) -> str:
        return ids.today(self.tool)

    def model(self, model_id: object, *, fresh: bool = False) -> dict:
        """The raw model record (cached); summarize it with `summarize_model` before keeping anything."""
        return self.http.get_json(model_url(model_id), fresh=fresh, what=f"model {model_id}")

    def version(self, version_id: object, *, fresh: bool = False) -> dict:
        """The raw model-version record (cached)."""
        return self.http.get_json(version_url(version_id), fresh=fresh, what=f"version {version_id}")

    def version_by_hash(self, sha256: str, *, fresh: bool = False) -> dict:
        """The raw model-version record a file's sha256 belongs to (cached); `NotFound` when the site has none."""
        return self.http.get_json(by_hash_url(sha256), fresh=fresh, what=f"the file {sha256[:16]}")

    def summary(self, model_id: object, *, fresh: bool = False) -> dict:
        """`summarize_model` of the model record, the licence dated today."""
        return summarize_model(self.model(model_id, fresh=fresh), checked_at=self._checked_at())

    def search(self, query: str | None = None, *, type_: str | None = None, limit: int = 20,
               cursor: str | None = None, sort: str | None = None, nsfw: bool | None = False,
               base_models: Iterable[str] = (), fresh: bool = False) -> SearchPage:
        """Search models over /api/v1/models; the answer's own next-page URL is never followed (cursor only)."""
        url = search_url(query, type_=type_, limit=limit, cursor=cursor, sort=sort, nsfw=nsfw,
                         base_models=base_models)
        answer = self.http.get_json(url, fresh=fresh, what=f"the search for {query!r}" if query else "the search")
        if not isinstance(answer, Mapping):
            raise NetError(f"{SITE}: the search answer is not an object")
        checked_at = self._checked_at()
        items = [summarize_model(m, checked_at=checked_at) for m in answer.get("items") or []
                 if isinstance(m, Mapping)]
        meta = answer.get("metadata") or {}
        cursor_next = meta.get("nextCursor") if isinstance(meta, Mapping) else None
        return SearchPage(items, None if cursor_next in (None, "") else str(cursor_next), url)

    def download(self, file: Mapping, folder: Path, *, progress: Callable[[int, int | None], None] | None = None,
                 allow_unverified: bool = False) -> Download:
        """Download a file summary (`summarize_file`) into `folder` under its own (checked) name."""
        url = file.get("download_url")
        if not url:
            raise Refused(f"refused: {file.get('name')!r} has no download URL under {DOWNLOAD_ROOT}")
        name = safe_file_name(file.get("name"))
        sha = file.get("sha256")
        return fetch_file(self.http, str(url), Path(folder) / name, check=sha256_check(sha) if sha else None,
                          size=file.get("bytes"), disk_floor_gb=self.disk_floor_gb, disk_free=self.disk_free,
                          progress=progress, allow_unverified=allow_unverified)
