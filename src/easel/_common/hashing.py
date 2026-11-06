"""sha256 helpers: files in chunks, bytes, text, and canonical JSON.

Every content id derives from a sha256 (`img-<sha[:12]>`, resources `<sha[:16]>`), downloads and copies are
verified against one, and a record's fingerprint is the sha256 of its canonical JSON (sorted keys, no spaces,
UTF-8), so the same data hashes the same on every machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHUNK = 1 << 20  # 1 MiB: model files are gigabytes; never read one into memory whole


def sha256_bytes(data: bytes) -> str:
    """The hex sha256 of `data`."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """The hex sha256 of `text` encoded as UTF-8."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk: int = CHUNK) -> str:
    """The hex sha256 of a file, read in chunks so a large model file never sits in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(obj: object) -> bytes:
    """`obj` as canonical JSON bytes: sorted keys, no spaces, UTF-8, so equal data gives equal bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_json(obj: object) -> str:
    """The hex sha256 of `obj`'s canonical JSON: a record's fingerprint."""
    return sha256_bytes(canonical_json(obj))


def short(sha256: str, n: int) -> str:
    """The first `n` hex characters of a digest (12 for images, 16 for resources)."""
    return sha256.lower()[:n]


def is_sha256(value: str) -> bool:
    """True when `value` is 64 hex characters (either case)."""
    return len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def verify_file(path: Path, expected: str) -> bool:
    """True when the file's sha256 equals `expected` (case-insensitive); a full digest or a prefix of >= 12."""
    want = expected.strip().lower()
    if len(want) < 12:
        raise ValueError(f"a sha256 to verify against needs at least 12 hex characters: {expected!r}")
    return sha256_file(path).startswith(want)
