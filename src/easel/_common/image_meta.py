"""Header-only image probe: format, size, colour mode and which metadata chunks exist. Pixels are never decoded.

Why it exists: image content is opaque to these tools. A registry needs only structure (bytes, sha256, width,
height, colour mode, which text chunks are present), and all of that sits in the first few hundred bytes or in
chunk headers. So this module reads chunk headers and seeks over everything else: no PIL, no decoder, no
thumbnail, no description. Text chunk *contents* are not read either (only the keyword and the byte count);
`_common.embedded_meta` is the one place that reads generation metadata, for the callers that store it.

Supported: PNG, JPEG, WebP, recognised by their first bytes (a misnamed file is probed by what it is). Anything
else is reported as `readable: False` with the reason, never guessed at.

    from <pkg>._common import image_meta   # <pkg>: the package this copy lives in
    info = image_meta.probe(path)          # {'format': 'png', 'width': 8, 'height': 8, ...}

`probe(path)` always returns the same keys: `path name suffix bytes hydration readable sha256 format image_format
width height bit_depth color_mode text_chunk_keys chunk_count`, plus `error` when unreadable. `image_format` is
the same value as `format` (the older key name, kept for ported callers). `sha256` is None with `sha256=False`.
`hydration` is `local`, `cloud-only`, `recall-on-open` or `unknown`: a cloud-synced folder can hold placeholder
files whose bytes are not on disk; hashing such a file downloads it. A path that does not exist raises
FileNotFoundError (a missing file is the caller's problem, not an unreadable image).
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import BinaryIO

from . import hashing

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt")
PNG_COLOUR = {0: "L", 2: "RGB", 3: "P", 4: "LA", 6: "RGBA"}
JPEG_FRAME_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF})
JPEG_COMPONENTS = {1: "L", 3: "RGB", 4: "CMYK"}
WEBP_TEXT_CHUNKS = (b"EXIF", b"XMP ")

# Windows file attributes of cloud placeholders (FILE_ATTRIBUTE_*).
_OFFLINE = 0x1000
_RECALL_ON_OPEN = 0x40000
_RECALL_ON_DATA_ACCESS = 0x400000


def sniff(head: bytes) -> str | None:
    """`png`, `jpeg` or `webp` from a file's first 12 bytes, else None."""
    if head.startswith(PNG_SIGNATURE):
        return "png"
    if head.startswith(b"\xff\xd8"):
        return "jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    return None


def placeholder_state(path: Path) -> str:
    """Whether a file's bytes are on disk, read from its attributes without opening it."""
    try:
        attrs = Path(path).stat().st_file_attributes  # Windows only
    except (AttributeError, OSError):
        return "unknown"
    if attrs & (_OFFLINE | _RECALL_ON_DATA_ACCESS):
        return "recall-on-open" if attrs & _RECALL_ON_OPEN else "cloud-only"
    return "local"


# --- PNG ----------------------------------------------------------------------------------------------------------

def _chunks(handle: BinaryIO) -> Iterator[tuple[bytes, int]]:
    """(type, length) of each PNG chunk after the signature, the handle left at the chunk's data."""
    while True:
        head = handle.read(8)
        if len(head) < 8:
            return
        length, kind = struct.unpack(">I4s", head)
        start = handle.tell()
        yield kind, length
        handle.seek(start + length + 4)  # data and CRC, whatever the consumer read
        if kind == b"IEND":
            return


def _probe_png(handle: BinaryIO) -> dict:
    handle.read(len(PNG_SIGNATURE))
    out: dict = {"format": "png", "text_chunk_keys": [], "chunk_count": 0}
    for kind, length in _chunks(handle):
        out["chunk_count"] += 1
        if out["chunk_count"] == 1:
            if kind != b"IHDR" or length != 13:
                raise ValueError("a PNG must start with a 13-byte IHDR chunk")
            width, height, depth, colour = struct.unpack(">IIBB", handle.read(10))
            if not width or not height:
                raise ValueError("IHDR gives a zero width or height")
            out.update(width=width, height=height, bit_depth=depth,
                       color_mode=PNG_COLOUR.get(colour, f"type{colour}"))
        elif kind in PNG_TEXT_CHUNKS:
            # The keyword only (1-79 bytes before a NUL); the value may hold a whole prompt and is not read.
            keyword = handle.read(min(length, 80)).split(b"\x00", 1)[0].decode("latin-1", "replace")
            out["text_chunk_keys"].append({"key": keyword, "bytes": length, "chunk": kind.decode("ascii")})
    if not out["chunk_count"]:
        raise ValueError("a PNG signature with no chunks")
    out["text_chunk_keys"].sort(key=lambda entry: entry["key"])
    return out


def pixels_sha256(path: Path) -> str | None:
    """sha256 of a PNG's IDAT chunks as stored (compressed, never decoded); None for anything but a whole PNG.

    Two renders of the same pixels differ as files when their text chunks differ (ComfyUI writes the graph
    there); their IDAT bytes do not. Equal values mean identical pixels; different values can still be the
    same pixels written by another encoder. A chunk is hashed in blocks of at most `hashing.CHUNK` bytes, never
    read whole: a damaged or truncated file whose chunk header claims more bytes than the file holds gives None
    (the caller then compares whole files), and nothing near the claimed size is ever allocated.
    """
    digest = hashlib.sha256()
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return None
        for kind, length in _chunks(handle):
            if kind != b"IDAT":
                continue
            if length > size - handle.tell():
                return None  # the header claims more than the file holds
            remaining = length
            while remaining:
                block = handle.read(min(remaining, hashing.CHUNK))
                if not block:
                    return None  # the file ended early (it changed while it was read)
                digest.update(block)
                remaining -= len(block)
    return digest.hexdigest()


# --- JPEG ---------------------------------------------------------------------------------------------------------

def _probe_jpeg(handle: BinaryIO) -> dict:
    handle.read(2)
    keys: list[dict] = []
    while True:
        byte = handle.read(1)
        if not byte:
            raise ValueError("no JPEG frame header before the end of the file")
        if byte != b"\xff":
            continue  # resynchronise on the next marker
        marker = handle.read(1)
        while marker == b"\xff":  # fill bytes
            marker = handle.read(1)
        if not marker:
            raise ValueError("a JPEG ends inside a marker")
        code = marker[0]
        if code in (0x00, 0x01) or 0xD0 <= code <= 0xD8:
            continue  # stuffing, TEM, restart markers, a second SOI: no length follows
        if code in (0xD9, 0xDA):
            raise ValueError("no JPEG frame header before the image data")
        raw_length = handle.read(2)
        if len(raw_length) < 2:
            raise ValueError("a JPEG ends inside a segment header")
        body = struct.unpack(">H", raw_length)[0] - 2
        if body < 0:
            raise ValueError("a JPEG segment with a length below 2")
        if code in JPEG_FRAME_MARKERS:
            frame = handle.read(6)
            if len(frame) < 6:
                raise ValueError("a JPEG ends inside its frame header")
            precision, height, width, components = struct.unpack(">BHHB", frame)
            return {"format": "jpeg", "width": width, "height": height, "bit_depth": precision,
                    "color_mode": JPEG_COMPONENTS.get(components, f"{components}ch"),
                    "text_chunk_keys": sorted(keys, key=lambda entry: entry["key"])}
        if code == 0xE1:
            peek = handle.read(min(body, 32))
            label = "APP1/EXIF" if peek.startswith(b"Exif\x00") else "APP1/XMP" if b"/xap/1.0/" in peek else "APP1"
            keys.append({"key": label, "bytes": body, "chunk": "APP1"})
            handle.seek(body - len(peek), 1)
            continue
        if code == 0xFE:
            keys.append({"key": "COM", "bytes": body, "chunk": "COM"})
        handle.seek(body, 1)


# --- WebP ---------------------------------------------------------------------------------------------------------

def _probe_webp(handle: BinaryIO, file_bytes: int) -> dict:
    riff = handle.read(12)
    end = min(file_bytes, 8 + struct.unpack("<I", riff[4:8])[0])
    out: dict = {"format": "webp", "bit_depth": 8, "text_chunk_keys": []}
    first = True
    while handle.tell() + 8 <= end:
        kind, size = struct.unpack("<4sI", handle.read(8))
        start = handle.tell()
        if first:
            first = False
            data = handle.read(min(size, 10))
            if kind == b"VP8X" and len(data) >= 10:
                out.update(width=int.from_bytes(data[4:7], "little") + 1,
                           height=int.from_bytes(data[7:10], "little") + 1,
                           color_mode="RGBA" if data[0] & 0x10 else "RGB")
            elif kind == b"VP8 " and len(data) >= 10 and data[3:6] == b"\x9d\x01\x2a":
                width, height = struct.unpack("<HH", data[6:10])
                out.update(width=width & 0x3FFF, height=height & 0x3FFF, color_mode="RGB")
            elif kind == b"VP8L" and len(data) >= 5 and data[0] == 0x2F:
                bits = int.from_bytes(data[1:5], "little")
                out.update(width=(bits & 0x3FFF) + 1, height=((bits >> 14) & 0x3FFF) + 1,
                           color_mode="RGBA" if bits >> 28 & 1 else "RGB")
            else:
                raise ValueError(f"unknown first WebP chunk {kind!r}")
        elif kind in WEBP_TEXT_CHUNKS:
            out["text_chunk_keys"].append({"key": kind.decode("ascii").strip(), "bytes": size,
                                           "chunk": kind.decode("ascii").strip()})
        handle.seek(start + size + (size & 1))  # payloads are padded to an even length
    if first:
        raise ValueError("a WebP file with no chunks")
    out["text_chunk_keys"].sort(key=lambda entry: entry["key"])
    return out


# --- the public interface -----------------------------------------------------------------------------------------

def probe(path: Path | str, *, sha256: bool = True) -> dict:
    """Structural metadata of one image file (see the module docstring for the keys). Never reads pixel data."""
    path = Path(path)
    size = path.stat().st_size
    info: dict = {"path": str(path), "name": path.name, "suffix": path.suffix.lower(), "bytes": size,
                  "hydration": placeholder_state(path), "readable": True,
                  "sha256": hashing.sha256_file(path) if sha256 else None,
                  "format": None, "image_format": None, "width": None, "height": None, "bit_depth": None,
                  "color_mode": None, "text_chunk_keys": [], "chunk_count": None}
    try:
        with path.open("rb") as handle:
            kind = sniff(handle.read(12))
            handle.seek(0)
            if kind == "png":
                info.update(_probe_png(handle))
            elif kind == "jpeg":
                info.update(_probe_jpeg(handle))
            elif kind == "webp":
                info.update(_probe_webp(handle, size))
            else:
                info.update(readable=False, error="unsupported format (not PNG, JPEG or WebP)")
    except (ValueError, struct.error, OSError) as exc:
        info.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    info["image_format"] = info["format"]
    return info


def iter_images(target: Path | str) -> Iterator[Path]:
    """`target` itself when it is a file, else every image file under it (by suffix), in sorted order."""
    target = Path(target)
    if target.is_file():
        yield target
        return
    for path in sorted(target.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
