"""_common.image_meta: header-only probes of PNG, JPEG and WebP files written byte by byte into tmp_path.

Every image here carries image data that no decoder could read (random bytes, a broken zlib stream): a probe
that tried to decode pixels would fail, so passing tests show that only headers are read.
"""

import hashlib
import struct
import zlib

import pytest

from easel._common import image_meta

JUNK = bytes(range(256)) * 4  # not a valid zlib stream, not valid entropy-coded data


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_bytes(width=8, height=8, colour=2, depth=8, texts=(), idat=JUNK, text_after=()) -> bytes:
    out = image_meta.PNG_SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 0))
    for kind, key, value in texts:
        out += chunk(kind, key + b"\x00" + value)
    out += chunk(b"IDAT", idat)
    for kind, key, value in text_after:
        out += chunk(kind, key + b"\x00" + value)
    return out + chunk(b"IEND", b"")


def jpeg_bytes(width=8, height=6, components=3) -> bytes:
    def segment(code: int, body: bytes) -> bytes:
        return bytes([0xFF, code]) + struct.pack(">H", len(body) + 2) + body

    return (b"\xff\xd8"
            + segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
            + segment(0xE1, b"Exif\x00\x00" + b"\x00" * 20)
            + segment(0xE1, b"xmp-namespace/xap/1.0/\x00<x/>")  # an XMP segment starts with its namespace
            + segment(0xFE, b"demo comment")
            + b"\xff\xff"  # fill bytes before a marker are legal
            + segment(0xC0, struct.pack(">BHHB", 8, height, width, components) + b"\x01\x22\x00" * components)
            + segment(0xDA, b"\x00" * 8) + JUNK + b"\xff\xd9")


def riff(chunks: bytes) -> bytes:
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WEBP" + chunks


def webp_chunk(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("<I", len(payload)) + payload + (b"\x00" if len(payload) % 2 else b"")


@pytest.fixture
def write(tmp_path):
    def _write(name: str, data: bytes):
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return _write


def test_png_header_fields_and_text_keys_without_values(write):
    prompt = b'{"1": {"class_type": "KSampler"}}'
    path = write("fix.png", png_bytes(texts=[(b"tEXt", b"prompt", prompt), (b"tEXt", b"workflow", b"{}")]))
    info = image_meta.probe(path)
    assert info["readable"] is True
    assert (info["format"], info["image_format"], info["width"], info["height"]) == ("png", "png", 8, 8)
    assert (info["bit_depth"], info["color_mode"], info["chunk_count"]) == (8, "RGB", 5)
    assert info["text_chunk_keys"] == [{"key": "prompt", "bytes": len(prompt) + 7, "chunk": "tEXt"},
                                       {"key": "workflow", "bytes": 11, "chunk": "tEXt"}]
    assert "KSampler" not in repr(info)  # the value of a text chunk is never read into the result
    assert info["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert info["bytes"] == path.stat().st_size and info["name"] == "fix.png" and info["suffix"] == ".png"


def test_probe_always_returns_the_same_keys(write):
    good = image_meta.probe(write("a.png", png_bytes()))
    bad = image_meta.probe(write("b.txt", b"plain text, not an image"))
    assert set(good) | {"error"} == set(bad)
    assert bad["readable"] is False and "unsupported" in bad["error"]
    assert bad["format"] is None and bad["width"] is None


def test_png_colour_types_and_text_after_the_image_data(write):
    path = write("g.png", png_bytes(width=3, height=5, colour=6, depth=16,
                                    text_after=[(b"zTXt", b"parameters", b"\x00" + zlib.compress(b"Steps: 20"))]))
    info = image_meta.probe(path, sha256=False)
    assert (info["width"], info["height"], info["color_mode"], info["bit_depth"]) == (3, 5, "RGBA", 16)
    assert info["text_chunk_keys"][0]["key"] == "parameters" and info["text_chunk_keys"][0]["chunk"] == "zTXt"
    assert info["sha256"] is None


def test_png_that_does_not_start_with_ihdr_is_unreadable(write):
    data = image_meta.PNG_SIGNATURE + chunk(b"tEXt", b"k\x00v") + chunk(b"IEND", b"")
    info = image_meta.probe(write("odd.png", data))
    assert info["readable"] is False and "IHDR" in info["error"]


def test_truncated_png_is_unreadable_not_a_crash(write):
    info = image_meta.probe(write("cut.png", png_bytes()[:20]))
    assert info["readable"] is False and info["error"]


def test_a_misnamed_file_is_probed_by_its_bytes(write):
    info = image_meta.probe(write("really-a-png.jpg", png_bytes()))
    assert info["format"] == "png" and info["suffix"] == ".jpg" and info["readable"]


def test_pixels_sha256_ignores_text_chunks_and_never_decodes(write):
    plain = write("plain.png", png_bytes())
    tagged = write("tagged.png", png_bytes(texts=[(b"tEXt", b"prompt", b"{}")]))
    other = write("other.png", png_bytes(idat=JUNK[::-1]))
    assert image_meta.pixels_sha256(plain) == image_meta.pixels_sha256(tagged) == hashlib.sha256(JUNK).hexdigest()
    assert image_meta.pixels_sha256(other) != image_meta.pixels_sha256(plain)
    assert image_meta.pixels_sha256(write("x.jpg", jpeg_bytes())) is None


def test_pixels_sha256_of_a_lying_idat_allocates_nothing_and_gives_none(write):
    """A 49-byte file that declares a 64 MiB IDAT: no read of the declared size, no MemoryError, None."""
    import tracemalloc

    head = image_meta.PNG_SIGNATURE + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    lying = write("lying.png", head + struct.pack(">I", 64 * 2**20) + b"IDAT" + b"\x00" * 8)
    tracemalloc.start()
    try:
        assert image_meta.pixels_sha256(lying) is None
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * 2**20


def test_pixels_sha256_hashes_a_large_idat_in_blocks(write, monkeypatch):
    monkeypatch.setattr(image_meta.hashing, "CHUNK", 7)  # many short reads give the same digest as one long one
    path = write("blocks.png", png_bytes())
    assert image_meta.pixels_sha256(path) == hashlib.sha256(JUNK).hexdigest()
    cut = write("cut-idat.png", path.read_bytes()[:-40])  # IEND and the end of IDAT missing
    assert image_meta.pixels_sha256(cut) is None


def test_jpeg_frame_header_and_metadata_segments(write):
    info = image_meta.probe(write("a.jpg", jpeg_bytes(width=8, height=6)))
    assert info["readable"] is True
    assert (info["format"], info["width"], info["height"], info["bit_depth"], info["color_mode"]) == \
        ("jpeg", 8, 6, 8, "RGB")
    assert [k["key"] for k in info["text_chunk_keys"]] == ["APP1/EXIF", "APP1/XMP", "COM"]


def test_grey_jpeg_and_jpeg_without_frame_header(write):
    assert image_meta.probe(write("g.jpg", jpeg_bytes(components=1)))["color_mode"] == "L"
    no_frame = b"\xff\xd8\xff\xda\x00\x04\x00\x00" + JUNK
    info = image_meta.probe(write("n.jpg", no_frame))
    assert info["readable"] is False and "frame header" in info["error"]


def test_webp_lossless_header(write):
    bits = (8 - 1) | ((6 - 1) << 14) | (1 << 28)  # 8x6 with alpha
    data = riff(webp_chunk(b"VP8L", b"\x2f" + struct.pack("<I", bits) + JUNK[:20]))
    info = image_meta.probe(write("l.webp", data))
    assert (info["format"], info["width"], info["height"], info["color_mode"]) == ("webp", 8, 6, "RGBA")


def test_webp_lossy_header(write):
    frame = b"\x00\x00\x00" + b"\x9d\x01\x2a" + struct.pack("<HH", 8, 6) + JUNK[:30]
    info = image_meta.probe(write("v.webp", riff(webp_chunk(b"VP8 ", frame))))
    assert (info["width"], info["height"], info["color_mode"]) == (8, 6, "RGB")


def test_webp_extended_header_lists_exif_and_xmp(write):
    vp8x = bytes([0x08 | 0x04]) + b"\x00\x00\x00" + (8 - 1).to_bytes(3, "little") + (6 - 1).to_bytes(3, "little")
    data = riff(webp_chunk(b"VP8X", vp8x) + webp_chunk(b"VP8L", b"\x2f" + JUNK[:9])
                + webp_chunk(b"EXIF", b"Exif\x00\x00abc") + webp_chunk(b"XMP ", b"<x/>"))
    info = image_meta.probe(write("x.webp", data))
    assert (info["width"], info["height"], info["color_mode"]) == (8, 6, "RGB")
    assert info["text_chunk_keys"] == [{"key": "EXIF", "bytes": 9, "chunk": "EXIF"},
                                       {"key": "XMP", "bytes": 4, "chunk": "XMP"}]


def test_unknown_webp_chunk_is_unreadable(write):
    info = image_meta.probe(write("u.webp", riff(webp_chunk(b"ABCD", b"1234"))))
    assert info["readable"] is False and "WebP" in info["error"]


def test_placeholder_state_of_an_ordinary_file(write):
    assert image_meta.placeholder_state(write("a.png", png_bytes())) in ("local", "unknown")
    assert image_meta.probe(write("b.png", png_bytes()))["hydration"] in ("local", "unknown")


def test_iter_images_by_suffix_sorted(tmp_path, write):
    write("b.png", png_bytes())
    write("a.JPG", jpeg_bytes())
    write("notes.txt", b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.webp").write_bytes(b"RIFF")
    names = [p.name for p in image_meta.iter_images(tmp_path)]
    assert names == ["a.JPG", "b.png", "c.webp"]
    assert list(image_meta.iter_images(tmp_path / "b.png")) == [tmp_path / "b.png"]
