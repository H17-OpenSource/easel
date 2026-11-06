"""Read the metadata a file carries about itself: safetensors headers, and the generation settings in images.

Why it exists: a model file's header names its architecture, training settings and tensors, and an image made
by ComfyUI or an A1111-style UI carries the exact recipe that made it (ComfyUI: the API graph under the PNG text
key `prompt` and the editor graph under `workflow`; A1111, Forge and most UIs: a `parameters` block, or the EXIF
UserComment in a JPEG or WebP). That is the best evidence of what a file is and how a picture was made, and it
can be read without loading a tensor or decoding a pixel.

What is read, and what never is:
- safetensors: the 8-byte little-endian header length and the JSON header after it. Tensor data is never read.
- PNG: the tEXt, zTXt and iTXt chunks (decompressed with a size limit); image data is skipped with a seek.
- JPEG/WebP: EXIF ImageDescription, XPComment and UserComment, XMP and the comment, through Pillow's header
  reader (opening reads headers; pixels load only on `.load()`, which this module never calls).
- Videos are not read here (their container tags need ffmpeg; later).

Privacy: the values returned can hold whole prompts, captions (`ss_tag_frequency`) or a model's private
training notes. Callers store them in the instance; they never print them. `probe_keys()` is the safe summary to
print: which keys exist, never their values.

    from <pkg>._common import embedded_meta   # <pkg>: the package this copy lives in
    head = embedded_meta.safetensors_header(path)    # {'metadata': {...}, 'tensor_count': 3, ...}; ValueError if bad
    api, ui = embedded_meta.comfy_graphs(embedded_meta.read_fields(png))
    recipe = embedded_meta.recipe(embedded_meta.read_fields(png))   # {'format': 'comfy', 'seed': 1001, ...}
"""

from __future__ import annotations

import json
import struct
import zlib
from collections import Counter
from pathlib import Path

from .image_meta import PNG_SIGNATURE, PNG_TEXT_CHUNKS, sniff

MAX_HEADER_BYTES = 100_000_000  # the reference safetensors reader's own limit
MAX_TEXT_BYTES = 16 * 1024 * 1024  # one PNG text chunk, before and after decompression
PRECISION = {"F16": "fp16", "BF16": "bf16", "F32": "fp32", "F64": "fp64", "F8_E4M3": "fp8", "F8_E5M2": "fp8"}
_DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}


# --- safetensors --------------------------------------------------------------------------------------------------

def _tensor_prefix(name: str) -> str:
    """The family of a tensor name: its first dotted part, or `lora_unet` / `lora_te1` for kohya-style names."""
    first = name.split(".", 1)[0]
    if first.startswith("lora_"):
        parts = first.split("_")
        return "_".join(parts[:2])
    return first


def safetensors_header(path: Path | str) -> dict:
    """The header of a safetensors file, summarised; ValueError when the file is not a well-formed one.

    Returns `header_bytes`, `file_bytes`, `data_bytes` (where the last tensor ends), `metadata` (the
    `__metadata__` map, str -> str, {} when absent), `tensor_count`, `param_count`, `dtypes` (sorted),
    `precision` (fp16, bf16, fp32, fp8 ... of the dtype holding the most parameters, or None), `prefixes`
    ({first name part: tensor count}), `tensor_names` (sorted) and `tensors` ({name: {dtype, shape}}). A
    missing file raises FileNotFoundError, as `open()` does; every problem with the bytes is a ValueError.
    """
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise ValueError("not a safetensors file: shorter than the 8-byte header length")
        (length,) = struct.unpack("<Q", head)
        if length < 2 or length > MAX_HEADER_BYTES:
            raise ValueError(f"not a safetensors file: implausible header length {length}")
        if 8 + length > size:
            raise ValueError("the file is shorter than its header length (a partial download?)")
        raw = handle.read(length)
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"not a safetensors file: the header is not JSON ({exc})") from exc
    if not isinstance(header, dict):
        raise ValueError("not a safetensors file: the header is not a JSON object")

    meta_raw = header.pop("__metadata__", None) or {}
    if not isinstance(meta_raw, dict):
        raise ValueError("__metadata__ is not an object")
    metadata = {str(k): v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for k, v in meta_raw.items()}

    tensors: dict[str, dict] = {}
    params_by_dtype: Counter[str] = Counter()
    data_end = 0
    for name, info in header.items():
        if not isinstance(info, dict):
            raise ValueError(f"tensor {name!r}: its entry is not an object")
        dtype, shape, offsets = info.get("dtype"), info.get("shape"), info.get("data_offsets")
        if not isinstance(dtype, str) or not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0
                                                                               for d in shape):
            raise ValueError(f"tensor {name!r}: dtype or shape missing or malformed")
        if (not isinstance(offsets, list) or len(offsets) != 2 or not all(isinstance(o, int) for o in offsets)
                or not 0 <= offsets[0] <= offsets[1]):
            raise ValueError(f"tensor {name!r}: data_offsets missing or malformed")
        count = 1
        for dim in shape:
            count *= dim
        if dtype in _DTYPE_BYTES and offsets[1] - offsets[0] != count * _DTYPE_BYTES[dtype]:
            raise ValueError(f"tensor {name!r}: data_offsets do not match its dtype and shape")
        params_by_dtype[dtype] += count
        data_end = max(data_end, offsets[1])
        tensors[name] = {"dtype": dtype, "shape": list(shape)}
    if 8 + length + data_end > size:
        raise ValueError("the file is shorter than its tensors need (a partial download?)")

    names = sorted(tensors)
    dominant = max(sorted(params_by_dtype), key=lambda d: params_by_dtype[d]) if params_by_dtype else None
    return {
        "header_bytes": length,
        "file_bytes": size,
        "data_bytes": data_end,
        "metadata": metadata,
        "tensor_count": len(tensors),
        "param_count": sum(params_by_dtype.values()),
        "dtypes": sorted(params_by_dtype),
        "precision": PRECISION.get(dominant, dominant.lower()) if dominant else None,
        "prefixes": dict(sorted(Counter(_tensor_prefix(n) for n in names).items())),
        "tensor_names": names,
        "tensors": {n: tensors[n] for n in names},
    }


def metadata_value(metadata: dict[str, str], key: str) -> object:
    """One `__metadata__` value, decoded when it is JSON text for an object or a list (kohya writes several so)."""
    value = metadata.get(key)
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


# --- images: reading the fields -------------------------------------------------------------------------------------

def _inflate(data: bytes) -> bytes:
    """zlib data, refusing to grow past MAX_TEXT_BYTES (a decompression bomb is an error, not a hang)."""
    inflater = zlib.decompressobj()
    out = inflater.decompress(data, MAX_TEXT_BYTES)
    if inflater.unconsumed_tail:
        raise ValueError("a compressed PNG text chunk is larger than the size limit")
    return out


def _png_text_value(kind: bytes, data: bytes) -> tuple[str, str]:
    key, _, rest = data.partition(b"\x00")
    keyword = key.decode("latin-1", "replace")
    if kind == b"tEXt":
        return keyword, rest.decode("latin-1")
    if kind == b"zTXt":  # compression method byte, then zlib data
        return keyword, _inflate(rest[1:]).decode("latin-1")
    # iTXt: compression flag, method, language NUL, translated keyword NUL, text (UTF-8)
    flag = rest[:1] == b"\x01"
    body = rest[2:].split(b"\x00", 2)[-1]
    return keyword, (_inflate(body) if flag else body).decode("utf-8", "replace")


def png_text(path: Path | str) -> dict[str, str]:
    """Every text chunk of a PNG, keyword -> text (the first chunk wins on a repeated keyword).

    Chunks anywhere in the file are read (ComfyUI writes them before the image data, other tools after it); a
    chunk that does not decode is skipped. A chunk whose header claims more bytes than the file holds ends the
    read (a truncated file): nothing is allocated for it. Not a PNG -> {}.
    """
    fields: dict[str, str] = {}
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as handle:
        if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
            return fields
        while True:
            head = handle.read(8)
            if len(head) < 8:
                break
            length, kind = struct.unpack(">I4s", head)
            if kind == b"IEND" or length > size - handle.tell():
                break
            if kind in PNG_TEXT_CHUNKS and length <= MAX_TEXT_BYTES:
                data = handle.read(length)
                handle.seek(4, 1)
                try:
                    key, value = _png_text_value(kind, data)
                except (ValueError, zlib.error, IndexError):
                    continue
                fields.setdefault(key, value)
            else:
                handle.seek(length + 4, 1)
    return fields


def user_comment(raw: object) -> str | None:
    """An EXIF UserComment as text: the 8-byte charset prefix (UNICODE, ASCII, or none) decides the decoding."""
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    data = bytes(raw)  # type: ignore[arg-type]
    prefix, body = data[:8], data[8:]
    if prefix.startswith(b"UNICODE"):
        # UTF-16 without a BOM; the byte order differs between writers, so read it from where the zeros are.
        big_endian = body[:1] == b"\x00"
        return body.decode("utf-16-be" if big_endian else "utf-16-le", "replace").rstrip("\x00")
    if prefix.startswith(b"ASCII"):
        return body.decode("ascii", "replace").rstrip("\x00")
    return data.decode("utf-8", "replace").strip("\x00")


def exif_text(path: Path | str) -> dict[str, str]:
    """JPEG/WebP text metadata through Pillow's header reader (no pixel is loaded)."""
    try:
        from PIL import Image
    except ImportError as exc:  # a declared dependency; say so plainly if the install is broken
        raise RuntimeError("reading JPEG/WebP metadata needs Pillow: reinstall with pip install -e .") from exc
    fields: dict[str, str] = {}
    with Image.open(path) as image:
        exif = image.getexif()
        for tag, name in ((0x010E, "ImageDescription"), (0x9C9C, "XPComment")):
            value = exif.get(tag)
            if value:
                fields[name] = (value.decode("utf-16-le", "replace").rstrip("\x00") if isinstance(value, bytes)
                                else str(value))
        try:
            comment = user_comment(exif.get_ifd(0x8769).get(0x9286))
        except (KeyError, AttributeError, TypeError):
            comment = None
        if comment:
            fields["UserComment"] = comment
        for key in ("xmp", "XML:com.adobe.xmp", "comment"):
            value = image.info.get(key)
            if value:
                fields[key] = value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
    return fields


def media_kind(path: Path | str) -> str:
    """`png`, `jpeg`, `webp`, `video` (an MP4-family or Matroska/WebM container) or `unknown`, from 12 bytes."""
    with Path(path).open("rb") as handle:
        head = handle.read(12)
    kind = sniff(head)
    if kind:
        return kind
    if head[4:8] == b"ftyp" or head.startswith(b"\x1a\x45\xdf\xa3"):
        return "video"
    return "unknown"


def read_fields(path: Path | str) -> dict[str, str]:
    """The text metadata of an image: PNG text chunks, or JPEG/WebP EXIF/XMP/comment; {} for anything else."""
    kind = media_kind(path)
    if kind == "png":
        return png_text(path)
    if kind in ("jpeg", "webp"):
        return exif_text(path)
    return {}


# --- images: understanding the fields -------------------------------------------------------------------------------

def _json_or_none(text: object) -> object:
    if not isinstance(text, str) or text.lstrip()[:1] not in ("{", "["):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _is_api_graph(graph: object) -> bool:
    return isinstance(graph, dict) and any(isinstance(n, dict) and "class_type" in n for n in graph.values())


def _is_ui_graph(graph: object) -> bool:
    return isinstance(graph, dict) and isinstance(graph.get("nodes"), list)


def comfy_graphs(fields: dict[str, str]) -> tuple[dict | None, dict | None]:
    """(API graph, editor workflow) from the `prompt` and `workflow` keys, or from one JSON value holding both."""
    api, ui = _json_or_none(fields.get("prompt")), _json_or_none(fields.get("workflow"))
    for value in fields.values():  # some writers nest both in one JSON comment
        nested = _json_or_none(value)
        if not isinstance(nested, dict):
            continue
        if api is None and "prompt" in nested:
            api = nested["prompt"] if isinstance(nested["prompt"], dict) else _json_or_none(nested["prompt"])
        if ui is None and "workflow" in nested:
            ui = nested["workflow"] if isinstance(nested["workflow"], dict) else _json_or_none(nested["workflow"])
    return (api if _is_api_graph(api) else None), (ui if _is_ui_graph(ui) else None)  # type: ignore[return-value]


def a1111_parameters(fields: dict[str, str]) -> str | None:
    """The A1111-style parameters block, from whichever field carries it."""
    for key in ("parameters", "UserComment", "ImageDescription", "Comment", "comment", "XPComment"):
        value = fields.get(key)
        if isinstance(value, str) and "Steps:" in value and ("Sampler:" in value or "CFG scale:" in value):
            return value
    return None


def _to_int(value: object) -> int | None:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _to_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _settings_line(line: str) -> dict[str, str]:
    """`Steps: 20, Sampler: Euler a, Lora hashes: "a: 1, b: 2"` -> {name: value}; a quoted value may hold commas."""
    out: dict[str, str] = {}
    pos, end = 0, len(line)
    while pos < end:
        colon = line.find(":", pos)
        if colon < 0:
            break
        name = line[pos:colon].strip(" ,")
        pos = colon + 1
        while pos < end and line[pos] == " ":
            pos += 1
        if pos < end and line[pos] == '"':
            close = pos + 1
            while close < end and line[close] != '"':
                close += 2 if line[close] == "\\" else 1
            value = line[pos + 1:close].replace('\\"', '"')
            comma = line.find(",", close)
        else:
            comma = line.find(",", pos)
            value = line[pos:comma if comma >= 0 else end].strip()
        if name:
            out[name] = value
        pos = end if comma < 0 else comma + 1
    return out


def parse_a1111(text: str) -> dict:
    """An A1111-style block -> the recipe fields: prompt, negative, the settings line, LoRAs with weights."""
    lines = text.strip().splitlines()
    starts = [i for i, line in enumerate(lines) if line.lstrip().startswith("Steps:")]
    starts = starts or [i for i, line in enumerate(lines) if "Steps:" in line]
    at = starts[-1] if starts else len(lines)  # no settings line: all of it is the prompt
    above = lines[:at]
    neg_at = next((i for i, line in enumerate(above) if line.startswith("Negative prompt:")), None)
    prompt_lines = above if neg_at is None else above[:neg_at]
    negative = "" if neg_at is None else "\n".join(above[neg_at:])[len("Negative prompt:"):].strip()
    settings = _settings_line(" ".join(lines[at:]))
    width = height = None
    if "x" in settings.get("Size", ""):
        width, height = (_to_int(p) for p in settings["Size"].split("x", 1))
    prompt = "\n".join(prompt_lines).strip()
    loras = []
    for pair in settings.get("Lora hashes", "").split(","):
        name, sep, short_hash = pair.partition(":")
        if not sep or not name.strip():
            continue
        name = name.strip()
        weight = None
        marker = f"<lora:{name}:"
        if marker in prompt:
            weight = _to_float(prompt.split(marker, 1)[1].split(">", 1)[0])
        loras.append({"name": name, "hash": short_hash.strip() or None, "weight": weight})
    return {
        "format": "a1111", "prompt": prompt, "negative": negative,
        "seed": _to_int(settings.get("Seed")), "steps": _to_int(settings.get("Steps")),
        "cfg": _to_float(settings.get("CFG scale")), "sampler": settings.get("Sampler"),
        "scheduler": settings.get("Schedule type"), "denoise": _to_float(settings.get("Denoising strength")),
        "clip_skip": _to_int(settings.get("Clip skip")), "width": width, "height": height,
        "model": settings.get("Model"), "model_hash": settings.get("Model hash"), "loras": loras,
    }


_SAMPLER_INPUTS = (("seed", "seed"), ("seed", "noise_seed"), ("steps", "steps"), ("cfg", "cfg"),
                   ("sampler", "sampler_name"), ("scheduler", "scheduler"), ("denoise", "denoise"))


def parse_comfy(api: dict) -> dict:
    """A ComfyUI API graph -> the recipe fields: the first sampler's settings, its prompts, model, LoRAs, size."""
    nodes = {str(k): v for k, v in api.items() if isinstance(v, dict)}
    out: dict = {"format": "comfy", "prompt": "", "negative": "", "seed": None, "steps": None, "cfg": None,
                 "sampler": None, "scheduler": None, "denoise": None, "clip_skip": None, "width": None,
                 "height": None, "model": None, "model_hash": None, "loras": []}

    def linked_text(link: object) -> str:
        if not (isinstance(link, list) and link):
            return ""
        inputs = (nodes.get(str(link[0])) or {}).get("inputs") or {}
        text = inputs.get("text", inputs.get("text_g"))
        return text if isinstance(text, str) else ""

    samplers = [n for n in nodes.values() if str(n.get("class_type", "")).startswith("KSampler")
                or n.get("class_type") in ("SamplerCustom", "SamplerCustomAdvanced")]
    if samplers:
        inputs = samplers[0].get("inputs") or {}
        for field, name in _SAMPLER_INPUTS:
            value = inputs.get(name)
            if out[field] is None and value is not None and not isinstance(value, list):
                out[field] = value
        out["prompt"], out["negative"] = linked_text(inputs.get("positive")), linked_text(inputs.get("negative"))
    for node in nodes.values():
        kind, inputs = str(node.get("class_type", "")), node.get("inputs") or {}
        if kind.startswith("CheckpointLoader") and isinstance(inputs.get("ckpt_name"), str):
            out["model"] = out["model"] or inputs["ckpt_name"]
        elif kind in ("UNETLoader", "UnetLoaderGGUF") and isinstance(inputs.get("unet_name"), str):
            out["model"] = out["model"] or inputs["unet_name"]
        elif "Lora" in kind and isinstance(inputs.get("lora_name"), str):
            weight = inputs.get("strength_model", inputs.get("strength"))
            out["loras"].append({"name": inputs["lora_name"], "hash": None,
                                 "weight": float(weight) if isinstance(weight, int | float) else None})
        elif kind == "CLIPSetLastLayer" and isinstance(inputs.get("stop_at_clip_layer"), int):
            out["clip_skip"] = out["clip_skip"] or abs(inputs["stop_at_clip_layer"])
        elif kind in ("EmptyLatentImage", "EmptySD3LatentImage") and isinstance(inputs.get("width"), int):
            out["width"] = out["width"] or inputs["width"]
            out["height"] = out["height"] or inputs.get("height")
    return out


def recipe(fields: dict[str, str]) -> dict:
    """The recipe an image's fields describe: ComfyUI first, then A1111, else `{'format': 'none'}`."""
    api, _ui = comfy_graphs(fields)
    if api is not None:
        return parse_comfy(api)
    block = a1111_parameters(fields)
    if block is not None:
        return parse_a1111(block)
    return {"format": "none"}


def probe_keys(path: Path | str) -> dict:
    """What a file carries, safe to print: the media kind, the field names, and whether a recipe is there."""
    kind = media_kind(path)
    fields = read_fields(path)
    api, ui = comfy_graphs(fields)
    return {"kind": kind, "keys": sorted(fields), "comfy_api_graph": api is not None,
            "comfy_workflow": ui is not None, "a1111": a1111_parameters(fields) is not None}
