"""What a model file is, read from its folder and its header, never from its weights.

Safety: a safetensors file is read through its JSON header only (`_common.embedded_meta.safetensors_header`);
pickle formats (.ckpt .pt .pth .bin) are never opened beyond hashing, because loading one runs code. Every
value comes back as a `Finding` with the reason it was chosen, so `easel registry explain` can say why a file
is classed as this type, family or rating.

The order of evidence (ported from the scanner the file registry replaces, and its lessons):
- type: the folder under `<models_root>` (data/folders.yaml) wins; the header refines a LoRA into a LyCORIS and
  types a file outside every known folder; a file neither can type is reported, never guessed;
- family: an architecture named in the header, then an SDXL derivative named in the metadata or the file name
  (their files still report the SDXL architecture, so this check comes before the generic one), then the
  architecture or kohya's base-model field, then tensor-name structure, then the file name (weakest);
- self-trained: never inferred. Training metadata (`ss_*` keys) is in most downloaded LoRAs too, so it proves
  nothing; only the person says so, in `registry/overrides.yaml`;
- rating: default-deny. Nothing known = `unknown`, which counts as not sfw; the generic cue words of
  data/content_markers.yaml move it, a non-sfw cue beats an sfw one, and a cue never overrides a rating that
  came from the source or the person (a disagreement is a RATING CONFLICT for the person to settle).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .shipped import DATA, DataError, load_yaml, schema_errors

SAFETENSORS_SUFFIXES = frozenset({".safetensors", ".sft"})
PICKLE_SUFFIXES = frozenset({".ckpt", ".pt", ".pth", ".bin"})
GGUF_SUFFIXES = frozenset({".gguf"})
MODEL_SUFFIXES = SAFETENSORS_SUFFIXES | PICKLE_SUFFIXES | GGUF_SUFFIXES
MARKER_KEYS = ("sfw_cues", "non_sfw_cues", "age_down_and_real_person_cues")
DTYPE_PRECISION = {"F16": "fp16", "BF16": "bf16", "F32": "fp32", "F64": "fp64", "F8_E4M3": "fp8", "F8_E5M2": "fp8"}


class HeaderUnavailable(Exception):
    """The header reader (`_common.embedded_meta`) is not installed in this build."""


@dataclass(frozen=True)
class Finding:
    """A value and the plain-words reason it was chosen."""

    value: object
    reason: str


@dataclass(frozen=True)
class Header:
    """A safetensors header split into its tensor table and its `__metadata__` strings."""

    tensors: Mapping[str, Mapping]
    metadata: Mapping[str, str]

    def meta(self, key: str) -> str:
        """A metadata value by case-insensitive key, or ""."""
        low = key.lower()
        for k, v in self.metadata.items():
            if k.lower() == low:
                return v
        return ""


# --- reading a header ---------------------------------------------------------------------------------------------

def split_header(raw: Mapping) -> Header:
    """Accept the raw header dict (tensor name -> info, plus `__metadata__`) or a {tensors, metadata} pair."""
    if isinstance(raw.get("tensors"), Mapping) and "__metadata__" not in raw:
        tensors, meta = raw["tensors"], raw.get("metadata") or {}
    else:
        meta = raw.get("__metadata__") or {}
        tensors = {k: v for k, v in raw.items() if k != "__metadata__" and isinstance(v, Mapping)}
    if not isinstance(meta, Mapping):
        meta = {}
    return Header(tensors=dict(tensors), metadata={str(k): "" if v is None else str(v) for k, v in meta.items()})


def read_header(path: Path) -> Header:
    """The header of a safetensors file. ValueError when it is malformed, HeaderUnavailable without the reader."""
    try:
        from easel._common.embedded_meta import safetensors_header
    except ImportError as exc:  # T1b builds the reader; until then a scan cannot read headers
        raise HeaderUnavailable("the header reader easel._common.embedded_meta.safetensors_header is not "
                                "installed yet") from exc
    raw = safetensors_header(Path(path))
    if not isinstance(raw, Mapping):
        raise ValueError("the header is not a JSON object")
    return split_header(raw)


# --- folders -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Folders:
    """data/folders.yaml: ComfyUI folder name -> folders under models_root, and resource type -> ComfyUI folder."""

    comfyui_folders: Mapping[str, tuple[str, ...]]
    type_folders: Mapping[str, str]
    source: str

    def type_for_comfy_folder(self, comfy: str) -> str | None:
        """The first resource type (in file order) that loads from ComfyUI folder `comfy`."""
        return next((t for t, f in self.type_folders.items() if f == comfy), None)

    def type_of(self, rel: str) -> Finding | None:
        """The type of a file at `rel` (relative to models_root): the longest listed folder that starts the path."""
        low = rel.lower()
        best: tuple[str, str] | None = None
        for comfy, dirs in self.comfyui_folders.items():
            rtype = self.type_for_comfy_folder(comfy)
            if rtype is None:
                continue
            for folder in dirs:
                folder = folder.strip("/")
                if low.startswith(folder.lower() + "/") and (best is None or len(folder) > len(best[1])):
                    best = (rtype, folder)
        if best is None:
            return None
        return Finding(best[0], f"folder {best[1]}/ ({self.source}: {best[0]} loads from {best[1]}/)")

    def folder_for(self, rtype: str) -> str | None:
        """The first folder under models_root a resource of `rtype` belongs in."""
        comfy = self.type_folders.get(rtype)
        dirs = self.comfyui_folders.get(comfy or "", ())
        return dirs[0] if dirs else None


def load_folders(home: Path | None = None) -> Folders:
    """`<home>/data/folders.yaml` when the instance has one (it replaces the shipped file), else data/folders.yaml."""
    path = home / "data" / "folders.yaml" if home is not None else None
    source = "<home>/data/folders.yaml"
    if path is None or not path.is_file():
        path, source = DATA / "folders.yaml", "data/folders.yaml"
    data = load_yaml(path, source)
    problems = schema_errors("folders", data)
    if problems or not isinstance(data, dict):
        raise DataError([f"{source} is not a valid easel/folders/1 record:"] + [f"  {p}" for p in problems])
    comfy = {str(k): tuple(str(d) for d in v) for k, v in (data.get("comfyui_folders") or {}).items()}
    return Folders(comfyui_folders=comfy, type_folders=dict(data.get("type_folders") or {}), source=source)


# --- words and cue lists ---------------------------------------------------------------------------------------------

_CAMEL_1 = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_CAMEL_2 = re.compile(r"(?<=[A-Z])(?=[A-Z][a-z])")
_NOT_WORD = re.compile(r"[^\w+]|_")


def words(text: str) -> tuple[str, ...]:
    """`text` as lower-case words: NFKC, camelCase split, and every separator (space _ - . / ...) a break."""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CAMEL_2.sub(" ", _CAMEL_1.sub(" ", text))
    return tuple(_NOT_WORD.sub(" ", text.casefold()).split())


def _contains(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i:i + n] == needle for i in range(len(haystack) - n + 1))


@dataclass(frozen=True)
class Markers:
    """The three generic cue lists of data/content_markers.yaml, each term pre-split into words."""

    sfw: tuple[tuple[str, ...], ...]
    non_sfw: tuple[tuple[str, ...], ...]
    subjects: tuple[tuple[str, ...], ...]
    source: str

    @staticmethod
    def hits(terms: Iterable[tuple[str, ...]], texts: Iterable[str]) -> list[str]:
        """The terms (joined by spaces) that occur as whole words in any of `texts`."""
        streams = [words(t) for t in texts if t]
        return [" ".join(term) for term in terms if any(_contains(s, term) for s in streams)]

    def subject_hits(self, text: str) -> list[str]:
        """Third-list terms in `text` (a prompt): age-down and real-person cues, refused under policy.subjects."""
        return self.hits(self.subjects, [text])


def load_markers(home: Path | None = None) -> Markers:
    """`<home>/data/content_markers.yaml` when present (it replaces the shipped file), else the shipped one."""
    path = home / "data" / "content_markers.yaml" if home is not None else None
    source = "<home>/data/content_markers.yaml"
    if path is None or not path.is_file():
        path, source = DATA / "content_markers.yaml", "data/content_markers.yaml"
    data = load_yaml(path, source)
    if not isinstance(data, dict) or any(not isinstance(data.get(k), list) for k in MARKER_KEYS):
        raise DataError([f"{source} needs three lists: {', '.join(MARKER_KEYS)}"])
    lists = [tuple(w for w in (words(str(t)) for t in data[k]) if w) for k in MARKER_KEYS]
    return Markers(sfw=lists[0], non_sfw=lists[1], subjects=lists[2], source=source)


def rating_from_markers(texts: Iterable[str], markers: Markers) -> tuple[dict, Finding]:
    """The `semantics.content` block a scan gives a new file, and why. Default-deny: no cue = unknown."""
    texts = list(texts)
    bad, good = markers.hits(markers.non_sfw, texts), markers.hits(markers.sfw, texts)
    if bad:
        return ({"rating": "unknown", "class": "nsfw", "basis": "markers"},
                Finding("nsfw", f"non-sfw cue {', '.join(bad)} ({markers.source})"))
    if good:
        return ({"rating": "sfw", "class": "sfw", "basis": "markers"},
                Finding("sfw", f"sfw cue {', '.join(good)} ({markers.source})"))
    return ({"rating": "unknown", "class": "unknown", "basis": "default"},
            Finding("unknown", "no source rating and no cue word: unknown, which counts as not sfw"))


# --- type and family from the header ---------------------------------------------------------------------------------

def _any(names: str, *needles: str) -> str | None:
    return next((n for n in needles if n in names), None)


def type_from_header(header: Header) -> Finding | None:
    """The resource type the tensor names show, or None. LoRA-family checks come first: their names embed UNet ones."""
    names = "\n".join(header.tensors).lower()
    module = header.meta("ss_network_module").lower()
    if hit := _any(names, ".hada_w1", ".lokr_w1", "hada_w1_a", "lokr_w1"):
        return Finding("lycoris", f"LyCORIS tensors ({hit.strip('.')})")
    if "lycoris" in module:
        return Finding("lycoris", f"ss_network_module {module}")
    if hit := _any(names, "lora_down.", "lora_up.", "lora_a.", "lora_b.", "lora_unet_", "lora_te"):
        return Finding("lora", f"LoRA tensors ({hit.strip('.')})")
    if "model.diffusion_model." in names and (hit := _any(names, "first_stage_model.", "conditioner.",
                                                            "cond_stage_model.")):
        return Finding("checkpoint", f"a diffusion model with {hit.strip('.')} in one file")
    if hit := _any(names, "control_model.", "input_hint_block", "controlnet_cond_embedding", "controlnet_"):
        return Finding("controlnet", f"ControlNet tensors ({hit.strip('.')})")
    if hit := _any(names, "image_proj.", "ip_adapter."):
        return Finding("ipadapter", f"IP-Adapter tensors ({hit.strip('.')})")
    if "vision_model." in names and "text_model." not in names:
        return Finding("clip_vision", "vision_model tensors only")
    if hit := _any(names, "model.diffusion_model.", "double_blocks.", "single_blocks.", "joint_blocks."):
        return Finding("unet", f"a diffusion model alone ({hit.strip('.')})")
    if hit := _any(names, "encoder.conv_in", "decoder.conv_in"):
        return Finding("vae", f"autoencoder tensors ({hit})")
    if hit := _any(names, "text_model.encoder.", "encoder.block.", "text_projection"):
        return Finding("text_encoder", f"text encoder tensors ({hit.strip('.')})")
    if hit := _any(names, "conv_first", ".rdb1.", "rrdb"):
        return Finding("upscaler", f"upscaler tensors ({hit.strip('.')})")
    if hit := _any(names, "emb_params", "string_to_param"):
        return Finding("embedding", f"embedding tensors ({hit})")
    if header.tensors and set(n.lower() for n in header.tensors) <= {"clip_l", "clip_g"}:
        return Finding("embedding", "clip_l/clip_g vectors only (an SDXL embedding)")
    return None


def _named(needle: str, *texts: str) -> bool:
    """`needle` in a text as a word part not glued to other letters (plain and camelCase-split forms)."""
    pattern = re.compile(rf"(?<![a-z]){re.escape(needle)}(?![a-z])")
    for text in texts:
        if not text:
            continue
        for form in (text.lower(), " ".join(_CAMEL_2.sub(" ", _CAMEL_1.sub(" ", text)).lower().split())):
            if pattern.search(form):
                return True
    return False


_ARCH_FAMILIES = (("flux", "flux1"), ("stable-diffusion-v3", "sd35"), ("sd3", "sd35"), ("hunyuan", "hunyuan_video"),
                  ("ltx", "ltxv"), ("qwen", "qwen_image"))
_DERIVATIVES = (("noobai", "noobai"), ("illustrious", "illustrious"), ("pony", "pony"))
_KOHYA_BASE = (("sdxl", "sdxl"), ("sd_v1", "sd15"), ("sd_v2", "sd21"), ("flux", "flux1"), ("sd3", "sd35"))
_FILE_NAME_FAMILIES = (("illustrious", "illustrious"), ("noobai", "noobai"), ("pony", "pony"), ("sd15", "sd15"),
                       ("sd1.5", "sd15"), ("sdxl", "sdxl"), ("xl", "sdxl"), ("flux", "flux1"), ("wan2.2", "wan22"),
                       ("wan", "wan21"), ("qwen", "qwen_image"))


def family_from_header(header: Header | None, file_name: str) -> Finding | None:
    """The model family, from the strongest evidence available; None when nothing names one."""
    if header is not None:
        arch = header.meta("modelspec.architecture").lower()
        for needle, fam in _ARCH_FAMILIES:
            if needle in arch:
                return Finding(fam, f"modelspec.architecture {arch}")
        if "wan" in arch:
            fam = "wan22" if ("2.2" in arch or "2_2" in arch) else "wan21"
            return Finding(fam, f"modelspec.architecture {arch}")
        base = header.meta("ss_base_model_version")
        hints = [base, header.meta("ss_sd_model_name"), header.meta("modelspec.title"),
                 header.meta("modelspec.description"), file_name]
        for needle, fam in _DERIVATIVES:
            if _named(needle, *hints):
                return Finding(fam, f"'{needle}' in the metadata or file name (an SDXL derivative)")
        if "stable-diffusion-xl" in arch or "sdxl" in arch:
            return Finding("sdxl", f"modelspec.architecture {arch}")
        if "stable-diffusion-v1" in arch:
            return Finding("sd15", f"modelspec.architecture {arch}")
        if "stable-diffusion-v2" in arch:
            return Finding("sd21", f"modelspec.architecture {arch}")
        for prefix, fam in _KOHYA_BASE:
            if base.lower().startswith(prefix):
                return Finding(fam, f"ss_base_model_version {base}")
        names = "\n".join(header.tensors).lower()
        if ("double_blocks" in names or "single_transformer_blocks" in names) and "t5" in names:
            return Finding("flux1", "double/single transformer blocks with a T5 encoder")
        if hit := _any(names, "lora_te2_", "text_encoder_2", "conditioner.embedders.1"):
            return Finding("sdxl", f"a second text encoder ({hit.strip('.')})")
        if "lora_te1_" in names or "lora_te_" in names:
            return Finding("sd15", "a single text encoder (lora_te1_/lora_te_ only)")
    for needle, fam in _FILE_NAME_FAMILIES:
        if _named(needle, file_name):
            return Finding(fam, f"'{needle}' in the file name (weakest evidence)")
    return None


# --- integrity and LoRA facts ------------------------------------------------------------------------------------------

def integrity_facts(header: Header) -> dict:
    """dtypes, tensor_count, param_count and precision (the float dtype holding the most parameters)."""
    by_params: Counter[str] = Counter()
    params = 0
    for info in header.tensors.values():
        dtype = str(info.get("dtype", "?"))
        n = 1
        for dim in info.get("shape") or []:
            try:
                n *= int(dim)
            except (TypeError, ValueError):
                n = 0
        by_params[dtype] += n
        params += n
    floats = [d for d, _ in sorted(by_params.items(), key=lambda kv: (-kv[1], kv[0])) if d in DTYPE_PRECISION]
    return {"dtypes": sorted(by_params), "tensor_count": len(header.tensors), "param_count": params,
            "precision": DTYPE_PRECISION[floats[0]] if floats else None}


def quantization(path: Path) -> str | None:
    """`none` for safetensors (or nf4 when the name says so), the GGUF quant from the name, None for pickles."""
    suffix, name = path.suffix.lower(), path.name
    if suffix in GGUF_SUFFIXES:
        match = re.search(r"(?i)(?<![a-z0-9])(i?q\d[_a-z0-9]*)", Path(name).stem)
        return f"GGUF-{match.group(1).upper()}" if match else "GGUF"
    if suffix in SAFETENSORS_SUFFIXES:
        return "nf4" if "nf4" in words(Path(name).stem) else "none"
    return None


def _int(text: str) -> int | None:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return None


def _float(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def lora_facts(header: Header | None, family: str | None) -> dict:
    """The `lora` block of a new LoRA/LyCORIS record: kohya's metadata, nothing guessed (no trigger words)."""
    module = header.meta("ss_network_module") if header else ""
    low = module.lower()
    names = " ".join(header.tensors).lower() if header else ""
    args = (header.meta("ss_network_args") if header else "").lower()
    if "hada_w" in names or ("lycoris" in low and "loha" in (low + args)):
        kind = "loha"
    elif "lokr_w" in names or ("lycoris" in low and "lokr" in (low + args)):
        kind = "lokr"
    elif "dora" in (low + args) or "dora_scale" in names:
        kind = "dora"
    elif "lycoris" in low:
        kind = None  # a LyCORIS algorithm the header does not name
    else:
        kind = "standard" if low or "lora_down" in names or "lora_a." in names else None
    dim = _int(header.meta("ss_network_dim")) if header else None
    clip_skip = _int(header.meta("ss_clip_skip")) if header else None
    return {"network_dim": dim if dim and dim > 0 else None,
            "network_alpha": _float(header.meta("ss_network_alpha")) if header else None,
            "network_module": module or None, "lora_type": kind, "trained_on_family": family,
            "trigger_words": [], "recommended_weight": None,
            "clip_skip": clip_skip if clip_skip and clip_skip > 0 else None}


def title_from_header(header: Header | None) -> str | None:
    """The model's own title: modelspec.title, else kohya's output name."""
    if header is None:
        return None
    return header.meta("modelspec.title") or header.meta("ss_output_name") or None


def marker_texts(file_name: str, header: Header | None, record: Mapping | None = None) -> list[str]:
    """The texts the cue words are looked for in: file name, the header's title fields, a record's title and tags."""
    texts = [Path(file_name).stem]
    if header is not None:
        texts += [header.meta("modelspec.title"), header.meta("ss_output_name"), header.meta("modelspec.tags")]
    if record:
        identity, semantics = record.get("identity") or {}, record.get("semantics") or {}
        texts += [identity.get("title") or "", *(semantics.get("tags") or [])]
    return [t for t in texts if t]


# --- one file ---------------------------------------------------------------------------------------------------------

@dataclass
class Classified:
    """Everything the folder and the header say about one file, each with its reason."""

    rel: str
    type: Finding | None
    family: Finding | None
    header: Header | None
    header_note: str
    content: dict = field(default_factory=dict)
    rating: Finding | None = None
    folder_type: Finding | None = None
    header_type: Finding | None = None


def classify(path: Path, rel: str, folders: Folders, markers: Markers, header: Header | None = None,
             header_note: str = "") -> Classified:
    """Classify one file: the type (folder, refined by the header), family, and the cue-word rating."""
    folder_type = folders.type_of(rel)
    header_type = type_from_header(header) if header is not None else None
    chosen = folder_type
    if folder_type is not None and folder_type.value == "lora" and header_type is not None \
            and header_type.value == "lycoris":
        chosen = Finding("lycoris", f"{folder_type.reason}; the header shows {header_type.reason}")
    elif folder_type is None and header_type is not None:
        chosen = Finding(header_type.value, f"the header: {header_type.reason} (not in a known folder)")
    family = family_from_header(header, Path(rel).name)
    content, rating = rating_from_markers(marker_texts(Path(rel).name, header), markers)
    return Classified(rel=rel, type=chosen, family=family, header=header, header_note=header_note, content=content,
                      rating=rating, folder_type=folder_type, header_type=header_type)
