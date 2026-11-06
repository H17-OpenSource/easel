"""Compile a resolved recipe into a ComfyUI API-format graph: {node_id: {class_type, inputs, _meta}}.

Why compile instead of converting saved workflows: what the browser editor saves is UI format, and turning it
into the API format `POST /prompt` takes needs the server's node definitions and breaks on custom nodes. The
common cases are built here from ComfyUI core nodes, which every server has; anything else is a workflow
template (`workflows/<id>/graph.json` + `slots.json`, filled by `fill_template`).

What it builds (one graph per prompt; the runner queues `count` prompts with seeds seed, seed+1, ...):
- txt2img and img2img (`task`), with a LoRA chain, clip skip, an optional VAE and embeddings in the prompts;
- a second pass chosen by `params.hires.mode`, which is required and never guessed: `none`, `latent` (upscale
  the latent, sample again) or `model` (decode, upscale model, encode, sample again);
- a standalone upscale (`task: upscale`, or `compile_upscale`);
- the face pass of ComfyUI-Impact-Pack (GPL-3.0; its detector node comes from ComfyUI-Impact-Subpack, AGPL-3.0,
  which installs ultralytics). Both are separate programs driven over HTTP: the graph names their node classes,
  nothing of theirs is imported. FaceDetailer's input names are read from the server's `/object_info` when it
  is given, with the `FACE_DETAILER_REQUIRED` constant as the offline fallback (`--dry-run`);
- MOSS-Image txt2img (`moss_txt2img`), experimental: only tested on one machine and one server version.

The input is a *resolved* spec: the recipe (schemas/recipe.schema.json) with file names in place of the
resource ids, which the recipe layer looks up in the registry. Names are as the server lists them, relative to
the folder of their type ("illustrious/demo-anime-xl.safetensors"); `resolve_names` maps them onto the server's
spelling: the same path with other separators or case is rewritten; a file found only by its name in another
folder may be a different file, so it is a problem line unless the caller passes `allow_basename=True` (then it
is rewritten and reported in `notes`). Nothing is substituted silently.

    {"task": "txt2img" | "img2img" | "upscale",
     "checkpoint": "<file>", "vae": "<file>" | None, "text_encoder": "<file>" (MOSS only),
     "loras": [{"name": "<file>", "weight": 0.8, "clip_weight": None, "slot": "lora.1"}],
     "embeddings": [{"name": "<file>", "weight": 1.0, "target": "positive" | "negative"}],
     "prompt": "...", "negative": "..." | None,
     "params": {sampler, scheduler, steps, cfg, width, height, clip_skip, seed, batch, denoise,
                hires: {mode, scale, denoise, steps, upscaler: "<file>"}},
     "init_image": "<name LoadImage takes>"                      (img2img: the uploaded image)
     "image": "<name>", "upscaler": "<file>", "scale": 2.0       (upscale)
     "detailer": {DETAILER_DEFAULTS' keys} | None,               (the face pass; {} = all defaults, None = none)
     "save": ["base", "hires", "detailed"]}                      (default: the last stage only)

Defaults come from the family's `inference` block (families/<id>.yaml; `inference_defaults` reads it along the
`inherits` chain): sampler, scheduler, steps, cfg, clip skip, the native size, the hires scale/denoise/steps,
the LoRA weight and the baseline negative. A value in the spec always wins; a value neither gives is an error.

The flow a runner follows (offline steps first; the server is asked only for what the graph uses):
    graph = compile_graph(spec, inference=inf, seed=1001, filename_prefix="<tool>/run-20300101-001")  # dry-run
    info = client.object_info_for(node_classes(graph))
    graph = compile_graph(spec, inference=inf, object_info=info, seed=1001, filename_prefix=...)  # live inputs
    problems = resolve_names(graph, info) + check(graph, info)                                # [] = queue it
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

HIRES_MODES = ("none", "latent", "model")
TASKS = ("txt2img", "img2img", "upscale")
SAVE_STAGES = ("base", "hires", "detailed")
IMG2IMG_DENOISE = 0.55  # an img2img pass without params.denoise keeps the composition and changes the detail
LATENT_MULTIPLE = 8  # SD 1.5 / SDXL latents are 1/8 of the image
MOSS_MULTIPLE = 16  # the Flux2-style latent of MOSS-Image
EMBEDDING_PREFIX = "embedding:"  # ComfyUI reads embeddings from the prompt text; there is no loader node

# Node class -> its inputs that name a model file, mapped onto the server's list by resolve_names.
NAME_INPUTS: dict[str, tuple[str, ...]] = {
    "CheckpointLoaderSimple": ("ckpt_name",),
    "LoraLoader": ("lora_name",),
    "LoraLoaderModelOnly": ("lora_name",),
    "VAELoader": ("vae_name",),
    "UpscaleModelLoader": ("model_name",),
    "UNETLoader": ("unet_name",),
    "CLIPLoader": ("clip_name",),
    "ControlNetLoader": ("control_net_name",),
    "UltralyticsDetectorProvider": ("model_name",),  # ComfyUI-Impact-Subpack
    "SAMLoader": ("model_name",),  # ComfyUI-Impact-Pack
}

# The face pass. Offline fallback for FaceDetailer's input names (the server's /object_info wins when given):
# every required input must be in the graph, the SAM ones too when no SAM model is used.
FACE_DETAILER_REQUIRED: tuple[str, ...] = (
    "image", "model", "clip", "vae", "guide_size", "guide_size_for", "max_size", "seed", "steps", "cfg",
    "sampler_name", "scheduler", "positive", "negative", "denoise", "feather", "noise_mask", "force_inpaint",
    "bbox_threshold", "bbox_dilation", "bbox_crop_factor", "sam_detection_hint", "sam_dilation", "sam_threshold",
    "sam_bbox_expansion", "sam_mask_hint_threshold", "sam_mask_hint_use_negative", "drop_size", "bbox_detector",
    "wildcard", "cycle",
)
FACE_DETAILER_OPTIONAL: tuple[str, ...] = ("sam_model_opt", "inpaint_model", "noise_mask_feather")  # the ones filled
DETAILER_CLASSES = ("FaceDetailer", "UltralyticsDetectorProvider", "SAMLoader")
DETAILER_PACKS = ("it needs ComfyUI-Impact-Pack (GPL-3.0) and ComfyUI-Impact-Subpack (AGPL-3.0, which installs "
                  "ultralytics) in the server's custom nodes, at the commits in runtime/comfyui.lock.json; without "
                  "them render without the face pass")
# The face pass's settings (a recipe's `detailer` block overrides any of them). sampler, scheduler and cfg
# default to the main pass's; `detector` is a file under the detector models folder ("bbox/<file>").
DETAILER_DEFAULTS: dict[str, object] = {
    "detector": "bbox/face_yolov8m.pt", "sam": None, "sam_device": "AUTO",
    "guide_size": 1024, "guide_size_for_bbox": True, "max_size": 1024, "steps": 20, "denoise": 0.35,
    "feather": 5, "noise_mask": True, "force_inpaint": True, "bbox_threshold": 0.4, "bbox_dilation": 10,
    "bbox_crop_factor": 3.0, "drop_size": 10, "noise_mask_feather": 20, "cycle": 1, "inpaint_model": False,
    "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93, "sam_bbox_expansion": 0,
    "sam_mask_hint_threshold": 0.7, "sam_mask_hint_use_negative": "False",
    "sampler": None, "scheduler": None, "cfg": None,
}
# A FaceDetailer input whose setting has another name (every other input is named like its setting).
_SETTING_OF_INPUT = {"guide_size_for": "guide_size_for_bbox", "sampler_name": "sampler", "bbox_detector": "detector",
                     "sam_model_opt": "sam"}


class GraphError(ValueError):
    """A spec that cannot become a graph, with the reason and what to change. `lines` are ready to print."""

    def __init__(self, *lines: str) -> None:
        self.lines = list(lines)
        super().__init__("\n".join(self.lines))


class Graph:
    """An API-format graph under construction: node ids are "1", "2", ... in the order the nodes are added."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict] = {}

    def add(self, class_type: str, inputs: Mapping[str, object], title: str | None = None) -> str:
        """Add a node and return its id; `title` becomes `_meta.title`, which the server ignores and people read."""
        node_id = str(len(self.nodes) + 1)
        node: dict = {"class_type": class_type, "inputs": dict(inputs)}
        if title:
            node["_meta"] = {"title": title}
        self.nodes[node_id] = node
        return node_id


def out(node_id: str, index: int = 0) -> list:
    """A link to output `index` of a node, as the API format writes it: [node_id, index]."""
    return [node_id, index]


# --- the family's defaults ----------------------------------------------------------------------------------------

def deep_merge(parent: Mapping, child: Mapping) -> dict:
    """`parent` overlaid by `child`: nested mappings merge, every other value (null included) is replaced."""
    merged = copy.deepcopy(dict(parent))
    for key, value in child.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def inference_defaults(family: str, dirs: Sequence[Path]) -> dict:
    """A family's `inference` block after inheritance (parent first, the child's keys win; the rule of
    `families check`). `dirs` are searched in order for `<family>.yaml` (an instance's families/ before the
    repo's)."""
    import yaml  # a core dependency of both tools; imported here so the module stays cheap to import

    chain: list[dict] = []
    current: str | None = family
    while current is not None:
        if current in (c.get("id") for c in chain):
            raise GraphError(f"family {family!r}: its inherits chain has a cycle at {current!r}")
        found = next((Path(d) / f"{current}.yaml" for d in dirs if (Path(d) / f"{current}.yaml").is_file()), None)
        if found is None:
            where = ", ".join(str(d) for d in dirs) or "no folder"
            raise GraphError(f"family {current!r}: no {current}.yaml in {where}")
        data = yaml.safe_load(found.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise GraphError(f"family {current!r}: {found.name} is not a YAML mapping")
        chain.append({"id": current, "inference": data.get("inference") or {}})
        parent = data.get("inherits")
        current = str(parent) if parent else None
    merged: dict = {}
    for item in reversed(chain):
        merged = deep_merge(merged, item["inference"])
    return merged


def effective_params(spec: Mapping, inference: Mapping | None = None) -> dict:
    """The spec's params with the family's defaults filled in where the spec leaves a value unset (None)."""
    params = dict(spec.get("params") or {})
    inf = inference or {}
    for key in ("sampler", "scheduler", "steps", "cfg", "clip_skip"):
        if params.get(key) is None and inf.get(key) is not None:
            params[key] = inf[key]
    native = (inf.get("resolution") or {}).get("native") or [None, None]
    for key, value in zip(("width", "height"), native, strict=False):
        if params.get(key) is None and value is not None:
            params[key] = value
    hires = params.get("hires")
    if not isinstance(hires, Mapping) or hires.get("mode") not in HIRES_MODES:
        got = hires.get("mode") if isinstance(hires, Mapping) else hires
        raise GraphError(f"params.hires.mode is required: one of {', '.join(HIRES_MODES)} (got {got!r})",
                         "fix: state it in the recipe; the compiler never guesses whether a second pass runs")
    hires = dict(hires)
    if hires["mode"] != "none":
        for key, value in (inf.get("hires") or {}).items():
            if key in ("scale", "denoise", "steps") and hires.get(key) is None:
                hires[key] = value
    params["hires"] = hires
    return params


# --- small checks ---------------------------------------------------------------------------------------------------

def _need(params: Mapping, key: str, kind: type, where: str = "params") -> object:
    value = params.get(key)
    if value is None:
        raise GraphError(f"{where}.{key} is not set and the family's inference block has no default",
                         f"fix: set {where}.{key} in the recipe")
    try:
        return kind(value)
    except (TypeError, ValueError):
        raise GraphError(f"{where}.{key} must be a {kind.__name__}, not {value!r}") from None


def _size(params: Mapping, multiple: int) -> tuple[int, int]:
    width, height = int(_need(params, "width", int)), int(_need(params, "height", int))
    for key, value in (("width", width), ("height", height)):
        if value < 64 or value % multiple:
            raise GraphError(f"params.{key} {value} must be at least 64 and a multiple of {multiple}",
                             "fix: pick a size from the family's resolution presets")
    return width, height


def _seed(params: Mapping, seed: int | None) -> int:
    value = params.get("seed") if seed is None else seed
    if value is None:
        raise GraphError("no seed: params.seed is null and no seed= was passed",
                         "fix: the runner picks the seed and passes it (seed, seed+1, ... for count > 1)")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GraphError(f"seed {value!r} must be a whole number, zero or more")
    return value


def hires_size(width: int, height: int, scale: float, multiple: int = LATENT_MULTIPLE) -> tuple[int, int]:
    """The model hires pass's exact size: both sides times `scale`, rounded down to `multiple`.
    832x1216 at 1.5 -> 1248x1824."""
    m = int(multiple)
    return int(round(width * scale, 6)) // m * m, int(round(height * scale, 6)) // m * m


def upscale_factor(model_name: str, default: int = 4) -> int:
    """An upscale model's native factor from its file name: "4x-demo.pth" -> 4, "demo_x2.pth" -> 2; `default`
    when the name does not say (4 is the common case; pass the real factor when it differs)."""
    stem = Path(str(model_name).replace("\\", "/")).name.lower()
    m = re.search(r"(?<![0-9a-z])([1-8])x(?![0-9])|(?<![0-9a-z])x([1-8])(?![0-9])", stem)
    return int(m.group(1) or m.group(2)) if m else int(default)


def detector_name(name: str) -> str:
    """The detector node lists "bbox/<file>" and "segm/<file>"; a bare file name means the bbox model."""
    text = str(name).replace("\\", "/")
    return text if text.startswith(("bbox/", "segm/")) else f"bbox/{text}"


def embedding_token(name: str, weight: float | None = 1.0) -> str:
    """The prompt text that loads an embedding: `embedding:<name>`, weighted as `(embedding:<name>:1.2)`.
    ComfyUI splits the prompt on spaces and reads weights from parentheses, so a name with whitespace,
    parentheses or a comma cannot be written into a prompt."""
    text = str(name).replace("\\", "/").strip()
    if not text or re.search(r"[\s(),]", text):
        raise GraphError(f"embedding name {name!r} cannot go into a prompt (empty, or it holds whitespace, "
                         "parentheses or a comma)", "fix: rename the file and rescan the registry")
    token = EMBEDDING_PREFIX + text
    w = 1.0 if weight is None else float(weight)
    return token if abs(w - 1.0) < 1e-9 else f"({token}:{w:g})"


def _join(text: str, tokens: Sequence[str]) -> str:
    return ", ".join([t for t in [text.strip(), *tokens] if t])


def prompts(spec: Mapping, inference: Mapping | None = None) -> tuple[str, str]:
    """(positive, negative) text: the spec's prompts with the embedding tokens appended; a negative of None
    becomes the family's `negative_baseline`."""
    prompt = spec.get("prompt")
    if prompt is None:
        raise GraphError("the spec has no prompt", "fix: set prompt (an empty string is allowed)")
    negative = spec.get("negative")
    if negative is None:
        negative = (inference or {}).get("negative_baseline") or ""
    tokens: dict[str, list[str]] = {"positive": [], "negative": []}
    for emb in spec.get("embeddings") or []:
        target = emb.get("target") or "positive"
        if target not in tokens:
            raise GraphError(f"embedding {emb.get('name')!r}: target must be positive or negative, not {target!r}")
        tokens[target].append(embedding_token(emb.get("name", ""), emb.get("weight")))
    return _join(str(prompt), tokens["positive"]), _join(str(negative), tokens["negative"])


def lora_chain(spec: Mapping, inference: Mapping | None = None) -> list[dict]:
    """The spec's LoRAs with weights filled: a missing weight is the family's `lora_default_weight`, a missing
    clip weight is the model weight."""
    default = (inference or {}).get("lora_default_weight")
    chain = []
    for i, lora in enumerate(spec.get("loras") or [], 1):
        name = lora.get("name")
        if not name:
            raise GraphError(f"loras[{i - 1}] has no file name")
        weight = lora.get("weight", None)
        if weight is None:
            weight = default
        if weight is None:
            raise GraphError(f"LoRA {name!r} has no weight and the family has no lora_default_weight",
                             "fix: set the LoRA's weight in the recipe")
        clip_weight = lora.get("clip_weight")
        chain.append({"name": str(name), "weight": float(weight),
                      "clip_weight": float(weight if clip_weight is None else clip_weight),
                      "slot": str(lora.get("slot") or f"lora.{i}")})
    return chain


def _stages(spec: Mapping, hires_mode: str, has_detailer: bool) -> list[str]:
    """The stages to save, in order. The default is the last stage the spec builds."""
    last = "detailed" if has_detailer else ("hires" if hires_mode != "none" else "base")
    want = list(spec.get("save") or [last])
    bad = [s for s in want if s not in SAVE_STAGES]
    if bad:
        raise GraphError(f"save: unknown stage(s) {bad}; use {', '.join(SAVE_STAGES)}")
    if "hires" in want and hires_mode == "none":
        raise GraphError("save 'hires' needs a second pass: params.hires.mode is none")
    if "detailed" in want and not has_detailer:
        raise GraphError("save 'detailed' needs the spec's detailer block")
    return [s for s in SAVE_STAGES if s in want]


# --- shared pieces -------------------------------------------------------------------------------------------------

def _ksampler(g: Graph, title: str, *, model: list, positive: list, negative: list, latent: list, seed: int,
              steps: int, cfg: float, sampler: str, scheduler: str, denoise: float) -> list:
    node = g.add("KSampler", {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler,
                              "scheduler": scheduler, "denoise": denoise, "model": model, "positive": positive,
                              "negative": negative, "latent_image": latent}, title)
    return out(node)


def _decode(g: Graph, samples: list, vae: list, title: str) -> list:
    return out(g.add("VAEDecode", {"samples": samples, "vae": vae}, title))


class _Saver:
    """SaveImage nodes per stage: one stage keeps the prefix as given, several get `<prefix>_<stage>`."""

    def __init__(self, g: Graph, prefix: str, stages: Sequence[str]) -> None:
        self.g, self.prefix, self.stages = g, prefix, list(stages)

    def __call__(self, stage: str, image: list) -> None:
        if stage not in self.stages:
            return
        prefix = self.prefix if len(self.stages) == 1 else f"{self.prefix}_{stage}"
        self.g.add("SaveImage", {"filename_prefix": prefix, "images": image}, f"save {stage}")


def _model_hires(g: Graph, image: list, hires: Mapping, size: tuple[int, int], multiple: int, *, model: list,
                 positive: list, negative: list, vae: list, seed: int, cfg: float, sampler: str,
                 scheduler: str) -> list:
    """Decoded image -> upscale model -> exact size -> encode -> sample again -> decode."""
    upscaler = hires.get("upscaler")
    if not upscaler:
        raise GraphError("params.hires.mode is model but params.hires.upscaler names no upscale model file",
                         "fix: choose an upscaler from the model registry (type upscaler) or use mode latent")
    scale = float(_need(hires, "scale", float, "params.hires"))
    loader = g.add("UpscaleModelLoader", {"model_name": str(upscaler)}, "upscale model")
    up = g.add("ImageUpscaleWithModel", {"upscale_model": out(loader), "image": image}, "model upscale")
    width, height = hires_size(size[0], size[1], scale, multiple)
    scaled = g.add("ImageScale", {"upscale_method": "lanczos", "width": width, "height": height, "crop": "disabled",
                                  "image": out(up)}, f"to x{scale:g}")
    encoded = g.add("VAEEncode", {"pixels": out(scaled), "vae": vae}, "hires encode")
    samples = _ksampler(g, "hires sampler", model=model, positive=positive, negative=negative, latent=out(encoded),
                        seed=seed, steps=int(_need(hires, "steps", int, "params.hires")), cfg=cfg, sampler=sampler,
                        scheduler=scheduler, denoise=float(_need(hires, "denoise", float, "params.hires")))
    return _decode(g, samples, vae, "hires decode")


# --- the face pass -------------------------------------------------------------------------------------------------

def _input_names(object_info: Mapping, cls: str, group: str) -> tuple[str, ...]:
    """A node class's input names in one group (required, optional), in the server's order when it gives one."""
    entry = object_info.get(cls) or {}
    order = (entry.get("input_order") or {}).get(group)
    if isinstance(order, list):
        return tuple(str(n) for n in order)
    return tuple(((entry.get("input") or {}).get(group) or {}).keys())


def detailer_inputs(object_info: Mapping | None, *, sam: bool = False) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """(required, optional, source) of FaceDetailer: from the server's /object_info when given (source
    "server"), else the constants (source "fallback"). With object_info, a missing node class is an error that
    names the node packs."""
    if object_info is None:
        return FACE_DETAILER_REQUIRED, FACE_DETAILER_OPTIONAL, "fallback"
    needed = ["FaceDetailer", "UltralyticsDetectorProvider"] + (["SAMLoader"] if sam else [])
    missing = [cls for cls in needed if cls not in object_info]
    if missing:
        raise GraphError(f"the face pass is unavailable: the server has no {', '.join(missing)} node",
                         f"fix: {DETAILER_PACKS}")
    required = _input_names(object_info, "FaceDetailer", "required")
    return required, _input_names(object_info, "FaceDetailer", "optional"), "server"


def _face_pass(g: Graph, settings: Mapping | None, image: list, *, model: list, clip: list, vae: list, positive: list,
               negative: list, params: Mapping, seed: int, object_info: Mapping | None) -> list:
    """The detector and FaceDetailer on `image`; returns the detailed image. Settings the server's FaceDetailer
    does not offer are refused when the spec set them and dropped when they are only our defaults."""
    given = {k: v for k, v in dict(settings or {}).items() if v is not None}
    unknown = sorted(set(given) - set(DETAILER_DEFAULTS))
    if unknown:
        raise GraphError(f"detailer: unknown setting(s) {unknown}", f"known: {', '.join(sorted(DETAILER_DEFAULTS))}")
    d = {**DETAILER_DEFAULTS, **given}
    required, optional, _source = detailer_inputs(object_info, sam=bool(d["sam"]))
    detector = g.add("UltralyticsDetectorProvider", {"model_name": detector_name(str(d["detector"]))}, "face detector")
    values: dict[str, object] = {
        "image": image, "model": model, "clip": clip, "vae": vae,
        "guide_size": float(d["guide_size"]), "guide_size_for": bool(d["guide_size_for_bbox"]),
        "max_size": float(d["max_size"]), "seed": seed, "steps": int(d["steps"]),
        "cfg": float(d["cfg"] if d["cfg"] is not None else params["cfg"]),
        "sampler_name": str(d["sampler"] or params["sampler"]), "scheduler": str(d["scheduler"] or params["scheduler"]),
        "positive": positive, "negative": negative, "denoise": float(d["denoise"]),
        "feather": int(d["feather"]), "noise_mask": bool(d["noise_mask"]), "force_inpaint": bool(d["force_inpaint"]),
        "bbox_threshold": float(d["bbox_threshold"]), "bbox_dilation": int(d["bbox_dilation"]),
        "bbox_crop_factor": float(d["bbox_crop_factor"]),
        "sam_detection_hint": str(d["sam_detection_hint"]), "sam_dilation": int(d["sam_dilation"]),
        "sam_threshold": float(d["sam_threshold"]), "sam_bbox_expansion": int(d["sam_bbox_expansion"]),
        "sam_mask_hint_threshold": float(d["sam_mask_hint_threshold"]),
        "sam_mask_hint_use_negative": str(d["sam_mask_hint_use_negative"]),
        "drop_size": int(d["drop_size"]), "bbox_detector": out(detector), "wildcard": "", "cycle": int(d["cycle"]),
        "inpaint_model": bool(d["inpaint_model"]), "noise_mask_feather": int(d["noise_mask_feather"]),
    }
    if d["sam"]:
        sam = g.add("SAMLoader", {"model_name": str(d["sam"]), "device_mode": str(d["sam_device"])}, "sam")
        values["sam_model_opt"] = out(sam)
    missing = [name for name in required if name not in values]
    if missing:
        raise GraphError(f"this server's FaceDetailer needs input(s) {missing} that the compiler does not fill",
                         "fix: install the node pack at the commit in runtime/comfyui.lock.json, or report it so "
                         "comfy_graph learns the new input")
    accepted = set(required) | set(optional)
    inputs = {}
    for name, value in values.items():
        if name in accepted:
            inputs[name] = value
        elif _SETTING_OF_INPUT.get(name, name) in given:
            raise GraphError(f"detailer setting {_SETTING_OF_INPUT.get(name, name)!r} has no FaceDetailer input "
                             f"{name!r} on this server", "fix: remove the setting or install the pinned node pack")
    return out(g.add("FaceDetailer", inputs, "face pass"))


# --- the compilers -------------------------------------------------------------------------------------------------

def _sd_graph(spec: Mapping, params: Mapping, inference: Mapping | None, *, prefix: str, seed: int,
              object_info: Mapping | None) -> dict:
    """SD 1.5 / SDXL family graph (sdxl_txt2img, sd15_txt2img): core nodes only, the face pass excepted."""
    task = spec.get("task") or "txt2img"
    checkpoint = spec.get("checkpoint")
    if not checkpoint:
        raise GraphError("the spec names no checkpoint file")
    g = Graph()
    ck = g.add("CheckpointLoaderSimple", {"ckpt_name": str(checkpoint)}, "checkpoint")
    model, clip, vae = out(ck, 0), out(ck, 1), out(ck, 2)
    if spec.get("vae"):
        vae = out(g.add("VAELoader", {"vae_name": str(spec["vae"])}, "vae"))
    clip_skip = int(params.get("clip_skip") or 1)
    if clip_skip > 1:
        clip = out(g.add("CLIPSetLastLayer", {"stop_at_clip_layer": -clip_skip, "clip": clip}, f"clip skip {clip_skip}"))
    for lora in lora_chain(spec, inference):
        node = g.add("LoraLoader", {"lora_name": lora["name"], "strength_model": lora["weight"],
                                    "strength_clip": lora["clip_weight"], "model": model, "clip": clip}, lora["slot"])
        model, clip = out(node, 0), out(node, 1)
    pos_text, neg_text = prompts(spec, inference)
    positive = out(g.add("CLIPTextEncode", {"text": pos_text, "clip": clip}, "positive"))
    negative = out(g.add("CLIPTextEncode", {"text": neg_text, "clip": clip}, "negative"))
    size = _size(params, LATENT_MULTIPLE)
    batch = int(params.get("batch") or 1)
    if task == "img2img":
        init = spec.get("init_image")
        if not init:
            raise GraphError("an img2img spec needs init_image (the name the server's LoadImage takes)",
                             "fix: upload the image first (Client.upload_image) and pass the returned name")
        loaded = g.add("LoadImage", {"image": str(init)}, "init image")
        fitted = g.add("ImageScale", {"upscale_method": "lanczos", "width": size[0], "height": size[1],
                                      "crop": "disabled", "image": out(loaded)}, "fit to size")
        latent = out(g.add("VAEEncode", {"pixels": out(fitted), "vae": vae}, "encode"))
        if batch > 1:
            latent = out(g.add("RepeatLatentBatch", {"samples": latent, "amount": batch}, "batch"))
        denoise = float(params["denoise"]) if params.get("denoise") is not None else IMG2IMG_DENOISE
    else:
        if params.get("denoise") is not None and abs(float(params["denoise"]) - 1.0) > 1e-9:
            raise GraphError(f"params.denoise {params['denoise']} applies to img2img; a txt2img pass starts from "
                             "noise and runs at 1.0", "fix: remove denoise or set task img2img")
        latent = out(g.add("EmptyLatentImage", {"width": size[0], "height": size[1], "batch_size": batch}, "latent"))
        denoise = 1.0
    steps, cfg = int(_need(params, "steps", int)), float(_need(params, "cfg", float))
    sampler, scheduler = str(_need(params, "sampler", str)), str(_need(params, "scheduler", str))
    common = {"model": model, "positive": positive, "negative": negative, "seed": seed, "cfg": cfg,
              "sampler": sampler, "scheduler": scheduler}
    samples = _ksampler(g, "sampler", latent=latent, steps=steps, denoise=denoise, **common)
    hires = params["hires"]
    stages = _stages(spec, hires["mode"], spec.get("detailer") is not None)
    save = _Saver(g, prefix, stages)
    last = stages[-1]
    if hires["mode"] == "none" or last == "base":
        image = _decode(g, samples, vae, "decode")
        save("base", image)
    elif hires["mode"] == "latent":
        if "base" in stages:
            save("base", _decode(g, samples, vae, "decode"))
        upscaled = g.add("LatentUpscaleBy", {"upscale_method": "nearest-exact",
                                             "scale_by": float(_need(hires, "scale", float, "params.hires")),
                                             "samples": samples}, "hires upscale")
        samples = _ksampler(g, "hires sampler", latent=out(upscaled),
                            steps=int(_need(hires, "steps", int, "params.hires")),
                            denoise=float(_need(hires, "denoise", float, "params.hires")), **common)
        image = _decode(g, samples, vae, "hires decode")
        save("hires", image)
    else:  # model
        image = _decode(g, samples, vae, "decode")
        save("base", image)
        image = _model_hires(g, image, hires, size, LATENT_MULTIPLE, vae=vae, **common)
        save("hires", image)
    if last == "detailed":
        image = _face_pass(g, spec.get("detailer"), image, model=model, clip=clip, vae=vae, positive=positive,
                           negative=negative, params={"cfg": cfg, "sampler": sampler, "scheduler": scheduler},
                           seed=seed, object_info=object_info)
        save("detailed", image)
    return g.nodes


def _moss_graph(spec: Mapping, params: Mapping, inference: Mapping | None, *, prefix: str, seed: int,
                object_info: Mapping | None) -> dict:
    """MOSS-Image txt2img (experimental; ComfyUI >= 0.43): the diffusion model through UNETLoader, model-only
    LoRAs, its own text encoder (CLIPLoader type moss_image) and VAE, a Flux2-style latent with sides that are
    multiples of 16. Its shift is built into the model, so there is no sampling node. No clip skip, no
    embeddings, no img2img, no latent hires pass (untested)."""
    if (spec.get("task") or "txt2img") != "txt2img":
        raise GraphError("the MOSS-Image compiler is txt2img only (experimental)")
    if int(params.get("clip_skip") or 1) > 1:
        raise GraphError("MOSS-Image has no clip skip: its text encoder is a language model",
                         "fix: set params.clip_skip to null")
    if spec.get("embeddings"):
        raise GraphError("MOSS-Image takes no embeddings (its text encoder is a language model)")
    hires = params["hires"]
    if hires["mode"] == "latent":
        raise GraphError("the MOSS-Image compiler supports hires mode none or model (latent is untested)")
    names = {key: spec.get(key) for key in ("checkpoint", "vae")}
    names["text_encoder"] = spec.get("text_encoder") or spec.get("clip")
    absent = [key for key, value in names.items() if not value]
    if absent:
        raise GraphError(f"a MOSS-Image spec names the diffusion model, text encoder and VAE files; missing: {absent}")
    g = Graph()
    model = out(g.add("UNETLoader", {"unet_name": str(names["checkpoint"]), "weight_dtype": "default"},
                      "diffusion model"))
    for lora in lora_chain(spec, inference):
        model = out(g.add("LoraLoaderModelOnly", {"model": model, "lora_name": lora["name"],
                                                  "strength_model": lora["weight"]}, lora["slot"]))
    encoder = out(g.add("CLIPLoader", {"clip_name": str(names["text_encoder"]), "type": "moss_image",
                                       "device": "default"}, "text encoder"))
    pos_text, neg_text = prompts(spec, inference)
    positive = out(g.add("CLIPTextEncode", {"text": pos_text, "clip": encoder}, "positive"))
    negative = out(g.add("CLIPTextEncode", {"text": neg_text, "clip": encoder}, "negative"))
    size = _size(params, MOSS_MULTIPLE)
    latent = out(g.add("EmptyFlux2LatentImage", {"width": size[0], "height": size[1],
                                                 "batch_size": int(params.get("batch") or 1)}, "latent"))
    if params.get("denoise") is not None and abs(float(params["denoise"]) - 1.0) > 1e-9:
        raise GraphError(f"params.denoise {params['denoise']} applies to img2img; txt2img runs at 1.0")
    common = {"model": model, "positive": positive, "negative": negative, "seed": seed,
              "cfg": float(_need(params, "cfg", float)), "sampler": str(_need(params, "sampler", str)),
              "scheduler": str(_need(params, "scheduler", str))}
    samples = _ksampler(g, "sampler", latent=latent, steps=int(_need(params, "steps", int)), denoise=1.0, **common)
    vae = out(g.add("VAELoader", {"vae_name": str(names["vae"])}, "vae"))
    stages = _stages(spec, hires["mode"], spec.get("detailer") is not None)
    save = _Saver(g, prefix, stages)
    image = _decode(g, samples, vae, "decode")
    save("base", image)
    if hires["mode"] == "model" and stages[-1] != "base":  # the hires size is rounded down to multiples of 16
        image = _model_hires(g, image, hires, size, MOSS_MULTIPLE, vae=vae, **common)
        save("hires", image)
    if stages[-1] == "detailed":
        image = _face_pass(g, spec.get("detailer"), image, model=model, clip=encoder, vae=vae, positive=positive,
                           negative=negative, params={"cfg": common["cfg"], "sampler": common["sampler"],
                                                      "scheduler": common["scheduler"]},
                           seed=seed, object_info=object_info)
        save("detailed", image)
    return g.nodes


COMPILERS = {"sdxl_txt2img": _sd_graph, "sd15_txt2img": _sd_graph, "moss_txt2img": _moss_graph}
EXPERIMENTAL = ("moss_txt2img",)


def compile_upscale(image: str, upscaler: str, scale: float | None = None, *, filename_prefix: str = "render",
                    factor: int | None = None) -> dict:
    """A model upscale of one image the server holds: the model's native factor, then resampled to `scale`
    overall (a 4x model at scale 2 adds an x0.5 step). `factor` overrides the factor read from the file name."""
    if not image or not upscaler:
        raise GraphError("an upscale needs the image name (as LoadImage takes it) and an upscaler file")
    native = int(factor) if factor else upscale_factor(upscaler)
    target = float(scale) if scale is not None else float(native)
    if target <= 0:
        raise GraphError(f"upscale scale {target:g} must be above 0")
    g = Graph()
    loaded = g.add("LoadImage", {"image": str(image)}, "source image")
    loader = g.add("UpscaleModelLoader", {"model_name": str(upscaler)}, "upscale model")
    result = g.add("ImageUpscaleWithModel", {"upscale_model": out(loader), "image": out(loaded)}, "upscale")
    if abs(target - native) > 1e-6:
        result = g.add("ImageScaleBy", {"upscale_method": "lanczos", "scale_by": round(target / native, 4),
                                        "image": out(result)}, f"to x{target:g}")
    g.add("SaveImage", {"filename_prefix": filename_prefix, "images": out(result)}, "save")
    return g.nodes


def compile_graph(spec: Mapping, *, inference: Mapping | None = None, object_info: Mapping | None = None,
                  seed: int | None = None, filename_prefix: str = "render") -> dict:
    """The API graph of one prompt of a resolved spec (see the module docstring for its shape).

    `inference` is the family's block (`inference_defaults`); its `graph_compiler` picks the compiler unless
    the spec names one in `compiler`. `object_info` is the server's /object_info for the graph's classes (only
    the face pass reads it; None = the offline fallback). `seed` overrides params.seed (the runner passes
    seed + i for the i-th image).
    """
    task = spec.get("task") or "txt2img"
    if task not in TASKS:
        raise GraphError(f"task {task!r} has no built-in compiler (have {', '.join(TASKS)})",
                         "fix: use a workflow template (workflows/<id>/), filled by fill_template")
    if task == "upscale":
        params = spec.get("params") or {}
        return compile_upscale(str(spec.get("image") or ""), str(spec.get("upscaler") or ""),
                               spec.get("scale", params.get("scale")), filename_prefix=filename_prefix)
    name = spec.get("compiler") or (inference or {}).get("graph_compiler")
    if not name:
        raise GraphError("no graph compiler: the family's inference block says graph_compiler: null (or no "
                         "family was given)", "fix: use a workflow template for this family")
    builder = COMPILERS.get(str(name))
    if builder is None:
        raise GraphError(f"unknown graph compiler {name!r} (have {', '.join(COMPILERS)})")
    params = effective_params(spec, inference)
    return builder(spec, params, inference, prefix=filename_prefix, seed=_seed(params, seed), object_info=object_info)


def save_nodes(graph: Mapping) -> dict[str, str]:
    """{stage: node id} of the graph's SaveImage nodes, from their titles ("save base" -> "base")."""
    found = {}
    for node_id, node in graph.items():
        if node.get("class_type") == "SaveImage":
            title = str((node.get("_meta") or {}).get("title") or "save")
            found[title.removeprefix("save").strip() or "image"] = str(node_id)
    return found


# --- templates ----------------------------------------------------------------------------------------------------

def fill_template(graph: Mapping, slots: Mapping | Sequence[Mapping], values: Mapping[str, object]) -> dict:
    """A copy of a template graph with its slots filled. `slots` is a slots.json (easel/workflow-slots/1) or
    its `slots` list: {name, node, input, required, default}. An unknown value name, a slot pointing at a
    missing node and a required slot left without value or default are errors."""
    items = slots.get("slots", []) if isinstance(slots, Mapping) else slots
    by_name = {str(s["name"]): s for s in items}
    filled = copy.deepcopy(dict(graph))
    unknown = sorted(set(values) - set(by_name))
    if unknown:
        raise GraphError(f"the template has no slot(s) {unknown} (slots: {', '.join(by_name) or 'none'})")
    for name, slot in by_name.items():
        node_id = str(slot["node"])
        if node_id not in filled:
            raise GraphError(f"slot {name!r} points at node {node_id}, which the graph does not have")
        if name in values:
            value = values[name]
        elif "default" in slot:
            value = slot["default"]
        elif slot.get("required"):
            raise GraphError(f"slot {name!r} is required and has no value and no default")
        else:
            continue
        filled[node_id].setdefault("inputs", {})[str(slot["input"])] = copy.deepcopy(value)
    return filled


# --- checking a graph against the server ---------------------------------------------------------------------------

def node_classes(graph: Mapping) -> list[str]:
    """The node classes a graph uses, sorted: what to ask /object_info for."""
    return sorted({str(node["class_type"]) for node in graph.values()})


def allowed_names(object_info: Mapping, cls: str, field: str) -> list[str]:
    """The server's choices for a combo input, in either shape it uses: the classic [[name, ...], {...}] or
    the newer ["COMBO", {"options": [name, ...]}]."""
    inputs = (object_info.get(cls) or {}).get("input") or {}
    spec = (inputs.get("required") or {}).get(field) or (inputs.get("optional") or {}).get(field)
    if not spec:
        return []
    if isinstance(spec[0], list):
        return [str(n) for n in spec[0]]
    if len(spec) > 1 and spec[0] == "COMBO" and isinstance(spec[1], Mapping):
        return [str(n) for n in spec[1].get("options") or []]
    return []


def _norm(name: str) -> str:
    return name.replace("\\", "/").lower()


MATCH_EXACT = "exact"
MATCH_SAME_PATH = "same-path"  # the same path with other separators or case
MATCH_BASENAME = "basename"  # only the file name agrees: possibly another file in another folder


def match_name_how(want: str, allowed: Iterable[str]) -> tuple[str | None, str]:
    """(the server's spelling of a model name, how it matched): exact, then the same path ignoring separator and
    case, then a unique file name in any folder; (None, "") when nothing (or more than one file) matches."""
    allowed = list(allowed)
    if want in allowed:
        return want, MATCH_EXACT
    norm = _norm(want)
    same = [a for a in allowed if _norm(a) == norm]
    if same:
        return same[0], MATCH_SAME_PATH
    base = norm.rsplit("/", 1)[-1]
    hits = [a for a in allowed if _norm(a).rsplit("/", 1)[-1] == base]
    return (hits[0], MATCH_BASENAME) if len(hits) == 1 else (None, "")


def match_name(want: str, allowed: Iterable[str]) -> str | None:
    """The server's spelling of a model name, however it matched (see `match_name_how`); None when none does.
    A caller that rewrites a graph must look at how it matched: `resolve_names` does."""
    return match_name_how(want, allowed)[0]


def resolve_names(graph: dict, object_info: Mapping, *, allow_basename: bool = False,
                  notes: list[str] | None = None) -> list[str]:
    """Rewrite every model name in `graph` to the server's spelling, in place; return one line per problem.

    An exact or same-path match is rewritten. A match by file name only is never taken silently: without
    `allow_basename` it is a problem line naming both paths and the graph keeps the wanted name; with it the name
    is rewritten and a line naming both goes into `notes` (when given) for the caller to print and record.
    """
    problems = []
    for node_id, node in graph.items():
        cls = node.get("class_type")
        for field in NAME_INPUTS.get(cls, ()):
            want = (node.get("inputs") or {}).get(field)
            if not isinstance(want, str):
                continue
            allowed = allowed_names(object_info, cls, field)
            if not allowed:
                problems.append(f"node {node_id} ({cls}): the server lists no {field} values (no such files, or "
                                "no such node)")
                continue
            got, how = match_name_how(want, allowed)
            if got is None:
                problems.append(f"node {node_id} ({cls}): {field} {want!r} is not on the server; report it to the "
                                "operator (the file is missing from the server's models folder for its type)")
            elif how == MATCH_BASENAME and not allow_basename:
                problems.append(f"node {node_id} ({cls}): {field} {want!r} is not on the server by that path; the "
                                f"server has {got!r}, which only shares the file name and may be another file; "
                                "name the file as the server lists it, or report it to the operator")
            else:
                if how == MATCH_BASENAME and notes is not None:
                    notes.append(f"node {node_id} ({cls}): {field} {want!r} matched by file name only: using {got!r}")
                node["inputs"][field] = got
    return problems


def check(graph: Mapping, object_info: Mapping) -> list[str]:
    """One line per mismatch between the graph and the server's node definitions: an unknown node class, an
    input the class does not offer, a required input left unset."""
    problems = []
    for node_id, node in graph.items():
        cls = str(node.get("class_type"))
        if cls not in object_info:
            problems.append(f"node {node_id}: the server has no node class {cls} (is its node pack installed?)")
            continue
        required = _input_names(object_info, cls, "required")
        optional = _input_names(object_info, cls, "optional")
        hidden = _input_names(object_info, cls, "hidden")
        given = (node.get("inputs") or {}).keys()
        for name in given:
            if name not in required and name not in optional and name not in hidden:
                problems.append(f"node {node_id} ({cls}): the server's {cls} has no input {name!r}")
        for name in required:
            if name not in given:
                problems.append(f"node {node_id} ({cls}): required input {name!r} is not set")
    return problems
