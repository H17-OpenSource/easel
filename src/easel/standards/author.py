"""Numbers out of an author's usage text: CFG, steps, sampler and scheduler, size, clip skip, weight, denoise.

Authors write their main recommendation first and their variants after (hires pass, detailer, inpainting), so
the first match per field wins and lines about a variant never set a base field. A range ("CFG 5-7") becomes
`{min, max}`. Sampler names in the A1111 spelling ("DPM++ 2M Karras") become ComfyUI's sampler and scheduler
names. Each value keeps the line it came from as its evidence (cut to 200 characters).
"""

from __future__ import annotations

import re
from collections.abc import Callable

# A1111 and site spellings -> ComfyUI sampler names (lower case, spaces as written).
SAMPLER_NAMES = {
    "euler a": "euler_ancestral", "euler_a": "euler_ancestral", "euler ancestral": "euler_ancestral",
    "euler_ancestral": "euler_ancestral", "euler": "euler", "heun": "heun", "lms": "lms", "ddim": "ddim",
    "lcm": "lcm", "dpm2": "dpm_2", "dpm2 a": "dpm_2_ancestral", "dpm++ 2s a": "dpmpp_2s_ancestral",
    "dpm++ 2m": "dpmpp_2m", "dpmpp_2m": "dpmpp_2m", "dpm++ sde": "dpmpp_sde", "dpmpp_sde": "dpmpp_sde",
    "dpm++ 2m sde": "dpmpp_2m_sde", "dpmpp_2m_sde": "dpmpp_2m_sde", "dpm++ 3m sde": "dpmpp_3m_sde",
    "dpmpp_3m_sde": "dpmpp_3m_sde", "uni_pc": "uni_pc", "unipc": "uni_pc", "er_sde": "er_sde", "er sde": "er_sde",
    "res_multistep": "res_multistep", "deis": "deis", "ipndm": "ipndm",
}
SCHEDULER_NAMES = ("karras", "exponential", "sgm_uniform", "simple", "normal", "beta", "ddim_uniform",
                   "linear_quadratic", "kl_optimal")
_NUMBER = r"(\d+(?:\.\d+)?)"
_UPTO = r"(?:\s*(?:-|–|~|to)\s*" + _NUMBER + r")?"
_VARIANT = re.compile(r"hires|hi-res|adetailer|detailer|upscal|refiner|inpaint|face", re.IGNORECASE)
_CFG = re.compile(r"\bcfg(?:\s*scale)?\s*[:=\-]?\s*" + _NUMBER + _UPTO)
_STEPS = re.compile(r"\b(?:sampling\s+)?steps\s*[:=\-]?\s*(\d+)(?:\s*(?:-|–|~|to)\s*(\d+))?")
_SAMPLER = re.compile(r"\bsampler\s*[:=\-]\s*([a-z0-9_+ .]+)")
_SIZE = re.compile(r"(\d{3,4})\s*[x×]\s*(\d{3,4})")
_CLIP_SKIP = re.compile(r"clip\s*skip\s*[:=\-]?\s*(?:of\s*)?(-?\d)")
_DENOISE = re.compile(r"denois\w*(?:\s*strength)?\s*[:=\-]?\s*" + _NUMBER + _UPTO)
_LORA_TAG = re.compile(r"<lora:[^:>]+:(\d(?:\.\d+)?)>")
_WEIGHT = re.compile(r"\b(?:weight|strength|str)\s*(?:of\s*)?[:=\-]?\s*(\d(?:\.\d+)?)(?:\s*(?:-|–|~|to)\s*(\d(?:\.\d+)?))?")
_SCORE_TAGS = re.compile(r"score_\d(?:_up)?")


def span(low: str, high: str | None) -> float | dict:
    """One number, or `{min, max}` for a range written either way round."""
    a = float(low)
    if high is None:
        return a
    b = float(high)
    return {"min": min(a, b), "max": max(a, b)}


def normalise_sampler(text: str | None) -> tuple[str | None, str | None]:
    """('DPM++ 2M Karras') -> ('dpmpp_2m', 'karras'); a name it does not know -> (None, None)."""
    if not text:
        return None, None
    name = re.sub(r"\s+", " ", text.strip().lower().replace("_", " " if " " in text else "_"))
    scheduler = next((s for s in SCHEDULER_NAMES if name.endswith((" " + s, "_" + s))), None)
    if scheduler:
        name = name[: -len(scheduler) - 1].strip()
    name = name.strip(" .,;")
    return SAMPLER_NAMES.get(name) or SAMPLER_NAMES.get(name.replace("_", " ")), scheduler


def parse(text: str | None, resource_type: str) -> dict[str, tuple[object, str]]:
    """{field: (value, evidence line)} from an author's usage text; the first match per field wins."""
    found: dict[str, tuple[object, str]] = {}

    def keep(field: str, value: object, line: str) -> None:
        if value is not None and field not in found:
            found[field] = (value, line.strip()[:200])

    readers: list[Callable[..., None]] = [_base_fields, _clip_skip, _denoise, _prefix]
    if resource_type in ("lora", "lycoris"):
        readers.append(_weight)
    for line in (text or "").splitlines():
        low = line.lower()
        base = not _VARIANT.search(line)
        for reader in readers:
            reader(line, low, base, keep)
    return found


def _base_fields(line: str, low: str, base: bool, keep) -> None:
    if not base:
        return
    if m := _CFG.search(low):
        keep("cfg", span(m.group(1), m.group(2)), line)
    if m := _STEPS.search(low):
        value = span(m.group(1), m.group(2))
        keep("steps", int(value) if isinstance(value, float) else
             {"min": int(value["min"]), "max": int(value["max"])}, line)
    if m := _SAMPLER.search(low):
        sampler, scheduler = normalise_sampler(m.group(1))
        keep("sampler", sampler, line)
        keep("scheduler", scheduler, line)
    if (m := _SIZE.search(low)) and all(256 <= int(g) <= 2048 for g in m.groups()):
        keep("width", int(m.group(1)), line)
        keep("height", int(m.group(2)), line)


def _clip_skip(line: str, low: str, base: bool, keep) -> None:
    if m := _CLIP_SKIP.search(low):
        keep("clip_skip", abs(int(m.group(1))), line)


def _denoise(line: str, low: str, base: bool, keep) -> None:
    if ("hires" in low or "hi-res" in low or (base and re.search(r"\bdenois", low))) and (m := _DENOISE.search(low)):
        keep("hires_denoise", span(m.group(1), m.group(2)), line)


def _prefix(line: str, low: str, base: bool, keep) -> None:
    if "score_9" in low and (tags := _SCORE_TAGS.findall(low)):
        keep("prefix", ", ".join(dict.fromkeys(tags)), line)


def _weight(line: str, low: str, base: bool, keep) -> None:
    if m := _LORA_TAG.search(low):
        keep("weight", float(m.group(1)), line)
    if (m := _WEIGHT.search(low)) and float(m.group(1)) <= 2.0:
        keep("weight", span(m.group(1), m.group(2)), line)
