"""Recipes: everything needed to make an image again, with the reason for every choice.

A recipe is one file, `<home>/recipes/<rcp-id>.json` (schemas/recipe.schema.json): the checkpoint and LoRAs by
resource id with their weights, the prompt and negative, sampler, scheduler, steps, CFG, clip skip, size, seed,
count, the hires mode, the content class and the policy in force, the request as it was asked, and
`rationale` (choice -> why, each why naming its source: requested, parent, manual, experience, author,
references or prior). `propose`, `revise` and `like` write a new recipe with status `proposed`; `approve`
(after the operator's yes) sets `approved`; a render sets `run`.

How a proposal is assembled, deterministically, from the instance's files:
- checkpoint: the one asked for (`--checkpoint <id or name>`), else the best installed checkpoint of a family
  that has a graph compiler, matching `--family` and the request's rating (sfw by default), scored by the
  request's words in its name, title and tags, then favourites; ties by file name. The runners-up are listed;
- LoRAs: only those asked for (`--lora <id or name>[:weight]`): a LoRA changes what is in the picture, so the
  operator names it. Each is checked against data/compat.yaml (a degraded or untested pair is a warning);
  compatible LoRAs whose name matches the request are suggested, never added;
- numbers (sampler, scheduler, steps, CFG, clip skip, size, LoRA weight, hires denoise): `--set` and the
  parent recipe win, then the resource's usage card (`standards/cards/<sha16>.yaml`, whose `chosen` block
  already applies manual > experience > author > references > prior), then the family's `inference` block;
- `hires.mode` is required and never guessed (`--hires none|latent|model`; a revision keeps its parent's);
- prompt: the family's (or card's) quality prefix, the LoRAs' trigger words, then the request text, which is
  never rewritten; negative: the card's or the family's baseline, plus `--negative-extra`;
- the seed is `--seed`, the parent's, or derived from the request text, so the same request proposes the same
  recipe (and documentation stays reproducible).

Content policy: a recipe is class `sfw` only when the request is sfw and every checkpoint and LoRA in it is
rated sfw (unknown counts as not sfw); otherwise only the person's own words go into its prompt
(`policy.prompt_authoring = "user-text-only"` is recorded), the prompt is not echoed, and its resources are
named by id only. Under `policy.subjects = "fictional-adults-only"` a request whose prompt, name or chosen
resources hold a term of the third list of data/content_markers.yaml (age-down and real-person cues) is
refused before anything is written (exit 2); so is `approve` of such a recipe.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import comfy_graph, hashing, ids, store
from easel.registry import classify
from easel.registry import compat as compat_rules
from easel.registry import resources as res
from easel.registry.resources import get_field

from .common import EXIT_USAGE, Studio, StudioError, check_record, read_json, value_text

RECIPE_SCHEMA = "easel/recipe/1"
STATUSES = ("proposed", "approved", "run", "archived")
HIRES_MODES = comfy_graph.HIRES_MODES
NEUTRAL_TYPES = {"vae", "text_encoder", "clip", "clip_vision", "upscaler", "detector", "sam", "controlnet",
                 "ipadapter", "unet", "model_patch", "style_model"}
COMPAT_OK = ("native", "works")
# Words that say nothing about which model to use (they are in half the trigger lists and every scene).
GENERIC = set("a an the of in on at with and or to for from by is are be this that it its as into over under very "
              "some 1girl 1boy girl boy woman man solo looking viewer smile portrait full body upper standing "
              "sitting background simple white black red blue green yellow hair eyes outdoors indoors day night "
              "light dark detailed high quality best masterpiece score".split())
# `--set` keys and their types; `hires.<key>` changes the hires block.
SETTABLE: dict[str, type] = {
    "sampler": str, "scheduler": str, "steps": int, "cfg": float, "width": int, "height": int, "clip_skip": int,
    "seed": int, "count": int, "denoise": float, "hires.mode": str, "hires.scale": float, "hires.denoise": float,
    "hires.steps": int,
}
PARAM_KEYS = ("sampler", "scheduler", "steps", "cfg", "clip_skip")
RCP_ID = re.compile(r"^rcp-\d{8}-\d{3,}$")


# --- the request --------------------------------------------------------------------------------------------------

@dataclass
class Request:
    """What was asked, in the person's terms. `pinned` holds a parent recipe whose choices a revision keeps."""

    text: str
    family: str | None = None
    checkpoint: str | None = None
    loras: list[str] = field(default_factory=list)
    size: str | None = None
    count: int | None = None
    seed: int | None = None
    hires: str | None = None
    upscaler: str | None = None
    name: str | None = None
    rating: str = "sfw"
    negative_extra: str | None = None
    sets: dict[str, object] = field(default_factory=dict)
    kind: str = "propose"
    from_recipe: str | None = None
    from_run: str | None = None
    from_output: int | None = None
    pinned: dict | None = None


def parse_sets(pairs: Sequence[str] | None) -> dict[str, object]:
    """`--set key=value` pairs as typed values; StudioError(2) for an unknown key or a value of the wrong type."""
    out: dict[str, object] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or key not in SETTABLE:
            raise StudioError(EXIT_USAGE, [f"--set {pair}: give key=value with one of: {', '.join(SETTABLE)}"])
        kind = SETTABLE[key]
        try:
            value: object = json.loads(raw) if kind is not str else raw
            if kind is int and not (isinstance(value, int) and not isinstance(value, bool)):
                raise ValueError
            if kind is float:
                if isinstance(value, bool) or not isinstance(value, int | float):
                    raise ValueError
                value = float(value)
        except ValueError:
            raise StudioError(EXIT_USAGE, [f"--set {pair}: {key} takes a {kind.__name__}"]) from None
        if key == "hires.mode" and value not in HIRES_MODES:
            raise StudioError(EXIT_USAGE, [f"--set {pair}: hires.mode is one of {', '.join(HIRES_MODES)}"])
        out[key] = value
    return out


# --- what the instance knows ----------------------------------------------------------------------------------------

@dataclass
class Context:
    """The registry (overrides applied), the compat rules, the cue words, and where the usage cards are."""

    studio: Studio
    records: dict[str, dict]
    rules: dict
    markers: classify.Markers
    warnings: list[str] = field(default_factory=list)


def load_context(studio: Studio) -> Context:
    reg = res.open_registry(studio.cfg, studio.tool)
    loaded = res.load_records(reg)
    records = res.effective(loaded.records, res.load_overrides(reg))
    warnings = [f"registry: {p}" for p in loaded.problems]
    rules = compat_rules.load_rules(studio.root, warnings)
    return Context(studio=studio, records=records, rules=rules, markers=classify.load_markers(studio.root),
                   warnings=warnings)


def content_class(record: Mapping) -> str:
    """`sfw` only when the record says so; unknown is not sfw (policy.unknown_rating_is). Neutral types are sfw."""
    if get_field(record, "identity.type") in NEUTRAL_TYPES:
        return "sfw"
    return "sfw" if get_field(record, "semantics.content.class") == "sfw" else "nsfw"


def label(record: Mapping) -> str:
    """A resource in output: `file (sha16)` when it is sfw, the id alone otherwise (no descriptive text)."""
    if content_class(record) == "sfw":
        return f"{get_field(record, 'identity.file_name')} ({record.get('id')})"
    return f"{record.get('id')} (named by id: content class not sfw)"


def card_chosen(studio: Studio, sha16: str) -> dict[str, dict]:
    """{field: {value, source, evidence}} of a resource's usage card; {} without a card."""
    path = studio.folder("standards") / "cards" / f"{sha16}.yaml"
    if not path.is_file():
        return {}
    import yaml

    try:
        data = store.read_record(path, "easel/usage-card/")
    except (OSError, ValueError, yaml.YAMLError):  # a broken card is no card; `easel standards build` rewrites it
        return {}
    evidence = {(v.get("field"), v.get("source")): v.get("evidence") for v in data.get("values") or []
                if isinstance(v, Mapping)}
    out = {}
    for name, chosen in (data.get("chosen") or {}).items():
        if isinstance(chosen, Mapping) and "value" in chosen:
            out[name] = {"value": chosen["value"], "source": chosen.get("source", "manual"),
                         "evidence": evidence.get((name, chosen.get("source")))}
    return out


def point(value: object) -> object:
    """A card value as one number: the middle of a {min, max} range, else the value itself."""
    if isinstance(value, Mapping) and "min" in value and "max" in value:
        return round((float(value["min"]) + float(value["max"])) / 2, 2)
    return value


def card_reason(chosen: Mapping) -> str:
    return f"{chosen['source']}" + (f" ({chosen['evidence']})" if chosen.get("evidence") else "") + " (usage card)"


def family_inference(studio: Studio, family: str) -> dict:
    try:
        return comfy_graph.inference_defaults(family, studio.families_dirs())
    except comfy_graph.GraphError as exc:
        raise StudioError(EXIT_USAGE, exc.lines) from exc


def _family_file(studio: Studio, family: str) -> str:
    return "<home>/families/" if (studio.root / "families" / f"{family}.yaml").is_file() else "families/"


def resolve_resource(records: Mapping[str, Mapping], key: str, types: Sequence[str]) -> dict:
    """The resource a request names: an id (16 hex digits or a unique prefix of 6+), or a unique name fragment."""
    pool = {sha: rec for sha, rec in records.items() if get_field(rec, "identity.type") in types}
    try:
        return dict(pool[res.resolve_id(pool, key)])
    except LookupError:
        pass
    want = key.strip().lower()
    hits = [rec for rec in pool.values()
            if want and (want in str(get_field(rec, "identity.file_name") or "").lower()
                         or want in str(get_field(rec, "identity.title") or "").lower())]
    exact = [rec for rec in hits if Path(str(get_field(rec, "identity.file_name"))).stem.lower() == want]
    if len(hits) == 1 or len(exact) == 1:
        return dict((exact or hits)[0])
    kinds = "/".join(types)
    if not hits:
        raise StudioError(EXIT_USAGE, [f"no {kinds} in the registry matches {key!r}",
                                       f"next: easel registry list --type {types[0]}"])
    raise StudioError(EXIT_USAGE, [f"{key!r} matches {len(hits)} {kinds} resources; give the id: "
                                   + ", ".join(sorted(str(h.get('id')) for h in hits)[:6])])


def _yes(value: object) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _words(text: str | None) -> set[str]:
    return {w for w in classify.words(text or "") if len(w) > 2 and w not in GENERIC}


def _name_words(record: Mapping) -> set[str]:
    parts = [get_field(record, "identity.file_name"), get_field(record, "identity.title"),
             " ".join(str(t) for t in get_field(record, "semantics.tags") or [])]
    return _words(" ".join(str(p) for p in parts if p))


def _score(record: Mapping, want: set[str]) -> tuple[float, list[str]]:
    score, why = 0.0, []
    hits = sorted(want & _name_words(record))
    if hits:
        score += 3 * min(len(hits), 3)
        why.append("matches " + ", ".join(hits[:4]))
    if get_field(record, "lifecycle.favorite"):
        score += 2
        why.append("favourite")
    if get_field(record, "lifecycle.test_verdict") == "rejected":
        score -= 4
        why.append("marked rejected")
    return score, why


# --- the subjects policy ----------------------------------------------------------------------------------------------

def subject_refusal(ctx_markers: classify.Markers, texts: Sequence[str], policy: str) -> list[str] | None:
    """The refusal lines when a text holds a third-list term under `policy.subjects`, else None."""
    if policy != "fictional-adults-only":
        return None
    hits = sorted({h for text in texts if text for h in ctx_markers.subject_hits(text)})
    if not hits:
        return None
    return [f"REFUSED: policy.subjects = fictional-adults-only: the request holds age-down or real-person terms "
            f"({', '.join(hits)}; {ctx_markers.source}); nothing was written",
            "next: tell the operator this request cannot be rendered; never reword around it"]


# --- assembling a proposal ----------------------------------------------------------------------------------------------

def _pick_checkpoint(ctx: Context, req: Request) -> tuple[dict, str, list[str]]:
    """(record, reason, runner-up lines)."""
    if req.checkpoint:
        record = resolve_resource(ctx.records, req.checkpoint, ("checkpoint",))
        reason = f"parent {req.from_recipe}" if req.pinned else "requested by name"
        return record, reason, []
    want = _words(req.text) | _words(req.name)
    asked_loras = [resolve_resource(ctx.records, _lora_spec(spec)[0], ("lora", "lycoris")) for spec in req.loras]
    cands = []
    for rec in ctx.records.values():
        if get_field(rec, "identity.type") != "checkpoint" or get_field(rec, "lifecycle.install_status") != "active":
            continue
        family = get_field(rec, "identity.family")
        if not family or (req.family and family != req.family):
            continue
        if req.rating == "sfw" and content_class(rec) != "sfw":
            continue
        try:
            inference = comfy_graph.inference_defaults(str(family), ctx.studio.families_dirs())
        except comfy_graph.GraphError:
            continue
        if not inference.get("graph_compiler"):
            continue
        score, why = _score(rec, want)
        verdicts = [compat_rules.pair(lora, rec, ctx.rules).verdict for lora in asked_loras]
        if "broken" in verdicts:
            continue  # a requested LoRA is known not to work on this checkpoint's family
        if verdicts:
            score += sum({"native": 3, "works": 1, "degraded": -2}.get(v, -1) for v in verdicts)
            why.append("the requested LoRAs are " + ", ".join(verdicts) + " on it (data/compat.yaml)")
        cands.append((score, str(get_field(rec, "identity.file_name") or "").lower(), rec, why))
    if not cands:
        where = f"family {req.family}, " if req.family else ""
        raise StudioError(EXIT_USAGE, [f"no installed checkpoint fits ({where}rating {req.rating}, a family with a "
                                       "graph compiler)",
                                       "next: easel registry list --type checkpoint"
                                       + (f" --family {req.family}" if req.family else "")])
    cands.sort(key=lambda c: (-c[0], c[1]))
    best = cands[0]
    reason = f"best match of {len(cands)}: score {best[0]:g}" + (f" ({'; '.join(best[3])})" if best[3] else
                                                                 " (no specific signal; first by name)")
    others = [f"{label(c[2])} score {c[0]:g}" for c in cands[1:4]]
    return best[2], reason, others


def _lora_spec(spec: str) -> tuple[str, float | None]:
    """`name[:weight]` -> (name, weight or None)."""
    name, sep, weight = spec.rpartition(":")
    if sep and name:
        try:
            return name, float(weight)
        except ValueError:
            pass
    return spec, None


def _size(req: Request, inference: Mapping, chosen: Mapping) -> tuple[int, int, str]:
    presets = (inference.get("resolution") or {}).get("presets") or {}
    spec = (req.size or "").strip().lower()
    if spec:
        if "x" in spec and all(p.isdigit() for p in spec.split("x", 1)):
            w, h = (int(p) for p in spec.split("x", 1))
            w, h = max(64, w - w % 8), max(64, h - h % 8)
            return w, h, "requested (rounded down to multiples of 8)" if (w, h) != tuple(
                int(p) for p in spec.split("x", 1)) else "requested"
        if spec in presets:
            w, h = presets[spec]
            return int(w), int(h), f"requested preset {spec} (family inference.resolution.presets)"
        raise StudioError(EXIT_USAGE, [f"--size {req.size}: give WxH or one of {', '.join(presets) or 'WxH only'}"])
    if "width" in chosen and "height" in chosen:
        return int(point(chosen["width"]["value"])), int(point(chosen["height"]["value"])), card_reason(chosen["width"])
    native = (inference.get("resolution") or {}).get("native")
    if native:
        return int(native[0]), int(native[1]), "prior: the family's native size (inference.resolution.native)"
    raise StudioError(EXIT_USAGE, ["no size: the family names no native resolution; pass --size WxH"])


def assemble(ctx: Context, req: Request) -> tuple[dict, list[str]]:
    """The recipe record a request makes (not yet saved, id unset) and its warnings."""
    studio = ctx.studio
    subjects = studio.policy("subjects", "fictional-adults-only")
    refusal = subject_refusal(ctx.markers, [req.text, req.name or ""], subjects)
    if refusal:  # before anything else, so no other message can hide it
        raise StudioError(EXIT_USAGE, refusal)
    warnings = list(ctx.warnings)
    pinned = req.pinned or {}
    pinned_params = dict(pinned.get("params") or {})
    if (req.sets.get("hires.mode") or req.hires or (pinned_params.get("hires") or {}).get("mode")) not in HIRES_MODES:
        raise StudioError(EXIT_USAGE, [f"hires.mode is required: pass --hires {'|'.join(HIRES_MODES)} "
                                       "(docs/WHEN-TO-APPLY.md, section Hires); nothing was written"])
    parent_word = f"parent {req.from_recipe}"
    ckpt, ck_reason, runners_up = _pick_checkpoint(ctx, req)
    family = str(get_field(ckpt, "identity.family") or "")
    if not family:
        raise StudioError(EXIT_USAGE, [f"checkpoint {ckpt.get('id')} has no family; the operator sets it",
                                       "next: easel registry explain " + str(ckpt.get("id"))])
    if req.family and family != req.family and req.checkpoint:
        warnings.append(f"the checkpoint is {family}, not {req.family} as asked")
    inference = family_inference(studio, family)
    if not inference.get("graph_compiler"):
        raise StudioError(EXIT_USAGE, [f"family {family} has no graph compiler yet (inference.graph_compiler)",
                                       "next: use a workflow template for this family (planned)"])
    fam_src = f"prior: {_family_file(studio, family)}{family}.yaml"
    chosen = card_chosen(studio, str(ckpt["id"]))
    rationale: dict[str, object] = {"checkpoint": ck_reason}

    # LoRAs, as asked
    loras, lora_why = [], {}
    parent_weights = {str(r["resource"]): r.get("weight") for r in pinned.get("resources") or []
                      if r.get("role") in ("lora", "lycoris")}
    for i, spec in enumerate(req.loras, 1):
        name, weight = _lora_spec(spec)
        rec = resolve_resource(ctx.records, name, ("lora", "lycoris"))
        pair = compat_rules.pair(rec, ckpt, ctx.rules)
        why = [parent_word if req.pinned and str(rec["id"]) in parent_weights else "requested by name"]
        why.append(f"{pair.verdict} on {family} ({pair.origin or 'no rule'}"
                   + (f": {pair.reason})" if pair.reason else ")"))
        if pair.verdict not in COMPAT_OK:
            warnings.append(f"{label(rec)} is {pair.verdict} on {family} (requested anyway)")
        lcard = card_chosen(studio, str(rec["id"]))
        if weight is not None:
            same = weight == parent_weights.get(str(rec["id"]))
            why.append("weight: " + (parent_word if req.pinned and same else "requested"))
        elif "weight" in lcard:
            weight = float(point(lcard["weight"]["value"]))
            why.append("weight: " + card_reason(lcard["weight"]))
        elif inference.get("lora_default_weight") is not None:
            weight = float(inference["lora_default_weight"])
            why.append(f"weight: {fam_src} lora_default_weight")
        else:
            raise StudioError(EXIT_USAGE, [f"{label(rec)}: no weight known; pass --lora {name}:<weight>"])
        triggers = [str(t) for t in (get_field(rec, "lora.trigger_words") or [])][:3]
        if "trigger" in lcard and isinstance(lcard["trigger"]["value"], str):
            triggers = [t.strip() for t in lcard["trigger"]["value"].split(",") if t.strip()][:3]
        slot = f"lora.{i}"
        loras.append({"record": rec, "weight": round(float(weight), 3), "slot": slot, "triggers": triggers})
        lora_why[slot] = why
    if lora_why:
        rationale["loras"] = lora_why

    # suggestions: compatible LoRAs whose name matches the request (never added)
    want = _words(req.text) | _words(req.name)
    taken = {lo["record"]["id"] for lo in loras}
    suggested = []
    for rec in ctx.records.values():
        if get_field(rec, "identity.type") not in ("lora", "lycoris") or rec["id"] in taken:
            continue
        if get_field(rec, "lifecycle.install_status") != "active":
            continue
        if req.rating == "sfw" and content_class(rec) != "sfw":
            continue
        pair = compat_rules.pair(rec, ckpt, ctx.rules)
        if pair.verdict not in COMPAT_OK:
            continue
        hits = sorted(want & _name_words(rec))
        if hits:
            suggested.append(f"{label(rec)} ({pair.verdict}; matches {', '.join(hits[:3])})")

    # numbers
    params: dict[str, object] = {}
    for key in PARAM_KEYS:
        if f"{key}" in req.sets:
            params[key], why = req.sets[key], "requested (--set)"
        elif key in pinned_params and pinned_params[key] is not None:
            params[key], why = pinned_params[key], parent_word
        elif key in chosen:
            params[key], why = point(chosen[key]["value"]), card_reason(chosen[key])
        elif inference.get(key) is not None:
            params[key], why = inference[key], fam_src
        else:
            params[key], why = None, "not set"
        rationale[key] = why
    if params["clip_skip"] is not None:
        params["clip_skip"] = int(params["clip_skip"])
    for key, kind in (("steps", int), ("cfg", float)):
        if params[key] is None:
            raise StudioError(EXIT_USAGE, [f"no {key}: neither the card nor the family gives one; pass --set {key}=<n>"])
        params[key] = kind(params[key])
    for key in ("sampler", "scheduler"):
        if not params[key]:
            raise StudioError(EXIT_USAGE, [f"no {key}: neither the card nor the family gives one; pass --set {key}=<name>"])
    width, height, size_why = _size(req, inference, chosen)
    if req.pinned and not req.size:
        width, height, size_why = int(pinned_params["width"]), int(pinned_params["height"]), parent_word
    for key, value in (("width", width), ("height", height)):
        if key in req.sets:
            value, size_why = int(req.sets[key]), "requested (--set)"
        params[key] = value
    rationale["size"] = size_why
    if "seed" in req.sets:
        seed, seed_why = int(req.sets["seed"]), "requested (--set)"
    elif req.seed is not None:
        same = req.pinned and req.seed == pinned_params.get("seed")
        seed, seed_why = int(req.seed), parent_word if same else "requested"
    else:
        seed = int(hashing.sha256_text(req.text)[:8], 16) % (2 ** 31)
        seed_why = "derived from the request text (pass --seed to choose another)"
    params["seed"] = seed
    rationale["seed"] = seed_why
    count = int(req.sets.get("count") or req.count or pinned_params.get("count") or 1)
    if count < 1:
        raise StudioError(EXIT_USAGE, ["--count must be 1 or more"])
    params["count"] = count
    if "denoise" in req.sets:
        params["denoise"] = float(req.sets["denoise"])
    elif pinned_params.get("denoise") is not None:
        params["denoise"] = float(pinned_params["denoise"])

    # hires: required, never guessed
    mode = req.sets.get("hires.mode") or req.hires or (pinned_params.get("hires") or {}).get("mode")
    if mode not in HIRES_MODES:
        raise StudioError(EXIT_USAGE, [f"hires.mode is required: pass --hires {'|'.join(HIRES_MODES)} "
                                       "(docs/WHEN-TO-APPLY.md, section Hires); nothing was written"])
    hires: dict[str, object] = {"mode": mode}
    kept = req.pinned and "hires.mode" not in req.sets and mode == (pinned_params.get("hires") or {}).get("mode")
    hires_why = parent_word if kept else "requested"
    upscaler_rec = None
    if mode != "none":
        defaults = inference.get("hires") or {}
        old = pinned_params.get("hires") or {}
        for key in ("scale", "denoise", "steps"):
            if f"hires.{key}" in req.sets:
                hires[key] = req.sets[f"hires.{key}"]
            elif old.get(key) is not None:
                hires[key] = old[key]
            elif key == "denoise" and "hires_denoise" in chosen:
                hires[key] = float(point(chosen["hires_denoise"]["value"]))
            elif defaults.get(key) is not None:
                hires[key] = defaults[key]
        if mode == "model":
            key = req.upscaler or old.get("upscaler")
            if key:
                upscaler_rec = resolve_resource(ctx.records, str(key), ("upscaler",))
            else:
                active = [r for r in ctx.records.values() if get_field(r, "identity.type") == "upscaler"
                          and get_field(r, "lifecycle.install_status") == "active"]
                if len(active) != 1:
                    raise StudioError(EXIT_USAGE, ["hires mode model needs an upscale model: pass --upscaler <id or "
                                                   "name>", "next: easel registry list --type upscaler"])
                upscaler_rec = dict(active[0])
            hires["upscaler"] = upscaler_rec["id"]
        hires_why += f"; scale, denoise and steps: {fam_src} inference.hires unless set"
    params["hires"] = hires
    rationale["hires"] = f"{mode}: {hires_why}"

    # prompt and negative
    same_as_parent = (req.pinned and req.text == (pinned.get("request") or {}).get("text")
                      and [lo["record"]["id"] for lo in loras] == [r["resource"] for r in pinned.get("resources") or []
                                                                    if r.get("role") in ("lora", "lycoris")]
                      and ckpt["id"] == pinned.get("checkpoint"))
    if same_as_parent:
        prompt, negative = str(pinned.get("prompt", "")), str(pinned.get("negative", ""))
        rationale["prompt"] = rationale["negative"] = parent_word
    else:
        if "prefix" in chosen and isinstance(chosen["prefix"]["value"], str):
            prefix, prefix_why = chosen["prefix"]["value"], card_reason(chosen["prefix"])
        else:
            prefix = str((inference.get("prompt") or {}).get("prefix") or "")
            prefix_why = f"{fam_src} inference.prompt.prefix" if prefix else "none"
        triggers = []
        for lo in loras:
            for t in lo["triggers"]:
                if t and t.lower() not in req.text.lower() and t not in triggers:
                    triggers.append(t)
        prompt = ", ".join(p for p in [prefix.strip(), *triggers, req.text.strip()] if p)
        rationale["prompt"] = (f"prefix: {prefix_why}; triggers: " + (", ".join(triggers) or "none")
                               + "; then the request text, unchanged")
        if "negative" in chosen and isinstance(chosen["negative"]["value"], str):
            negative, neg_why = chosen["negative"]["value"], card_reason(chosen["negative"])
        else:
            negative = str(inference.get("negative_baseline") or "")
            neg_why = f"{fam_src} inference.negative_baseline"
        if req.negative_extra:
            negative = ", ".join(p for p in [negative, req.negative_extra.strip()] if p)
            neg_why += "; plus the requested extra"
        rationale["negative"] = neg_why

    # content class and policy
    used = [ckpt] + [lo["record"] for lo in loras]
    klass = "sfw" if req.rating == "sfw" and all(content_class(r) == "sfw" for r in used) else "nsfw"
    if req.rating == "sfw" and klass != "sfw":
        warnings.append("a requested resource is not rated sfw: the recipe is not sfw, so only the person's own "
                        "words go into its prompt (user-text-only)")
    authoring = studio.policy("prompt_authoring", "sfw-only")
    policy = {"prompt_authoring": "sfw-only" if klass == "sfw" and authoring == "sfw-only" else "user-text-only"}
    policy["subjects"] = "fictional-adults-only"
    refusal = subject_refusal(ctx.markers, [prompt, req.name or "", *(
        str(get_field(r, field_name) or "") for r in used for field_name in ("identity.file_name", "identity.title"))],
        subjects)
    if refusal:
        raise StudioError(EXIT_USAGE, refusal)
    if (inference.get("prompt") or {}).get("style") == "tags" and "," not in req.text and len(req.text.split()) > 6:
        warnings.append(f"{family} reads comma-separated tags best; the request text is prose")

    # licences and alternatives
    licence: dict[str, str] = {}
    for rec in used + ([upscaler_rec] if upscaler_rec else []):
        lic = rec.get("licence") or {}
        line = (f"{lic.get('status', 'unknown')} - {value_text(lic.get('name'))}; commercial_use: "
                f"{value_text(lic.get('commercial_use'))}; credit required {_yes(lic.get('credit_required'))}; "
                f"derivatives {_yes(lic.get('derivatives_allowed'))}")
        if lic.get("commercial_use") == "none":
            line += "; FLAG commercial_use: none (no commercial use of any kind)"
        if lic.get("status", "unknown") == "unknown":
            line += "; unknown: never invent one, ask the operator"
        licence[str(rec["id"])] = line
    rationale["licence"] = licence
    if runners_up:
        rationale["other_checkpoints"] = runners_up
    if suggested:
        rationale["suggested_loras"] = suggested[:5]
    if warnings:
        rationale["warnings"] = warnings

    resources = [{"role": "checkpoint", "resource": ckpt["id"], "slot": "", "weight": None, "reason": ck_reason}]
    for lo in loras:
        resources.append({"role": get_field(lo["record"], "identity.type"), "resource": lo["record"]["id"],
                          "slot": lo["slot"], "weight": lo["weight"], "reason": "; ".join(lora_why[lo["slot"]])})
    if upscaler_rec:
        resources.append({"role": "upscaler", "resource": upscaler_rec["id"], "slot": "hires", "weight": None,
                          "reason": "the hires pass's upscale model"})
    request: dict[str, object] = {"kind": req.kind, "text": req.text, "from_recipe": req.from_recipe,
                                  "from_run": req.from_run, "from_output": req.from_output}
    for key, value in (("name", req.name), ("checkpoint", req.checkpoint), ("loras", list(req.loras) or None),
                       ("count", req.count), ("size", req.size), ("seed", req.seed), ("rating", req.rating),
                       ("hires", req.hires), ("negative_extra", req.negative_extra), ("set", req.sets or None),
                       ("family", req.family)):
        if value is not None:
            request[key] = value
    recipe = {"schema": RECIPE_SCHEMA, "id": None, "legacy_id": None, "created_at": ids.now_iso(studio.tool),
              "name": req.name, "status": "proposed", "task": "txt2img", "family": family,
              "checkpoint": ckpt["id"], "workflow": None, "prompt": prompt, "negative": negative,
              "params": params, "resources": resources, "policy": policy, "content_class": klass,
              "request": request, "rationale": rationale, "parent": req.from_recipe, "approved_at": None,
              "notes": None}
    return recipe, warnings


# --- the files ---------------------------------------------------------------------------------------------------------

def recipe_path(studio: Studio, rcp_id: str) -> Path:
    return studio.recipes / f"{rcp_id}.json"


def load(studio: Studio, rcp_id: str) -> dict:
    """One recipe record; StudioError(2) when there is none."""
    if not RCP_ID.match(rcp_id or ""):
        raise StudioError(EXIT_USAGE, [f"not a recipe id: {rcp_id!r} (rcp-YYYYMMDD-NNN)"])
    path = recipe_path(studio, rcp_id)
    if not path.is_file():
        raise StudioError(EXIT_USAGE, [f"no recipe {rcp_id} in {studio.shown(studio.recipes)}/",
                                       "next: easel recipe list"])
    data = read_json(path, f"recipe {rcp_id}")
    if not str(data.get("schema", "")).startswith("easel/recipe/"):
        raise StudioError(EXIT_USAGE, [f"{studio.shown(path)} is not an easel/recipe record"])
    return data


def all_recipes(studio: Studio) -> list[dict]:
    """Every readable recipe, newest id first."""
    out = []
    for path in sorted(studio.recipes.glob("rcp-*.json"), reverse=True):
        try:
            out.append(store.read_record(path, "easel/recipe/"))
        except (OSError, ValueError):
            continue
    return out


def save_new(studio: Studio, recipe: Mapping) -> dict:
    """Give a proposal the next free `rcp-YYYYMMDD-NNN` and write it, schema-checked, under the recipes lock."""
    data = copy.deepcopy(dict(recipe))
    data["id"] = f"rcp-{ids.stamp(studio.tool)}-000"
    check_record("recipe", data, "the recipe")
    with store.lock(studio.recipes, studio.tool):
        existing = [p.stem for p in studio.recipes.glob("rcp-*.json")]
        data["id"] = ids.next_seq_id("rcp", existing, studio.tool)
        store.write_record(recipe_path(studio, data["id"]), RECIPE_SCHEMA, data)
    return data


def update(studio: Studio, recipe: Mapping) -> dict:
    """Rewrite an existing recipe (status changes), schema-checked."""
    data = dict(recipe)
    check_record("recipe", data, f"recipe {data.get('id')}")
    with store.lock(studio.recipes, studio.tool):
        store.write_record(recipe_path(studio, str(data["id"])), RECIPE_SCHEMA, data)
    return data


# --- the verbs' logic --------------------------------------------------------------------------------------------------

def propose(studio: Studio, req: Request) -> tuple[dict, Context]:
    ctx = load_context(studio)
    recipe, _warnings = assemble(ctx, req)
    return save_new(studio, recipe), ctx


def revise_request(parent: Mapping, *, text: str | None = None, checkpoint: str | None = None,
                   loras: Sequence[str] = (), lora_weights: Sequence[str] = (), drop: Sequence[str] = (),
                   seed: int | None = None, size: str | None = None, count: int | None = None,
                   hires: str | None = None, upscaler: str | None = None, name: str | None = None,
                   negative_extra: str | None = None, sets: Mapping[str, object] | None = None,
                   kind: str = "revise", from_run: str | None = None, from_output: int | None = None) -> Request:
    """A request that keeps the parent's checkpoint, LoRAs (and weights), seed, size and hires, changing only what
    is named, so the new render compares one change."""
    asked = dict(parent.get("request") or {})
    params = dict(parent.get("params") or {})
    weights = {r["resource"]: r.get("weight") for r in parent.get("resources") or []
               if r.get("role") in ("lora", "lycoris")}

    def match(key: str) -> list[str]:
        hits = [sha for sha in weights if sha.startswith(key.lower())]
        if not hits:
            raise StudioError(EXIT_USAGE, [f"{key!r} is not a LoRA id of {parent.get('id')} (its LoRAs: "
                                           f"{', '.join(weights) or 'none'})"])
        return hits

    for pair in lora_weights:
        key, sep, raw = pair.partition("=")
        try:
            value = float(raw)
        except ValueError:
            raise StudioError(EXIT_USAGE, [f"--lora-weight {pair}: give <lora id>=<weight>"]) from None
        if not sep:
            raise StudioError(EXIT_USAGE, [f"--lora-weight {pair}: give <lora id>=<weight>"])
        for sha in match(key):
            weights[sha] = value
    for key in drop:
        for sha in match(key):
            weights.pop(sha)
    specs = [f"{sha}:{w:g}" if isinstance(w, int | float) else sha for sha, w in weights.items()] + list(loras)
    old_hires = params.get("hires") or {}
    return Request(
        text=text if text is not None else str(asked.get("text", "")),
        family=parent.get("family"), checkpoint=checkpoint or parent.get("checkpoint"), loras=specs,
        size=size, count=count if count is not None else params.get("count"),
        seed=seed if seed is not None else params.get("seed"),
        hires=hires or old_hires.get("mode"), upscaler=upscaler or old_hires.get("upscaler"),
        name=name if name is not None else parent.get("name"),
        rating=str(asked.get("rating") or ("sfw" if parent.get("content_class") == "sfw" else "nsfw")),
        negative_extra=negative_extra if negative_extra is not None else asked.get("negative_extra"),
        sets=dict(sets or {}), kind=kind, from_recipe=str(parent.get("id")), from_run=from_run,
        from_output=from_output, pinned=dict(parent))


def approve(studio: Studio, rcp_id: str) -> tuple[dict, str]:
    """(recipe, line): proposed -> approved; approved or run stay as they are; archived refuses."""
    recipe = load(studio, rcp_id)
    status = recipe.get("status")
    if status == "archived":
        raise StudioError(EXIT_USAGE, [f"{rcp_id} is archived; propose a revision instead (easel recipe revise "
                                       f"{rcp_id} ...)"])
    markers = classify.load_markers(studio.root)
    refusal = subject_refusal(markers, [str(recipe.get("prompt") or ""), str(recipe.get("name") or "")],
                              studio.policy("subjects", "fictional-adults-only"))
    if refusal:
        raise StudioError(EXIT_USAGE, [refusal[0].replace("nothing was written", f"{rcp_id} stays {status}"),
                                       refusal[1]])
    if status in ("approved", "run"):
        return recipe, f"{rcp_id} is already {status}"
    recipe["status"] = "approved"
    recipe["approved_at"] = ids.now_iso(studio.tool)
    return update(studio, recipe), f"{rcp_id} approved"


# --- lines -------------------------------------------------------------------------------------------------------------

def describe(recipe: Mapping, records: Mapping[str, Mapping], *, explain: bool = False) -> list[str]:
    """What `propose`, `revise` and `show` print: the choices, then with `explain` the why of each and licences."""
    p = recipe.get("params") or {}
    hires = p.get("hires") or {}
    klass = recipe.get("content_class")
    policy = recipe.get("policy") or {}

    def name_of(sha: object) -> str:
        rec = records.get(str(sha))
        return label(rec) if rec else f"{sha} (not in the registry)"

    lines = [f"{recipe.get('id')} {recipe.get('status')}: {recipe.get('family')}, {recipe.get('task')}, content "
             f"{klass}, prompt authoring {policy.get('prompt_authoring')}"
             + (f", parent {recipe['parent']}" if recipe.get("parent") else ""),
             f"  checkpoint  {name_of(recipe.get('checkpoint'))}"]
    for r in recipe.get("resources") or []:
        if r.get("role") in ("lora", "lycoris"):
            lines.append(f"  {r.get('slot', ''):<10}  {name_of(r.get('resource'))} @ {value_text(r.get('weight'))}")
    lines.append(f"  sampler     {p.get('sampler')} / {p.get('scheduler')}, {p.get('steps')} steps, CFG "
                 f"{value_text(p.get('cfg'))}, clip skip {value_text(p.get('clip_skip'))}")
    extra = ""
    if hires.get("mode") and hires.get("mode") != "none":
        extra = (f" x{value_text(hires.get('scale'))} denoise {value_text(hires.get('denoise'))} "
                 f"{value_text(hires.get('steps'))} steps")
        if hires.get("upscaler"):
            extra += f" with {name_of(hires['upscaler'])}"
    lines.append(f"  size        {p.get('width')}x{p.get('height')}, count {p.get('count', 1)}, seed "
                 f"{value_text(p.get('seed'))}, hires {hires.get('mode')}{extra}")
    if klass == "sfw":
        lines.append(f"  prompt      {recipe.get('prompt')}")
        lines.append(f"  negative    {recipe.get('negative')}")
    else:
        lines.append("  prompt      (not echoed: the recipe is not sfw; its prompt is the operator's text in the "
                     "recipe file)")
    why = recipe.get("rationale") or {}
    if explain:
        lines.append("why:")
        lines.append(f"  checkpoint  {why.get('checkpoint', '-')}")
        for slot, reasons in (why.get("loras") or {}).items():
            lines.append(f"  {slot:<10}  " + "; ".join(reasons if isinstance(reasons, list) else [str(reasons)]))
        for key in (*PARAM_KEYS, "size", "seed", "hires", "prompt", "negative"):
            if key in why:
                lines.append(f"  {key:<10}  {why[key]}")
        for key, title in (("other_checkpoints", "other checkpoints"),
                           ("suggested_loras", "suggested LoRAs, not added (name one with --lora)")):
            if why.get(key):
                lines.append(f"  {title}: " + "; ".join(str(x) for x in why[key]))
        lines.append("licence:")
        for sha, text in (why.get("licence") or {}).items():
            lines.append(f"  {name_of(sha)}: {text}")
    for warning in why.get("warnings") or []:
        lines.append(f"  WARNING     {warning}")
    return lines


def list_line(recipe: Mapping) -> str:
    return (f"{recipe.get('id')}  {str(recipe.get('status')):<9} {str(recipe.get('family')):<12} "
            f"{str(recipe.get('content_class')):<5} {recipe.get('name') or ''}").rstrip()
