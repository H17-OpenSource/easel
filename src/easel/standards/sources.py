"""Every source's values for every resource: experience, author, references and prior (manual lives in the card).

- experience: `experience.reduce` over `<home>/runs/` (weight of the kept rows, keep rate, and the settings of the
  checkpoints that produced keepers); `n` is the number of verdicts the value rests on;
- author: the usage text in `registry/resources/<sha16>.usage.yaml` (`author.parse`), then the facts the record
  carries (trigger words, the file's clip skip, a recommended weight range);
- references: `<home>/references/<source>/<id>/resources.yaml` linked to the registry (the resolved id, else the
  site's version id, else a unique sha256 prefix of at least 10 digits, else a unique file stem); LoRA weights
  as a median, checkpoint settings from the reference's recipe as a median (numbers) or the most common value;
- prior: the family file (`families/<id>.yaml`, the instance's `<home>/families/` winning), its `inference`
  block after inheritance: sampler, scheduler, steps, cfg, clip skip, prefix, negative, hires denoise and the
  native size for a checkpoint; `lora_default_weight` for a LoRA.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from easel import families_check
from easel.registry import resources as res
from easel.registry.shipped import REPO, reason

from . import author, experience

LORA_TYPES = ("lora", "lycoris")
CHECKPOINT_TYPES = ("checkpoint", "unet")


@dataclass(frozen=True)
class Value:
    """One value of one field from one source, with what it rests on."""

    field: str
    source: str
    value: object
    evidence: str | None
    n: int | None = None


@dataclass
class Gathered:
    """Every resource's values (manual excluded), the problems met, and what the sources held."""

    values: dict[str, list[Value]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    runs: int = 0
    verdicts: int = 0
    references: int = 0
    linked: int = 0

    def add(self, sha16: str, value: Value) -> None:
        self.values.setdefault(sha16, []).append(value)


def _median(values: list) -> object:
    mid = statistics.median(values)
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values) and float(mid).is_integer():
        return int(mid)
    return mid


def _mode(values: list) -> object:
    return Counter(values).most_common(1)[0][0]


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


# --- families (prior) --------------------------------------------------------------------------------------------

def load_families(home: Path | None) -> dict[str, tuple[dict, str]]:
    """{family id: (the family after inheritance, its file as output names it)}; broken chains are left out."""
    report = families_check.Report()
    raw = families_check.load_families(REPO / "families", report)
    labels = {fid: f"families/{fid}.yaml" for fid in raw}
    if home is not None and (Path(home) / "families").is_dir():
        own = families_check.load_families(Path(home) / "families", report)
        raw.update(own)
        labels.update({fid: f"<home>/families/{fid}.yaml" for fid in own})
    cache: dict[str, dict | None] = {}
    out = {}
    for fid in raw:
        resolved = families_check.resolve(fid, raw, cache)
        if resolved is not None:
            out[fid] = (resolved, labels[fid])
    return out


def prior_values(record: Mapping, families: Mapping[str, tuple[dict, str]]) -> list[Value]:
    """The family defaults for a checkpoint or a LoRA (nothing for other types or an unknown family)."""
    rtype = res.get_field(record, "identity.type")
    if rtype in LORA_TYPES:
        fid = res.get_field(record, "lora.trained_on_family") or res.get_field(record, "identity.family")
    elif rtype in CHECKPOINT_TYPES:
        fid = res.get_field(record, "identity.family")
    else:
        return []
    if not fid or fid not in families:
        return []
    family, label = families[fid]
    inf = family.get("inference") if isinstance(family.get("inference"), Mapping) else {}
    if rtype in LORA_TYPES:
        pairs = [("weight", inf.get("lora_default_weight"), "inference.lora_default_weight")]
    else:
        prompt = inf.get("prompt") if isinstance(inf.get("prompt"), Mapping) else {}
        hires = inf.get("hires") if isinstance(inf.get("hires"), Mapping) else {}
        native = (inf.get("resolution") or {}).get("native") if isinstance(inf.get("resolution"), Mapping) else None
        size = native if isinstance(native, list) and len(native) == 2 else [None, None]
        pairs = [("cfg", inf.get("cfg"), "inference.cfg"), ("steps", inf.get("steps"), "inference.steps"),
                 ("sampler", inf.get("sampler"), "inference.sampler"),
                 ("scheduler", inf.get("scheduler"), "inference.scheduler"),
                 ("clip_skip", inf.get("clip_skip"), "inference.clip_skip"),
                 ("width", size[0], "inference.resolution.native"), ("height", size[1], "inference.resolution.native"),
                 ("prefix", prompt.get("prefix"), "inference.prompt.prefix"),
                 ("negative", inf.get("negative_baseline"), "inference.negative_baseline"),
                 ("hires_denoise", hires.get("denoise"), "inference.hires.denoise")]
    return [Value(f, "prior", v, f"{label} {where}") for f, v, where in pairs if v not in (None, "")]


# --- author ------------------------------------------------------------------------------------------------------

def usage_text(reg: res.Registry, sha16: str, problems: list[str]) -> str:
    """The author's usage rules from `<sha16>.usage.yaml` (every entry, in file order), or ''."""
    path = reg.resources_dir / f"{sha16}.usage.yaml"
    if not path.is_file():
        return ""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        problems.append(f"{reg.shown(path)}: not readable: {reason(exc)}")
        return ""
    entries = data.get("entries") if isinstance(data, Mapping) else None
    return "\n".join(str(e.get("usage_rules")) for e in entries or [] if isinstance(e, Mapping) and e.get("usage_rules"))


def author_values(record: Mapping, text: str) -> list[Value]:
    """What the author's text says, then the facts the record carries (each field once: the text wins)."""
    rtype = str(res.get_field(record, "identity.type") or "")
    found = author.parse(text, rtype)
    triggers = [t for t in res.get_field(record, "lora.trigger_words") or [] if isinstance(t, str) and t.strip()]
    if triggers:
        found.setdefault("trigger", (", ".join(triggers), "lora.trigger_words"))
    clip = res.get_field(record, "lora.clip_skip")
    if clip:
        found.setdefault("clip_skip", (int(clip), "lora.clip_skip (the file's training metadata)"))
    weight = res.get_field(record, "lora.recommended_weight")
    if isinstance(weight, Mapping) and _number(weight.get("min")):
        found.setdefault("weight", ({"min": weight["min"], "max": weight.get("max", weight["min"])},
                                    "lora.recommended_weight"))
    return [Value(f, "author", v, why) for f, (v, why) in found.items()]


# --- references --------------------------------------------------------------------------------------------------

def _linker(records: Mapping[str, Mapping]):
    by_version: dict[str, str] = {}
    stems: dict[str, list[str]] = {}
    shas = [(str(res.get_field(r, "integrity.sha256") or "").lower(), sha) for sha, r in records.items()]
    for sha, rec in records.items():
        version = res.get_field(rec, "provenance.version_id")
        if version:
            by_version[str(version)] = sha
        stems.setdefault(Path(str(res.get_field(rec, "identity.file_name") or "")).stem.lower(), []).append(sha)

    def link(entry: Mapping) -> str | None:
        resolved = (entry.get("resolved") or {}).get("resource")
        if resolved in records:
            return str(resolved)
        if entry.get("model_version_id") and str(entry["model_version_id"]) in by_version:
            return by_version[str(entry["model_version_id"])]
        short = str(entry.get("hash") or "").lower()
        if len(short) >= 10:
            hits = {sha for full, sha in shas if full.startswith(short)}
            if len(hits) == 1:
                return hits.pop()
        if entry.get("name"):
            stem = Path(str(entry["name"]).replace("\\", "/")).stem.lower()
            if len(stems.get(stem, [])) == 1:
                return stems[stem][0]
        return None

    return link


def reference_values(references_dir: Path, records: Mapping[str, Mapping], out: Gathered) -> None:
    """LoRA weights and checkpoint settings from the collected references, added to `out`."""
    if not references_dir.is_dir():
        return
    link = _linker(records)
    weights: dict[str, list[float]] = {}
    params: dict[str, dict[str, list]] = {}
    for path in sorted(references_dir.glob("*/*/resources.yaml")):
        try:
            used = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
            ref_path = path.with_name("reference.yaml")
            ref = yaml.safe_load(ref_path.read_text(encoding="utf-8-sig")) if ref_path.is_file() else {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            out.problems.append(f"references/{path.parent.parent.name}/{path.parent.name}: not readable: "
                                f"{reason(exc)}")
            continue
        out.references += 1
        recipe = (ref or {}).get("recipe") if isinstance(ref, Mapping) else None
        recipe = recipe if isinstance(recipe, Mapping) else {}
        for entry in used.get("resources") or [] if isinstance(used, Mapping) else []:
            if not isinstance(entry, Mapping):
                continue
            sha = link(entry)
            if sha is None:
                continue
            out.linked += 1
            rtype = res.get_field(records[sha], "identity.type")
            if entry.get("role") in LORA_TYPES and _number(entry.get("weight")):
                weights.setdefault(sha, []).append(float(entry["weight"]))
            if rtype in CHECKPOINT_TYPES:
                for key in ("cfg", "steps", "sampler", "scheduler", "clip_skip"):
                    value = recipe.get(key)
                    if key == "sampler" and value:
                        value, sched = author.normalise_sampler(str(value))
                        value = value or recipe.get(key)
                        if sched and not recipe.get("scheduler"):
                            params.setdefault(sha, {}).setdefault("scheduler", []).append(sched)
                    if value is not None and value != "":
                        params.setdefault(sha, {}).setdefault(key, []).append(value)
    for sha, ws in weights.items():
        out.add(sha, Value("weight", "references", round(statistics.median(ws), 2),
                           f"median of {len(ws)} (range {min(ws):g}-{max(ws):g})", len(ws)))
    for sha, fields in params.items():
        for key, vals in fields.items():
            numeric = key in ("cfg", "steps") and all(_number(v) for v in vals)
            out.add(sha, Value(key, "references", _median(vals) if numeric else _mode(vals),
                               f"{'median' if numeric else 'most common'} of {len(vals)}", len(vals)))


# --- experience --------------------------------------------------------------------------------------------------

def experience_values(exp: experience.Experience) -> list[Value]:
    """One experience row as card values: weight and keep rate (n = verdicts), settings of kept outputs (n = kept)."""
    out: list[Value] = []
    if exp.reviews:
        if exp.kept_avg_weight is not None:
            out.append(Value("weight", "experience", exp.kept_avg_weight,
                             f"{exp.kept} kept / {exp.reviews} reviewed", exp.reviews))
        out.append(Value("keep_rate", "experience", experience.round_half_up(exp.kept / exp.reviews, 2),
                         f"{exp.kept} kept, {exp.rejected} rejected of {exp.reviews}", exp.reviews))
    for key, vals in exp.settings.items():
        numeric = key in ("cfg", "steps") and all(_number(v) for v in vals)
        out.append(Value(key, "experience", _median(vals) if numeric else _mode(vals),
                         f"from {len(vals)} kept output(s)", len(vals)))
    return out


# --- all of it ---------------------------------------------------------------------------------------------------

def instance_path(cfg, key: str, default: str) -> Path:
    """A folder of the instance: the config value (absolute as written, else under the instance root)."""
    path = Path(str(cfg.get(key, default))).expanduser()
    return path if path.is_absolute() or cfg.root is None else cfg.root / path


def gather(reg: res.Registry, cfg, records: Mapping[str, Mapping], only: Iterable[str] | None = None) -> Gathered:
    """Every non-manual value of every resource in `records` (or of the ids in `only`)."""
    out = Gathered()
    wanted = set(records) if only is None else set(only) & set(records)
    root = cfg.root
    runs = experience.read_runs(instance_path(cfg, "paths.runs", "runs"), out.problems)
    out.runs = len(runs)
    out.verdicts = sum(len(r.current) for r in runs)
    for sha, exp in experience.reduce(runs).items():
        if sha in wanted:
            for value in experience_values(exp):
                out.add(sha, value)
    for sha in sorted(wanted):
        for value in author_values(records[sha], usage_text(reg, sha, out.problems)):
            out.add(sha, value)
    linked = Gathered()
    reference_values(instance_path(cfg, "paths.references", "references"), records, linked)
    out.references, out.linked, out.problems = linked.references, linked.linked, out.problems + linked.problems
    for sha, values in linked.values.items():
        if sha in wanted:
            for value in values:
                out.add(sha, value)
    families = load_families(root)
    for sha in sorted(wanted):
        for value in prior_values(records[sha], families):
            out.add(sha, value)
    return out
