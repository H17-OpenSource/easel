"""What our own runs say about a resource: its current verdicts, counted per resource.

Inputs, per run folder `<home>/runs/<run-id>/`: `manifest.json` (the resources the run used, with role, slot and
weight) and `reviews.jsonl` (the verdicts; the latest line per output is the current verdict, as `easel runs reviews
--current` shows it). For every resource used by a run, every current verdict of that run counts:
- `reviews`: the number of distinct current verdicts on runs that used it;
- `kept` / `rejected`: how many of those rows say keep / reject (a resource in two slots of one run counts twice,
  because the join is per slot);
- `kept_avg_weight`: the mean weight the resource had in the kept rows, rounded to 3 places (null when none).
Checkpoint settings that produced keepers (cfg, steps, sampler, scheduler from `run.json`, one value per kept
verdict) come from the same join, for the checkpoint of each run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from easel._common import store
from easel.registry.shipped import reason

SETTINGS = ("cfg", "steps", "sampler", "scheduler")


@dataclass
class Run:
    """One run folder: its id, the resources it used, its parameters and its current verdicts."""

    id: str
    resources: list[dict]
    params: dict
    current: list[dict]


@dataclass
class Experience:
    """One resource's row of the reduction, plus the kept checkpoint settings."""

    resource: str
    reviews: int = 0
    kept: int = 0
    rejected: int = 0
    kept_avg_weight: float | None = None
    settings: dict[str, list] = field(default_factory=dict)

    def as_row(self) -> tuple:
        """(resource, reviews, kept, rejected, kept_avg_weight): the row the tests compare."""
        return (self.resource, self.reviews, self.kept, self.rejected, self.kept_avg_weight)


def round_half_up(value: float, places: int) -> float:
    """Round like SQLite's ROUND (half away from zero), not like Python's banker's rounding."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def current_verdicts(path: Path, run_id: str) -> list[dict]:
    """The latest review line per output of one run (an output `n` of null is a verdict on the whole run)."""
    latest = store.latest_by_key(path, lambda r: (r["run"], r.get("n")))
    return [line for (run, _n), line in latest.items() if run == run_id]


def read_runs(runs_dir: Path, problems: list[str] | None = None) -> list[Run]:
    """Every run folder that has a manifest; a folder whose files cannot be read is listed in `problems`."""
    out: list[Run] = []
    if not runs_dir.is_dir():
        return out
    for folder in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            params_path = folder / "run.json"
            params = json.loads(params_path.read_text(encoding="utf-8-sig")) if params_path.is_file() else {}
        except (OSError, ValueError) as exc:
            if problems is not None:
                problems.append(f"runs/{folder.name}: not readable: {reason(exc)}")
            continue
        run_id = str(manifest.get("run") or folder.name)
        resources = [r for r in manifest.get("resources") or [] if isinstance(r, Mapping) and r.get("resource")]
        out.append(Run(run_id, [dict(r) for r in resources], params if isinstance(params, dict) else {},
                       current_verdicts(folder / "reviews.jsonl", run_id)))
    return out


def reduce(runs: Iterable[Run]) -> dict[str, Experience]:
    """The rows by resource id, with the kept checkpoint settings attached."""
    runs = list(runs)
    rows: dict[str, list[tuple[float | None, dict]]] = {}
    for run in runs:
        for used in run.resources:
            for verdict in run.current:
                rows.setdefault(str(used["resource"]), []).append((used.get("weight"), verdict))
    out: dict[str, Experience] = {}
    for resource, joined in rows.items():
        exp = Experience(resource)
        exp.reviews = len({(v.get("run"), v.get("n")) for _w, v in joined})
        exp.kept = sum(1 for _w, v in joined if v.get("verdict") == "keep")
        exp.rejected = sum(1 for _w, v in joined if v.get("verdict") == "reject")
        weights = [float(w) for w, v in joined if v.get("verdict") == "keep" and isinstance(w, int | float)]
        exp.kept_avg_weight = round_half_up(sum(weights) / len(weights), 3) if weights else None
        out[resource] = exp
    for run in runs:
        kept = [v for v in run.current if v.get("verdict") == "keep"]
        for used in run.resources:
            if used.get("role") != "checkpoint" or not kept:
                continue
            exp = out.setdefault(str(used["resource"]), Experience(str(used["resource"])))
            for key in SETTINGS:
                if run.params.get(key) is not None:
                    exp.settings.setdefault(key, []).extend([run.params[key]] * len(kept))
    return out
