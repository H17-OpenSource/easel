"""The run ledger: `easel runs list|show|compare|reviews|review`.

Every run is a folder `<home>/runs/<run-id>/` with `run.json`, `graph.json`, `manifest.json`, `outputs/NNN.png`
and `reviews.jsonl`, the operator's verdicts (schemas/run-review-event.schema.json), one JSON line per verdict,
appended under the run folder's store lock. Every writer of the folder takes that same lock: `runs review`,
`runs import-legacy --with-reviews`, `easel note <run-id>` and the renderer's `run.json` updates (the review page,
later, writes through it too).

The current verdict: the latest line per (run, output), where an output of null means the whole run. A changed
mind replaces a verdict; it never adds a second one.
`current_reviews` reduces every run's file that way; a torn last line (a writer stopped mid-line) is skipped.

Verdicts are the operator's: `review` records `--source operator` unless `policy.reviewer` lets the assistant
read sfw images, and then the assistant's line is stored under `policy.second_source_name`, never as the
operator's. Nothing here opens or describes an image: `compare` hashes the PNG pixel stream without decoding
it (`_common.image_meta.pixels_sha256`) and compares files byte for byte otherwise.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from easel._common import hashing, ids, image_meta, store

from .common import EXIT_USAGE, Studio, StudioError, read_json, value_text

REVIEW_SCHEMA = "easel/run-review-event/1"
RUN_ID = re.compile(r"^run-\d{8}-[a-z0-9][a-z0-9_-]*$")
VERDICTS = ("keep", "maybe", "reject")
CHECK_VALUES = ("pass", "fail", "na")
COMPARE_FIELDS = ("recipe", "seed", "count", "width", "height", "steps", "cfg", "sampler", "scheduler", "denoise",
                  "prompt", "negative", "workflow")


def run_folder(studio: Studio, run_id: str) -> Path:
    if not RUN_ID.match(run_id or ""):
        raise StudioError(EXIT_USAGE, [f"not a run id: {run_id!r} (run-YYYYMMDD-NNN)"])
    return studio.runs / run_id


def load_run(studio: Studio, run_id: str) -> dict:
    """One run's `run.json`; StudioError(2) when there is none."""
    folder = run_folder(studio, run_id)
    if not (folder / "run.json").is_file():
        raise StudioError(EXIT_USAGE, [f"no run {run_id} in {studio.shown(studio.runs)}/", "next: easel runs list"])
    return read_json(folder / "run.json", f"run {run_id}")


def all_runs(studio: Studio) -> list[dict]:
    """Every readable run, newest id first."""
    out = []
    if not studio.runs.is_dir():
        return out
    for folder in sorted(studio.runs.iterdir(), reverse=True):
        if folder.is_dir() and RUN_ID.match(folder.name) and (folder / "run.json").is_file():
            try:
                out.append(store.read_record(folder / "run.json", "easel/run/"))
            except (OSError, ValueError):
                continue
    return out


# --- verdicts ----------------------------------------------------------------------------------------------------

def reviews_file(studio: Studio, run_id: str) -> Path:
    return run_folder(studio, run_id) / "reviews.jsonl"


def review_events(studio: Studio, run_ids: Iterable[str]) -> list[dict]:
    """Every verdict line of the runs, in file order per run (the history behind the current verdicts)."""
    events = []
    for run_id in run_ids:
        events.extend(r for r in store.read_jsonl(reviews_file(studio, run_id)).records
                      if str(r.get("schema", "")).startswith("easel/run-review-event/"))
    return events


def current(events: Iterable[Mapping]) -> dict[tuple[str, int | None], dict]:
    """The current verdicts: the latest event per (run, output); output None is the whole run."""
    latest: dict[tuple[str, int | None], dict] = {}
    for event in events:
        if "run" not in event or "n" not in event:
            continue
        latest[(str(event["run"]), event["n"])] = dict(event)
    return latest


def current_reviews(studio: Studio, run_ids: Iterable[str] | None = None) -> dict[tuple[str, int | None], dict]:
    """The current verdicts of the given runs (every run when None)."""
    ids_ = [str(r["id"]) for r in all_runs(studio)] if run_ids is None else list(run_ids)
    return current(review_events(studio, sorted(ids_)))


def verdict_of(cur: Mapping[tuple[str, int | None], Mapping], run_id: str, n: int) -> Mapping | None:
    """An output's current verdict: its own, else the whole run's."""
    return cur.get((run_id, n)) or cur.get((run_id, None))


def unreviewed_outputs(studio: Studio, run: Mapping) -> list[int]:
    """The output numbers of a run without a current verdict (a whole-run verdict covers them all)."""
    cur = current_reviews(studio, [str(run["id"])])
    return [int(o["n"]) for o in run.get("outputs") or [] if verdict_of(cur, str(run["id"]), int(o["n"])) is None]


def parse_checks(text: str | None) -> dict[str, str]:
    """`--checks face=pass,hands=na` as a mapping; StudioError(2) for another value."""
    checks: dict[str, str] = {}
    for part in (text or "").split(","):
        if not part.strip():
            continue
        key, _, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or value not in CHECK_VALUES:
            raise StudioError(EXIT_USAGE, [f"--checks {part}: give <key>=pass|fail|na, the key in lower case"])
        checks[key] = value
    return checks


def review(studio: Studio, run_id: str, *, verdict: str, outputs: Sequence[int] | None, note: str | None,
           checks: Mapping[str, str] | None = None, source: str = "operator", via: str = "cli") -> list[dict]:
    """Append one verdict per output (or one for the whole run) to the run's reviews.jsonl; returns the lines."""
    from .common import schema_errors

    if verdict not in VERDICTS:
        raise StudioError(EXIT_USAGE, [f"--verdict must be one of {', '.join(VERDICTS)}"])
    run = load_run(studio, run_id)
    second = studio.policy("second_source_name", "assistant")
    if source != "operator":
        if source != second:
            raise StudioError(EXIT_USAGE, [f"--source is operator or {second} (policy.second_source_name)"])
        if studio.policy("reviewer", "human-only") != "assistant-may-read-sfw":
            raise StudioError(EXIT_USAGE, ["policy.reviewer = human-only: only the operator's verdict is recorded",
                                           "next: ask the operator for their verdict"])
    numbers = sorted({int(n) for n in outputs or []})
    known = {int(o["n"]) for o in run.get("outputs") or []}
    bad = [n for n in numbers if n not in known]
    if bad:
        raise StudioError(EXIT_USAGE, [f"--output {', '.join(map(str, bad))}: {run_id} has outputs "
                                       f"{', '.join(map(str, sorted(known))) or 'none'} (easel runs show {run_id})"])
    lines = []
    for n in numbers or [None]:
        event = {"schema": REVIEW_SCHEMA, "at": ids.now_iso(studio.tool), "v": 1, "source": source,
                 "by": "easel runs review", "run": run_id, "n": n, "verdict": verdict, "checks": dict(checks or {}),
                 "note": note, "via": via, "legacy_id": None}
        problems = schema_errors("run-review-event", event)
        if problems:
            raise StudioError(EXIT_USAGE, ["the verdict was not recorded: " + problems[0]])
        lines.append(event)
    folder = run_folder(studio, run_id)
    data = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    with store.lock(folder, studio.tool):
        store.append_bytes_locked(folder / "reviews.jsonl", data.encode("utf-8"))
    return lines


# --- lines ---------------------------------------------------------------------------------------------------------

def verdict_counts(cur: Mapping[tuple[str, int | None], Mapping], run: Mapping) -> str:
    counts: dict[str, int] = {}
    for o in run.get("outputs") or []:
        v = verdict_of(cur, str(run["id"]), int(o["n"]))
        word = (v or {}).get("verdict") or "unreviewed"
        counts[word] = counts.get(word, 0) + 1
    order = [*VERDICTS, "unreviewed"]
    return ", ".join(f"{counts[w]} {w}" for w in order if counts.get(w)) or "no outputs"


def list_lines(studio: Studio, limit: int | None = None) -> list[str]:
    runs = all_runs(studio)[: limit or None]
    cur = current_reviews(studio, [str(r["id"]) for r in runs])
    return [f"{r['id']}  {str(r.get('status')):<9} recipe {r.get('recipe') or '-':<16} "
            f"{len(r.get('outputs') or [])} output(s)  {verdict_counts(cur, r)}" for r in runs]


def show_lines(studio: Studio, run_id: str) -> list[str]:
    run = load_run(studio, run_id)
    cur = current_reviews(studio, [run_id])
    head = (f"{run_id}  {run.get('status')}  recipe {run.get('recipe') or '-'}  job {run.get('job') or '-'}  "
            f"seed {value_text(run.get('seed'))}  {value_text(run.get('width'))}x{value_text(run.get('height'))}")
    if run.get("steps") is not None:
        head += (f"  {run['steps']} steps CFG {value_text(run.get('cfg'))} {run.get('sampler')}/"
                 f"{run.get('scheduler')}")
    lines = [head]
    if run.get("legacy_id") is not None or run.get("imported_from"):
        lines.append("  imported from the host runner (legacy id kept in run.json)")
    for key in ("notes", "error"):
        if run.get(key):
            lines.append(f"  {key}: {str(run[key])[:400]}")
    whole = cur.get((run_id, None))
    if whole:
        lines.append(f"  whole run: {whole.get('verdict')} ({whole.get('source')})"
                     + (f" - {whole['note']}" if whole.get("note") else ""))
    for o in run.get("outputs") or []:
        v = cur.get((run_id, int(o["n"])))
        verdict = (f"{v.get('verdict')} ({v.get('source')})" + (f" - {v['note']}" if v.get("note") else "")) if v \
            else ("(whole-run verdict)" if whole else "(not reviewed)")
        lines.append(f"  output {o['n']}: {o.get('file')}  {value_text(o.get('width'))}x{value_text(o.get('height'))}"
                     f"  seed {value_text(o.get('seed'))}  {verdict}")
    if run.get("manifest"):
        lines.append(f"  manifest: {studio.shown(run_folder(studio, run_id) / str(run['manifest']))}")
    return lines


def review_line(event: Mapping) -> str:
    where = "whole run" if event.get("n") is None else f"output {event['n']}"
    return (f"{event.get('run')}  {where:<10} {str(event.get('verdict')):<7} {event.get('source')}  {event.get('at')}"
            + (f"  {event['note']}" if event.get("note") else ""))


# --- compare -------------------------------------------------------------------------------------------------------

def _resources(studio: Studio, run: Mapping) -> list[tuple]:
    path = run_folder(studio, str(run["id"])) / str(run.get("manifest") or "manifest.json")
    if not path.is_file():
        return []
    manifest = read_json(path, f"the manifest of {run['id']}")
    return sorted((r.get("role"), r.get("slot") or "", r.get("sha256") or r.get("resource"), r.get("weight"))
                  for r in manifest.get("resources") or [])


def _fingerprint(path: Path) -> tuple[str, str | None]:
    """('pixels', hash of the PNG pixel stream) or ('file', the file's sha256); never decoded."""
    pixels = image_meta.pixels_sha256(path)
    if pixels is not None:
        return "pixels", pixels
    return "file", hashing.sha256_file(path) if path.is_file() else None


def compare(studio: Studio, run_a: str, run_b: str) -> tuple[bool, list[str]]:
    """(reproduced exactly, lines): the settings that differ, then pixel identity per output."""
    a, b = load_run(studio, run_a), load_run(studio, run_b)
    diffs = [f for f in COMPARE_FIELDS if a.get(f) != b.get(f)]
    if _resources(studio, a) != _resources(studio, b):
        diffs.append("resources")
    shown = [("prompt or negative (not echoed)" if f in ("prompt", "negative") else f) for f in diffs]
    lines = [f"{run_a} vs {run_b}: settings " + ("identical" if not diffs else
                                                   "differ in " + ", ".join(dict.fromkeys(shown)))]
    outs_a = {int(o["n"]): o for o in a.get("outputs") or []}
    outs_b = {int(o["n"]): o for o in b.get("outputs") or []}
    same = 0
    numbers = sorted(set(outs_a) | set(outs_b))
    for n in numbers:
        if n not in outs_a or n not in outs_b:
            lines.append(f"  output {n}: only in {run_a if n in outs_a else run_b}")
            continue
        fa = run_folder(studio, run_a) / str(outs_a[n]["file"])
        fb = run_folder(studio, run_b) / str(outs_b[n]["file"])
        (ka, ha), (kb, hb) = _fingerprint(fa), _fingerprint(fb)
        if ha is None or hb is None:
            lines.append(f"  output {n}: a file is missing")
        elif ka == kb and ha == hb:
            same += 1
            lines.append(f"  output {n}: identical {'pixels' if ka == 'pixels' else 'files'}")
        else:
            lines.append(f"  output {n}: different {'pixel data' if ka == kb == 'pixels' else 'files'}")
    reproduced = not diffs and numbers and same == len(numbers)
    if reproduced:
        lines.append("reproduced exactly")
    elif diffs:
        lines.append("not an exact reproduction (expected: the settings differ)")
    else:
        lines.append("not an exact reproduction (same settings: check the server version, node packs and GPU)")
    return bool(reproduced), lines
