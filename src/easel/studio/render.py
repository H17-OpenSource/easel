"""`easel run <rcp>`: an approved recipe -> its graph -> the scheduler -> ComfyUI -> `runs/<run-id>/`.

The steps, offline first (the server is asked only for what the graph uses):
1. resolve the recipe's resource ids to the file names the server lists (`identity.path` without its type folder,
   data/folders.yaml) and read the family's `inference` block (the instance's `families/` first);
2. `--dry-run`: compile the first prompt's graph and print it; nothing is queued, no server or GPU is asked, and
   it works on a proposed recipe too. `--draft` renders without the second (hires) pass, for a fast first look;
3. otherwise the recipe must be approved, every file must be on disk, and the job waits for room through the
   scheduler (`_common.scheduler`: the family's `host_ram_gb.eval` and the disk floor; `--no-wait` checks
   once and exits 3). Under `policy.render = "legacy-runner"` it refuses (exit 2): the host's runner renders;
4. the run is recorded at queue time: `runs/<run-id>/run.json` (status queued), `graph.json` (the first
   prompt's graph; the others differ only in seed and file prefix), `manifest.json` (resources with their
   sha256 and weights, the graph's hash, the server version, the seeds);
5. one prompt per image (seeds seed, seed+1, ...), each queued and waited for; every saved image is fetched
   byte for byte into `outputs/NNN.png` and probed header-only. A failure is recorded in `run.json`
   (`failed`, `oom`, `cancelled`, with the error) and exits 1; success sets the recipe's status to `run`.

When the job runner (`easel jobs run`) calls `execute`, it already holds the grant, so the scheduler is skipped.
"""

from __future__ import annotations

import copy
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import comfy_client, comfy_graph, hashing, ids, image_meta, scheduler, store
from easel.registry import classify
from easel.registry.resources import get_field

from . import lifecycle
from . import recipe as recipes
from .common import (
    EXIT_OK,
    EXIT_PROBLEMS,
    EXIT_USAGE,
    EXIT_WAITING,
    Studio,
    StudioError,
    check_record,
    dump,
    refuse_on_legacy,
)

RUN_SCHEMA = "easel/run/1"
MANIFEST_SCHEMA = "easel/run-manifest/1"
RUN_ID = re.compile(r"^run-\d{8}-[a-z0-9][a-z0-9_-]*$")
DEFAULT_TIMEOUT_S = 900
Say = Callable[[str], None]


@dataclass
class Plan:
    """A recipe resolved for the compiler: the spec, the family's defaults, the seeds, the manifest's resources."""

    recipe: dict
    spec: dict
    inference: dict
    seeds: list[int]
    draft: bool
    resources: list[dict] = field(default_factory=list)
    files: list[tuple[str, Path]] = field(default_factory=list)  # (label, file on disk) to check before a render


def server_name(record: Mapping, folders: classify.Folders) -> str:
    """The name the server lists: the path under `<models_root>` without its type's folder (forward slashes)."""
    rel = str(get_field(record, "identity.path") or get_field(record, "identity.file_name") or "").replace("\\", "/")
    rtype = str(get_field(record, "identity.type") or "")
    comfy = folders.type_folders.get(rtype)
    for folder in sorted(folders.comfyui_folders.get(comfy or "", ()), key=len, reverse=True):
        prefix = folder.strip("/") + "/"
        if rel.lower().startswith(prefix.lower()):
            return rel[len(prefix):]
    return rel.rsplit("/", 1)[-1]


def model_roots(studio: Studio) -> list[Path]:
    """`paths.models_root`, then every `paths.extra_model_roots` entry."""
    roots = [studio.cfg.path("paths.models_root")]
    for extra in studio.cfg.get("paths.extra_model_roots", []) or []:
        p = Path(str(extra)).expanduser()
        roots.append(p if p.is_absolute() else studio.root / p)
    return roots


def make_plan(studio: Studio, recipe: Mapping, *, count: int | None = None, seed: int | None = None,
              draft: bool = False) -> Plan:
    """Resolve a recipe for the compiler; StudioError(2) when a resource is missing from the registry."""
    ctx_records = recipes.load_context(studio).records
    folders = classify.load_folders(studio.root)
    params = copy.deepcopy(dict(recipe.get("params") or {}))
    hires = dict(params.get("hires") or {})
    if draft:
        hires = {"mode": "none"}
    spec: dict = {"task": recipe.get("task") or "txt2img", "prompt": recipe.get("prompt", ""),
                  "negative": recipe.get("negative", ""), "loras": [], "embeddings": [], "vae": None,
                  "detailer": None}
    resources, files = [], []
    for item in recipe.get("resources") or []:
        sha = str(item.get("resource"))
        record = ctx_records.get(sha)
        if record is None:
            raise StudioError(EXIT_USAGE, [f"{recipe.get('id')}: resource {sha} ({item.get('role')}) is not in the "
                                           "registry", "next: easel registry scan --hash, or revise the recipe"])
        name = server_name(record, folders)
        role = item.get("role")
        if role == "checkpoint":
            spec["checkpoint"] = name
        elif role in ("lora", "lycoris"):
            spec["loras"].append({"name": name, "weight": item.get("weight"), "slot": item.get("slot") or None})
        elif role == "vae":
            spec["vae"] = name
        elif role == "embedding":
            target = "negative" if str(item.get("slot") or "").startswith("negative") else "positive"
            spec["embeddings"].append({"name": name.rsplit(".", 1)[0], "weight": item.get("weight"),
                                       "target": target})
        elif role == "upscaler":
            if hires.get("upscaler") == sha:
                hires["upscaler"] = name
            if draft:
                continue
        resources.append({"role": role, "slot": item.get("slot") or "", "resource": sha,
                          "sha256": get_field(record, "integrity.sha256"),
                          "file_name": str(get_field(record, "identity.file_name") or name),
                          "weight": item.get("weight")})
        if get_field(record, "identity.path"):
            files.append((recipes.label(record), Path(str(get_field(record, "identity.path")))))
    if "checkpoint" not in spec:
        raise StudioError(EXIT_USAGE, [f"{recipe.get('id')} names no checkpoint resource"])
    params["hires"] = hires
    params.pop("count", None)
    spec["params"] = params
    inference = recipes.family_inference(studio, str(recipe.get("family")))
    first = int(seed if seed is not None else (params.get("seed") or 0))
    n = int(count if count is not None else (recipe.get("params") or {}).get("count") or 1)
    if n < 1:
        raise StudioError(EXIT_USAGE, ["--count must be 1 or more"])
    return Plan(recipe=dict(recipe), spec=spec, inference=inference, seeds=[first + i for i in range(n)],
                draft=draft, resources=resources, files=files)


def compile_one(plan: Plan, seed: int, prefix: str, object_info: Mapping | None = None) -> dict:
    return comfy_graph.compile_graph(plan.spec, inference=plan.inference, object_info=object_info, seed=seed,
                                     filename_prefix=prefix)


def dry_run(studio: Studio, recipe: Mapping, *, count: int | None = None, seed: int | None = None,
            draft: bool = False) -> tuple[dict, str]:
    """(the first prompt's graph, the summary line): compiled offline, nothing queued."""
    plan = make_plan(studio, recipe, count=count, seed=seed, draft=draft)
    graph = compile_one(plan, plan.seeds[0], f"easel/dry-run/{recipe.get('id')}")
    seeds = f"seed {plan.seeds[0]}" if len(plan.seeds) == 1 else f"seeds {plan.seeds[0]}..{plan.seeds[-1]}"
    note = "" if len(plan.seeds) == 1 else "; the other prompts differ only in their seed"
    hires = (plan.spec["params"].get("hires") or {}).get("mode")
    summary = (f"(dry run: {len(graph)} nodes, {len(plan.seeds)} image(s), {seeds}, hires {hires}"
               f"{', draft' if draft else ''}{note}; nothing queued, no server asked)")
    return graph, summary


# --- the ledger records ----------------------------------------------------------------------------------------------

def run_record(run_id: str, plan: Plan, *, job_id: str | None, tool: str) -> dict:
    params = plan.spec["params"]
    return {"schema": RUN_SCHEMA, "id": run_id, "legacy_id": None, "recipe": plan.recipe.get("id"),
            "workflow": plan.recipe.get("workflow"), "job": job_id, "created_at": ids.now_iso(tool),
            "status": "queued", "prompt_id": None, "seed": plan.seeds[0], "count": len(plan.seeds),
            "prompt": str(plan.recipe.get("prompt", "")), "negative": str(plan.recipe.get("negative", "")),
            "width": params.get("width"), "height": params.get("height"), "steps": params.get("steps"),
            "cfg": params.get("cfg"), "sampler": params.get("sampler"), "scheduler": params.get("scheduler"),
            "denoise": params.get("denoise"), "started_at": None, "finished_at": None, "duration_s": None,
            "gpu_peak_mb": None, "graph": "graph.json", "manifest": "manifest.json", "outputs": [],
            "imported_from": None, "error": None,
            "notes": "draft: rendered without the hires pass" if plan.draft else None}


def write_run(folder: Path, run: dict, tool: str) -> None:
    """Write `run.json` under the run folder's store lock, the lock `easel note` and `runs review` take.

    The renderer owns every field but `notes`: once the file exists, the notes on disk win and are copied into
    `run`, so a `easel note <run-id>` written while the render runs is never overwritten by the next update.
    """
    path = folder / "run.json"
    with store.lock(folder, tool):
        if path.is_file():
            try:
                on_disk = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                on_disk = None
            if isinstance(on_disk, dict) and "notes" in on_disk:
                run["notes"] = on_disk["notes"]
        check_record("run", run, f"run {run.get('id')}")
        store.write_record(path, RUN_SCHEMA, run)


def write_manifest(folder: Path, run_id: str, plan: Plan, graph_text: str, version: str | None, tool: str) -> None:
    manifest = {"schema": MANIFEST_SCHEMA, "run": run_id, "written_at": ids.now_iso(tool),
                "graph_sha256": hashing.sha256_text(graph_text),
                "comfyui": {"version": version, "node_packs": []},
                "resources": plan.resources, "seeds": plan.seeds}
    check_record("run-manifest", manifest, f"the manifest of {run_id}")
    store.write_record(folder / "manifest.json", MANIFEST_SCHEMA, manifest)


# --- a real render -------------------------------------------------------------------------------------------------------

def check_files(studio: Studio, plan: Plan) -> None:
    roots = model_roots(studio)
    missing = [f"{label} is not on disk under <models_root> ({rel.as_posix()})" for label, rel in plan.files
               if not any((root / rel).is_file() for root in roots)]
    if missing:
        raise StudioError(EXIT_USAGE, ["cannot run:", *(f"  {m}" for m in missing),
                                       "next: easel registry scan --hash (a moved file), or revise the recipe"])


def ready_recipe(studio: Studio, rcp_id: str) -> dict:
    """The recipe of a real render: refused on an adopted workspace, and only once the operator approved it."""
    refuse_on_legacy(studio, rcp_id)
    recipe = recipes.load(studio, rcp_id)
    if recipe.get("status") not in ("approved", "run"):
        raise StudioError(EXIT_USAGE, [f"{rcp_id} is {recipe.get('status')}: it renders once the operator approved "
                                       "it", f"next: ask the operator; on their yes: easel recipe approve {rcp_id}"])
    return recipe


def execute(studio: Studio, rcp_id: str, *, count: int | None = None, seed: int | None = None,
            draft: bool = False, by: str = "operator", minutes: int = 15, no_wait: bool = False,
            timeout_s: float = DEFAULT_TIMEOUT_S, job_id: str | None = None, granted: bool = False,
            say: Say = print) -> tuple[int, str | None]:
    """Render a recipe; (exit code, run id or None). `granted` = the caller already holds the scheduler's grant."""
    recipe = ready_recipe(studio, rcp_id)
    plan = make_plan(studio, recipe, count=count, seed=seed, draft=draft)
    check_files(studio, plan)
    if granted:
        return _render(studio, plan, job_id=job_id, timeout_s=timeout_s, say=say)
    family = str(recipe.get("family"))
    sched = scheduler.from_config(studio.cfg, families_dir=studio.families_dir_for(family), say=say)
    ticket = sched.plan(scheduler.Request(title=rcp_id, tool=studio.tool, family=family, step="eval",
                                          minutes=minutes, by=by))
    readied = sched.ready(ticket)  # a machine scheduler readies a job when it is next; locally a room check
    outcome = readied if readied.status == "gone" else sched.wait(ticket, no_wait=no_wait)
    if outcome.status != "granted":
        say(outcome.line)
        return (EXIT_WAITING if outcome.status == "waiting" else EXIT_USAGE), None
    say(f"room: {outcome.line}")
    sched.start(ticket)
    code, run_id = EXIT_PROBLEMS, None
    try:
        code, run_id = _render(studio, plan, job_id=job_id, timeout_s=timeout_s, say=say)
    finally:
        sched.done(ticket, exit_code=code)
    return code, run_id


def _render(studio: Studio, plan: Plan, *, job_id: str | None, timeout_s: float, say: Say) -> tuple[int, str | None]:
    client, started_here = lifecycle.ensure_server(studio, say)
    try:
        return _render_on(studio, plan, client, job_id=job_id, timeout_s=timeout_s, say=say)
    finally:
        if started_here:
            lifecycle.stop(studio)
            say("comfyui: stopped (easel started it for this run)")


def _render_on(studio: Studio, plan: Plan, client: comfy_client.Client, *, job_id: str | None, timeout_s: float,
               say: Say) -> tuple[int, str | None]:
    tool = studio.tool
    stats = client.system_stats()
    version = (stats.get("system") or {}).get("comfyui_version")
    run_id, folder = ids.claim_seq_dir(studio.runs, "run", tool)
    run = run_record(run_id, plan, job_id=job_id, tool=tool)
    graphs = []
    first = compile_one(plan, plan.seeds[0], f"easel/{run_id}/001")
    info = client.object_info_for(comfy_graph.node_classes(first))
    for i, seed in enumerate(plan.seeds, 1):
        graph = compile_one(plan, seed, f"easel/{run_id}/{i:03d}", info)
        problems = comfy_graph.resolve_names(graph, info) + comfy_graph.check(graph, info)
        if problems:
            graph_text = dump(graph) + "\n"
            store.write_text(folder / "graph.json", graph_text)
            write_manifest(folder, run_id, plan, graph_text, version, tool)
            run.update(status="failed", error="; ".join(problems)[:4000], finished_at=ids.now_iso(tool))
            write_run(folder, run, tool)
            raise StudioError(EXIT_PROBLEMS, [f"{run_id}: the server cannot run this graph:",
                                              *(f"  {p}" for p in problems),
                                              "fix: report to the operator: a missing model is not where the server "
                                              "looks; a missing node class means its node pack is not installed"])
        graphs.append(graph)
    graph_text = dump(graphs[0]) + "\n"
    store.write_text(folder / "graph.json", graph_text)
    write_manifest(folder, run_id, plan, graph_text, version, tool)
    write_run(folder, run, tool)
    say(f"{run_id}: recorded at queue time in {studio.shown(folder)}/ ({len(graphs)} prompt(s))")
    began = time.monotonic()
    run["started_at"] = ids.now_iso(tool)
    outputs: list[dict] = []
    try:
        for i, (seed, graph) in enumerate(zip(plan.seeds, graphs, strict=True), 1):
            prompt_id = client.queue(graph)
            run.update(status="running", prompt_id=prompt_id)
            write_run(folder, run, tool)
            say(f"{run_id}: prompt {i}/{len(graphs)} queued as {prompt_id}; waiting (timeout {timeout_s:g} s)")
            entry = client.wait(prompt_id, timeout=timeout_s)
            for image in comfy_client.output_images(entry):
                data = client.view(image["filename"], image["subfolder"], image["type"])
                n = len(outputs) + 1
                suffix = Path(image["filename"]).suffix.lower() or ".png"
                rel = f"outputs/{n:03d}{suffix}"
                store.write_bytes(folder / rel, data)
                meta = image_meta.probe(folder / rel)
                outputs.append({"n": n, "file": rel, "sha256": meta.get("sha256"), "bytes": meta.get("bytes"),
                                "width": meta.get("width"), "height": meta.get("height"), "seed": seed,
                                "metadata_stripped": False})
            run["outputs"] = outputs
            write_run(folder, run, tool)
    except comfy_client.ComfyError as exc:
        status = getattr(exc, "status", "failed") if isinstance(exc, comfy_client.ComfyFailed) else "failed"
        run.update(status=status, error=" ".join(exc.lines)[:4000], outputs=outputs, finished_at=ids.now_iso(tool),
                   duration_s=round(time.monotonic() - began, 1))
        write_run(folder, run, tool)
        raise StudioError(EXIT_PROBLEMS, [f"{run_id}: {status}", *exc.lines,
                                          f"next: easel runs show {run_id}; then docs/STUCK.md"]) from exc
    run.update(status="ok", outputs=outputs, finished_at=ids.now_iso(tool),
               duration_s=round(time.monotonic() - began, 1))
    write_run(folder, run, tool)
    if plan.recipe.get("status") == "approved":
        plan.recipe["status"] = "run"
        recipes.update(studio, plan.recipe)
    say(f"{run_id}: {len(outputs)} output(s) in {studio.shown(folder)}/outputs/")
    say(f"next: ask the operator for a verdict per output, then easel runs review {run_id} --verdict "
        "keep|maybe|reject --output <n> --note \"<their words>\"")
    return EXIT_OK, run_id
