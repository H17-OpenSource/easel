"""`easel next`: which stage the instance is in, the runbook to read, and the next command.

It prints three lines, `stage: <id>`, `read: docs/runbooks/<id>.md`, `next: <command>` (`--stage` prints the id
alone, `--json` all of it plus why). The stage follows from the instance's state, first match wins:
1. `01-setup`: no instance file, a configuration problem, an unset `${VAR}`, a missing models folder, or (native
   only) no ComfyUI server where `comfyui.url` points while easel does not start one itself. An adopted instance
   skips the server check: the host runner starts the server per job;
2. `02-resources`: the registry has no resources yet;
3. on an adopted instance (`instance.adopted = true` or `policy.render = "legacy-runner"`) rendering and review go
   through the host, so every later stage is `90-adopt-existing`;
4. `04-review-and-learn`: a finished run has outputs without a current verdict;
5. `03-render`: a job is open, an approved recipe has no run yet, or nothing is in flight (propose one).
"""

from __future__ import annotations

from pathlib import Path

from easel._common import config

from . import jobs, ledger
from . import recipe as recipes
from .common import TOOL, Studio, is_adopted
from .lifecycle import client

PROPOSE = 'easel recipe propose --text "<the scene>" --family <family> --hires <none|latent|model> --explain'


def decide(tool: str = TOOL, workspace: str | Path | None = None) -> dict:
    """{stage, read, next, why} for the instance."""
    def out(stage: str, nxt: str, why: str) -> dict:
        return {"stage": stage, "read": f"docs/runbooks/{stage}.md", "next": nxt, "why": why}

    try:
        cfg = config.load(tool, workspace, strict=False)
    except config.ConfigError as exc:
        return out("01-setup", f"{tool} doctor", exc.lines[0])
    if cfg.file is None or cfg.root is None:
        from easel.cli import instance_not_found

        return out("01-setup", f"{tool} doctor", instance_not_found(tool))
    if cfg.unset:
        return out("01-setup", f"{tool} doctor", cfg.unset_lines()[0])
    studio = Studio(cfg=cfg, tool=tool, root=cfg.root)
    adopted = is_adopted(cfg)
    try:
        models = cfg.path("paths.models_root")
    except config.ConfigError as exc:
        return out("01-setup", f"{tool} doctor", exc.lines[0])
    if not models.is_dir():
        return out("01-setup", f"{tool} doctor", "paths.models_root does not exist")
    if not adopted and not bool(cfg.get("comfyui.manage_lifecycle", False)):
        try:
            answered = client(studio).ping() is not None
        except Exception:  # noqa: BLE001  (a malformed URL is a setup problem too)
            answered = False
        if not answered:
            return out("01-setup", f"{tool} doctor", "no ComfyUI server answers at comfyui.url")

    from easel.registry import resources as res

    if not res.load_records(res.open_registry(cfg, tool)).records:
        return out("02-resources", f"{tool} registry scan --hash", "the registry has no resources yet")

    runs = ledger.all_runs(studio)
    for run in runs:
        if run.get("status") == "ok" and ledger.unreviewed_outputs(studio, run):
            stage = "90-adopt-existing" if adopted else "04-review-and-learn"
            return out(stage, f"{tool} runs show {run['id']}", f"run {run['id']} has outputs without a verdict")
    if adopted:
        approved = [r for r in recipes.all_recipes(studio) if r.get("status") == "approved"]
        if approved:
            rcp = approved[-1]["id"]
            return out("90-adopt-existing", f"{tool} recipe to-legacy {rcp}",
                       f"{rcp} is approved; the host runner renders it")
        return out("90-adopt-existing", PROPOSE, "adopted workspace: propose, then print the host lines")
    open_jobs = [j for j in jobs.all_jobs(studio) if j.get("status") in (*jobs.OPEN, "running")]
    if open_jobs:
        job = open_jobs[0]
        return out("03-render", f"{tool} jobs status {job['id']}", f"job {job['id']} is {job.get('status')}")
    ran = {str(r.get("recipe")) for r in runs}
    approved = [r for r in recipes.all_recipes(studio) if r.get("status") == "approved" and r.get("id") not in ran]
    if approved:
        rcp = approved[-1]["id"]
        return out("03-render", f"{tool} run {rcp} --dry-run", f"{rcp} is approved and has no run yet")
    return out("03-render", PROPOSE, "nothing is in flight: propose a recipe for the operator's request")
