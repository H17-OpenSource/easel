"""GPU jobs: `easel jobs submit|status|cancel|run`. Submit a render, walk away; the runner does the rest.

Why: an agent must never run GPU work itself, and a render may have to wait for free RAM (other programs may
need it). `submit` checks everything that can be checked now (an approved recipe, the files on disk, the verb
allowed by `jobs.allowed`), plans the job with the scheduler, writes `<home>/jobs/<job-id>.json`
(schemas/job.schema.json) and starts the runner hidden if none runs, then returns at once. The runner
(`easel jobs run`, started by `submit`; an agent never types it) takes the open jobs oldest first: it waits for
the grant (the family's `host_ram_gb.eval` and the disk floor, or an external scheduler configured in
`[scheduler]`), renders through the same code as
`easel run`, tells the scheduler it is done, and exits when the queue is empty, so nothing stays resident.

Job states: planned -> waiting -> running -> done | failed; cancelled from any state before running. Every
change is a line in the job's `history`; the render's own lines go to `jobs/<job-id>.log`. Reading a job's
state once (`status`) is not a retry. A running render is not stopped halfway: `cancel` refuses it.

Refusals (exit 2): `policy.render = "legacy-runner"` (the host's runner renders; `easel recipe to-legacy`
prints its lines) and a verb missing from `jobs.allowed` (an adopted instance allows none).
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

from easel._common import background, ids, scheduler, store

from . import render
from .common import (
    EXIT_OK,
    EXIT_PROBLEMS,
    EXIT_USAGE,
    EXIT_WAITING,
    Studio,
    StudioError,
    check_record,
    read_json,
    refuse_on_legacy,
)

JOB_SCHEMA = "easel/job/1"
JOB_ID = re.compile(r"^job-\d{8}-\d{3,}$")
OPEN = ("planned", "waiting", "ready")
FINISHED = ("done", "failed", "cancelled")
LOG_TAIL = 8
Say = Callable[[str], None]


def say_stderr(line: str) -> None:
    """Notes of the scheduler while a verb that returns at once runs: stderr, so stdout keeps the verb's answer."""
    print(line, file=sys.stderr, flush=True)


def job_path(studio: Studio, job_id: str) -> Path:
    if not JOB_ID.match(job_id or ""):
        raise StudioError(EXIT_USAGE, [f"not a job id: {job_id!r} (job-YYYYMMDD-NNN)"])
    return studio.jobs / f"{job_id}.json"


def load_job(studio: Studio, job_id: str) -> dict:
    path = job_path(studio, job_id)
    if not path.is_file():
        raise StudioError(EXIT_USAGE, [f"no job {job_id} in {studio.shown(studio.jobs)}/", "next: easel jobs status"])
    return read_json(path, f"job {job_id}")


def all_jobs(studio: Studio) -> list[dict]:
    """Every readable job record, oldest id first."""
    out = []
    for path in sorted(studio.jobs.glob("job-*.json")):
        if JOB_ID.match(path.stem):
            try:
                out.append(store.read_record(path, "easel/job/"))
            except (OSError, ValueError):
                continue
    return out


def write_job(studio: Studio, job: Mapping) -> None:
    check_record("job", job, f"job {job.get('id')}")
    store.write_record(job_path(studio, str(job["id"])), JOB_SCHEMA, job)


def note(job: dict, status: str, line: str | None, tool: str) -> None:
    """Move a job to `status` and keep the line in its history."""
    job["status"] = status
    job.setdefault("history", []).append({"at": ids.now_iso(tool), "status": status, "note": line})


def allowed_entry(studio: Studio, verb: str) -> dict | None:
    """The `jobs.allowed` entry for (easel, verb), or None."""
    for entry in studio.cfg.get("jobs.allowed", []) or []:
        if isinstance(entry, Mapping) and entry.get("tool") == studio.tool and entry.get("verb") == verb:
            return dict(entry)
    return None


def refuse_unless_allowed(studio: Studio, verb: str) -> dict:
    entry = allowed_entry(studio, verb)
    if entry is None:
        raise StudioError(EXIT_USAGE, [f"REFUSED: {studio.tool} {verb} is not in jobs.allowed of this instance "
                                       "(easel.toml [jobs]); a scheduler may start only the verbs listed there",
                                       "next: the operator decides whether to allow it; never edit the list yourself"])
    return entry


def stop_file(studio: Studio, job_id: str) -> Path:
    return studio.jobs / f"{job_id}.stop"


def _scheduler(studio: Studio, family: str | None, say: Say) -> scheduler.Scheduler:
    return scheduler.from_config(studio.cfg, families_dir=studio.families_dir_for(family), say=say)


def ticket_for(sched: scheduler.Scheduler, studio: Studio, job: Mapping) -> scheduler.Ticket:
    """The scheduler's ticket of a stored job (its request, its need, its ref at the scheduler)."""
    request = scheduler.Request(title=str(job.get("title") or job["id"]), tool=str(job.get("tool", studio.tool)),
                                family=job.get("family"), step="eval", minutes=int(job.get("minutes") or 15),
                                by=str(job.get("by") or "operator"), pausable=bool(job.get("pausable")),
                                stop_file=stop_file(studio, str(job["id"])))
    local = sched if isinstance(sched, scheduler.NullScheduler) else sched.local
    return scheduler.Ticket(request=request, need=local.need(request),
                            ref=(job.get("scheduler") or {}).get("ref"), stop_file=request.stop_file)


def parse_run_args(args: list[str]) -> dict:
    """The arguments a `run` job stores: [rcp, --count N, --seed N, --draft]."""
    if not args:
        raise StudioError(EXIT_USAGE, ["the job names no recipe"])
    out: dict[str, object] = {"recipe": args[0], "count": None, "seed": None, "draft": False}
    rest = list(args[1:])
    while rest:
        word = rest.pop(0)
        if word in ("--count", "--seed") and rest:
            out[word[2:]] = int(rest.pop(0))
        elif word == "--draft":
            out["draft"] = True
        else:
            raise StudioError(EXIT_USAGE, [f"the job's arguments hold {word!r}, which a run job does not take"])
    return out


# --- submit --------------------------------------------------------------------------------------------------------

def submit(studio: Studio, rcp_id: str, *, count: int | None = None, seed: int | None = None, draft: bool = False,
           by: str = "operator", minutes: int = 15, say: Say = say_stderr) -> dict:
    """Check, plan and write one `run` job (status waiting); the runner renders it later."""
    refuse_on_legacy(studio, rcp_id)
    entry = refuse_unless_allowed(studio, "run")
    recipe = render.ready_recipe(studio, rcp_id)
    plan = render.make_plan(studio, recipe, count=count, seed=seed, draft=draft)
    render.check_files(studio, plan)
    render.compile_one(plan, plan.seeds[0], "easel/check")  # a recipe that cannot become a graph is refused now
    family = str(recipe.get("family"))
    args = [rcp_id, "--count", str(len(plan.seeds))]
    if seed is not None:
        args += ["--seed", str(seed)]
    if draft:
        args.append("--draft")
    sched = _scheduler(studio, family, say)
    with store.lock(studio.jobs, studio.tool):
        job_id = ids.next_seq_id("job", [p.stem for p in studio.jobs.glob("job-*.json")], studio.tool)
        request = scheduler.Request(title=rcp_id, tool=studio.tool, family=family, step="eval", minutes=minutes,
                                    by=by, pausable=bool(entry.get("pausable")), stop_file=stop_file(studio, job_id))
        ticket = sched.plan(request)
        job: dict = {"schema": JOB_SCHEMA, "id": job_id, "created_at": ids.now_iso(studio.tool), "tool": studio.tool,
                     "verb": "run", "args": args, "title": rcp_id, "by": by, "family": family,
                     "host_ram_gb": float(ticket.need.ram_gb), "minutes": int(minutes),
                     "pausable": bool(entry.get("pausable")), "status": "planned",
                     "scheduler": {"kind": sched.kind, "ref": ticket.ref},
                     "target": {"recipe": rcp_id, "run": None}, "started_at": None, "finished_at": None,
                     "exit_code": None, "log": f"{studio.shown(studio.jobs)}/{job_id}.log", "history": []}
        note(job, "waiting", f"needs {ticket.need.ram_gb:g} GB free ({ticket.need.source}) and the disk floor",
             studio.tool)
        write_job(studio, job)
    return job


def runner_pid_file(studio: Studio) -> Path:
    return studio.jobs / "runner.pid"


def runner_line(studio: Studio) -> str:
    line = background.status_line(runner_pid_file(studio))
    return "runner: " + (line if line.startswith("running") else "not running (the next submit starts it)")


def _env_for_runner(studio: Studio) -> dict[str, str]:
    """The variables the runner needs: EASEL_* and every ${VAR} the instance file names (a WMI start inherits
    only the user's default environment)."""
    names = set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", repr(studio.cfg.file_values)))
    names |= {str(k) for k in (studio.cfg.file_values.get("env_aliases") or {})}
    return {k: v for k, v in os.environ.items() if k.upper().startswith("EASEL_") or k in names}


def ensure_runner(studio: Studio) -> str:
    """Start the runner hidden unless one runs; the line to print."""
    info = background.read(runner_pid_file(studio))
    if info and info.get("alive"):
        return f"runner: already running (pid {info['pid']}); it takes this job in turn"
    cmd = [sys.executable, "-m", "easel", "jobs", "run", "--workspace", str(studio.cfg.file)]
    started = background.start(cmd, studio.jobs / "runner.log", runner_pid_file(studio), cwd=studio.root,
                               env=_env_for_runner(studio))
    return f"runner: started (pid {started.pid}, {started.status_line}); log {studio.shown(studio.jobs)}/runner.log"


# --- status, cancel -------------------------------------------------------------------------------------------------

def job_line(job: Mapping) -> str:
    target = job.get("target") or {}
    return (f"{job['id']}  {str(job.get('status')):<9} {job.get('title')}  by {job.get('by')}  "
            f"{job.get('host_ram_gb'):g} GB  ~{job.get('minutes')} min  run {target.get('run') or '-'}")


def status_lines(studio: Studio, job_id: str | None, limit: int = 15) -> list[str]:
    lines = [runner_line(studio)]
    if job_id is None:
        jobs = all_jobs(studio)[-limit:]
        lines += [job_line(j) for j in jobs] or ["no jobs"]
        return lines
    job = load_job(studio, job_id)
    lines.append(job_line(job))
    lines.append(f"  command: {job.get('tool')} {job.get('verb')} {' '.join(job.get('args') or [])}")
    for item in job.get("history") or []:
        lines.append(f"  {item.get('at')}  {item.get('status')}" + (f"  {item['note']}" if item.get("note") else ""))
    if job.get("exit_code") is not None:
        lines.append(f"  exit code: {job['exit_code']}")
    log = studio.jobs / f"{job_id}.log"
    if log.is_file():
        tail = [t for t in log.read_text(encoding="utf-8", errors="replace").splitlines() if t.strip()][-LOG_TAIL:]
        if tail:
            lines.append(f"  log tail ({studio.shown(log)}):")
            lines += [f"    {t}" for t in tail]
    return lines


def cancel(studio: Studio, job_id: str, say: Say = say_stderr) -> str:
    with store.lock(studio.jobs, studio.tool):
        job = load_job(studio, job_id)
        status = job.get("status")
        if status == "running":
            raise StudioError(EXIT_PROBLEMS, [f"{job_id} is running: a render is not stopped halfway",
                                              "next: let it finish, or ask the operator"])
        if status in FINISHED:
            raise StudioError(EXIT_PROBLEMS, [f"{job_id} is {status}: nothing to cancel"])
        note(job, "cancelled", "cancelled with easel jobs cancel", studio.tool)
        write_job(studio, job)
    sched = _scheduler(studio, job.get("family"), say)
    outcome = sched.cancel(ticket_for(sched, studio, job))
    return f"{job_id} cancelled ({outcome.line})"


# --- the runner -----------------------------------------------------------------------------------------------------

def next_open(studio: Studio) -> dict | None:
    return next((j for j in all_jobs(studio) if j.get("status") in OPEN), None)


def _update(studio: Studio, job_id: str, change: Callable[[dict], None]) -> dict:
    with store.lock(studio.jobs, studio.tool):
        job = load_job(studio, job_id)
        change(job)
        write_job(studio, job)
    return job


def run_job(studio: Studio, job: Mapping, *, no_wait: bool = False, say: Say = print) -> int:
    """Wait for the grant, render, record the outcome; the exit code of the job (3 = still waiting)."""
    tool, job_id = studio.tool, str(job["id"])
    if job.get("verb") != "run" or allowed_entry(studio, str(job.get("verb"))) is None:
        why = f"{job.get('tool')} {job.get('verb')} is not a job this runner may start (jobs.allowed)"
        _update(studio, job_id, lambda j: (note(j, "failed", why, tool), j.update(exit_code=EXIT_USAGE)))
        say(f"{job_id}: {why}")
        return EXIT_USAGE
    try:
        run_args = parse_run_args(list(job.get("args") or []))
    except StudioError as exc:
        why = exc.lines[0]
        _update(studio, job_id, lambda j: (note(j, "failed", why, tool), j.update(exit_code=EXIT_USAGE)))
        return EXIT_USAGE
    sched = _scheduler(studio, job.get("family"), say)
    ticket = ticket_for(sched, studio, job)
    readied = sched.ready(ticket)  # a machine scheduler readies a job when it is next; locally a room check
    outcome = readied if readied.status == "gone" else sched.wait(ticket, no_wait=no_wait)
    if outcome.status == "waiting":
        if job.get("status") != "waiting" or no_wait:
            _update(studio, job_id, lambda j: note(j, "waiting", outcome.line, tool))
        say(f"{job_id}: {outcome.line}")
        return EXIT_WAITING
    if outcome.status != "granted":
        _update(studio, job_id, lambda j: (note(j, "cancelled", outcome.line, tool), j.update(exit_code=EXIT_USAGE)))
        say(f"{job_id}: {outcome.line}")
        return EXIT_USAGE
    current = load_job(studio, job_id)
    if current.get("status") == "cancelled":
        sched.cancel(ticket)
        say(f"{job_id}: cancelled while it waited")
        return EXIT_OK
    _update(studio, job_id, lambda j: (note(j, "running", outcome.line, tool),
                                       j.update(started_at=ids.now_iso(tool))))
    sched.start(ticket)
    log_path = studio.jobs / f"{job_id}.log"

    def log(line: str) -> None:
        say(f"{job_id}: {line}")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"{ids.now_iso(tool)} {line}\n")

    code, run_id = EXIT_PROBLEMS, None
    try:
        code, run_id = render.execute(studio, str(run_args["recipe"]), count=run_args["count"],
                                      seed=run_args["seed"], draft=bool(run_args["draft"]), by=str(job.get("by")),
                                      job_id=job_id, granted=True, say=log)
    except StudioError as exc:
        code = exc.code
        for line in exc.lines:
            log(line)
    finally:
        sched.done(ticket, exit_code=code)
        final = "done" if code == EXIT_OK else "failed"

        def finish(j: dict) -> None:
            note(j, final, None if code == EXIT_OK else f"exit {code}; see {studio.shown(log_path)}", tool)
            j.update(exit_code=code, finished_at=ids.now_iso(tool))
            j.setdefault("target", {})["run"] = run_id or (j.get("target") or {}).get("run")

        _update(studio, job_id, finish)
    log(f"{final} (exit {code})" + (f"; run {run_id}" if run_id else ""))
    return code


def run_queue(studio: Studio, *, no_wait: bool = False, once: bool = False, say: Say = print) -> int:
    """Take the open jobs oldest first until none is left (or one with `once`); 3 when one still waits."""
    refuse_on_legacy(studio)
    last = EXIT_OK
    while True:
        job = next_open(studio)
        if job is None:
            say("queue empty: the runner exits")
            return last
        code = run_job(studio, job, no_wait=no_wait, say=say)
        if code == EXIT_WAITING or once:
            return code
        last = code
