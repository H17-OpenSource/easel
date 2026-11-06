"""`easel jobs submit|status|cancel|run`: GPU work is submitted, never run by hand.

`submit <rcp>` checks the recipe now, plans the job with the scheduler, writes `jobs/<job-id>.json`, starts the
runner hidden when none runs, and returns at once. `status` reads a job's state (once per check-in; reading is
not a retry). `cancel` cancels a job that has not started. `run` is the runner itself (submit starts it).
Exit codes: 0 ok; 1 nothing to cancel, or a job failed; 2 a refusal (adopted workspace, a verb not in
jobs.allowed, an unapproved recipe) or a usage error; 3 still waiting (`run --no-wait`); 4 a lock is held.
"""

from __future__ import annotations

import argparse

from easel.studio.common import add_workspace_arg, guarded, open_studio

HELP = "submit, inspect and cancel GPU jobs"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `jobs` and its actions."""
    from easel._common.scheduler import add_wait_args

    parser = subparsers.add_parser(
        "jobs", help=HELP,
        description="GPU jobs: submit a render and walk away; the runner waits for room (the family's host RAM, "
                    "the disk floor, or an external scheduler configured in [scheduler]), renders, and exits when "
                    "the queue is empty.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    submit = actions.add_parser(
        "submit", help="queue the render of an approved recipe; returns at once",
        description="Check the recipe (approved, files on disk, a graph it can become), plan the job with the "
                    "scheduler, write jobs/<job-id>.json and start the runner hidden when none runs. Refused on "
                    "an adopted workspace and when jobs.allowed does not list easel run.")
    submit.add_argument("recipe", metavar="<rcp>", help="the approved recipe")
    submit.add_argument("--count", type=int, metavar="<n>", help="images (default the recipe's count)")
    submit.add_argument("--seed", type=int, metavar="<n>", help="the first seed (default the recipe's)")
    submit.add_argument("--draft", action="store_true", help="render without the hires pass")
    submit.add_argument("--by", default="operator", metavar="<who>", help="who asks: operator or the agent's name")
    submit.add_argument("--minutes", type=int, default=15, metavar="<n>", help="expected GPU time (default 15)")
    submit.add_argument("--no-runner", action="store_true", help="queue only; do not start the runner")
    add_workspace_arg(submit)
    submit.set_defaults(func=_submit)

    status = actions.add_parser("status", help="the runner and the jobs; with a job id its history and log tail",
                                description="Read the jobs' state once. With a job id: its history, exit code, "
                                            "run and the last lines of its log.")
    status.add_argument("job", nargs="?", metavar="<job-id>", help="one job")
    status.add_argument("--limit", type=int, default=15, metavar="<n>", help="at most this many jobs (default 15)")
    add_workspace_arg(status)
    status.set_defaults(func=_status)

    cancel = actions.add_parser("cancel", help="cancel a job that has not started",
                                description="Cancel a planned or waiting job and tell the scheduler. A running "
                                            "render is not stopped halfway.")
    cancel.add_argument("job", metavar="<job-id>", help="the job")
    add_workspace_arg(cancel)
    cancel.set_defaults(func=_cancel)

    run = actions.add_parser("run", help="the runner: render the open jobs oldest first (submit starts it)",
                             description="The runner: take the open jobs oldest first, wait for each grant, render, "
                                         "record the outcome; exit when the queue is empty. `easel jobs submit` "
                                         "starts it hidden; you do not type this.")
    run.add_argument("--once", action="store_true", help="take one job, then exit")
    add_wait_args(run)
    add_workspace_arg(run)
    run.set_defaults(func=_run)


@guarded
def _submit(args: argparse.Namespace) -> int:
    from easel.studio import jobs

    studio = open_studio(args.tool, args.workspace)
    job = jobs.submit(studio, args.recipe, count=args.count, seed=args.seed, draft=args.draft, by=args.by,
                      minutes=args.minutes)
    print(f"submitted {job['id']}: easel run {' '.join(job['args'])}; waits for {job['host_ram_gb']:g} GB free "
          f"(family {job['family']}) and the disk floor; ~{job['minutes']} min")
    print("runner: not started (--no-runner); the next submit starts it" if args.no_runner
          else jobs.ensure_runner(studio))
    print(f"next: easel jobs status {job['id']} (once per check-in)")
    return 0


@guarded
def _status(args: argparse.Namespace) -> int:
    from easel.studio import jobs

    studio = open_studio(args.tool, args.workspace)
    for line in jobs.status_lines(studio, args.job, args.limit):
        print(line)
    return 0


@guarded
def _cancel(args: argparse.Namespace) -> int:
    from easel.studio import jobs

    studio = open_studio(args.tool, args.workspace)
    print(jobs.cancel(studio, args.job))
    return 0


@guarded
def _run(args: argparse.Namespace) -> int:
    from easel.studio import jobs

    studio = open_studio(args.tool, args.workspace)
    return jobs.run_queue(studio, no_wait=args.no_wait, once=args.once, say=lambda line: print(line, flush=True))
