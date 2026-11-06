"""`easel comfy start|stop|status`: a ComfyUI server that easel manages (native instances only).

`status` only reads: is a server answering at `comfyui.url`, its version, devices and queue, and whether easel
started it. `start` and `stop` act only when the instance lets easel manage the server
(`comfyui.manage_lifecycle = true`, `comfyui.dir`, `comfyui.python`); on an adopted workspace they refuse
(exit 2), because the host runner starts the server per job. A start waits for room first (free RAM and the
disk floor); under `scheduler.kind = "command"` it refuses, since the job runner then starts the server inside
each job's grant. `start` and `stop` are the operator's verbs: an agent reads `status` and submits renders with
`easel jobs submit`. Exit codes: 0 ok; 1 the server is down (status on a native instance) or did not come up;
2 a refusal; 3 no room yet (`start --no-wait`).
"""

from __future__ import annotations

import argparse

from easel.studio.common import add_workspace_arg, guarded, open_studio

HELP = "start, stop and check a managed ComfyUI; native instances only"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `comfy` and its actions."""
    from easel._common.scheduler import add_wait_args

    parser = subparsers.add_parser(
        "comfy", help=HELP,
        description="The ComfyUI server: status reads; start and stop need comfyui.manage_lifecycle = true and "
                    "refuse on an adopted workspace.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")
    status = actions.add_parser("status", help="is a server answering at comfyui.url: version, devices, queue",
                                description="Ask the server at comfyui.url for its version, devices and queue. "
                                            "Reads only.")
    add_workspace_arg(status)
    status.set_defaults(func=_status)
    start = actions.add_parser("start", help="start the managed server hidden, after waiting for room",
                               description="Start the server hidden (outputs under <runtime>/comfyui/), after "
                                           "waiting for free RAM and the disk floor. Native instances with "
                                           "comfyui.manage_lifecycle = true only. The operator's verb: agents "
                                           "submit renders with `easel jobs submit` instead.")
    start.add_argument("--family", metavar="<family>", help="wait for this family's host_ram_gb.eval "
                                                            "(default limits.host_ram_floor_gb)")
    start.add_argument("--wait", type=float, default=300, metavar="<s>", help="seconds to wait for the server to "
                                                                              "answer (default 300)")
    add_wait_args(start)
    add_workspace_arg(start)
    start.set_defaults(func=_start)
    stop = actions.add_parser("stop", help="stop the server easel started",
                              description="Stop the server easel started (its pid file). The operator's verb: an "
                                          "agent never stops a GPU process; it reports to the operator.")
    add_workspace_arg(stop)
    stop.set_defaults(func=_stop)


@guarded
def _status(args: argparse.Namespace) -> int:
    from easel.studio import lifecycle

    studio = open_studio(args.tool, args.workspace)
    up, lines = lifecycle.status_lines(studio)
    for line in lines:
        print(line)
    return 0 if up or studio.adopted else 1


@guarded
def _start(args: argparse.Namespace) -> int:
    from easel._common import scheduler
    from easel.studio import lifecycle
    from easel.studio.common import EXIT_USAGE, EXIT_WAITING, StudioError

    studio = open_studio(args.tool, args.workspace)
    lifecycle.check_can_manage(studio)
    the_client = lifecycle.client(studio)
    if the_client.ping() is not None:
        print(f"comfyui: already running at {the_client.url}")
        return 0
    if str(studio.cfg.get("scheduler.kind", "none")) != "none":
        raise StudioError(EXIT_USAGE, ["scheduler.kind = command: the job runner starts the server inside each "
                                       "job's grant; easel comfy start is for a machine without a scheduler",
                                       "next: easel jobs submit <rcp>"])
    say = lambda line: print(line, flush=True)  # noqa: E731
    sched = scheduler.from_config(studio.cfg, families_dir=studio.families_dir_for(args.family), say=say)
    ticket = sched.plan(scheduler.Request(title="comfy start", tool=studio.tool, family=args.family, step="eval"))
    outcome = sched.wait(ticket, no_wait=args.no_wait)
    if outcome.status != "granted":
        print(outcome.line)
        return EXIT_WAITING
    lifecycle.start(studio, say, wait_s=args.wait)
    return 0


@guarded
def _stop(args: argparse.Namespace) -> int:
    from easel.studio import lifecycle
    from easel.studio.common import REFUSED_LEGACY, StudioError

    studio = open_studio(args.tool, args.workspace)
    if studio.legacy_runner:
        raise StudioError(2, [REFUSED_LEGACY, "the host runner stops the server after each job"])
    pid = lifecycle.stop(studio)
    if pid is None:
        print("no pid file: easel did not start the running server (it is the operator's: report it)")
        return 1
    print(f"stopped pid {pid}")
    return 0
