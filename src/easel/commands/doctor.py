"""`easel doctor`: check the install, the instance and ComfyUI; one ok|warn|FAIL line per check, then `next:`.

Exit 0 when no line is a FAIL (warn lines are reported, they do not block), 1 otherwise. It only reads: it never
starts a server, writes a record or changes the configuration. The checks are in `easel.studio.doctor`.
"""

from __future__ import annotations

import argparse

from easel.studio.common import add_workspace_arg

HELP = "check the install, the instance and ComfyUI: ok|warn|FAIL per line, then next:"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `doctor`."""
    parser = subparsers.add_parser(
        "doctor", help=HELP,
        description="Check the install, the instance file (every ${VAR} set), the models folder, the registry, the "
                    "scheduler, jobs.allowed, the policy, the [adopt] templates of an adopted instance and the "
                    "ComfyUI server. On an adopted instance a stopped server is expected and never a FAIL.")
    add_workspace_arg(parser)
    parser.set_defaults(func=_doctor)


def _doctor(args: argparse.Namespace) -> int:
    from easel.studio import doctor

    code, lines = doctor.report(args.tool, args.workspace)
    for line in lines:
        print(line)
    return code
