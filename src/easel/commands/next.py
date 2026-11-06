"""`easel next`: the current stage, the runbook to read and the next command (three lines).

`--stage` prints the stage id alone (the entry condition of every runbook), `--json` the stage, the runbook,
the next command and why. On an adopted instance the server is not checked and the render and review stages are
`90-adopt-existing`. The routing rules are in `easel.studio.stage`.
"""

from __future__ import annotations

import argparse
import json

from easel.studio.common import add_workspace_arg

HELP = "print the current stage, the runbook to read and the next command"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `next`."""
    parser = subparsers.add_parser(
        "next", help=HELP,
        description="Print three lines: stage: <id>, read: docs/runbooks/<id>.md, next: <command>. Read only the "
                    "runbook it names.")
    shape = parser.add_mutually_exclusive_group()
    shape.add_argument("--stage", action="store_true", help="print the stage id alone, for example 03-render")
    shape.add_argument("--json", action="store_true", help="print stage, read, next and why as JSON")
    add_workspace_arg(parser)
    parser.set_defaults(func=_next)


def _next(args: argparse.Namespace) -> int:
    from easel.studio import stage

    answer = stage.decide(args.tool, args.workspace)
    if args.stage:
        print(answer["stage"])
    elif args.json:
        print(json.dumps(answer, indent=2, ensure_ascii=False))
    else:
        print(f"stage: {answer['stage']}")
        print(f"read: {answer['read']}")
        print(f"next: {answer['next']}")
    return 0
