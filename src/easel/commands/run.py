"""`easel run <rcp>`: render an approved recipe through the scheduler; `--dry-run` prints the graph only.

`--dry-run` compiles the first prompt's ComfyUI graph offline and prints it (JSON), then one summary line; it
needs no server and no GPU and works on any recipe. Without it this is GPU work: an agent never types it, it
submits the render instead (`easel jobs submit <rcp>`); on an adopted workspace it refuses (exit 2) and points
at `easel recipe to-legacy <rcp>`. Exit codes: 0 ok; 1 the render failed (the run records why); 2 a refusal or
usage error; 3 still waiting for room (`--no-wait`) or the scheduler cannot be reached; 4 a lock is held.
"""

from __future__ import annotations

import argparse
import json

from easel.studio.common import add_workspace_arg, guarded, open_studio

HELP = "render an approved recipe through the scheduler; --dry-run prints the graph"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `run`."""
    from easel._common.scheduler import add_wait_args

    parser = subparsers.add_parser(
        "run", help=HELP,
        description="Render an approved recipe: wait for room through the scheduler, queue one prompt per image on "
                    "ComfyUI, and record runs/<run-id>/ (run.json, graph.json, manifest.json, outputs/). --dry-run "
                    "prints the graph and renders nothing. Agents submit renders with `easel jobs submit` instead.")
    parser.add_argument("recipe", metavar="<rcp>", help="the recipe id")
    parser.add_argument("--count", type=int, metavar="<n>", help="images (default the recipe's count)")
    parser.add_argument("--seed", type=int, metavar="<n>", help="the first seed (default the recipe's)")
    parser.add_argument("--dry-run", action="store_true", help="print the graph; queue nothing, ask no server")
    parser.add_argument("--draft", action="store_true", help="render without the hires pass, for a fast first look")
    parser.add_argument("--by", default="operator", metavar="<who>", help="who asks (kept in the scheduler's job)")
    parser.add_argument("--minutes", type=int, default=15, metavar="<n>", help="expected wall clock (default 15)")
    parser.add_argument("--timeout", type=float, default=900, metavar="<s>", help="seconds to wait per prompt "
                                                                                  "(default 900)")
    add_wait_args(parser)
    add_workspace_arg(parser)
    parser.set_defaults(func=_run)


@guarded
def _run(args: argparse.Namespace) -> int:
    from easel.studio import recipe as recipes
    from easel.studio import render

    studio = open_studio(args.tool, args.workspace)
    if args.dry_run:
        recipe = recipes.load(studio, args.recipe)
        graph, summary = render.dry_run(studio, recipe, count=args.count, seed=args.seed, draft=args.draft)
        print(json.dumps(graph, indent=2, ensure_ascii=False))
        print(summary)
        return 0
    code, _run_id = render.execute(studio, args.recipe, count=args.count, seed=args.seed, draft=args.draft,
                                   by=args.by, minutes=args.minutes, no_wait=args.no_wait, timeout_s=args.timeout,
                                   say=lambda line: print(line, flush=True))
    return code
