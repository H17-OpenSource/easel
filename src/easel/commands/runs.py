"""`easel runs list|show|compare|reviews|review|import-legacy`: the run ledger.

Read verbs never write. `review` appends the operator's verdict to `runs/<run-id>/reviews.jsonl` under the run
folder's store lock; the latest verdict per output counts (`reviews --current`).
`import-legacy` (an adopted workspace) lives in `easel.legacy` and is imported only when it runs.
Exit codes: 0 ok; 1 `compare` found the runs differ; 2 a usage error or an unknown id; 4 a lock is held.
"""

from __future__ import annotations

import argparse
import json

from easel.studio.common import add_workspace_arg, guarded, open_studio

HELP = "list, show, compare and review runs; import-legacy on an adopted instance"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `runs` and its actions."""
    from easel.studio.bridge import register_bridge

    parser = subparsers.add_parser(
        "runs", help=HELP,
        description="The run ledger: runs/<run-id>/ with run.json, graph.json, manifest.json, outputs/ and the "
                    "operator's verdicts in reviews.jsonl.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    lst = actions.add_parser("list", help="one line per run: status, recipe, outputs, current verdicts",
                             description="One line per run, newest first, with the counts of current verdicts.")
    lst.add_argument("--limit", type=int, default=20, metavar="<n>", help="at most this many runs (default 20)")
    lst.add_argument("--json", action="store_true", help="a JSON list of the run records")
    add_workspace_arg(lst)
    lst.set_defaults(func=_list)

    show = actions.add_parser("show", help="one run: settings, outputs numbered as --output takes them, verdicts",
                              description="One run: its settings, every output with its number (what review "
                                          "--output takes) and its current verdict. Never opens an image.")
    show.add_argument("run", metavar="<run-id>", help="the run id")
    show.add_argument("--json", action="store_true", help="the run record as JSON")
    add_workspace_arg(show)
    show.set_defaults(func=_show)

    compare = actions.add_parser(
        "compare", help="did a rerun reproduce? the settings that differ, then pixel identity per output",
        description="Compare two runs: the settings and resources that differ, then per output whether the PNG "
                    "pixel streams are identical (hashed, never decoded). Exit 0 when reproduced exactly, else 1.")
    compare.add_argument("run_a", metavar="<run-id>", help="the first run")
    compare.add_argument("run_b", metavar="<run-id>", help="the second run")
    add_workspace_arg(compare)
    compare.set_defaults(func=_compare)

    reviews = actions.add_parser(
        "reviews", help="the verdicts of the runs; --current keeps the latest per output",
        description="The operator's verdicts. --current: the latest per output (a whole-run verdict is its own "
                    "row); without it every verdict in order.")
    reviews.add_argument("run", nargs="?", metavar="<run-id>", help="only this run")
    reviews.add_argument("--current", action="store_true", help="the latest verdict per output only")
    reviews.add_argument("--json", action="store_true", help="JSON lines instead of text")
    add_workspace_arg(reviews)
    reviews.set_defaults(func=_reviews)

    review = actions.add_parser(
        "review", help="record the operator's verdict on a run or on outputs (their words in --note)",
        description="Append the operator's verdict to the run's reviews.jsonl: keep, maybe or reject, for the "
                    "whole run or for the outputs named with --output. Quote the operator in --note; never add a "
                    "verdict or a note of your own.")
    review.add_argument("run", metavar="<run-id>", help="the run id")
    review.add_argument("--verdict", required=True, choices=("keep", "maybe", "reject"), help="the operator's verdict")
    review.add_argument("--output", type=int, action="append", metavar="<n>",
                        help="the output number (repeatable; default: the whole run)")
    review.add_argument("--note", metavar="<text>", help="the operator's words, verbatim")
    review.add_argument("--checks", metavar="<key>=pass|fail|na,...", help="checklist results the operator gave")
    review.add_argument("--source", default="operator", metavar="<who>",
                        help="operator (default), or policy.second_source_name when policy.reviewer allows it")
    review.add_argument("--via", default="cli", choices=("cli", "chat"), help="how the verdict arrived")
    add_workspace_arg(review)
    review.set_defaults(func=_review)

    register_bridge(actions, "import-legacy", "add_import_legacy_args", "run_import_legacy",
                     "read a host run (and with --with-reviews its review rows) into runs/<run-id>/ (adopted "
                     "workspace)")


@guarded
def _list(args: argparse.Namespace) -> int:
    from easel.studio import ledger

    studio = open_studio(args.tool, args.workspace)
    if args.json:
        print(json.dumps(ledger.all_runs(studio)[: args.limit], indent=2, ensure_ascii=False))
        return 0
    lines = ledger.list_lines(studio, args.limit)
    for line in lines or ["no runs"]:
        print(line)
    return 0


@guarded
def _show(args: argparse.Namespace) -> int:
    from easel.studio import ledger

    studio = open_studio(args.tool, args.workspace)
    if args.json:
        print(json.dumps(ledger.load_run(studio, args.run), indent=2, ensure_ascii=False))
        return 0
    for line in ledger.show_lines(studio, args.run):
        print(line)
    return 0


@guarded
def _compare(args: argparse.Namespace) -> int:
    from easel.studio import ledger

    studio = open_studio(args.tool, args.workspace)
    same, lines = ledger.compare(studio, args.run_a, args.run_b)
    for line in lines:
        print(line)
    return 0 if same else 1


@guarded
def _reviews(args: argparse.Namespace) -> int:
    from easel.studio import ledger

    studio = open_studio(args.tool, args.workspace)
    run_ids = [args.run] if args.run else sorted(str(r["id"]) for r in ledger.all_runs(studio))
    if args.run:
        ledger.load_run(studio, args.run)
    events = ledger.review_events(studio, run_ids)
    rows = list(ledger.current(events).values()) if args.current else events
    for row in rows:
        print(json.dumps(row, ensure_ascii=False) if args.json else ledger.review_line(row))
    if not rows and not args.json:
        print("0 verdicts" + (f" for {args.run}" if args.run else ""))
    return 0


@guarded
def _review(args: argparse.Namespace) -> int:
    from easel.studio import ledger

    studio = open_studio(args.tool, args.workspace)
    lines = ledger.review(studio, args.run, verdict=args.verdict, outputs=args.output, note=args.note,
                          checks=ledger.parse_checks(args.checks), source=args.source, via=args.via)
    where = "the whole run" if lines[0]["n"] is None else "output " + ", ".join(str(x["n"]) for x in lines)
    print(f"recorded {args.verdict} on {args.run} {where} ({args.source})")
    print("next: easel runs reviews --current, then easel standards build")
    return 0
