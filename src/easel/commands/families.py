"""`easel families check`: the argparse layer over `easel.families_check`, which holds the rules."""

import argparse
from pathlib import Path

HELP = "check the model family files: families check"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `families` and its one action, `check`."""
    parser = subparsers.add_parser(
        "families",
        help=HELP,
        description="Check the model family files under families/.",
    )
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")
    check = actions.add_parser(
        "check",
        help="check every family file: own licence summary with a url, host RAM, inheritance, known keys",
        description=(
            "Check every families/*.yaml file. Prints one line per problem and exits 1 when there is any; "
            "unknown keys and missing card urls are warnings, which fail only with --strict."
        ),
    )
    check.add_argument("--dir", type=Path, default=None, metavar="<folder>",
                       help="the folder that holds the family files; default: families/ of this checkout")
    check.add_argument("--strict", action="store_true", help="treat warnings as problems")
    check.set_defaults(func=_check)


def _check(args: argparse.Namespace) -> int:
    from easel import families_check

    return families_check.run(families_dir=args.dir, strict=args.strict)
