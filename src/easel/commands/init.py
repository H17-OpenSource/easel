"""`easel init <dir>`: the argparse layer over `easel.init`, which holds the rules and writes the files."""

from __future__ import annotations

import argparse
import sys

HELP = "create an easel home: easel.toml, registry/, runs/, recipes/, journal/"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `init`."""
    parser = subparsers.add_parser(
        "init", help=HELP,
        description="Create an easel home: easel.toml (from easel.example.toml), the registry/, runs/, recipes/ "
                    "and journal/ folders, a .gitignore that ignores everything but README.md, and a README. "
                    "Refuses a folder inside a repository checkout, a folder that already holds easel.toml, and "
                    "any other non-empty folder unless --force-existing. Exit 0 when created, 2 when refused.")
    parser.add_argument("dir", metavar="<dir>", help="the instance folder to create, for example ~/easel-home")
    parser.add_argument("--force-existing", action="store_true",
                        help="add the instance files to a non-empty folder that holds no easel.toml yet; no "
                             "existing file is overwritten")
    parser.set_defaults(func=_init)


def _init(args: argparse.Namespace) -> int:
    from easel import init

    try:
        lines = init.init(args.dir, force_existing=args.force_existing)
    except init.InitError as exc:
        for line in exc.lines:
            print(line, file=sys.stderr)
        return exc.code
    for line in lines:
        print(line)
    return 0
