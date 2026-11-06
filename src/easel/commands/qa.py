"""`qa docs|agents|tests`: the argparse layer over the package's `qa` module, which holds the checks.

The same file serves both packages; the tool name comes from the package it runs in.
"""

import argparse
from pathlib import Path

HELP = "check docs, agent files and tests against the repo's rules"
PKG = __name__.split(".")[0]


def _root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None, metavar="<folder>",
                        help="the repository checkout to check; default: the one around the current folder")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `qa` and its three actions: docs, agents, tests."""
    parser = subparsers.add_parser(
        "qa",
        help=HELP,
        description=("Check this repository against the rules in qa.toml. Each action prints one line per finding, "
                     "one summary line per check and a total; exit 0 clean, 1 findings, 2 usage error."),
    )
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")
    docs = actions.add_parser(
        "docs",
        help="runbook sections, commands against --help, planned markers, paths, GPU commands, absolute paths, "
             "output dates",
        description=(f"Check the docs: the seven runbook sections and the entry-condition table; every `{PKG} ...` "
                     "and `python tools/<x>.py` command a doc shows against --help (the sister tool's commands "
                     "against its own CLI when it is installed), and on a fenced command line every option the "
                     "verb requires; a fenced command line that names a planned verb or subcommand ends with "
                     "`# planned: <task>`, and a built one carries no such marker; repository paths and links; no "
                     "direct GPU command on a fenced command line; no absolute path; in docs/runbooks/ and "
                     "docs/reference/formats.md no fenced line dated before 2030 and no frozen-clock warning; "
                     "placeholders on the command lines of adopt runbooks."),
    )
    docs.add_argument("--allow-pending", action="store_true",
                      help="tolerate the <!-- pending: fixture output --> marker (overnight builds only; CI never)")
    docs.add_argument("--no-help-cache", action="store_true",
                      help="rebuild every --help answer instead of reading the on-disk help cache")
    _root_arg(docs)
    docs.set_defaults(func=_docs)
    agents = actions.add_parser(
        "agents",
        help="AGENTS.md: under the line limit, its seven parts in order, the Never rules, planned commands marked; "
             "host files point at it",
        description=("Check AGENTS.md and the host files (CLAUDE.md, .github/copilot-instructions.md). A line of "
                     "AGENTS.md that names a planned command says `planned: <task>` on the same line."),
    )
    _root_arg(agents)
    agents.set_defaults(func=_agents)
    tests = actions.add_parser(
        "tests",
        help="every test file defines a test; test file names are unique; tests/ holds no __init__.py",
        description="Check the test files under tests/.",
    )
    _root_arg(tests)
    tests.set_defaults(func=_tests)


def _qa():
    import importlib

    return importlib.import_module(f"{PKG}.qa")


def _docs(args: argparse.Namespace) -> int:
    qa = _qa()
    cache = None if args.no_help_cache else qa.default_cache_file()
    return qa.run("docs", root=args.root, allow_pending=args.allow_pending, cache_file=cache)


def _agents(args: argparse.Namespace) -> int:
    return _qa().run("agents", root=args.root)


def _tests(args: argparse.Namespace) -> int:
    return _qa().run("tests", root=args.root)
