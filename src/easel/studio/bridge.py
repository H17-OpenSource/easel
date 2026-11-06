"""Subcommands whose code lives in `easel.legacy` (the adopted-workspace bridges, another task).

`recipe to-legacy` and `runs import-legacy` are registered by the studio's verbs, but their options and their
handlers come from `easel.legacy` (`add_to_legacy_args`/`run_to_legacy`, `add_import_legacy_args`/
`run_import_legacy`). The module is imported only when the verb's parser is built; while it is not there the
subcommand is a placeholder that prints `planned: T4` and exits 2 (marked with `set_defaults(planned=...)`,
the convention `qa docs` reads).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util

from .common import EXIT_USAGE

MODULE = "easel.legacy"
TASK = "T4"


def legacy_module():
    """`easel.legacy`, or None while it is not built (or does not import)."""
    try:
        if importlib.util.find_spec(MODULE) is None:
            return None
        return importlib.import_module(MODULE)
    except ImportError:
        return None


def _planned(args: argparse.Namespace) -> int:
    print(f"planned: {TASK}")
    return EXIT_USAGE


def register_bridge(actions: argparse._SubParsersAction, name: str, add: str, run: str, text: str) -> None:
    """Add subcommand `name`: its options from `easel.legacy.<add>(parser)` and handler `easel.legacy.<run>(args)`
    when the module is there, else the planned placeholder."""
    parser = actions.add_parser(name, help=text, description=text[0].upper() + text[1:] + ".")
    module = legacy_module()
    if module is not None and callable(getattr(module, add, None)) and callable(getattr(module, run, None)):
        getattr(module, add)(parser)

        def handler(args: argparse.Namespace) -> int:
            loaded = legacy_module()
            if loaded is None or not callable(getattr(loaded, run, None)):
                return _planned(args)
            code = getattr(loaded, run)(args)
            return 0 if code is None else int(code)

        parser.set_defaults(func=handler)
        return
    parser.add_argument("rest", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    parser.set_defaults(func=_planned, planned=TASK)
