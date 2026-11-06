"""`easel index`: rebuild the generated views of the instance (today the registry's `INDEX.md` and `index.jsonl`).

Generated views carry no timestamp, so an unchanged registry rebuilds byte-identically; the records are never
touched. The same rebuild runs after every writing registry command and as `easel registry index`.
Exit codes: 0 ok; 2 a configuration or data error; 4 the registry lock is held by another process.
"""

from __future__ import annotations

import argparse
import sys

HELP = "rebuild the generated views: index.jsonl, INDEX.md"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the top-level `index` verb."""
    from easel._common.config import add_workspace_arg

    parser = subparsers.add_parser(
        "index", help=HELP,
        description="Rebuild the generated views from the records: registry/INDEX.md and registry/index.jsonl "
                    "(the person's overrides applied). Nothing else is written.")
    add_workspace_arg(parser)
    parser.set_defaults(func=_index)


def _index(args: argparse.Namespace) -> int:
    from easel._common import config
    from easel._common.config import ConfigError
    from easel.registry import resources as res
    from easel.registry.shipped import DataError

    try:
        cfg = config.load(args.tool, args.workspace, strict=True)
        reg = res.open_registry(cfg, args.tool)
        loaded = res.load_records(reg)
        records = res.effective(loaded.records, res.load_overrides(reg))
    except ConfigError as exc:
        print("\n".join(exc.lines), file=sys.stderr)
        return exc.code
    except DataError as exc:
        print("\n".join(exc.lines), file=sys.stderr)
        return 2
    for problem in loaded.problems:
        print(f"warning: {problem}", file=sys.stderr)
    idx, md = res.rebuild_index(reg, records)
    print(f"wrote {reg.shown(md)} and {reg.shown(idx)} ({len(records)} resource(s))")
    return 0
