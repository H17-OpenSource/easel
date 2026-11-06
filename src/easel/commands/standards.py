"""`easel standards build|explain|report|set|unset`: the argparse layer over `easel.standards`.

`explain` and `report` only read. `build` rewrites the cards under `<home>/standards/cards/` from the evidence
(keeping the manual entries); `set` and `unset` change one manual entry and append a changelog line. Precedence:
manual > experience (from 3 verdicts on) > author > references > prior.
Exit codes: 0 ok; 2 usage, configuration or data error, or an unknown id; 4 the registry lock is held.
"""

from __future__ import annotations

import argparse
import json
import sys

HELP = "usage standards per resource: build, explain, report, set, unset"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `standards` and its actions."""
    from easel._common.config import add_workspace_arg

    parser = subparsers.add_parser(
        "standards", help=HELP,
        description="How each resource should be used (weight, cfg, steps, sampler, ...), kept per source in "
                    "standards/cards/<sha16>.yaml. Precedence: manual > experience (from 3 verdicts on) > author "
                    "> references > prior.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    build = actions.add_parser(
        "build", help="rebuild every card from runs, verdicts, usage texts, references and family priors",
        description="Rebuild the cards: experience from the current verdicts on runs, author from the usage "
                    "texts and the records, references from the collected references, prior from the family "
                    "files. Manual entries are kept. A card is rewritten only when a value changed.")
    add_workspace_arg(build)
    build.set_defaults(func=_guarded(_build))

    explain = actions.add_parser(
        "explain", help="for one resource, every field with all five sources and which one wins",
        description="Read-only, from the evidence as it is now (no build needed): per field the manual, "
                    "experience, author, references and prior value with its evidence and n, and the winner.")
    explain.add_argument("sha16", metavar="<sha16>", help="the resource id (16 hex digits, or a unique prefix of 6)")
    explain.add_argument("--json", action="store_true", help="the fields and sources as JSON")
    add_workspace_arg(explain)
    explain.set_defaults(func=_guarded(_explain))

    report = actions.add_parser(
        "report", help="a Markdown table of the cards: which sources each checkpoint, LoRA and embedding has",
        description="Read the built cards and print (or with --out write) a Markdown table: type, family, "
                    "rating, resource, sources, fields. Resources that are not sfw are named by id only.")
    report.add_argument("--out", metavar="<file>", help="write the table to this file (relative to the instance)")
    add_workspace_arg(report)
    report.set_defaults(func=_guarded(_report))

    setter = actions.add_parser(
        "set", help="record the person's value for one field of one resource (manual wins over every source)",
        description="Record a manual value. <value>: a number, `0.6-0.8` for a range, JSON, or plain text. "
                    "--note is required: the person's reason in their words (never invented).")
    setter.add_argument("sha16", metavar="<sha16>", help="the resource id")
    setter.add_argument("field", metavar="<field>", help="weight, cfg, steps, sampler, scheduler, clip_skip, width, "
                                                         "height, prefix, negative, trigger, hires_denoise")
    setter.add_argument("value", metavar="<value>", help="the value")
    setter.add_argument("--note", required=True, metavar="<why>", help="the person's reason, quoted")
    add_workspace_arg(setter)
    setter.set_defaults(func=_guarded(_set))

    unsetter = actions.add_parser(
        "unset", help="remove the person's value for one field; the next source takes over",
        description="Remove a manual value; the next source in the precedence wins again.")
    unsetter.add_argument("sha16", metavar="<sha16>", help="the resource id")
    unsetter.add_argument("field", metavar="<field>", help="the field whose manual value goes")
    unsetter.add_argument("--note", metavar="<why>", help="the person's reason, quoted (optional)")
    add_workspace_arg(unsetter)
    unsetter.set_defaults(func=_guarded(_unset))


def _err(lines) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _guarded(handler):
    """Map configuration, data and lookup errors to exit 2 with their lines; a held lock stays exit 4."""

    def run(args: argparse.Namespace) -> int:
        from easel._common.config import ConfigError
        from easel.registry.shipped import DataError

        try:
            return handler(args)
        except ConfigError as exc:
            _err(exc.lines)
            return exc.code
        except DataError as exc:
            _err(exc.lines)
            return 2
        except LookupError as exc:
            _err([str(exc).strip("'\"")])
            return 2

    run.__doc__ = handler.__doc__
    return run


def _open(args: argparse.Namespace):
    from easel._common import config
    from easel.registry import resources as res

    cfg = config.load(args.tool, args.workspace, strict=True)
    return cfg, res.open_registry(cfg, args.tool)


def _build(args: argparse.Namespace) -> int:
    from easel.standards import cards

    cfg, reg = _open(args)
    report = cards.build(reg, cfg)
    _err(report.warnings)
    for line in report.lines(reg):
        print(line)
    return 0


def _explain(args: argparse.Namespace) -> int:
    from easel.standards import cards

    cfg, reg = _open(args)
    result = cards.explain(reg, cfg, args.sha16, as_json=args.json)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        for line in result:
            print(line)
    return 0


def _report(args: argparse.Namespace) -> int:
    from pathlib import Path

    from easel._common import store
    from easel.standards import cards

    cfg, reg = _open(args)
    text = "\n".join(cards.report(reg)) + "\n"
    if not args.out:
        print(text, end="")
        return 0
    out = Path(args.out).expanduser()
    if not out.is_absolute() and cfg.root is not None:
        out = cfg.root / out
    store.write_text(out, text)
    print(f"wrote {reg.shown(out)}")
    return 0


def _set(args: argparse.Namespace) -> int:
    from easel.standards import cards

    cfg, reg = _open(args)
    for line in cards.set_manual(reg, cfg, args.sha16, args.field, args.value, args.note):
        print(line)
    return 0


def _unset(args: argparse.Namespace) -> int:
    from easel.standards import cards

    cfg, reg = _open(args)
    for line in cards.unset_manual(reg, cfg, args.sha16, args.field, args.note):
        print(line)
    return 0
