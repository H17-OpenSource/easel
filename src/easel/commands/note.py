"""`easel note <id> ["<text>"]`: the argparse layer over `easel.note`.

Exit codes: 0 ok; 2 usage, configuration or data error, or an id that names nothing; 4 a lock is held.
"""

from __future__ import annotations

import argparse
import sys

HELP = "attach a short note to a resource, recipe or run"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `note`."""
    from easel._common.config import add_workspace_arg

    parser = subparsers.add_parser(
        "note", help=HELP,
        description="Attach a short note in the operator's words to a resource (sha16), a recipe (rcp-...), a "
                    "workflow template or a run (run-...). Without text, print the notes the id has. Notes are "
                    "appended, never replaced. A verdict is not a note: use `easel runs review`.")
    parser.add_argument("id", help="a run id, a recipe id, a resource sha16 (or 6+ hex digits of it) or a workflow id")
    parser.add_argument("text", nargs="?", help="the note, quoted; leave it out to print the id's notes")
    parser.add_argument("--assistant", action="store_true",
                        help="the note is the assistant's own, stored under policy.second_source_name, not the "
                             "operator's words")
    add_workspace_arg(parser)
    parser.set_defaults(func=_run)


def _err(lines) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _run(args: argparse.Namespace) -> int:
    from easel import note
    from easel._common import config, store
    from easel.registry.shipped import DataError, shown

    try:
        cfg = config.load(args.tool, args.workspace, strict=True)
        target = note.find_target(cfg, args.id)
        if args.text is None:
            lines = note.notes_of(cfg, target)
            for line in lines:
                print(line)
            print(f"{len(lines)} note{'' if len(lines) == 1 else 's'} on {target.kind} {target.id}")
            return 0
        source = str(cfg.get("policy.second_source_name", "assistant")) if args.assistant else "operator"
        note.add(cfg, target, args.text, source=source, tool=args.tool)
    except config.ConfigError as exc:
        _err(exc.lines)
        return exc.code
    except (note.NoteError, DataError) as exc:
        _err(exc.lines)
        return 2
    except store.Held as held:
        try:
            from easel._common import journal

            journal.note(f"stuck: {held.line}", tool=args.tool, journal_dir=cfg.path("paths.journal"), timeout=2.0)
        except Exception:  # noqa: BLE001  (journalling a held lock must never hide the exit 4)
            pass
        raise
    print(f"note added to {target.kind} {target.id} ({shown(target.file, cfg.root)}, by {source})")
    return 0
