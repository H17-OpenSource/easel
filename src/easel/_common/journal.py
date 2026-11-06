"""The day's journal (`journal note`, `journal stuck`, `journal report`) and the one STUCK template.

The journal is the one place an agent writes prose. Each day is `<instance>/journal/<YYYY-MM-DD>.md`, appended
under the journal folder's store lock, with times from the tool's clock (`<TOOL>_NOW` freezes it):
- `note "<text>"` adds `- HH:MM <text>`; `--pitfall P --rule R` adds the indented line `Pitfall: P Rule: R`;
- `stuck "<line>"` adds a STUCK block in the shape of STUCK_TEMPLATE (fields not given keep their placeholders,
  so the gaps show) and prints it, ready to paste as the one hand-off message; `stuck --template` only prints
  the empty template (the docs generator reads it so);
- `report [--date D] [--out F]` collects the day's STUCK blocks and bullets into one Markdown text.

STUCK_TEMPLATE is the single source of the STUCK shape: `docs/STUCK.md`, the trial brief and the pitfall issue
template are rendered from it by `tools/gen_formats_doc.py`, and a test checks that they agree.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from . import config, ids, store

STUCK_TEMPLATE = """\
## STUCK <date from the clock> <stage>
command: <exact>
exit code: <n>
output (last 20 lines): ...
expected: <runbook file:line> says ...
tried: 1) ... 2) ...
hypothesis: ...
need: <operator yes | image judgement | maintainer>
"""

STUCK_HEADER = STUCK_TEMPLATE.splitlines()[0]
# (label, placeholder) per field line, in template order, derived from the template so the two cannot drift.
STUCK_FIELDS: tuple[tuple[str, str], ...] = tuple(
    tuple(line.split(": ", 1)) for line in STUCK_TEMPLATE.splitlines()[1:]  # type: ignore[misc]
)
OUTPUT_LINES = 20
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def field_key(label: str) -> str:
    """The keyword for a template label: `exit code` -> `exit_code`, `output (last 20 lines)` -> `output`."""
    return label.split(" (", 1)[0].replace(" ", "_")


STUCK_KEYS: tuple[str, ...] = tuple(field_key(label) for label, _ in STUCK_FIELDS)


def last_lines(text: str, n: int = OUTPUT_LINES) -> list[str]:
    """The last `n` lines of `text`, trailing blank lines dropped."""
    lines = text.rstrip("\n").splitlines()
    return lines[-n:]


def render_stuck(when: str, stage: str | None = None, summary: str | None = None, **values: object) -> str:
    """A STUCK block: the template with the given fields filled in; `output` is indented, its last 20 lines only.

    `values` keys are STUCK_KEYS (command, exit_code, output, expected, tried, hypothesis, need); `tried` may be
    a list, rendered `1) ... 2) ...`.
    """
    unknown = set(values) - set(STUCK_KEYS)
    if unknown:
        raise ValueError(f"not STUCK fields: {', '.join(sorted(unknown))}")
    header = STUCK_HEADER.replace("<date from the clock>", when).replace("<stage>", stage or "<stage>")
    out = [header]
    if summary:
        out.append(f"summary: {summary}")
    for (label, placeholder), key in zip(STUCK_FIELDS, STUCK_KEYS, strict=True):
        value = values.get(key)
        if value is None or value == [] or value == "":
            out.append(f"{label}: {placeholder}")
        elif key == "output":
            out.append(f"{label}:")
            out.extend(f"    {line}" for line in last_lines(str(value)))
        elif key == "tried" and isinstance(value, list | tuple):
            out.append(f"{label}: " + " ".join(f"{i}) {item}" for i, item in enumerate(value, 1)))
        else:
            out.append(f"{label}: {value}")
    return "\n".join(out) + "\n"


# --- writing --------------------------------------------------------------------------------------------------

def day_file(journal_dir: Path, day: str) -> Path:
    return Path(journal_dir) / f"{day}.md"


def _append(journal_dir: Path, tool: str, text: str, timeout: float | None) -> Path:
    """Append to today's file under the journal's store lock, writing the day header first on a new file."""
    journal_dir = Path(journal_dir)
    path = day_file(journal_dir, ids.today(tool))
    with store.lock(journal_dir, tool, timeout):
        if not path.exists() or path.stat().st_size == 0:
            text = f"# Journal {ids.today(tool)}\n\n" + text.lstrip("\n")
        store.append_bytes_locked(path, text.encode("utf-8"))
    return path


def note(text: str, *, tool: str, journal_dir: Path, pitfall: str | None = None, rule: str | None = None,
         timeout: float | None = None) -> Path:
    """Append `- HH:MM <text>` (and an indented `Pitfall: ... Rule: ...` line) to today's journal."""
    text = text.strip()
    if not text:
        raise ValueError("a journal note needs text")
    if (pitfall is None) != (rule is None):
        raise ValueError("a pitfall needs its rule: pass both --pitfall and --rule")
    lines = text.splitlines()
    entry = f"- {ids.now(tool).strftime('%H:%M')} {lines[0]}\n" + "".join(f"  {line}\n" for line in lines[1:])
    if pitfall is not None:
        entry += f"  Pitfall: {pitfall.strip()} Rule: {str(rule).strip()}\n"
    return _append(journal_dir, tool, entry, timeout)


def stuck(summary: str, *, tool: str, journal_dir: Path, stage: str | None = None, timeout: float | None = None,
          **values: object) -> tuple[Path, str]:
    """Append a STUCK block to today's journal; returns (file, block)."""
    when = ids.now(tool).strftime("%Y-%m-%d %H:%M")
    block = render_stuck(when, stage, summary.strip() or None, **values)
    return _append(journal_dir, tool, "\n" + block + "\n", timeout), block


# --- reading --------------------------------------------------------------------------------------------------

def split_day(text: str) -> tuple[list[str], list[str]]:
    """(STUCK blocks, bullets) of one day's journal text, each with its continuation lines.

    An open block or bullet continues through every line that starts with whitespace, so a blank line of captured
    output (written as an indented blank) or of a multi-line note never cuts it short; only an empty line, a `#`
    heading, the next bullet or the next STUCK block ends it.
    """
    blocks: list[list[str]] = []
    bullets: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if line.startswith("## STUCK"):
            current = [line]
            blocks.append(current)
        elif line.startswith("- "):
            current = [line]
            bullets.append(current)
        elif current is not None and line[:1].isspace():
            current.append(line)
        elif not line.strip() or line.startswith("#"):
            current = None
        elif current is not None:
            current.append(line)

    def joined(lines: list[str]) -> str:
        while len(lines) > 1 and not lines[-1].strip():
            lines = lines[:-1]
        return "\n".join(lines)

    return [joined(b) for b in blocks], [joined(b) for b in bullets]


def report(*, tool: str, journal_dir: Path, day: str | None = None) -> str:
    """The day's STUCK blocks and bullets as one Markdown text (the trial report, the hand-off's tail)."""
    day = day or ids.today(tool)
    if not _DAY.match(day):
        raise ValueError(f"a day is YYYY-MM-DD: {day!r}")
    path = day_file(journal_dir, day)
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    blocks, bullets = split_day(text)
    out = [f"# Journal report {day}", ""]
    if not path.exists():
        out += [f"no journal file for {day}", ""]
    out += [f"## STUCK notes: {len(blocks)}", ""]
    for block in blocks:
        out += ["#" + block, ""]  # one level down under the report's own heading
    out += [f"## Bullets: {len(bullets)}", ""]
    out += bullets
    return "\n".join(out).rstrip("\n") + "\n"


# --- the `journal` verb ---------------------------------------------------------------------------------------

def _where(args: argparse.Namespace) -> tuple[Path, Path | None]:
    """(journal folder, instance root or None): `--dir` wins, else the instance's `paths.journal`."""
    if args.dir:
        return Path(args.dir).expanduser(), None
    cfg = config.load(args.tool, args.workspace, strict=False)
    if cfg.file is None:
        raise config.ConfigError(2, [cfg.not_found_line(), "pass --dir <folder> or --workspace <instance>"])
    value = cfg.get("paths.journal", "journal")
    folder = Path(str(value)).expanduser()
    return (folder if folder.is_absolute() else cfg.root / folder), cfg.root


def _shown(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.resolve().relative_to(root).as_posix()
        except ValueError:
            pass
    return str(path)


def _run(args: argparse.Namespace, action) -> int:
    try:
        return action(*_where(args))
    except config.ConfigError as exc:
        for line in exc.lines:
            print(line, file=sys.stderr)
        return exc.code
    except (ValueError, OSError) as exc:  # IdError is a ValueError
        print(f"refused: {exc}", file=sys.stderr)
        return 2


def _cmd_note(args: argparse.Namespace) -> int:
    def act(folder: Path, root: Path | None) -> int:
        path = note(args.text, tool=args.tool, journal_dir=folder, pitfall=args.pitfall, rule=args.rule)
        print(f"noted in {_shown(path, root)}")
        return 0

    return _run(args, act)


def _cmd_stuck(args: argparse.Namespace) -> int:
    if args.template:  # the one STUCK shape, for generators; nothing is read or written
        print(STUCK_TEMPLATE, end="")
        return 0
    if not (args.line or "").strip():
        print(f"refused: `{args.tool} journal stuck` needs a line saying what is stuck (or --template)",
              file=sys.stderr)
        return 2

    def act(folder: Path, root: Path | None) -> int:
        output = args.output
        if output == "-":
            output = sys.stdin.read()
        if args.output_file:
            output = Path(args.output_file).read_text(encoding="utf-8", errors="replace")
        path, block = stuck(args.line, tool=args.tool, journal_dir=folder, stage=args.stage, command=args.command,
                            exit_code=args.exit_code, output=output, expected=args.expected,
                            tried=args.tried or None, hypothesis=args.hypothesis, need=args.need)
        print(block, end="")
        print(f"recorded in {_shown(path, root)}")
        return 0

    return _run(args, act)


def _cmd_report(args: argparse.Namespace) -> int:
    def act(folder: Path, root: Path | None) -> int:
        text = report(tool=args.tool, journal_dir=folder, day=args.date)
        if not args.out:
            print(text, end="")
            return 0
        out = Path(args.out).expanduser()
        if not out.is_absolute():
            out = (root or Path.cwd()) / out
        store.write_text(out, text)
        print(f"report written to {_shown(out, root)}")
        return 0

    return _run(args, act)


def _place_args(parser: argparse.ArgumentParser) -> None:
    config.add_workspace_arg(parser)
    parser.add_argument("--dir", metavar="FOLDER", default=None,
                        help="write to this journal folder instead of the instance's paths.journal")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `journal note`, `journal stuck` and `journal report` to the tool's command line."""
    parser = subparsers.add_parser(
        "journal", help="write the day's journal: note, stuck, report",
        description="The day's journal in the instance: journal/<YYYY-MM-DD>.md. Times come from the clock.")
    inner = parser.add_subparsers(dest="journal_action", metavar="<action>")

    p = inner.add_parser("note", help="add one timestamped bullet to today's journal")
    p.add_argument("text", help="what happened, in plain words; quote the user when it is their verdict")
    p.add_argument("--pitfall", help="a pitfall met on the way; needs --rule")
    p.add_argument("--rule", help="the rule that avoids the pitfall next time")
    _place_args(p)
    p.set_defaults(func=_cmd_note)

    s = inner.add_parser("stuck", help="add a STUCK block to today's journal and print it for the hand-off",
                         description="Record being stuck in the one STUCK shape; fields left out keep their "
                                     "placeholders so the gaps show.")
    s.add_argument("line", nargs="?", default=None, help="one line: what is stuck")
    s.add_argument("--template", action="store_true",
                   help="print the empty STUCK template and exit; writes nothing")
    s.add_argument("--stage", help="the runbook stage, for example 05-captions")
    s.add_argument("--command", help="the exact command that failed")
    s.add_argument("--exit-code", dest="exit_code", help="its exit code")
    s.add_argument("--output", help="its output text; - reads it from stdin; the last 20 lines are kept")
    s.add_argument("--output-file", dest="output_file", help="read the output from this file instead")
    s.add_argument("--expected", help="what the runbook says should have happened: <file:line> says ...")
    s.add_argument("--tried", action="append", help="one thing tried; repeat for each, in order")
    s.add_argument("--hypothesis", help="the best guess at the cause")
    s.add_argument("--need", help="operator yes, image judgement or maintainer")
    _place_args(s)
    s.set_defaults(func=_cmd_stuck)

    r = inner.add_parser("report", help="collect a day's STUCK blocks and bullets into one text")
    r.add_argument("--date", help="the day, YYYY-MM-DD; default today")
    r.add_argument("--out", help="write the report to this file, relative to the instance root; default: print it")
    _place_args(r)
    r.set_defaults(func=_cmd_report)
