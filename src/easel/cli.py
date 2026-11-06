"""The `easel` command line: one table of verbs, each verb in its own module, imported only when it runs."""

# How to add or build a verb (keep this block short and true):
# 1. Every top-level verb has exactly one row in COMMANDS: (verb, "module.path", help, planned-task).
# 2. A built verb is a module that exposes `register(subparsers)`. It calls
#    `subparsers.add_parser("<verb>", help=...)`, adds its subverbs and options, and sets
#    `set_defaults(func=handler)`. `handler(args)` returns an int exit code (None means 0); `args.tool` is "easel".
# 3. Default home: `easel/commands/<verb>.py` (thin argparse layer; logic lives in the domain modules). Verbs
#    shared with kiln live in `easel._common.<module>` (the source of truth, synced to kiln).
# 4. Until its module exists, a verb prints `planned: <task> (<help>)` and exits 2. Building it = creating the
#    module; edit the row only to move the module or fix the help. One module per verb; lazy import means a
#    broken verb cannot break the others, so keep module-level imports cheap.
# 5. tests/test_cli.py checks every row. Exit codes: 0 ok, 1 a check found problems, 2 usage error, refusal or
#    planned verb, 3 deferred by a guard, 4 a lock or file is held by another process.

import argparse
import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Sequence
from typing import NamedTuple

from . import __version__

PKG = "easel"


class Command(NamedTuple):
    verb: str
    module: str
    help: str
    planned: str  # the task that builds it: a runbook task id, "session-2", "v0.2" or "later"


COMMANDS: tuple[Command, ...] = (
    # start, orientation, configuration
    Command("init", "easel.commands.init", "create an easel home: easel.toml, registry/, runs/, recipes/, journal/", "session-2"),
    Command("doctor", "easel.commands.doctor", "check the install, the instance and ComfyUI: ok|warn|FAIL per line, then next:", "T3"),
    Command("next", "easel.commands.next", "print the current stage, the runbook to read and the next command", "T3"),
    Command("config", "easel._common.config", "show or set configuration values with their source", "T1a"),
    Command("privacy", "easel._common.privacy", "scan files for private values before they leave the machine", "T1a"),
    Command("journal", "easel._common.journal", "write the day's journal: note, stuck, report", "T1a"),
    Command("qa", "easel.commands.qa", "check docs, agent files and tests against the repo's rules", "T13"),
    # shipped data and the model registry
    Command("families", "easel.commands.families", "check the model family files: families check", "T6"),
    Command("registry", "easel.commands.registry", "the model registry: scan, list, show, explain, compatible, index, add, download, audit, dups, register-upstream", "T2"),
    Command("index", "easel.commands.index", "rebuild the generated views: index.jsonl, INDEX.md", "T2"),
    Command("standards", "easel.commands.standards", "usage standards per resource: build, explain, report, set, unset", "T2b"),
    Command("search", "easel.commands.search", "search Civitai and Hugging Face for candidates; vet a candidate", "T4"),
    # recipes, rendering, review
    Command("recipe", "easel.commands.recipe", "propose, revise, like, show, list and approve recipes; to-legacy on an adopted instance", "T3"),
    Command("run", "easel.commands.run", "render an approved recipe through the scheduler; --dry-run prints the graph", "T3"),
    Command("runs", "easel.commands.runs", "list, show, compare and review runs; import-legacy on an adopted instance", "T3"),
    Command("jobs", "easel.commands.jobs", "submit, inspect and cancel GPU jobs", "T3"),
    Command("comfy", "easel.commands.comfy", "start, stop and check a managed ComfyUI; native instances only", "T3"),
    Command("workflows", "easel.commands.workflows", "register, validate, approve and check workflow templates", "T4"),
    Command("lessons", "easel.commands.lessons", "record a lesson as finding, evidence and rule; render the lessons docs", "T4"),
    Command("note", "easel.commands.note", "attach a short note to a resource, recipe or run", "T4"),
    Command("review", "easel.commands.review", "serve the local review page, blurred until revealed", "session-2"),
    Command("references", "easel.commands.references", "capture reference images and the resources they used", "session-2"),
    Command("runtime", "easel.commands.runtime", "install ComfyUI and the node packs pinned in runtime/comfyui.lock.json", "v0.2"),
    Command("agents", "easel.commands.agents", "generate host adapter files from AGENTS.md", "later"),
)

_HELP_FLAGS = {"-h", "--help", "--version"}


def version() -> str:
    """The installed distribution's version, or the source tree's `__version__` when not installed."""
    try:
        return importlib.metadata.version(PKG)
    except importlib.metadata.PackageNotFoundError:
        return __version__


def module_exists(name: str) -> bool:
    """True when the verb's module can be found; a missing parent package (for example `_common`) counts as absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def planned_task(verb: str) -> str | None:
    """None when `verb` is built, else the task its COMMANDS row names (so a hint never sends anyone to a stub)."""
    cmd = next((c for c in COMMANDS if c.verb == verb), None)
    if cmd is None:
        return "later"
    return None if module_exists(cmd.module) else cmd.planned


def new_instance_hint(tool: str = PKG) -> str:
    """How an instance file comes to exist: `<tool> init <folder>` once init is built, else the operator's copy."""
    task = planned_task("init")
    if task is None:
        return f"create one with `{tool} init <folder>`, or pass --workspace <folder>"
    return (f"the operator copies {tool}.example.toml to <home>/{tool}.toml and sets {tool.upper()}_HOME=<home> "
            f"(`{tool} init` is planned: {task}), or pass --workspace <folder>")


def instance_not_found(tool: str = PKG) -> str:
    """The no-instance line in words: where discovery looks, never the absolute folders it tried (those would
    put the machine's paths into output that lands in hand-offs and docs)."""
    return (f"no {tool}.toml found (looked at --workspace, ${tool.upper()}_HOME, the current folder and its parents, "
            f"~/.config/{tool}/)")


def _first_verb(argv: Sequence[str]) -> tuple[str | None, bool]:
    """The first non-option word, and whether a top-level -h/--help/--version came before it."""
    for i, word in enumerate(argv):
        if not word.startswith("-"):
            return word, any(a in _HELP_FLAGS for a in argv[:i])
    return None, any(a in _HELP_FLAGS for a in argv)


def build_parser(verb: str | None = None) -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    """The top-level parser. Only `verb`'s module is imported; every other verb gets a help-only placeholder."""
    parser = argparse.ArgumentParser(
        prog=PKG,
        description="easel finds models, writes recipes and renders images with ComfyUI, with plain files as the registry. Agents: read AGENTS.md first.",
        epilog="A verb marked [planned: <task>] prints that line and exits 2 until it is built.",
    )
    parser.add_argument("--version", action="version", version=f"{PKG} {version()}")
    parser.set_defaults(tool=PKG, func=None)
    sub = parser.add_subparsers(dest="verb", metavar="<verb>", title="verbs")
    for cmd in COMMANDS:
        built = module_exists(cmd.module)
        if built and cmd.verb == verb:
            importlib.import_module(cmd.module).register(sub)
            if verb not in sub.choices:
                raise RuntimeError(f"{cmd.module}.register() did not add the verb {verb!r}")
        else:
            label = cmd.help if built else f"{cmd.help} [planned: {cmd.planned}]"
            sub.add_parser(cmd.verb, help=label, add_help=False)
    return parser, sub


def main(argv: Sequence[str] | None = None) -> int:
    """Run one easel command and return its exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    by_verb = {cmd.verb: cmd for cmd in COMMANDS}
    verb, help_first = _first_verb(args_list)
    cmd = by_verb.get(verb) if verb else None
    if cmd is not None and not help_first and not module_exists(cmd.module):
        print(f"planned: {cmd.planned} ({cmd.help})")
        return 2
    parser, sub = build_parser(cmd.verb if cmd else None)
    args = parser.parse_args(args_list)
    if args.verb is None:
        parser.print_help()
        return 2
    if args.func is None:  # a built verb called without one of its subverbs
        sub.choices[args.verb].print_help()
        return 2
    code = args.func(args)
    return 0 if code is None else int(code)


if __name__ == "__main__":
    sys.exit(main())
