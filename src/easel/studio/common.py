"""What every studio verb shares: the instance's folders, its policy, the adopted-workspace refusal, the errors.

`open_studio(tool, workspace)` loads the instance configuration once and names its folders (each a `paths.*`
key, relative to the instance root unless absolute): `recipes/`, `runs/`, `jobs/`, `runtime/`, `standards/`
and the registry. Keys an older instance file leaves out fall back to the default folder names (`DEFAULT_FOLDERS`).

An instance is *adopted* when `instance.adopted = true` or `policy.render = "legacy-runner"`: an existing
workspace keeps its own render runner until cutover. Under `legacy-runner` the GPU verbs (`easel run`,
`easel jobs submit|run`, `easel comfy start|stop`) refuse with exit 2 and the one line of `REFUSED_LEGACY`,
pointing at the host commands `easel recipe to-legacy <rcp>` prints (docs/runbooks/90-adopt-existing.md).

`StudioError(code, lines)` is the one failure type of the studio: `guarded(handler)` prints its lines on stderr
and returns its code, and maps the shared modules' errors the same way (config, data, graph, ComfyUI,
scheduler). A held store lock (exit 4) is written into today's journal, then re-raised.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from easel._common import config, store

TOOL = "easel"
REPO = Path(__file__).resolve().parents[3]  # src/easel/studio/common.py -> the repository root
REPO_FAMILIES = REPO / "families"
REFUSED_LEGACY = "adopted workspace: run the lines `easel recipe to-legacy <rcp>` prints (runbook 90)"
EXIT_OK, EXIT_PROBLEMS, EXIT_USAGE, EXIT_WAITING = 0, 1, 2, 3
DEFAULT_FOLDERS = {"recipes": "recipes", "runs": "runs", "jobs": "jobs", "runtime": "runtime",
                   "standards": "standards", "registry": "registry", "journal": "journal"}


class StudioError(Exception):
    """A refusal or failure with its exit code and the lines to print (what happened, then `next:` or `fix:`)."""

    def __init__(self, code: int, lines: str | Iterable[str]) -> None:
        self.code = int(code)
        self.lines = [lines] if isinstance(lines, str) else list(lines)
        super().__init__("\n".join(self.lines))


@dataclass
class Studio:
    """One loaded instance: its configuration and the folders the studio reads and writes."""

    cfg: config.Config
    tool: str
    root: Path

    def folder(self, name: str) -> Path:
        """The folder of `paths.<name>`, relative to the instance root unless written absolute."""
        value = Path(str(self.cfg.get(f"paths.{name}", DEFAULT_FOLDERS.get(name, name)))).expanduser()
        return value if value.is_absolute() else self.root / value

    @property
    def recipes(self) -> Path:
        return self.folder("recipes")

    @property
    def runs(self) -> Path:
        return self.folder("runs")

    @property
    def jobs(self) -> Path:
        return self.folder("jobs")

    @property
    def runtime(self) -> Path:
        return self.folder("runtime")

    @property
    def journal(self) -> Path:
        return self.folder("journal")

    def shown(self, path: Path) -> str:
        """A path for output: relative to the instance root (never absolute), else the file name."""
        try:
            return Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        except (ValueError, OSError):
            return Path(path).name

    def policy(self, key: str, default: str) -> str:
        return str(self.cfg.get(f"policy.{key}", default) or default)

    @property
    def adopted(self) -> bool:
        return is_adopted(self.cfg)

    @property
    def legacy_runner(self) -> bool:
        return legacy_runner(self.cfg)

    def families_dirs(self) -> list[Path]:
        """Where family files are looked up: the instance's `families/` first, then the repository's."""
        return [self.root / "families", REPO_FAMILIES]

    def families_dir_for(self, family: str | None) -> Path | None:
        """The folder whose `<family>.yaml` the scheduler reads (the instance's when it has one), else None."""
        if family and (self.root / "families" / f"{family}.yaml").is_file():
            return self.root / "families"
        return None


def open_studio(tool: str = TOOL, workspace: str | Path | None = None, *, strict: bool = True) -> Studio:
    """Load the instance configuration; StudioError(2) when no instance file is found."""
    try:
        cfg = config.load(tool, workspace, strict=strict)
    except config.ConfigError as exc:
        raise StudioError(exc.code, exc.lines) from exc
    if cfg.file is None or cfg.root is None:
        from easel.cli import instance_not_found, new_instance_hint

        raise StudioError(EXIT_USAGE, [instance_not_found(tool), f"next: {new_instance_hint(tool)}"])
    return Studio(cfg=cfg, tool=tool, root=cfg.root)


def is_adopted(cfg: config.Config) -> bool:
    """True on an adopted workspace: `instance.adopted = true` or `policy.render = "legacy-runner"`."""
    return bool(cfg.get("instance.adopted", False)) or legacy_runner(cfg)


def legacy_runner(cfg: config.Config) -> bool:
    """True when the host's own runner renders (`policy.render = "legacy-runner"`)."""
    return str(cfg.get("policy.render", "native")) == "legacy-runner"


def refuse_on_legacy(studio: Studio, recipe_id: str | None = None) -> None:
    """Exit 2 with the to-legacy hint when this instance renders through the host's runner."""
    if studio.legacy_runner:
        lines = [REFUSED_LEGACY]
        lines.append(f"next: easel recipe to-legacy {recipe_id}" if recipe_id else
                     "next: read docs/runbooks/90-adopt-existing.md")
        raise StudioError(EXIT_USAGE, lines)


# --- records ------------------------------------------------------------------------------------------------------

def schema_errors(name: str, data: object) -> list[str]:
    """Problems of `data` against `schemas/<name>.schema.json` (the registry's shared validator)."""
    from easel.registry.shipped import schema_errors as errors

    return errors(name, data)


def check_record(name: str, data: Mapping, what: str) -> None:
    """StudioError(2) naming the first problems when `data` fails its schema; nothing is written then."""
    problems = schema_errors(name, data)
    if problems:
        raise StudioError(EXIT_USAGE, [f"{what}: not written, it fails schemas/{name}.schema.json:",
                                       *(f"  {p}" for p in problems[:5])])


def read_json(path: Path, what: str) -> dict:
    """A JSON record; StudioError(2) when it is missing or unreadable."""
    try:
        return store.read_record(path)
    except FileNotFoundError:
        raise StudioError(EXIT_USAGE, f"no {what}") from None
    except (OSError, ValueError) as exc:
        from easel.registry.shipped import reason

        raise StudioError(EXIT_USAGE, f"{what} is not readable: {reason(exc)}") from exc


def dump(obj: object) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def value_text(value: object) -> str:
    """A value for a line: numbers as written, None as `-`."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# --- the argparse layer -------------------------------------------------------------------------------------------

def err(lines: Iterable[str]) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def guarded(handler: Callable[[argparse.Namespace], int | None]) -> Callable[[argparse.Namespace], int]:
    """Map every known failure to its exit code and lines; a held lock is journalled, then re-raised (exit 4)."""

    def run(args: argparse.Namespace) -> int:
        from easel._common import comfy_graph, scheduler
        from easel._common.comfy_client import ComfyError
        from easel.registry.shipped import DataError

        try:
            code = handler(args)
        except StudioError as exc:
            err(exc.lines)
            return exc.code
        except config.ConfigError as exc:
            err(exc.lines)
            return exc.code
        except (DataError, comfy_graph.GraphError) as exc:
            err(exc.lines)
            return EXIT_USAGE
        except scheduler.SchedulerError as exc:
            err(exc.lines)
            return exc.code
        except ComfyError as exc:
            err(exc.lines)
            return EXIT_PROBLEMS
        except store.Held as held:
            journal_held(args, held)
            raise
        return EXIT_OK if code is None else int(code)

    run.__doc__ = handler.__doc__
    return run


def journal_held(args: argparse.Namespace, held: store.Held) -> None:
    """Best effort: the exit-4 line into today's journal of the instance (never raises)."""
    try:
        from easel._common import journal

        cfg = config.load(getattr(args, "tool", TOOL), getattr(args, "workspace", None), strict=False)
        journal.note(f"stuck: {held.line}", tool=getattr(args, "tool", TOOL), journal_dir=cfg.path("paths.journal"),
                     timeout=2.0)
    except Exception:  # noqa: BLE001  (journalling a held lock must never hide the exit 4)
        pass


def add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    config.add_workspace_arg(parser)
