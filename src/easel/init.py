"""`easel init <dir>`: create an easel home (an instance folder) from the shipped example configuration.

What it writes, and why each piece is there:
- `.gitignore` with `*` and `!README.md`, written first: an instance holds private files (paths, model names,
  verdicts), so if the folder ever becomes a repository nothing but the README can be committed by accident;
- `README.md`: what the folder is, what each part holds and how the tools find it;
- the folders the `[paths]` keys of the example name: `registry/`, `runs/`, `recipes/`, `journal/` (relative to
  the instance; a value that is absolute or holds a `${VAR}` lies outside the folder and is left alone);
- `easel.toml`, written last: the text of `easel.example.toml` from this checkout, every key and comment kept,
  with a first line that says it is this instance's copy. `easel config set` then edits it in place.
  Writing it last keeps one rule true: a folder that holds `easel.toml` is a finished home.

Refusals (exit 2, nothing written):
- a target inside a repository checkout (a `.git`, `.hg` or `.svn` at or above it, or a `pyproject.toml` beside
  a `src/` folder): a checkout is shared, an instance is private;
- a target that is a file;
- a folder that already holds a non-empty `easel.toml`: it is a home already, so point the tools at it instead;
- any other non-empty folder, unless `--force-existing`: then the missing pieces are added and no existing file
  is overwritten (an empty `easel.toml` counts as missing).

The output names what was created and kept, how the tools find the folder, and ends with
`next: easel doctor --workspace <dir>`. The folder is printed as it was typed (never resolved), so the output
holds no absolute path unless the person typed one.

This file has a twin in the other tool (same rules, other constants); keep the two in step.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import config, ids, store

# --- the constants that differ between the two tools' copies -------------------------------------------------
PKG = "easel"
ARTICLE = "an"  # "an easel home", "a kiln home"
FOLDERS: dict[str, str] = {  # [paths] key -> what the folder holds (a README row)
    "registry": "one record per model file; `easel registry scan --hash` fills it",
    "runs": "one folder per render run: its graph, outputs and reviews",
    "recipes": "one file per recipe; `easel recipe propose` writes them",
    "journal": "the day's journal; `easel journal note`, `stuck` and `report` write it",
}
WHAT = "the configuration, model registry, recipes, render runs and journal of one machine"
# --------------------------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]  # src/<pkg>/init.py -> the repository checkout
EXAMPLE = REPO / f"{PKG}.example.toml"
CONFIG_NAME = f"{PKG}.toml"
HOME_VAR = f"{PKG.upper()}_HOME"
GITIGNORE = "*\n!README.md\n"
CHECKOUT_MARKERS = (".git", ".hg", ".svn")
EXIT_REFUSED = 2


class InitError(Exception):
    """A refusal: `code` is the exit code, `lines` say what is wrong and what to do; nothing was written."""

    def __init__(self, code: int, lines: list[str]) -> None:
        super().__init__("\n".join(lines))
        self.code = code
        self.lines = lines


@dataclass
class Target:
    """The checked target folder and what init did there."""

    path: Path
    shown: str  # as typed on the command line
    created: list[str] = field(default_factory=list)
    kept: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def quoted(text: str) -> str:
    """`text` ready to paste into a shell line: wrapped in double quotes when it holds a space or a quote."""
    return f'"{text}"' if any(ch in text for ch in ' \t"\'') else text


def checkout_above(folder: Path) -> tuple[Path, str] | None:
    """The nearest repository checkout at or above `folder` and the marker that shows it, or None."""
    for candidate in (folder, *folder.parents):
        for marker in CHECKOUT_MARKERS:
            if (candidate / marker).exists():
                return candidate, marker
        if (candidate / "pyproject.toml").is_file() and (candidate / "src").is_dir():
            return candidate, "pyproject.toml and src/"
    return None


def has_config(path: Path) -> bool:
    """True when `path` is a config file with content (an empty or blank file does not count)."""
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8-sig").strip())
    except (OSError, UnicodeDecodeError):
        return True  # unreadable: never overwrite it


def example_text(example: Path) -> str:
    """The shipped example configuration, checked: it parses and passes the config schema and rules."""
    if not example.is_file():
        raise InitError(EXIT_REFUSED, [
            f"refused: {example.name} is not next to this install, so there is nothing to start {CONFIG_NAME} "
            "from; nothing was written",
            f"install {PKG} from a checkout (pip install -e .), then run {PKG} init again"])
    text = example.read_text(encoding="utf-8-sig")
    try:
        values = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InitError(EXIT_REFUSED, [f"refused: {example.name} is not valid TOML ({exc}); nothing was written",
                                       "report it to the maintainer"]) from exc
    problems = config.validate(values, config.load_schema())
    if problems:
        raise InitError(EXIT_REFUSED, [f"refused: {example.name} does not pass the config schema; nothing was "
                                       "written; report it to the maintainer", *problems])
    return text


def instance_text(text: str, example_name: str) -> str:
    """The example's text with its first line replaced by a line that says this is the instance's own copy."""
    from easel import cli

    head = (f"# {CONFIG_NAME}: this instance's configuration, written by `{PKG} init` ({PKG} {cli.version()}, "
            f"{ids.today(PKG)}) from {example_name}; change values with `{PKG} config set <section.key> <value>`.")
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith(f"# {example_name}"):
        lines = lines[1:]
    return head + "\n" + "".join(lines)


def readme_text(folders: dict[str, str]) -> str:
    """The instance README: what the folder is, what each part holds, how the tools find it."""
    from easel import cli

    rows = [f"| `{CONFIG_NAME}` | the configuration; `{PKG} config show` prints every value with its source, "
            f"`{PKG} config set <section.key> <value>` changes one |"]
    rows += [f"| `{rel}/` | {FOLDERS[key]} |" for key, rel in folders.items()]
    return "\n".join([
        f"# {PKG} home",
        "",
        f"This folder is {ARTICLE} {PKG} instance: {WHAT}. It is private and lives outside every repository "
        "checkout. Its `.gitignore` ignores everything except this README, so nothing else can be committed by "
        "accident.",
        "",
        "| path | what it holds |",
        "|---|---|",
        *rows,
        "",
        f"{PKG} finds this folder when you work inside it, when `{HOME_VAR}` names it, or with "
        f"`--workspace <this folder>`. Start with `{PKG} doctor`, then `{PKG} next`.",
        "",
        f"Created by `{PKG} init` ({PKG} {cli.version()}) on {ids.today(PKG)}.",
        "",
    ])


def folder_layout(text: str) -> tuple[dict[str, str], list[str]]:
    """The `[paths]` folders inside the instance ({key: relative folder}) and a note per folder left out."""
    paths = tomllib.loads(text).get("paths") or {}
    root = str(paths.get("root", ".") or ".")
    out: dict[str, str] = {}
    notes: list[str] = []
    for key in FOLDERS:
        value = str(paths.get(key, key) or key)
        if "${" in value or Path(value).expanduser().is_absolute() or config.ABSOLUTE.search(value):
            notes.append(f"paths.{key} lies outside the instance (<{key}>): not created here")
            continue
        out[key] = (Path(root) / value).as_posix()
    return out, notes


def check_target(raw_target: str, *, force_existing: bool = False) -> Target:
    """Check the target folder; InitError when init must refuse. Nothing is written here."""
    shown = str(raw_target)
    path = Path(raw_target).expanduser().resolve()
    found = checkout_above(path)
    if found is not None:
        checkout, marker = found
        raise InitError(EXIT_REFUSED, [
            f"refused: {shown} lies inside a repository checkout ({checkout.name}/, found by {marker}); nothing "
            "was written",
            "an instance holds private files and lives outside every checkout: pick another folder, for example "
            f"~/{PKG}-home"])
    if path.exists() and not path.is_dir():
        raise InitError(EXIT_REFUSED, [f"refused: {shown} is a file, not a folder; nothing was written",
                                       f"pick a new or empty folder, for example ~/{PKG}-home"])
    if has_config(path / CONFIG_NAME):
        raise InitError(EXIT_REFUSED, [
            f"refused: {shown} is already {ARTICLE} {PKG} home (it holds {CONFIG_NAME}); nothing was changed",
            f"use it: {PKG} doctor --workspace {quoted(shown)}, or set {HOME_VAR} to it"])
    entries = sorted(p.name for p in path.iterdir()) if path.is_dir() else []
    if entries and not force_existing:
        count = f"{len(entries)} {'entry' if len(entries) == 1 else 'entries'}"
        raise InitError(EXIT_REFUSED, [
            f"refused: {shown} is not empty ({count}) and holds no {CONFIG_NAME}; nothing was written",
            f"pick a new or empty folder, for example ~/{PKG}-home, or pass --force-existing to add the instance "
            "files to this folder (no existing file is overwritten)"])
    return Target(path=path, shown=shown)


def _write_once(target: Target, name: str, text: str) -> None:
    """Write `name` under the target unless it exists (then it is kept as it is)."""
    path = target.path / name
    if path.exists():
        target.kept.append(name)
        return
    store.write_text(path, text)
    target.created.append(name)


def init(raw_target: str, *, force_existing: bool = False, example: Path | None = None) -> list[str]:
    """Create the home at `raw_target` (`~` is expanded) and return the lines to print; InitError refuses."""
    target = check_target(raw_target, force_existing=force_existing)
    example = EXAMPLE if example is None else Path(example)
    text = example_text(example)
    folders, target.notes = folder_layout(text)
    target.path.mkdir(parents=True, exist_ok=True)

    _write_once(target, ".gitignore", GITIGNORE)
    if ".gitignore" in target.kept:
        kept = (target.path / ".gitignore").read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if "*" not in (line.strip() for line in kept):
            target.notes.append("warn: the existing .gitignore has no `*` line, so this folder's private files "
                                "could be committed if it ever becomes a repository; tell the operator")
    _write_once(target, "README.md", readme_text(folders))
    for rel in folders.values():
        folder = target.path / rel
        if folder.is_dir():
            target.kept.append(f"{rel}/")
        else:
            folder.mkdir(parents=True, exist_ok=True)
            target.created.append(f"{rel}/")
    config_file = target.path / CONFIG_NAME
    if has_config(config_file):  # appeared while init ran: never overwrite it
        target.kept.append(CONFIG_NAME)
    else:
        store.write_text(config_file, instance_text(text, example.name))
        target.created.append(CONFIG_NAME)

    lines = [f"{PKG} home: {target.shown}", f"created: {', '.join(target.created) or 'nothing'}"]
    if target.kept:
        lines.append(f"kept (already there, not changed): {', '.join(target.kept)}")
    lines += target.notes
    lines.append(f"{PKG} finds it when you work inside it, when {HOME_VAR} names it, or with --workspace "
                 f"{quoted(target.shown)}")
    lines.append(f"next: {PKG} doctor --workspace {quoted(target.shown)}")
    return lines
