"""`<tool> qa docs|agents|tests`: mechanical checks of the docs, the agent rule file and the tests.

Why: agents, often weaker models, work from the docs. A doc that names a flag the CLI does not have, a file that
moved, a direct GPU command or an output block with a real date makes them fail or improvise. People miss that
drift; a mechanical check does not. Everything repository-specific (the seven runbook headings, the forbidden
commands, the line limits) is data in `qa.toml` at the repository root. This module holds only the logic and is
the same file in both packages; the tool name comes from the package it runs in.

- `qa docs`: the seven runbook sections in order and the entry-condition table; every `<tool> ...` command a doc
  shows (inline code or a fenced command line) against the CLI's own parser, and every `python tools/<x>.py`
  line against that script's `--help`, both through a help cache; on a fenced command line also every option
  the verb declares required (`required=True` or a required mutually exclusive group); repository paths and
  relative links that must exist; no direct GPU command on a fenced command line, and (in the files
  `gpu_actions_in` names) no table action that starts, stops or restarts GPU work; no absolute path anywhere;
  in the output scope (runbooks and the formats page) no fenced line with a date before the minimum year and no
  frozen-clock warning; concrete values instead of placeholders on the command lines of adopt runbooks; the
  planned markers (below); and the marker `<!-- pending: fixture output -->`, which only `--allow-pending`
  tolerates (overnight builds; CI never passes it). A fenced line ending in ` \\` continues on the next.
- `qa agents`: AGENTS.md is shorter than its limit, has its parts in order and names the rules its "Never" part
  must carry; a line of it that names a planned command says `planned: <task>`; the host files are short and
  point at it.
- `qa tests`: every test file defines a test, test file names are unique across tests/ and tests/ holds no
  `__init__.py` (without one, two test files with the same name collide).

Planned steps. A verb that is not built yet prints `planned: <task>` and exits 2; so does a subcommand of a built
verb that is registered as a placeholder with `set_defaults(planned="<task>")`. A subcommand that is not
registered at all is a finding. The one rule for agents is: a step marked planned is not built: report it once
and stop at that stage. So a fenced command line that names a planned verb or subcommand must end with the shell
comment `# planned: <task>`, the task the CLI prints; a marker on a line whose commands are all built is stale
and a finding too, because it would stop an agent at a step that works. Inline mentions are counted, not
marked. When `[docs] planned_rule` is set, every file `planned_rule_in` names carries that sentence.

The sister tool (`[docs] sister`): its commands in these docs are checked against its own CLI when its package
is importable (both tools are usually installed side by side); otherwise they are counted as not checked.

Output: one line per finding (`<check>: <file>:<line>: <what>`), one summary line per check, then the total.
Exit 0 clean, 1 findings, 2 usage error (no qa.toml, an unknown key, a wrong type).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ._common import config

PKG = __name__.split(".")[0]
QA_FILE = "qa.toml"
EXIT_FINDINGS = 1
EXIT_USAGE = 2

# The keys qa.toml may hold, with their types; anything else is a usage error, so a typo cannot switch a rule off.
KNOWN: dict[str, dict[str, type]] = {
    "docs": {
        "tool": str, "sister": str, "include": list, "exclude": list, "runbooks": list, "adopt_runbooks": list,
        "output_scope": list, "sections": list, "output_sections": list, "entry_section": str,
        "entry_header": str, "entry_tools": list, "pending_marker": str, "min_year": int, "clock_warning": str,
        "path_roots": list, "path_ignore": list, "gpu": list, "adopt_concrete": list, "planned_rule": str,
        "planned_rule_in": list, "gpu_actions": list, "gpu_actions_in": list,
    },
    "agents": {
        "file": str, "max_lines": int, "parts": list, "never_section": str, "never_mentions": list,
        "hosts": list, "host_max_lines": int,
    },
    "tests": {"folders": list, "pattern": str},
}

FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
HEADING = re.compile(r"^(#{1,6})\s+\S")
SPAN = re.compile(r"(`+)(.+?)\1")
LINK = re.compile(r"\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
PLACEHOLDER = re.compile(r"<[^<>\n]*>")
WORD = re.compile(r"[a-z][a-z0-9_-]*(?:\|[a-z][a-z0-9_-]*)*")
FLAG = re.compile(r"--[a-z][a-z0-9-]*")
HELP_FLAG = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?!\d)")
ID_DATE = re.compile(r"\b[a-z]+-((?:19|20)\d{2})(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])-\d{3}\b", re.IGNORECASE)
SCRIPT = re.compile(r"tools/[\w-]+\.py")
PLANNED_COMMENT = re.compile(r"planned:\s*(?P<task>[\w.-]+)")  # the whole comment of a fenced command line
PLANNED_TEXT = re.compile(r"(?<![\w-])planned:\s*(?P<task>[\w.-]+)")  # anywhere on an AGENTS.md line
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "_gen", "node_modules"}
PYTHONS = {"python", "python3", "py", "PY", "python.exe"}


class QaError(Exception):
    """A usage problem (exit 2): no qa.toml, an unknown key or a wrong type. `lines` are printed as they are."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__("\n".join(lines))
        self.lines = lines


@dataclass(frozen=True, order=True)
class Finding:
    """One problem: the check that found it, the file (repository-relative) and line, and what to fix."""

    check: str
    file: str
    line: int
    message: str

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}" if self.line else self.file
        return f"{self.check}: {where}: {self.message}"


@dataclass
class Report:
    """What one action found, plus the counts its summary lines print."""

    action: str
    checks: tuple[str, ...]
    findings: list[Finding] = field(default_factory=list)
    files: int = 0
    notes: dict[str, str] = field(default_factory=dict)

    def add(self, check: str, file: str, line: int, message: str) -> None:
        finding = Finding(check, file, line, message)
        if finding not in self.findings:
            self.findings.append(finding)

    def lines(self) -> list[str]:
        """The finding lines (sorted), one summary line per check, then the total line."""
        out = [str(f) for f in sorted(self.findings)]
        for check in self.checks:
            count = sum(1 for f in self.findings if f.check == check)
            state = f"{count} finding(s)" if count else "ok"
            note = self.notes.get(check)
            out.append(f"{check:<9} {state}" + (f" ({note})" if note else ""))
        total = f"{self.action}: {len(self.findings)} finding(s) in {self.files} file(s)"
        if self.notes.get("_total"):
            total += f"; {self.notes['_total']}"
        out.append(total)
        return out


# --- qa.toml ---------------------------------------------------------------------------------------------------

def find_root(start: Path | None = None) -> Path:
    """The repository checkout: upward from `start` (default: the current folder) to a folder holding qa.toml
    next to `src/<pkg>/`, else the checkout this module was installed from."""
    here = (start or Path.cwd()).resolve()
    for folder in (here, *here.parents):
        if (folder / QA_FILE).is_file() and (folder / "src" / PKG).is_dir():
            return folder
    own = Path(__file__).resolve().parents[2]
    if (own / QA_FILE).is_file():
        return own
    raise QaError([f"qa: no {QA_FILE} found from {here} upward; run it inside a {PKG} checkout or pass --root <folder>"])


def load_rules(root: Path) -> dict[str, dict]:
    """qa.toml as {section: {key: value}}, every key known and of the right type (else QaError, exit 2)."""
    path = root / QA_FILE
    if not path.is_file():
        raise QaError([f"qa: {QA_FILE} is missing at the repository root ({root.name}/)"])
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise QaError([f"qa: {QA_FILE} cannot be read: {exc}"]) from exc
    problems = []
    for section, values in data.items():
        if section not in KNOWN or not isinstance(values, dict):
            problems.append(f"qa: {QA_FILE}: unknown section [{section}] (known: {', '.join(KNOWN)})")
            continue
        for key, value in values.items():
            want = KNOWN[section].get(key)
            if want is None:
                problems.append(f"qa: {QA_FILE}: unknown key {section}.{key}")
            elif not isinstance(value, want) or (want is int and isinstance(value, bool)):
                problems.append(f"qa: {QA_FILE}: {section}.{key} must be a {want.__name__}")
    if problems:
        raise QaError(problems)
    return {section: dict(data.get(section, {})) for section in KNOWN}


def _need(rules: dict, section: str, key: str) -> object:
    value = rules[section].get(key)
    if value is None:
        raise QaError([f"qa: {QA_FILE}: {section}.{key} is required for `{PKG} qa {section}`"])
    return value


# --- reading markdown -------------------------------------------------------------------------------------------

@dataclass
class Block:
    """A fenced block: the line of its opening fence, its inner lines, and whether it shows output."""

    start: int
    lines: list[tuple[int, str]]
    output: bool
    closed: bool = True


@dataclass
class Doc:
    path: Path
    rel: str
    text: str
    prose: list[tuple[int, str]]
    headings: list[tuple[int, str]]
    blocks: list[Block]

    def section(self, heading: str) -> list[tuple[int, str]]:
        """The prose lines under the `## ` heading `heading`, up to the next `## ` heading."""
        start = next((n for n, h in self.headings if h == heading), None)
        if start is None:
            return []
        end = next((n for n, h in self.headings if n > start and h.startswith("## ")), 10**9)
        return [(n, line) for n, line in self.prose if start < n < end]


def parse_doc(path: Path, rel: str, output_sections: list[str]) -> Doc:
    """Split a markdown file into prose lines, headings and fenced blocks (a block under an output heading
    such as `## 4. Expected output` is an output block; every other block holds command lines)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    prose, headings, blocks = [], [], []
    fence, section, current = "", "", None
    for n, line in enumerate(text.splitlines(), 1):
        match = FENCE.match(line)
        if fence:
            stripped = line.strip()
            if stripped.startswith(fence) and set(stripped) == {fence[0]}:
                fence, current = "", None
            else:
                current.lines.append((n, line))
            continue
        if match:
            fence = match.group(1)
            current = Block(n, [], output=section in output_sections)
            blocks.append(current)
            continue
        prose.append((n, line))
        if HEADING.match(line):
            heading = line.rstrip()
            headings.append((n, heading))
            if heading.startswith("## "):
                section = heading
    if fence and current is not None:
        current.closed = False
    return Doc(path, rel, text, prose, headings, blocks)


def spans(line: str) -> list[str]:
    """The inline code spans of a prose line; a table's escaped `\\|` reads as `|`."""
    return [m.group(2).strip().replace("\\|", "|") for m in SPAN.finditer(line)]


def _glob(root: Path, patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_file() and not (set(path.relative_to(root).parts) & SKIP_DIRS):
                found.add(path)
    return sorted(found)


def _rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


# --- commands: the CLI's own parser as a help cache ---------------------------------------------------------------

def _structure(parser: argparse.ArgumentParser) -> dict:
    """What `--help` shows for one parser, as data: its options, its subcommands (recursively), the choices of
    its first positional argument (None when it has none or takes any value), `required`: the option groups a
    command line must carry (one option of each: an option declared `required=True`, or a required mutually
    exclusive group; an option the verb checks by itself is invisible here, so a verb declares what it needs),
    and `planned`: the task named by `set_defaults(planned="<task>")`, the convention for a subcommand of a
    built verb that is not built yet."""
    options: set[str] = set()
    subcommands: dict[str, dict] = {}
    choices: list[str] | None = None
    required: list[list[str]] = []
    seen_positional = False
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in action.choices.items():
                subcommands[name] = _structure(child)
        elif action.option_strings:
            options.update(action.option_strings)
            if action.required:
                required.append(sorted(action.option_strings))
        elif not seen_positional:
            seen_positional = True
            choices = [str(c) for c in action.choices] if action.choices else None
    for group in parser._mutually_exclusive_groups:
        if group.required:
            required.append(sorted(o for a in group._group_actions for o in a.option_strings))
    planned = parser.get_default("planned")
    return {"options": sorted(options), "subcommands": subcommands, "choices": choices, "required": required,
            "planned": planned if isinstance(planned, str) and planned else None}


class HelpCache:
    """`--help` of a package's verbs (from its CLI's parser, verb by verb; this package by default, or the sister
    tool) and of this repository's `tools/*.py` scripts (by running `--help`), kept in memory and, when
    `cache_file` is given, on disk under a fingerprint of the package's source files, this module and the
    scripts, so an edit to any of them invalidates it. Importing `<pkg>.cli` may raise ImportError."""

    def __init__(self, root: Path, cache_file: Path | None = None, pkg: str = PKG) -> None:
        self.root = root
        self.pkg = pkg
        self.cli = importlib.import_module(f"{pkg}.cli")
        self.cache_file = cache_file
        self.fingerprint = self._fingerprint()
        self.data: dict = {"fingerprint": self.fingerprint, "verbs": {}, "scripts": {}, "top": None}
        self.dirty = False
        if cache_file is not None and cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("fingerprint") == self.fingerprint:
                    self.data = cached
            except (OSError, ValueError):
                pass

    def _fingerprint(self) -> str:
        digest = hashlib.sha256(f"{PKG} {self.pkg} {sys.version_info[:2]}".encode())
        package = Path(self.cli.__file__).resolve().parent
        sources = (sorted(package.rglob("*.py")) + sorted((self.root / "tools").glob("*.py"))
                   + [Path(__file__).resolve()])  # this module: a sister's cache must follow its format too
        for path in sources:
            if "__pycache__" in path.parts:
                continue
            stat = path.stat()
            digest.update(f"{path.as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
        return digest.hexdigest()[:32]

    def top(self) -> dict:
        """The top-level options and every verb with its state: built, planned (with its task) or failing."""
        if self.data.get("top") is None:
            parser, _sub = self.cli.build_parser(None)
            verbs = {}
            for cmd in self.cli.COMMANDS:
                verbs[cmd.verb] = {"planned": None if self.cli.module_exists(cmd.module) else cmd.planned}
            self.data["top"] = {"options": _structure(parser)["options"], "verbs": verbs}
            self.dirty = True
        return self.data["top"]

    def verb(self, name: str) -> dict:
        """{"tree": structure} for a built verb, {"planned": task} for an unbuilt one, {"error": text} when its
        parser cannot be built (for example an import error in its module)."""
        if name not in self.data["verbs"]:
            state = self.top()["verbs"].get(name)
            if state is None:
                entry: dict = {"unknown": True}
            elif state["planned"]:
                entry = {"planned": state["planned"]}
            else:
                try:
                    _parser, sub = self.cli.build_parser(name)
                    entry = {"tree": _structure(sub.choices[name])}
                except Exception as exc:  # a broken verb is a finding, never a crash of qa
                    entry = {"error": f"{type(exc).__name__}: {exc}"}
            self.data["verbs"][name] = entry
            self.dirty = True
        return self.data["verbs"][name]

    def script(self, rel: str) -> dict:
        """{"options": [...]} from `python <rel> --help`, or {"error": ...}. Only argparse scripts are run."""
        if rel not in self.data["scripts"]:
            path = self.root / rel
            text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
            if "argparse" not in text or "__main__" not in text:
                entry: dict = {"skip": "not an argparse script"}
            else:
                try:
                    done = subprocess.run([sys.executable, str(path), "--help"], capture_output=True, text=True,
                                          encoding="utf-8", errors="replace", timeout=60, cwd=self.root)
                    if done.returncode == 0:
                        entry = {"options": sorted(set(HELP_FLAG.findall(done.stdout)) | {"-h", "--help"})}
                    else:
                        entry = {"error": f"exit {done.returncode}"}
                except (OSError, subprocess.TimeoutExpired) as exc:
                    entry = {"error": type(exc).__name__}
            self.data["scripts"][rel] = entry
            self.dirty = True
        return self.data["scripts"][rel]

    def save(self) -> None:
        if self.cache_file is None or not self.dirty:
            return
        tmp = self.cache_file.with_name(f"{self.cache_file.name}.tmp-{os.getpid()}")
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.cache_file)
        except OSError:
            tmp.unlink(missing_ok=True)  # a cache that cannot be written only costs time


def default_cache_file() -> Path:
    """The on-disk help cache: one file per package in the system temp folder (no repository file)."""
    return Path(tempfile.gettempdir()) / f"{PKG}-qa" / "help-cache.json"


def sister_cache(sister: str, cache_file: Path | None, root: Path) -> HelpCache | None:
    """The sister tool's help cache (beside this package's cache file), or None when it is not importable."""
    if not sister or sister == PKG:
        return None
    try:
        if importlib.util.find_spec(f"{sister}.cli") is None:
            return None
        file = cache_file.with_name(f"{cache_file.stem}-{sister}{cache_file.suffix}") if cache_file else None
        return HelpCache(root, file, pkg=sister)
    except Exception:  # a sister that cannot even be imported is "not checked", never a crash of qa
        return None


def split_comment(text: str) -> tuple[str, str | None]:
    """(the command, its trailing shell comment without the `#`, or None). As in a POSIX shell, `#` starts a
    comment at the start of the line or after whitespace, never inside quotes or inside a word (`a.md#b`). A
    `<placeholder>` is one word here too, so an apostrophe inside one opens no quote."""
    masked = PLACEHOLDER.sub(lambda m: "x" * len(m.group(0)), text)
    quote = ""
    for i, char in enumerate(masked):
        if quote:
            if char == quote:
                quote = ""
        elif char in "'\"":
            quote = char
        elif char == "#" and (i == 0 or masked[i - 1].isspace()):
            return text[:i].rstrip(), text[i + 1:].strip()
    return text, None


def words_of(text: str) -> list[str]:
    """Shell words of a command line; every `<placeholder>` (even one with spaces) is one word `<ph>`."""
    flat = PLACEHOLDER.sub("<ph>", text.replace("\\|", "|"))
    try:
        return shlex.split(flat, posix=True)
    except ValueError:
        return flat.split()


def _flags_in(words: list[str]) -> list[str]:
    flags = []
    for word in words:
        for part in word.split("|"):
            part = part.strip("[](){},")
            if FLAG.fullmatch(part.split("=", 1)[0]):
                flags.append(part.split("=", 1)[0])
    return flags


@dataclass
class Checked:
    """What `check_command` found for one command: its state ("checked", "planned" or "skipped"), its problems,
    and every planned command it names (`<tool> <verb> [<subcommand>]` -> the task the CLI prints)."""

    state: str
    problems: list[str] = field(default_factory=list)
    planned: dict[str, str] = field(default_factory=dict)


def check_command(words: list[str], helps: HelpCache, *, required: bool = False) -> Checked:
    """Check one `<tool> ...` command (its words after the tool name) against `helps`' CLI. `required` also
    asks for every option the verb requires (fenced command lines, which an agent may run as written)."""
    pkg = helps.pkg
    if not words:
        return Checked("skipped")
    top = helps.top()
    if not WORD.fullmatch(words[0]):  # `kiln --version`, `kiln <verb> --help`, `kiln <x.y.z>`
        known = set(top["options"]) | {"-h", "--help"}
        bad = [f for f in _flags_in(words) if f not in known]
        return Checked("checked", [f"`{pkg} {f}` -- the CLI has no such top-level option" for f in bad])
    result = Checked("checked")
    reached_built = False
    for verb in (v for v in words[0].split("|") if v):
        entry = helps.verb(verb)
        if "unknown" in entry:
            result.problems.append(f"`{pkg} {verb}` -- no such verb (see `{pkg} --help`)")
        elif "planned" in entry:
            result.planned[f"{pkg} {verb}"] = entry["planned"]
        elif "error" in entry:
            result.problems.append(f"`{pkg} {verb} --help` fails: {entry['error']}")
        else:
            problems, planned, built = _walk(verb, entry["tree"], words[1:], pkg=pkg, required=required)
            result.problems += problems
            result.planned.update(planned)
            reached_built = reached_built or built
    if result.planned and not reached_built and not result.problems:
        result.state = "planned"
    return result


def _walk(verb: str, tree: dict, words: list[str], *, pkg: str = PKG,
          required: bool = False) -> tuple[list[str], dict[str, str], bool]:
    """Follow the subcommand words through the parser tree, then check the flags (and with `required` the
    required options) against the last parser. Returns (problems, planned, built): `planned` maps each planned
    subcommand the words name (`<pkg> <verb> <sub>`) to its task; `built` is True when the walk ends on a built
    parser."""
    problems: list[str] = []
    if tree.get("planned"):
        return problems, {f"{pkg} {verb}": tree["planned"]}, False
    planned: dict[str, str] = {}
    paths: list[tuple[tuple[str, ...], dict]] = [((verb,), tree)]
    i = 0
    stopped_on_placeholder = False
    while i < len(words) and any(node["subcommands"] for _, node in paths):
        word = words[i]
        if not WORD.fullmatch(word):
            stopped_on_placeholder = word.startswith("<")
            break
        names = [n for n in word.split("|") if n]
        nxt = []
        for prefix, node in paths:
            for name in names:
                if name in node["subcommands"]:
                    nxt.append(((*prefix, name), node["subcommands"][name]))
                else:
                    has = ", ".join(sorted(node["subcommands"])) or "none"
                    problems.append(f"`{pkg} {' '.join((*prefix, name))}` -- no such subcommand (has: {has})")
        for prefix, node in nxt:
            if node.get("planned"):
                planned[f"{pkg} {' '.join(prefix)}"] = node["planned"]
        built = [(prefix, node) for prefix, node in nxt if not node.get("planned")]
        if not built:  # nothing matched (problems say so), or every match is a planned subcommand
            return problems, planned, False
        paths, i = built, i + 1
    rest = words[i:]
    flags = _flags_in(rest)
    asks_help = bool({"-h", "--help"} & set(flags))
    for prefix, node in paths:
        shown = f"{pkg} {' '.join(prefix)}"
        if rest and node["choices"] and WORD.fullmatch(rest[0]):
            for name in (n for n in rest[0].split("|") if n):
                if name not in node["choices"]:
                    problems.append(f"`{pkg} {' '.join((*prefix, name))}` -- not one of {', '.join(node['choices'])}")
        known = set(node["options"]) | {"-h", "--help"}
        if stopped_on_placeholder:  # `kiln journal <action> --help`: a flag of any child counts
            for child in node["subcommands"].values():
                known |= set(child["options"])
        for flag in flags:
            if flag not in known:
                problems.append(f"`{shown} {flag}` -- no such flag (see `{shown} --help`)")
        if required and not asks_help and not stopped_on_placeholder:
            for group in node.get("required", []):
                if not any(option in flags for option in group):
                    problems.append(f"`{shown}` needs {' or '.join(group)}: a command line carries every option "
                                    f"the verb requires (see `{shown} --help`)")
    return problems, planned, True


def _command_words(text: str, tool: str) -> list[str] | None:
    """The words after the tool name when `text` is a `<tool> ...` or `python -m <tool> ...` command, else None."""
    words = words_of(text)
    if words[:1] == ["$"]:
        words = words[1:]
    if len(words) >= 3 and words[0] in PYTHONS and words[1] == "-m" and words[2] == tool:
        return words[3:]
    if words[:1] == [tool]:
        return words[1:]
    return None


def _script_words(text: str) -> tuple[str, list[str]] | None:
    """(`tools/<x>.py`, the words after it) when `text` runs a repository script with python, else None."""
    words = words_of(text)
    if len(words) >= 2 and words[0] in PYTHONS and SCRIPT.fullmatch(words[1]):
        return words[1], words[2:]
    return None


# --- qa docs ----------------------------------------------------------------------------------------------------

DOC_CHECKS = ("fences", "sections", "entry", "pending", "commands", "planned", "paths", "gpu", "abspath", "dates",
              "adopt")


def check_docs(root: Path, rules: dict, *, allow_pending: bool = False, cache_file: Path | None = None) -> Report:
    """Every docs check over the files qa.toml's `[docs] include` names (see the module docstring)."""
    d = rules["docs"]
    tool = d.get("tool", PKG)
    sister = d.get("sister", "")
    report = Report("docs", DOC_CHECKS)
    excluded = set(_glob(root, d.get("exclude", [])))
    files = [p for p in _glob(root, _need(rules, "docs", "include")) if p not in excluded]
    runbooks = set(_glob(root, d.get("runbooks", [])))
    adopt = set(_glob(root, d.get("adopt_runbooks", [])))
    scope = set(_glob(root, d.get("output_scope", [])))
    action_files = set(_glob(root, d.get("gpu_actions_in", [])))
    docs = [parse_doc(p, _rel(root, p), d.get("output_sections", [])) for p in files]
    report.files = len(docs)
    helps = HelpCache(root, cache_file)
    sister_helps = sister_cache(sister, cache_file, root)
    counts = {"checked": 0, "planned": 0, "skipped": 0, "pending": 0, "sister": 0, "sister_unchecked": 0}

    _check_planned_rule(report, root, d)
    for doc in docs:
        for block in doc.blocks:
            if not block.closed:
                report.add("fences", doc.rel, block.start, "a code fence that is never closed")
        if doc.path in runbooks:
            _check_sections(report, "sections", doc, _need(rules, "docs", "sections"))
            _check_entry(report, doc, d, tool)
        _check_pending(report, doc, d, allow_pending, counts)
        _check_commands(report, doc, [(tool, helps), (sister, sister_helps)], counts)
        _check_paths(report, doc, root, d)
        _check_gpu(report, doc, d.get("gpu", []))
        if doc.path in action_files:
            _check_gpu_actions(report, doc, d.get("gpu_actions", []))
        for n, hit in config.absolute_paths(doc.text):
            report.add("abspath", doc.rel, n, f"an absolute path ({hit}...); write <home>, <models_root> or "
                                              f"~/{tool}-home instead")
        if doc.path in scope:
            _check_dates(report, doc, d)
        if doc.path in adopt:
            _check_adopt(report, doc, d)
    helps.save()
    if sister_helps is not None:
        sister_helps.save()

    note = f"{counts['checked']} checked, {counts['planned']} name a planned verb or subcommand"
    if sister and sister_helps is not None:
        note += f"; {counts['sister']} {sister} command(s) among them, checked against {sister}'s CLI"
    elif sister and counts["sister_unchecked"]:
        note += f"; {counts['sister_unchecked']} {sister} command(s) not checked ({sister} is not importable)"
    report.notes["commands"] = note
    if allow_pending:
        report.notes["pending"] = f"{counts['pending']} marker(s) tolerated by --allow-pending"
        report.notes["_total"] = f"{counts['pending']} pending marker(s) tolerated (--allow-pending)"
    return report


def _check_sections(report: Report, check: str, doc: Doc, want: list[str]) -> None:
    """The `## ` headings are exactly `want`, each once, in order."""
    got = [(n, h) for n, h in doc.headings if h.startswith("## ")]
    names = [h for _, h in got]
    for heading in want:
        if heading not in names:
            report.add(check, doc.rel, 0, f"missing the section heading {heading!r}")
        elif names.count(heading) > 1:
            report.add(check, doc.rel, 0, f"the section heading {heading!r} appears {names.count(heading)} times")
    for n, heading in got:
        if heading not in want:
            report.add(check, doc.rel, n, f"an extra section heading {heading!r}; only these, in order: "
                                         + " | ".join(want))
    order = [h for h in dict.fromkeys(names) if h in want]
    if order != [h for h in want if h in order]:
        report.add(check, doc.rel, 0, "the section headings are out of order; the order is: " + " | ".join(want))


def _cells(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return [c.strip() for c in re.split(r"(?<!\\)\|", body)]


def _check_entry(report: Report, doc: Doc, d: dict, tool: str) -> None:
    """Section 2 holds a `| command | literal line |` table; each row is a command span plus a literal span;
    `<tool> next --stage` must print the runbook's own id; no `jq`."""
    heading = d.get("entry_section")
    if not heading:
        return
    lines = doc.section(heading)
    if not lines:
        return  # a missing section is the sections check's finding
    header = " ".join(d.get("entry_header", "| command | literal line |").split())
    tools = d.get("entry_tools", [tool])
    rows = [(n, line) for n, line in lines if line.strip().startswith("|")]
    start = next((i for i, (_, line) in enumerate(rows) if " ".join(line.split()) == header), None)
    if start is None:
        report.add("entry", doc.rel, lines[0][0], f"section 2 has no `{header}` table")
    else:
        data = [(n, line) for n, line in rows[start + 1:] if not re.fullmatch(r"[\s|:-]+", line)]
        if not data:
            report.add("entry", doc.rel, rows[start][0], "the entry-condition table has no row")
        for n, line in data:
            cells = _cells(line)
            whole = [SPAN.fullmatch(c) for c in cells]
            if len(cells) != 2 or not all(whole):
                report.add("entry", doc.rel, n, "an entry condition is two code cells: `command` | `literal line`")
                continue
            command, literal = (m.group(2).strip() for m in whole)
            words = words_of(command)
            if not words or words[0] not in tools:
                report.add("entry", doc.rel, n, f"the command cell does not start with {' or '.join(tools)}")
            elif words[1:] == ["next", "--stage"] and literal != doc.path.stem:
                report.add("entry", doc.rel, n, f"`{command}` prints {literal!r} here, but this runbook is "
                                                f"{doc.path.stem!r}")
    for n, line in lines:
        if re.search(r"\bjq\b", line):
            report.add("entry", doc.rel, n, "an entry condition uses jq; write the command and its literal line")


def _check_pending(report: Report, doc: Doc, d: dict, allow: bool, counts: dict) -> None:
    marker = d.get("pending_marker")
    if not marker:
        return
    for n, line in enumerate(doc.text.splitlines(), 1):
        if marker in line:
            counts["pending"] += 1
            if not allow:
                report.add("pending", doc.rel, n, "fixture output not filled in yet; run the documented commands on "
                                                  "the demo under the frozen clock and replace the marker")


def _fenced_commands(block: Block) -> list[tuple[int, str]]:
    """(first line, text) of each command in a fenced block; a line ending in ` \\` continues on the next."""
    out: list[tuple[int, str]] = []
    start, parts = 0, []
    for n, line in block.lines:
        text = line.strip()
        if not text and not parts:
            continue
        start = start or n
        if text.endswith("\\"):
            parts.append(text[:-1].rstrip())
            continue
        parts.append(text)
        out.append((start, " ".join(p for p in parts if p)))
        start, parts = 0, []
    if parts:
        out.append((start, " ".join(p for p in parts if p)))
    return out


def _command_lines(doc: Doc) -> list[tuple[int, str, bool]]:
    """(line, text, fenced) for every inline code span and every command of a fenced command block."""
    out = [(n, span, False) for n, line in doc.prose for span in spans(line)]
    out += [(n, text, True) for b in doc.blocks if not b.output for n, text in _fenced_commands(b)]
    return out


def _check_planned_rule(report: Report, root: Path, d: dict) -> None:
    """Every file `planned_rule_in` names carries the sentence `planned_rule` (line breaks and runs of spaces
    count as one space), so the rule for a planned step reads the same everywhere."""
    sentence = " ".join(str(d.get("planned_rule") or "").split())
    if not sentence:
        return
    for pattern in d.get("planned_rule_in", []):
        paths = _glob(root, [pattern])
        if not paths and not any(ch in pattern for ch in "*?["):
            report.add("planned", pattern, 0, "missing; it must carry the planned-step rule")
        for path in paths:
            text = " ".join(path.read_text(encoding="utf-8", errors="replace").split())
            if sentence not in text:
                report.add("planned", _rel(root, path), 0, f"does not carry the planned-step rule: {sentence!r}")


def _check_planned_mark(report: Report, rel: str, n: int, result: Checked, mark: str | None) -> None:
    """A fenced command line that names a planned command ends with `# planned: <task>`, the task the CLI prints;
    a marker on a line whose commands are all built is stale."""
    if result.planned:
        named = ", ".join(f"`{name}` (planned: {task})" for name, task in sorted(result.planned.items()))
        tasks = sorted(set(result.planned.values()))
        if mark is None:
            report.add("planned", rel, n, f"names {named}, which is not built; end the line with "
                                          f"`# planned: {tasks[0]}` (a step marked planned is not built: report "
                                          f"it once and stop at that stage)")
        elif mark not in tasks:
            report.add("planned", rel, n, f"says `# planned: {mark}`, but the CLI prints planned: "
                                          f"{' or '.join(tasks)} for {named}; write that task")
    elif mark is not None and result.state == "checked" and not result.problems:
        report.add("planned", rel, n, f"says `# planned: {mark}`, but every command on it is built now; drop the "
                                      "marker so agents run the step")


def _check_commands(report: Report, doc: Doc, tools: list[tuple[str, HelpCache | None]], counts: dict) -> None:
    """Commands of this tool (first in `tools`) and of the sister tool against their CLIs; scripts of this repo."""
    helps = tools[0][1]
    for n, text, fenced in _command_lines(doc):
        command, comment = split_comment(text)
        mark = PLANNED_COMMENT.fullmatch(comment) if comment is not None else None
        matched = False
        for k, (name, cache) in enumerate(tools):
            words = _command_words(command, name) if name else None
            if words is None:
                continue
            matched = True
            if cache is None:
                counts["sister_unchecked"] += 1
                break
            result = check_command(words, cache, required=fenced)
            counts[result.state] += 1
            counts["sister"] += 1 if k else 0
            for problem in result.problems:
                report.add("commands", doc.rel, n, problem)
            if fenced:
                _check_planned_mark(report, doc.rel, n, result, mark.group("task") if mark else None)
            break
        if matched:
            continue
        script = _script_words(command)
        if script is not None:
            rel, rest = script
            if not (helps.root / rel).is_file():
                continue  # the paths check reports a missing script
            entry = helps.script(rel)
            if "error" in entry:
                report.add("commands", doc.rel, n, f"`python {rel} --help` fails ({entry['error']})")
            elif "options" in entry:
                counts["checked"] += 1
                for flag in _flags_in(rest):
                    if flag not in entry["options"]:
                        report.add("commands", doc.rel, n, f"`python {rel} {flag}` -- no such flag "
                                                           f"(see `python {rel} --help`)")


def _path_texts(doc: Doc, sister: str) -> list[tuple[int, str]]:
    """Code spans and fenced command lines that may name repository paths. A span right after "<sister>'s" (for
    example "easel's `runtime/comfyui.lock.json`") names a file of the sister repository and is left out."""
    out = []
    owner = f"{sister.lower()}'s" if sister else None
    for n, line in doc.prose:
        for m in SPAN.finditer(line):
            if owner and line[: m.start()].rstrip().lower().endswith(owner):
                continue
            out.append((n, m.group(2).strip()))
    out += [(n, line.strip()) for b in doc.blocks if not b.output for n, line in b.lines if line.strip()]
    return out


def _check_paths(report: Report, doc: Doc, root: Path, d: dict) -> None:
    """Repository paths in code spans and on command lines exist; relative links resolve inside the repo."""
    roots = d.get("path_roots", [])
    ignore = tuple(d.get("path_ignore", []))
    if roots:
        pattern = re.compile(r"(?<![\w./~<>$-])((?:" + "|".join(re.escape(r) for r in roots) + r")/[\w./-]*)")
        for n, text in _path_texts(doc, d.get("sister", "")):
            for m in pattern.finditer(text):
                if m.end() < len(text) and text[m.end()] in "<*{$":
                    continue  # the path continues with a placeholder or a glob
                path = m.group(1).rstrip(".,:;")
                if path.startswith(ignore) or (root / path).exists():
                    continue
                report.add("paths", doc.rel, n, f"names {path}, which does not exist")
    resolved_root = root.resolve()
    for n, line in doc.prose:
        clean = SPAN.sub("", line)
        for m in LINK.finditer(clean):
            target = m.group(1)
            if re.match(r"[a-z][a-z0-9+.-]*:", target, re.IGNORECASE) or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            dest = (doc.path.parent / target).resolve()
            if not dest.is_relative_to(resolved_root):
                report.add("paths", doc.rel, n, f"links to {target}, outside the repository")
            elif not dest.exists():
                report.add("paths", doc.rel, n, f"links to {target}, which does not exist")


def _check_gpu(report: Report, doc: Doc, rules: list) -> None:
    """No fenced command line runs GPU work directly (prose may name the refused verbs on purpose)."""
    for block in doc.blocks:
        if block.output:
            continue
        for n, line in block.lines:
            words = words_of(line)
            for rule in rules:
                unless = rule.get("unless")
                if unless and unless in words:
                    continue
                command, regex = rule.get("command"), rule.get("regex")
                hit = (command and words[: len(command.split())] == command.split()) or (
                    regex and re.search(regex, line))
                if hit:
                    why = rule.get("why", "GPU work is submitted, never run directly")
                    report.add("gpu", doc.rel, n, f"a direct GPU command `{line.strip()}`; {why}")


def _check_gpu_actions(report: Report, doc: Doc, rules: list) -> None:
    """The action cell (the last one) of a table row tells an agent what to do, so it never says to start, stop,
    restart or suspend GPU work: that is the operator's, or goes through the tool's submit verb. The rules
    (`[[docs.gpu_actions]]`: `regex`, `why`) come from qa.toml and apply to the files `gpu_actions_in` names."""
    for n, line in doc.prose:
        if not line.strip().startswith("|") or re.fullmatch(r"[\s|:-]+", line):
            continue
        cells = _cells(line)
        action = cells[-1] if cells else ""
        for rule in rules:
            m = re.search(rule.get("regex", r"(?!)"), action, re.IGNORECASE)
            if m:
                why = rule.get("why", "the action is to report to the operator, or the tool's submit verb")
                report.add("gpu", doc.rel, n, f"a table action that runs GPU work by hand ({m.group(0)!r}); {why}")


def _check_dates(report: Report, doc: Doc, d: dict) -> None:
    """Fenced lines in the output scope carry dates from the frozen clock only, and never its warning."""
    min_year = d.get("min_year", 2030)
    warning = d.get("clock_warning", "clock frozen").lower()
    for block in doc.blocks:
        for n, line in block.lines:
            years = [int(y) for y in DATE.findall(line) + ID_DATE.findall(line)]
            old = [y for y in years if y < min_year]
            if old:
                report.add("dates", doc.rel, n, f"a fenced line dated {old[0]}; doc output runs under the frozen "
                                                f"clock, so every date is {min_year} or later")
            if warning and warning in line.lower():
                report.add("dates", doc.rel, n, "the frozen-clock warning in a fenced block; it goes to stderr and "
                                                "doc generation keeps stdout only")


def _check_adopt(report: Report, doc: Doc, d: dict) -> None:
    """Command lines of an adopt runbook carry placeholders, never concrete values (they come from the instance)."""
    tools = set(d.get("entry_tools", [d.get("tool", PKG)]))
    for n, text, fenced in _command_lines(doc):
        words = words_of(text)
        if not fenced and not (words[:1] and words[0] in tools):
            continue  # prose spans that are not commands (literal lines, keys, file names) are free
        for rule in d.get("adopt_concrete", []):
            m = re.search(rule.get("regex", r"(?!)"), text)
            if m:
                report.add("adopt", doc.rel, n, f"{rule.get('what', 'a concrete value')} ({m.group(0)}) on a command "
                                               "line of an adopt runbook; write a placeholder such as <version> or "
                                               "<run-id>: the values come from the instance")


# --- qa agents --------------------------------------------------------------------------------------------------

AGENT_CHECKS = ("length", "parts", "never", "planned", "hosts")


def _check_agent_planned(report: Report, name: str, doc: Doc, root: Path, rules: dict) -> None:
    """A line of AGENTS.md that names a planned command (this tool's, or the sister's when importable) says
    `planned: <task>` on the same line, with the task the CLI prints. Agents read this file first, so a planned
    step there must never look runnable. (No stale-marker check here: prose may call a feature planned that is
    no command.)"""
    d = rules["docs"]
    tools = [d.get("tool", PKG), d.get("sister", "")]
    caches: dict[int, HelpCache | None] = {}  # built on first use: a file without commands imports nothing
    by_line: dict[int, list[Checked]] = {}
    for n, text, _fenced in _command_lines(doc):
        command, _comment = split_comment(text)
        for k, tool in enumerate(tools):
            words = _command_words(command, tool) if tool else None
            if words is None:
                continue
            if k not in caches:
                try:
                    caches[k] = HelpCache(root) if k == 0 else sister_cache(tool, None, root)
                except ImportError:
                    caches[k] = None
            cache = caches[k]
            if cache is not None:
                by_line.setdefault(n, []).append(check_command(words, cache))
            break
    lines = doc.text.splitlines()
    for n, results in sorted(by_line.items()):
        marks = {m.group("task") for m in PLANNED_TEXT.finditer(lines[n - 1])}
        planned = {cmd: task for r in results for cmd, task in r.planned.items()}
        if planned:
            tasks = sorted(set(planned.values()))
            named = ", ".join(f"`{cmd}` (planned: {task})" for cmd, task in sorted(planned.items()))
            if not marks:
                report.add("planned", name, n, f"names {named} without `planned: {tasks[0]}` on the same line (a "
                                               "step marked planned is not built: report it once and stop at "
                                               "that stage)")
            elif not marks & set(tasks):
                report.add("planned", name, n, f"says planned: {', '.join(sorted(marks))}, but the CLI prints "
                                               f"planned: {' or '.join(tasks)} for {named}")


def check_agents(root: Path, rules: dict) -> Report:
    """AGENTS.md: fewer lines than the limit, its parts in order, the Never rules, planned commands marked; the
    host files point at it."""
    a = rules["agents"]
    report = Report("agents", AGENT_CHECKS)
    name = a.get("file", "AGENTS.md")
    path = root / name
    if not path.is_file():
        report.add("length", name, 0, "missing")
        return report
    report.files = 1
    doc = parse_doc(path, name, [])
    count = len(doc.text.splitlines())
    limit = a.get("max_lines", 80)
    if count >= limit:
        report.add("length", name, 0, f"{count} lines; it must stay under {limit}: move detail into the runbooks")
    _check_sections(report, "parts", doc, _need(rules, "agents", "parts"))
    never = a.get("never_section")
    if never:
        body = " ".join(" ".join(line.split()) for _, line in doc.section(never)).lower()
        for phrase in a.get("never_mentions", []):
            if phrase.lower() not in body:
                report.add("never", name, 0, f"the part {never!r} does not mention {phrase!r}")
    _check_agent_planned(report, name, doc, root, rules)
    for host in a.get("hosts", []):
        host_path = root / host
        if not host_path.is_file():
            report.add("hosts", host, 0, "missing")
            continue
        report.files += 1
        lines = host_path.read_text(encoding="utf-8", errors="replace").splitlines()
        limit = a.get("host_max_lines", 3)
        if len(lines) > limit:
            report.add("hosts", host, 0, f"{len(lines)} lines; a host file has at most {limit} and points at {name}")
        if name not in "\n".join(lines):
            report.add("hosts", host, 0, f"does not point at {name}")
    return report


# --- qa tests ---------------------------------------------------------------------------------------------------

TEST_CHECKS = ("has-test", "names", "layout")


def _defines_test(body: list[ast.stmt]) -> bool:
    """A `test_*` function or a `Test*` class with one, at module level or inside if/try/with blocks."""
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
            return True
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test") and _defines_test(node.body):
            return True
        if isinstance(node, ast.If | ast.With | ast.For | ast.While):
            if _defines_test(node.body) or _defines_test(getattr(node, "orelse", [])):
                return True
        if isinstance(node, ast.Try):
            if any(_defines_test(part) for part in (node.body, node.orelse, node.finalbody,
                                                    *(h.body for h in node.handlers))):
                return True
    return False


def check_tests(root: Path, rules: dict) -> Report:
    """Every test file defines a test; test file names are unique; tests/ holds no __init__.py."""
    t = rules["tests"]
    report = Report("tests", TEST_CHECKS)
    pattern = t.get("pattern", "test_*.py")
    by_name: dict[str, list[str]] = {}
    for folder in t.get("folders", ["tests"]):
        base = root / folder
        if not base.is_dir():
            report.add("layout", folder, 0, "the test folder is missing")
            continue
        for init in sorted(base.rglob("__init__.py")):
            if not set(init.relative_to(root).parts) & SKIP_DIRS:
                report.add("layout", _rel(root, init), 0, "tests/ holds no __init__.py (test files are found by "
                                                          "their unique names)")
        for path in sorted(base.rglob(pattern)):
            rel = _rel(root, path)
            if set(path.relative_to(root).parts) & SKIP_DIRS:
                continue
            report.files += 1
            by_name.setdefault(path.name, []).append(rel)
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
            except (SyntaxError, UnicodeDecodeError) as exc:
                report.add("has-test", rel, getattr(exc, "lineno", 0) or 0, f"does not parse: {exc}")
                continue
            if not _defines_test(tree.body):
                report.add("has-test", rel, 0, "defines no test (a test_* function or a Test* class)")
    for name, rels in sorted(by_name.items()):
        if len(rels) > 1:
            for rel in rels:
                report.add("names", rel, 0, f"another test file is also named {name} ({', '.join(rels)}); "
                                            "test file names are unique across tests/")
    return report


# --- entry point ------------------------------------------------------------------------------------------------

def run(action: str, *, root: Path | None = None, allow_pending: bool = False,
        cache_file: Path | None = None) -> int:
    """Run one action, print its lines and return the exit code (0 clean, 1 findings, 2 usage error)."""
    try:
        base = find_root() if root is None else Path(root).resolve()
        rules = load_rules(base)
        if action == "docs":
            report = check_docs(base, rules, allow_pending=allow_pending, cache_file=cache_file)
        elif action == "agents":
            report = check_agents(base, rules)
        elif action == "tests":
            report = check_tests(base, rules)
        else:
            raise QaError([f"qa: unknown action {action!r} (docs, agents, tests)"])
    except QaError as exc:
        for line in exc.lines:
            print(line, file=sys.stderr)
        return EXIT_USAGE
    for line in report.lines():
        print(line)
    return EXIT_FINDINGS if report.findings else 0
