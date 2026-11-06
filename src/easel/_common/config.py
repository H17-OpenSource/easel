"""One TOML per tool at the instance root: discovery, overrides, `${VAR}` expansion, validation, show and set.

The same module serves easel and kiln; the tool name is always a parameter (`load(tool)`), so the file is
`easel.toml` or `kiln.toml` and the variables are `EASEL_...` or `KILN_...`.

Discovery order (the first that applies wins): `--workspace <folder or file>`, `$<TOOL>_HOME`, upward from the
current folder, `~/.config/<tool>/`. An explicit `--workspace` or `<TOOL>_HOME` that names no file is an error
(exit 2), never a silent fall-through.

Value precedence, lowest first: the schema's `default`s, the instance TOML, `[env_aliases]` (an instance maps its
own old variable names: `"OLD_NAME" = "section.key"`), then `<TOOL>_<SECTION>__<KEY>` (for example
`EASEL_COMFYUI__URL`). A value from the environment is read as a TOML literal (`50`, `true`, `[1, 2]`) unless
the schema says the key is a string; anything that is not a TOML literal stays a string.

`${VAR}` in a string value is replaced from the environment at load time (`$${VAR}` keeps a literal `${VAR}`). An
unset variable is the line `FAIL: ${MODELS_ROOT} unset (paths.models_root)`: `doctor` prints it as a FAIL line
(load with `strict=False`, read `Config.unset_lines()`); every other command exits 2 with it (`strict=True`, the
default). `config show` and `config set` load non-strictly, so a broken value can still be seen and fixed.

Validation: `<repo>/schemas/config.schema.json` when it exists (an unknown key or a wrong type is exit 2), plus
two rules that hold with or without a schema: a `venv_dir` is a folder, never an `.exe`; and a shipped file (one
inside this tool's repository checkout, or any `*.example.toml`) holds no absolute path (exit 3, the rule of the
old template renderer: a path baked into a shipped file cannot travel; use `${VAR}` instead). Absolute paths
belong only in the instance TOML.

`config set <section.key> <value>` edits the instance TOML in place (comments, layout and CRLF or LF line endings
kept when the edit is a simple one-line change; otherwise the file is rewritten without comments, with its line
endings, and the command says so), validates
the result first and writes it atomically under the instance's store lock; a refused value leaves the file
byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import os
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from . import store

EXIT_USAGE = 2
EXIT_ABSOLUTE = 3
DEFAULT = object()  # "use the repo's schemas/config.schema.json" (a sentinel, so None can mean "no schema")
_MISSING = object()

_VARREF = re.compile(r"\$\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
# A literal absolute path: a drive letter path, a UNC share, or a POSIX path under a per-machine root.
ABSOLUTE = re.compile(r"(?<![\w$])(?:[A-Za-z]:[\\/]|\\\\[A-Za-z0-9._$-]+\\|/(?:home|Users|mnt|opt|var|srv|root|tmp|media|Volumes)/)")
_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")
_HEADER = re.compile(r"^\s*\[(?!\[)\s*([^\]]+?)\s*\]\s*(?:#.*)?$")


class ConfigError(Exception):
    """A configuration problem with the exit code it maps to (2 usage/refusal, 3 absolute path in a shipped file)."""

    def __init__(self, code: int, lines: list[str]) -> None:
        super().__init__("\n".join(lines))
        self.code = code
        self.lines = lines


# --- the loaded configuration -----------------------------------------------------------------------------------

@dataclass
class Config:
    """A loaded configuration: nested `values` (overrides applied, `${VAR}` expanded) and where each came from."""

    tool: str
    file: Path | None
    root: Path | None
    found_by: str
    values: dict = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    unset: list[tuple[str, str]] = field(default_factory=list)
    searched: list[Path] = field(default_factory=list)
    file_values: dict = field(default_factory=dict)

    def unset_lines(self) -> list[str]:
        """One `FAIL: ${VAR} unset (section.key)` line per unset variable."""
        return [f"FAIL: ${{{var}}} unset ({key})" for var, key in self.unset]

    def not_found_line(self) -> str:
        where = ", ".join(str(p) for p in self.searched) or "nowhere"
        return f"no {self.tool}.toml found; searched: {where}"

    def get(self, key: str, default: object = _MISSING) -> object:
        """The value at a dotted key; exit-2 ConfigError if its `${VAR}` is unset or it is missing without default."""
        bad = [line for (var, k), line in zip(self.unset, self.unset_lines(), strict=True)
               if k == key or k.startswith(key + ".")]
        if bad:
            raise ConfigError(EXIT_USAGE, bad)
        node: object = self.values
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(EXIT_USAGE, [f"config key {key} is not set" +
                                                   ("" if self.file else f"; {self.not_found_line()}")])
                return default
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """A path value: absolute as written, `~` expanded, otherwise relative to the instance root."""
        p = Path(str(self.get(key))).expanduser()
        if p.is_absolute():
            return p
        if self.root is None:
            raise ConfigError(EXIT_USAGE, [self.not_found_line(), f"create one with `{self.tool} init <folder>`"])
        return self.root / p

    def flat(self) -> list[tuple[str, object]]:
        """Every leaf as (dotted key, value), in file order, then defaults and environment-only keys."""
        return list(flatten(self.values))


# --- small helpers ----------------------------------------------------------------------------------------------

def flatten(values: Mapping, prefix: str = ""):
    """Yield (dotted key, leaf value) for every non-table value; arrays are leaves."""
    for key, value in values.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from flatten(value, dotted + ".")
        else:
            yield dotted, value


def _set_in(values: dict, dotted: str, value: object) -> None:
    parts = dotted.split(".")
    node = values
    for i, part in enumerate(parts[:-1]):
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(EXIT_USAGE, [f"{'.'.join(parts[: i + 1])} is a value, not a section"])
        node = child
    node[parts[-1]] = value


def _deep_merge(base: dict, over: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def check_key(dotted: str) -> str:
    """A dotted key is non-empty segments of letters, digits, `_` and `-` (TOML bare keys)."""
    if not dotted or not all(_BARE_KEY.match(p) for p in dotted.split(".")):
        raise ConfigError(EXIT_USAGE, [f"not a key of the form section.key: {dotted!r}"])
    return dotted


def absolute_paths(text: str) -> list[tuple[int, str]]:
    """(line number, matched start) for every literal absolute path in `text`."""
    return [(n, m.group(0)) for n, line in enumerate(text.splitlines(), 1) for m in ABSOLUTE.finditer(line)]


def check_shipped_text(name: str, text: str) -> list[str]:
    """Problem lines for a shipped file's text: every literal absolute path, with the fix."""
    return [f"REFUSED: {name}:{n} holds a literal absolute path ({hit}...); replace it with a ${{VAR}} so the file "
            "travels" for n, hit in absolute_paths(text)]


def is_shipped(path: Path, tool: str) -> bool:
    """True for a `*.example.toml` and for any file inside this tool's repository checkout."""
    path = Path(path).resolve()
    if path.name.endswith(".example.toml"):
        return True
    return any((d / "pyproject.toml").is_file() and (d / "src" / tool).is_dir() for d in path.parents)


def _rules(values: Mapping) -> list[str]:
    """The rules that hold with or without a schema."""
    problems = []
    for dotted, value in flatten(values):
        if dotted.rsplit(".", 1)[-1] == "venv_dir" and isinstance(value, str) and value.lower().endswith(".exe"):
            problems.append(f"{dotted}: a venv_dir is a folder, not an interpreter ({value}); "
                            "put an interpreter in the python key instead")
    return problems


# --- the schema -------------------------------------------------------------------------------------------------

def default_schema_path() -> Path:
    """`<repo>/schemas/config.schema.json` for the repository this module sits in (src layout)."""
    return Path(__file__).resolve().parents[3] / "schemas" / "config.schema.json"


def load_schema(schema_file: object = DEFAULT) -> dict | None:
    """The config schema: the repo's when `DEFAULT` and present, `None` for none, or the given file."""
    if schema_file is None:
        return None
    path = default_schema_path() if schema_file is DEFAULT else Path(str(schema_file))
    if not path.is_file():
        if schema_file is DEFAULT:
            return None
        raise ConfigError(EXIT_USAGE, [f"config schema not found: {path}"])
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(EXIT_USAGE, [f"config schema unreadable: {path}: {exc}"]) from exc


def _resolve(schema: dict, node: object) -> dict:
    """Follow a local `$ref` (`#/$defs/...`) once or more; anything else is returned as it is."""
    seen = 0
    while isinstance(node, dict) and isinstance(node.get("$ref"), str) and node["$ref"].startswith("#/") and seen < 20:
        target: object = schema
        for part in node["$ref"][2:].split("/"):
            target = target.get(part, {}) if isinstance(target, dict) else {}
        node, seen = target, seen + 1
    return node if isinstance(node, dict) else {}


def schema_defaults(schema: dict | None, node: dict | None = None) -> dict:
    """The `default` of every property in the schema, nested like the TOML."""
    if not schema:
        return {}
    node = _resolve(schema, schema if node is None else node)
    out: dict = {}
    for name, sub in (node.get("properties") or {}).items():
        sub = _resolve(schema, sub)
        if "default" in sub:
            out[name] = copy.deepcopy(sub["default"])
        elif "properties" in sub:
            nested = schema_defaults(schema, sub)
            if nested:
                out[name] = nested
    return out


def schema_at(schema: dict | None, dotted: str) -> dict | None:
    """The sub-schema of a dotted key, or None when the schema does not describe it."""
    if not schema:
        return None
    node = _resolve(schema, schema)
    for part in dotted.split("."):
        props = node.get("properties") or {}
        if part in props:
            node = _resolve(schema, props[part])
        elif isinstance(node.get("additionalProperties"), dict):
            node = _resolve(schema, node["additionalProperties"])
        else:
            return None
    return node


def validate(values: Mapping, schema: dict | None) -> list[str]:
    """Problem lines from the schema (`key: message`) plus the always-on rules; empty when valid."""
    problems = _rules(values)
    if schema is None:
        return problems
    try:
        from jsonschema import validators
    except ImportError as exc:  # a declared dependency; say so plainly if the install is broken
        raise ConfigError(EXIT_USAGE, ["the jsonschema package is missing: reinstall with pip install -e ."]) from exc
    validator = validators.validator_for(schema)(schema)
    for error in sorted(validator.iter_errors(dict(values)), key=lambda e: [str(p) for p in e.absolute_path]):
        where = ".".join(str(p) for p in error.absolute_path) or "top level"
        problems.append(f"{where}: {error.message}")
    return problems


def parse_value(raw: str, sub_schema: dict | None = None) -> object:
    """A command-line or environment string as a config value: a TOML literal, else the string itself.

    When the schema says the key is a string (and nothing else), the text is kept exactly as given.
    """
    kinds = (sub_schema or {}).get("type")
    kinds = [kinds] if isinstance(kinds, str) else list(kinds or [])
    if kinds and set(kinds) <= {"string"}:
        return raw
    try:
        value = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        return raw
    if isinstance(value, _dt.date | _dt.time):  # TOML dates come back as objects; keep the text
        return raw
    return value


# --- ${VAR} expansion -------------------------------------------------------------------------------------------

def expand(values: object, env: Mapping[str, str], key: str = "") -> tuple[object, list[tuple[str, str]]]:
    """Replace `${VAR}` in every string (nested); return the result and the (VAR, dotted key) pairs left unset."""
    unset: list[tuple[str, str]] = []
    if isinstance(values, dict):
        out = {}
        for k, v in values.items():
            out[k], more = expand(v, env, f"{key}.{k}" if key else k)
            unset += more
        return out, unset
    if isinstance(values, list):
        items = []
        for v in values:
            item, more = expand(v, env, key)
            items.append(item)
            unset += more
        return items, unset
    if not isinstance(values, str):
        return values, unset

    def one(match: re.Match) -> str:
        if match.group(1):
            return "${" + match.group(1) + "}"
        name = match.group(2)
        if name in env:
            return env[name]
        if (name, key) not in unset:
            unset.append((name, key))
        return match.group(0)

    return _VARREF.sub(one, values), unset


# --- discovery and loading --------------------------------------------------------------------------------------

def find(tool: str, workspace: str | Path | None = None, *, env: Mapping[str, str] | None = None,
         cwd: Path | None = None, user_dir: Path | None = None) -> tuple[Path | None, str, list[Path]]:
    """Locate `<tool>.toml`: (file or None, how it was found, the places searched)."""
    env = os.environ if env is None else env
    name = f"{tool}.toml"
    searched: list[Path] = []

    def candidate(p: str | Path) -> Path:
        p = Path(p).expanduser()
        return p if p.suffix.lower() == ".toml" and not p.is_dir() else p / name

    home_var = f"{tool.upper()}_HOME"
    for given, how in ((workspace, "--workspace"), (env.get(home_var) or None, home_var)):
        if given is None:
            continue
        c = candidate(given)
        searched.append(c)
        if c.is_file():
            return c, how, searched
        raise ConfigError(EXIT_USAGE, [f"no {name} at {c} (named by {how})",
                                       f"create one with `{tool} init <folder>` or fix {how}"])
    here = Path.cwd() if cwd is None else Path(cwd)
    for folder in (here, *here.parents):
        c = folder / name
        searched.append(c)
        if c.is_file():
            return c, "the current folder or a parent", searched
    c = (Path.home() / ".config" / tool if user_dir is None else Path(user_dir)) / name
    searched.append(c)
    if c.is_file():
        return c, "the user config folder", searched
    return None, "nothing", searched


def read_toml(path: Path) -> dict:
    """Parse a TOML file (a UTF-8 BOM is tolerated); exit-2 ConfigError when it does not parse."""
    try:
        return tomllib.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(EXIT_USAGE, [f"{path}: not readable TOML: {exc}"]) from exc


def load(tool: str, workspace: str | Path | None = None, *, strict: bool = True, env: Mapping[str, str] | None = None,
         cwd: Path | None = None, user_dir: Path | None = None, schema_file: object = DEFAULT) -> Config:
    """Find, read, override, validate and expand `tool`'s configuration (see the module docstring)."""
    env = os.environ if env is None else env
    file, found_by, searched = find(tool, workspace, env=env, cwd=cwd, user_dir=user_dir)
    file_values: dict = {}
    if file is not None:
        if is_shipped(file, tool):
            refused = check_shipped_text(file.name, file.read_text(encoding="utf-8-sig"))
            if refused:
                raise ConfigError(EXIT_ABSOLUTE, refused)
        file_values = read_toml(file)
    schema = load_schema(schema_file)
    values = schema_defaults(schema)
    sources = {k: "default" for k, _ in flatten(values)}
    values = _deep_merge(values, file_values)
    sources.update({k: "file" for k, _ in flatten(file_values)})

    aliases = file_values.get("env_aliases") or {}
    if not isinstance(aliases, dict):
        raise ConfigError(EXIT_USAGE, ["env_aliases is a table of \"OLD_NAME\" = \"section.key\""])
    for old, target in aliases.items():
        if old in env:
            dotted = check_key(str(target))
            _set_in(values, dotted, parse_value(env[old], schema_at(schema, dotted)))
            sources[dotted] = f"env alias {old}"
    prefix = f"{tool.upper()}_"
    for name in sorted(env):
        if not name.upper().startswith(prefix) or "__" not in name[len(prefix):]:
            continue
        parts = name[len(prefix):].lower().split("__")
        if not all(_BARE_KEY.match(p) for p in parts):
            continue
        dotted = ".".join(parts)
        _set_in(values, dotted, parse_value(env[name], schema_at(schema, dotted)))
        sources[dotted] = f"env {name}"

    # Without an instance file there is nothing of the user's to validate (and a schema's required keys would
    # only bury the real message, "no <tool>.toml found"); the always-on rules still apply.
    problems = validate(values, schema if file is not None else None)
    if problems:
        where = file if file is not None else f"{tool} configuration"
        raise ConfigError(EXIT_USAGE, [f"{where}: {p}" for p in problems])
    expanded, unset = expand(values, env)
    root = None
    if file is not None:
        root_value = expanded.get("paths", {}).get("root", ".") if isinstance(expanded.get("paths"), dict) else "."
        root = (file.parent / Path(str(root_value)).expanduser()).resolve()
    cfg = Config(tool=tool, file=file, root=root, found_by=found_by, values=expanded, sources=sources,
                 unset=unset, searched=searched, file_values=file_values)
    if strict and cfg.unset:
        raise ConfigError(EXIT_USAGE, cfg.unset_lines())
    return cfg


# --- TOML text -------------------------------------------------------------------------------------------------

def toml_key(key: str) -> str:
    return key if _BARE_KEY.match(key) else json.dumps(key, ensure_ascii=False)


def toml_value(value: object) -> str:
    """One value as TOML text (strings with a backslash and no quote become literal strings: Windows paths)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "nan"
        if value in (float("inf"), float("-inf")):
            return "inf" if value > 0 else "-inf"
        return repr(value)
    if isinstance(value, str):
        if "\\" in value and "'" not in value and not any(ord(c) < 32 or ord(c) == 127 for c in value):
            return f"'{value}'"
        return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{toml_key(str(k))} = {toml_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, _dt.date | _dt.time):
        return value.isoformat()
    raise ConfigError(EXIT_USAGE, [f"TOML cannot hold this value: {value!r}"])


def _header_name(match: re.Match) -> tuple[str, ...]:
    return tuple(p.strip().strip("\"'") for p in match.group(1).split("."))


def _split_comment(rest: str) -> tuple[str, str]:
    """Split the text after `key =` into (value text, trailing comment) where that is unambiguous."""
    for i, ch in enumerate(rest):
        if ch == "#":
            try:
                tomllib.loads("v = " + rest[:i])
            except tomllib.TOMLDecodeError:
                continue
            return rest[:i].rstrip(), rest[i:]
    return rest.rstrip(), ""


def edit_toml_text(text: str, dotted: str, value: object) -> str | None:
    """`text` with `dotted` set to `value`, comments and layout kept; None when no simple one-line edit fits."""
    parts = dotted.split(".")
    table, key = tuple(parts[:-1]), parts[-1]
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    headers = [(i, _header_name(m)) for i, line in enumerate(lines) if (m := _HEADER.match(line))]
    if table:
        starts = [i for i, name in headers if name == table]
        if not starts:
            tail = "" if not text or text.endswith(("\n", "\r\n")) else newline
            block = f"{newline}[{'.'.join(toml_key(p) for p in table)}]{newline}{toml_key(key)} = {toml_value(value)}{newline}"
            return text + tail + block
        start = starts[0] + 1
    else:
        start = 0
    end = next((i for i, _ in headers if i >= start), len(lines))
    key_line = re.compile(rf"^(\s*)({re.escape(key)}|\"{re.escape(key)}\"|'{re.escape(key)}')\s*=(.*?)(\r?\n)?$")
    last_entry = None
    for i in range(start, end):
        m = key_line.match(lines[i])
        if m:
            value_text, comment = _split_comment(m.group(3))
            try:
                tomllib.loads("v = " + value_text)
            except tomllib.TOMLDecodeError:
                return None  # a multi-line value: leave it to the full rewrite
            ending = m.group(4) or ""
            head = f"{m.group(1)}{m.group(2)} = {toml_value(value)}"
            if comment:  # keep the comment, in its old column when the new value leaves room
                column = len(lines[i]) - len(ending) - len(comment)
                head = (head.ljust(column) if column > len(head) + 1 else head + "  ") + comment
            lines[i] = head + ending
            return "".join(lines)
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            last_entry = i
    insert_at = (last_entry + 1) if last_entry is not None else start
    new_line = f"{toml_key(key)} = {toml_value(value)}{newline}"
    if insert_at > 0 and not lines[insert_at - 1].endswith(("\n", "\r\n")):
        lines[insert_at - 1] += newline
    lines.insert(insert_at, new_line)
    return "".join(lines)


def set_value(tool: str, dotted: str, raw: str, *, workspace: str | Path | None = None,
              env: Mapping[str, str] | None = None, cwd: Path | None = None, user_dir: Path | None = None,
              schema_file: object = DEFAULT) -> tuple[Path, object, bool, str]:
    """Write `dotted = raw` into the instance TOML; returns (file, value, comments kept, how the file was found).

    Validated before writing (schema and rules on the whole file plus defaults); refused -> exit-2 ConfigError
    and the file is unchanged. Written atomically under the instance's store lock.
    """
    check_key(dotted)
    env = os.environ if env is None else env
    file, how, searched = find(tool, workspace, env=env, cwd=cwd, user_dir=user_dir)
    if file is None:
        where = ", ".join(str(p) for p in searched)
        raise ConfigError(EXIT_USAGE, [f"no {tool}.toml found; searched: {where}",
                                       f"create one with `{tool} init <folder>` or pass --workspace <folder>"])
    schema = load_schema(schema_file)
    value = parse_value(raw, schema_at(schema, dotted))
    with store.lock(file.parent, tool):
        # Bytes, not read_text: text mode would turn CRLF into LF, and a Windows-edited file must stay CRLF.
        text = file.read_bytes().decode("utf-8-sig")
        current = read_toml(file)
        updated = copy.deepcopy(current)
        _set_in(updated, dotted, value)
        problems = validate(_deep_merge(schema_defaults(schema), updated), schema)
        if problems:
            raise ConfigError(EXIT_USAGE, [f"refused: {dotted} = {raw}; {file.name} is unchanged", *problems])
        new_text = edit_toml_text(text, dotted, value)
        kept = True
        try:
            if new_text is None or tomllib.loads(new_text) != updated:
                raise ValueError
        except (ValueError, tomllib.TOMLDecodeError):
            import tomli_w

            new_text = f"# rewritten by `{tool} config set`: the comments of the earlier file were not kept\n"
            new_text += tomli_w.dumps(updated)
            if "\r\n" in text:  # the rewrite keeps the file's line endings too
                new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
            kept = False
        if is_shipped(file, tool):
            refused = check_shipped_text(file.name, new_text)
            if refused:
                raise ConfigError(EXIT_ABSOLUTE, refused)
        store.write_text(file, new_text)
    return file, value, kept, how


# --- config show ------------------------------------------------------------------------------------------------

def _redact(value: object, key: str, root: Path | None) -> object:
    if isinstance(value, list):
        return [_redact(v, key, root) for v in value]
    if isinstance(value, dict):
        return {k: _redact(v, key, root) for k, v in value.items()}
    if not isinstance(value, str) or "://" in value:
        return value
    p = Path(value).expanduser()
    if not (p.is_absolute() or ABSOLUTE.search(value)):
        return value
    if root is not None:
        try:
            rel = p.resolve().relative_to(root)
            return rel.as_posix() or "."
        except (ValueError, OSError):
            pass
    return f"<{key.rsplit('.', 1)[-1]}>"


def show_lines(cfg: Config, *, redact_paths: bool = False, key: str | None = None) -> list[str]:
    """The `config show` lines: where the file is, then `key = value  # source` per value."""
    if cfg.file is None:
        head = [cfg.not_found_line()]
    elif redact_paths:
        try:
            shown = cfg.file.resolve().relative_to(cfg.root).as_posix() if cfg.root else cfg.file.name
        except ValueError:
            shown = cfg.file.name
        head = [f"file: {shown} (found by {cfg.found_by})", "root: <root>"]
    else:
        head = [f"file: {cfg.file} (found by {cfg.found_by})", f"root: {cfg.root}"]
    unset_by_key: dict[str, list[str]] = {}
    for var, k in cfg.unset:
        unset_by_key.setdefault(k, []).append(var)
    rows = []
    for dotted, value in cfg.flat():
        if key and not (dotted == key or dotted.startswith(key + ".")):
            continue
        shown_value = _redact(value, dotted, cfg.root) if redact_paths else value
        source = cfg.sources.get(dotted, "file")
        if dotted in unset_by_key:
            source += ", unset: " + " ".join("${" + v + "}" for v in unset_by_key[dotted])
        rows.append((f"{dotted} = {toml_value(shown_value)}", source))
    width = min(max((len(r[0]) for r in rows), default=0), 60)
    return head + [f"{left.ljust(width)}  # {source}" for left, source in rows]


# --- the `config` verb ------------------------------------------------------------------------------------------

def add_workspace_arg(parser: argparse.ArgumentParser) -> None:
    """The `--workspace` option every instance-reading verb takes."""
    parser.add_argument("--workspace", metavar="PATH", default=None,
                        help="the instance folder or its TOML file; default: the tool's HOME variable, then upward "
                             "from the current folder, then the user config folder")


def _fail(exc: ConfigError) -> int:
    for line in exc.lines:
        print(line, file=sys.stderr)
    return exc.code


def _cmd_show(args: argparse.Namespace) -> int:
    try:
        cfg = load(args.tool, args.workspace, strict=False)
        if args.key:
            check_key(args.key)
        lines = show_lines(cfg, redact_paths=args.redact_paths, key=args.key)
    except ConfigError as exc:
        return _fail(exc)
    for line in lines:
        print(line)
    if args.key and len(lines) <= (1 if cfg.file is None else 2):
        print(f"no value at {args.key}", file=sys.stderr)
        return EXIT_USAGE
    for line in cfg.unset_lines():
        print(line, file=sys.stderr)
    return EXIT_USAGE if (cfg.file is None or cfg.unset) else 0


def _cmd_set(args: argparse.Namespace) -> int:
    try:
        file, value, kept, how = set_value(args.tool, args.key, args.value, workspace=args.workspace)
    except ConfigError as exc:
        return _fail(exc)
    except store.Held as held:
        _journal_held(args, held)
        raise
    print(f"set {args.key} = {toml_value(value)} in {file.name} (found by {how})")  # no absolute path: doc-safe
    if not kept:
        print("note: the file was rewritten as plain TOML; its earlier comments were not kept")
    return 0


def _journal_held(args: argparse.Namespace, held: store.Held) -> None:
    """Best effort: put the exit-4 line into today's journal of the instance (never raises)."""
    try:
        from . import journal

        cfg = load(args.tool, args.workspace, strict=False)
        journal.note(f"stuck: {held.line}", tool=args.tool, journal_dir=cfg.path("paths.journal"), timeout=2.0)
    except (ConfigError, OSError, store.Held):
        pass


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `config show` and `config set` to the tool's command line."""
    parser = subparsers.add_parser(
        "config", help="show or set configuration values with their source",
        description="Show or set the instance configuration. Values come from the schema defaults, the instance "
                    "TOML, [env_aliases] and <TOOL>_<SECTION>__<KEY> variables, in rising precedence.")
    inner = parser.add_subparsers(dest="config_action", metavar="<action>")
    show = inner.add_parser("show", help="print every value with its source: default, file, env or env alias",
                            description="Print every value with its source. Exit 2 when no instance TOML is found "
                                        "or a ${VAR} is unset.")
    show.add_argument("key", nargs="?", help="only this dotted key or section, for example comfyui.url or paths")
    show.add_argument("--redact-paths", action="store_true",
                      help="print paths inside the instance relative to it and other absolute paths as <key>")
    add_workspace_arg(show)
    show.set_defaults(func=_cmd_show)
    put = inner.add_parser("set", help="write one value into the instance TOML, validated first",
                           description="Write one value into the instance TOML atomically. The new file is "
                                       "validated first; a refused value leaves the file unchanged and exits 2.")
    put.add_argument("key", help="a dotted key, for example comfyui.url")
    put.add_argument("value", help="a TOML literal such as 50, true or [1, 2]; anything else is a string")
    add_workspace_arg(put)
    put.set_defaults(func=_cmd_set)
