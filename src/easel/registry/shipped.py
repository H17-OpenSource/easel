"""Where the shipped data and schemas are, and the two small readers every registry module shares.

The repository ships `data/` (folders, compat rules, content markers) and `schemas/` beside `src/`. An instance
may hold its own copies under `<home>/data/`; each reader says whether the instance file adds to the shipped
one (compat rules) or replaces it (folders, markers). Validation uses `schemas/<name>.schema.json` when the
schema is there; an install without the schemas folder skips that check rather than refusing to work.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # src/easel/registry/shipped.py -> the repository root
DATA = REPO / "data"
SCHEMAS = REPO / "schemas"
EXIT_PROBLEMS = 1
EXIT_USAGE = 2


class DataError(Exception):
    """A data file that cannot be used (unreadable, not YAML, or not valid against its schema): exit 2."""

    def __init__(self, lines: list[str]) -> None:
        super().__init__("\n".join(lines))
        self.lines = lines


def load_yaml(path: Path, what: str) -> object:
    """Parse a YAML file; DataError naming `what` when it is missing or does not parse."""
    import yaml

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise DataError([f"{what} not found: {Path(path).name}"]) from exc
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DataError([f"{what} is not readable YAML: {reason(exc)}"]) from exc


def reason(exc: BaseException) -> str:
    """An exception as one line for a person, without the absolute paths its text may carry.

    An OSError's text ends in the file's full path, and store.read_record puts the path into its ValueError, so
    printing `str(exc)` would leak `<home>`'s location into output that ends up in docs and hand-offs. An OSError
    gives its reason alone (`strerror`); any other text keeps its first line with each absolute path as `<path>`.
    """
    import re

    from easel._common.config import ABSOLUTE

    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    lines = str(exc).splitlines()
    text = lines[0] if lines else type(exc).__name__
    span = re.compile(f"(?:{ABSOLUTE.pattern})[^'\"\\n]*?(?=: |'|\"|$)")
    return span.sub("<path>", text)


@cache
def _validator(name: str):
    path = SCHEMAS / f"{name}.schema.json"
    if not path.is_file():
        return None
    from jsonschema import Draft202012Validator

    return Draft202012Validator(json.loads(path.read_text(encoding="utf-8")))


def schema_errors(name: str, data: object) -> list[str]:
    """One line per problem of `data` against `schemas/<name>.schema.json` (none when the schema is absent)."""
    validator = _validator(name)
    if validator is None:
        return []
    errors = sorted(validator.iter_errors(data), key=lambda e: [str(p) for p in e.absolute_path])
    return [f"at /{'/'.join(map(str, e.absolute_path))}: {e.message[:200]}" for e in errors[:10]]


def shown(path: Path, root: Path | None) -> str:
    """A path for output: relative to `root` when inside it (doc-safe, no absolute path), else the file name."""
    path = Path(path)
    for base in (root, REPO):
        if base is None:
            continue
        try:
            return path.resolve().relative_to(Path(base).resolve()).as_posix()
        except (ValueError, OSError):
            continue
    return path.name
