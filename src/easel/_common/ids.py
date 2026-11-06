"""The clock and the id shapes, in one place so tests and doc generation can freeze both.

The clock: `now(tool)` is the current local time, unless `<TOOL>_NOW` holds an ISO date-time (tests and doc
generation set `2030-01-01T00:00:00+00:00`). While it is set, every id, journal line and `at` field follows it,
and the first call in a process prints `clock frozen (<TOOL>_NOW set)` to stderr, so a live session cannot use a
frozen clock without seeing it. The line goes to stderr only: doc generation keeps stdout and `qa docs` rejects
any output block that contains it.

Id shapes (plan, file-based registries):
- images `img-<sha256[:12]>`, resources `<sha256[:16]>` (`image_id`, `resource_name`);
- recipes `rcp-YYYYMMDD-NNN`, jobs `job-YYYYMMDD-NNN` (`next_seq_id`, `claim_seq_dir`);
- runs `run-YYYYMMDD-<slug>`, claimed by an exclusive mkdir (`claim_run_dir`);
- decisions `dec-NNNN` (`next_decision_id`).
For example, under the frozen clock the first recipe is `rcp-20300101-001` and the demo run is
`run-20300101-001`.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

_warned: set[str] = set()

SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class IdError(ValueError):
    """A malformed clock value, slug or hash; the message says which and how to fix it."""


def now_var(tool: str) -> str:
    """The name of the variable that freezes `tool`'s clock, for example `EASEL_NOW`."""
    return f"{tool.upper()}_NOW"


def now(tool: str) -> datetime:
    """The current time as an aware datetime: `<TOOL>_NOW` when set (warning once on stderr), else local time."""
    var = now_var(tool)
    raw = os.environ.get(var, "").strip()
    if not raw:
        return datetime.now().astimezone()
    try:
        frozen = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise IdError(f"{var} is not an ISO date-time, for example 2030-01-01T00:00:00+00:00: {raw!r}") from exc
    if frozen.tzinfo is None:
        frozen = frozen.replace(tzinfo=UTC)
    if var not in _warned:
        _warned.add(var)
        print(f"clock frozen ({var} set)", file=sys.stderr, flush=True)
    return frozen


def now_iso(tool: str) -> str:
    """`now(tool)` as an ISO string with seconds and offset: the shape of every `at` field."""
    return now(tool).isoformat(timespec="seconds")


def today(tool: str) -> str:
    """The date part of `now(tool)` as YYYY-MM-DD (journal file names)."""
    return now(tool).strftime("%Y-%m-%d")


def stamp(tool: str) -> str:
    """The date part of `now(tool)` as YYYYMMDD (the middle of every dated id)."""
    return now(tool).strftime("%Y%m%d")


# --- content ids ------------------------------------------------------------------------------------------------

def _sha(sha256: str) -> str:
    value = sha256.strip().lower()
    if not SHA256.match(value):
        raise IdError(f"not a sha256 hex digest (64 hex characters): {sha256!r}")
    return value


def image_id(sha256: str) -> str:
    """`img-` plus the first 12 hex characters of the image file's sha256."""
    return "img-" + _sha(sha256)[:12]


def resource_name(sha256: str) -> str:
    """The first 16 hex characters of a model file's sha256: the stem of `registry/resources/<sha16>.yaml`."""
    return _sha(sha256)[:16]


# --- dated sequence ids -----------------------------------------------------------------------------------------

def check_slug(slug: str) -> str:
    """Return `slug` if it is lowercase letters, digits and dashes (at most 64), else raise IdError."""
    if not SLUG.match(slug):
        raise IdError(f"a slug is lowercase letters, digits and dashes, starting with a letter or digit: {slug!r}")
    return slug


def next_seq_id(prefix: str, existing: Iterable[str], tool: str, width: int = 3) -> str:
    """The next `<prefix>-YYYYMMDD-NNN` for today, one above the highest of today's ids in `existing`."""
    check_slug(prefix)
    day = stamp(tool)
    shape = re.compile(rf"^{re.escape(prefix)}-{day}-(\d{{{width},}})(?:\.[A-Za-z0-9]+)?$")
    numbers = [int(m.group(1)) for m in map(shape.match, existing) if m]
    return f"{prefix}-{day}-{(max(numbers) + 1 if numbers else 1):0{width}d}"


def claim_seq_dir(parent: Path, prefix: str, tool: str, width: int = 3) -> tuple[str, Path]:
    """Create the folder of the next free `<prefix>-YYYYMMDD-NNN` under `parent` and return (id, folder).

    The exclusive mkdir is the claim: two processes asking at once get two different ids.
    """
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(1000):
        new = next_seq_id(prefix, (p.name for p in parent.iterdir()), tool, width)
        target = parent / new
        try:
            target.mkdir()
        except FileExistsError:
            continue
        return new, target
    raise IdError(f"could not claim a new {prefix} id under {parent} after 1000 tries")


def claim_run_dir(parent: Path, slug: str, tool: str) -> tuple[str, Path]:
    """Create `parent/run-YYYYMMDD-<slug>` exclusively and return (run id, folder).

    An existing folder of that name raises FileExistsError: a run id is never reused; pick another slug.
    """
    run_id = f"run-{stamp(tool)}-{check_slug(slug)}"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / run_id
    target.mkdir()  # exclusive: FileExistsError when the id is taken
    return run_id, target


def next_decision_id(existing: Iterable[str]) -> str:
    """The next `dec-NNNN`, one above the highest in `existing` (file stems or ids)."""
    shape = re.compile(r"^dec-(\d{4,})(?:\.[A-Za-z0-9]+)?$")
    numbers = [int(m.group(1)) for m in map(shape.match, existing) if m]
    return f"dec-{(max(numbers) + 1 if numbers else 1):04d}"
