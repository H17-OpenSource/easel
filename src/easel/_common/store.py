"""The single write path for everything a tool keeps on disk: records, event logs, generated views, the lock.

Three kinds of file (plan, file-based registries, "Storage model"):
1. Records: one YAML (human) or JSON (tool) file per thing, first key `schema: "<repo>/<name>/<int>"`. Written
   atomically: `<path>.tmp-<pid>`, fsync, `os.replace`. Windows refuses to replace a file another process has
   open (a backup tool, a review page or an antivirus), so the replace retries on PermissionError: about 20
   attempts, 50 ms doubling to 500 ms, at most 5 s. Then the tool exits 4 with the STUCK line
   `another process holds <file> (backup, review page, antivirus); retry in a minute`. `replace(src, dst)` is
   the same retrying rename for a file written some other way (a finished download).
2. Event logs: append-only `*.jsonl`; every line carries `at`, `v` and `source`; one `write()` per line under
   the store lock. Readers reduce latest-per-key (`latest_by_key`) and skip a torn last line, so a writer killed
   mid-line never breaks a reader. The next append starts on a fresh line: in a `*.jsonl` file a torn fragment
   that is not JSON is moved to `<file>.torn` (appended there, one per line) and cut from the log, so the log
   stays clean and the fragment stays findable (`JsonlRead.torn_file`); other files just get a newline first.
3. Generated views: `index.jsonl` + `INDEX.md` ("generated, do not edit"), rebuilt from the records.

The lock is an OS lock on an open handle of the persistent file `<store>/.lock` (`msvcrt.locking` on Windows,
`fcntl.flock` elsewhere). It dies with the process, so a crashed or killed writer never strands it and PID reuse
cannot fool it; the `{pid, host, at, tool}` written inside are diagnostics for the STUCK message only. A writer
waits up to 30 s, then exits 4. Only contention is waited out: a folder whose file system refuses locks (some
network or FUSE mounts) fails at once with `LockUnsupported`, also exit 4, naming the error. Inside one process
the lock is re-entrant for the thread holding it, and other threads wait their turn.

`Held` is the exit-4 exception: a SystemExit subclass, so a generic `except Exception` cannot swallow it; a
caller that can record it (for example `journal stuck`) catches `store.Held` by name and reads `.line`.
"""

from __future__ import annotations

import errno
import json
import os
import socket
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from . import ids

EXIT_HELD = 4
LOCK_NAME = ".lock"
LOCK_TIMEOUT_S = 30.0
LOCK_POLL_S = 0.05
REPLACE_ATTEMPTS = 20
REPLACE_FIRST_DELAY_S = 0.05
REPLACE_MAX_DELAY_S = 0.5
REPLACE_BUDGET_S = 5.0
# Windows locks a byte range, and a locked range cannot be read by anyone else. The lock therefore sits on one
# byte far past the diagnostics, so a waiting process can still read who holds it.
_WIN_LOCK_OFFSET = 1 << 30
_DIAG_SIZE = 256

HELD_FILE = "another process holds {file} (backup, review page, antivirus); retry in a minute"
HELD_LOCK = ("another process holds the store lock {lock} {holder}; waited {waited:.0f} s; "
             "docs/STUCK.md section Locks says how to confirm the holder is gone")
NO_LOCKS = ("the folder of {lock} does not support file locks ({error}); keep the instance on a local disk, or "
            "report it to the operator")
TORN_SUFFIX = ".torn"
_TAIL_BLOCK = 1 << 16  # a log's last line is found by reading backwards in blocks of this size
# The errors a non-blocking lock attempt gives when another process holds the lock: flock says EWOULDBLOCK or
# EAGAIN, msvcrt.locking says EACCES (EDEADLK after its own retries). Anything else is not contention.
_CONTENTION = frozenset({errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK})

T = TypeVar("T")


class Held(SystemExit):
    """A file or the store lock stayed held by another process: the tool exits 4 after printing `.line`."""

    def __init__(self, line: str, path: Path) -> None:
        super().__init__(EXIT_HELD)
        self.line = line
        self.path = path

    def __str__(self) -> str:
        return self.line


class LockUnsupported(Held):
    """The folder's file system refuses OS locks: not contention, so nothing is waited for (exit 4 at once)."""


def _held(line: str, path: Path, kind: type[Held] = Held) -> Held:
    """Print the STUCK line on stderr and return the exception to raise."""
    print(f"STUCK: {line}", file=sys.stderr, flush=True)
    return kind(line, path)


def _retry_permission(action: Callable[[], T], path: Path, budget_s: float | None = None) -> T:
    """Run `action`, retrying on PermissionError (about 20 tries, 50 ms doubling to 500 ms, <= 5 s), then Held."""
    budget = REPLACE_BUDGET_S if budget_s is None else budget_s
    delay = REPLACE_FIRST_DELAY_S
    start = time.monotonic()
    for attempt in range(1, REPLACE_ATTEMPTS + 1):
        try:
            return action()
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS or time.monotonic() - start + delay > budget:
                break
            time.sleep(delay)
            delay = min(delay * 2, REPLACE_MAX_DELAY_S)
    raise _held(HELD_FILE.format(file=path), path)


def replace(src: Path, dst: Path, *, budget_s: float | None = None) -> Path:
    """`os.replace(src, dst)` with the PermissionError retry of every atomic write; exit 4 (Held) when it stays
    refused. For a file finished some other way, for example a verified download that an antivirus still scans."""
    dst = Path(dst)
    _retry_permission(lambda: os.replace(src, dst), dst, budget_s)
    return dst


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _fsync_dir(folder: Path) -> None:
    """Make a rename durable on POSIX (Windows has no directory handles for this; NTFS journals the rename)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(folder, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _tmp_path(path: Path) -> Path:
    suffix = f".tmp-{os.getpid()}"
    if threading.current_thread() is not threading.main_thread():
        suffix += f"-{threading.get_ident()}"
    return path.with_name(path.name + suffix)


# --- atomic records ---------------------------------------------------------------------------------------------

def write_bytes(path: Path, data: bytes, *, budget_s: float | None = None) -> Path:
    """Write `data` to `path` atomically: a temp file beside it, fsync, then os.replace with the retry loop."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(path)
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _retry_permission(lambda: os.replace(tmp, path), path, budget_s)
    except BaseException:
        _unlink_quiet(tmp)
        raise
    _fsync_dir(path.parent)
    return path


def write_text(path: Path, text: str, *, budget_s: float | None = None) -> Path:
    """Write UTF-8 text atomically; newlines are written as given (LF), never translated."""
    return write_bytes(path, text.encode("utf-8"), budget_s=budget_s)


def dump_json(obj: object) -> str:
    """The JSON text a store writes: two-space indent, UTF-8 kept readable, trailing newline."""
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def dump_yaml(obj: object) -> str:
    """The YAML text a store writes: key order kept, UTF-8 kept readable, block style."""
    import yaml

    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, default_flow_style=False)


def write_json(path: Path, obj: object, *, budget_s: float | None = None) -> Path:
    """Write `obj` as JSON atomically."""
    return write_text(path, dump_json(obj), budget_s=budget_s)


def write_yaml(path: Path, obj: object, *, budget_s: float | None = None) -> Path:
    """Write `obj` as YAML atomically."""
    return write_text(path, dump_yaml(obj), budget_s=budget_s)


def write_record(path: Path, schema: str, data: Mapping[str, object], *, budget_s: float | None = None) -> Path:
    """Write a record with `schema` as its first key; YAML for .yaml/.yml, JSON for .json."""
    record = {"schema": schema, **{k: v for k, v in data.items() if k != "schema"}}
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml"):
        return write_yaml(path, record, budget_s=budget_s)
    if suffix == ".json":
        return write_json(path, record, budget_s=budget_s)
    raise ValueError(f"a record is .yaml, .yml or .json, not {suffix or 'no suffix'}: {path}")


def read_record(path: Path, schema_prefix: str | None = None) -> dict:
    """Read a YAML or JSON record; with `schema_prefix` (for example "<tool>/resource/"), check its schema key."""
    path = Path(path)
    text = _retry_permission(lambda: path.read_text(encoding="utf-8-sig"), path)  # a BOM from an editor is fine
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        import yaml

        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"a record is a mapping at the top level: {path}")
    if schema_prefix is not None and not str(data.get("schema", "")).startswith(schema_prefix):
        raise ValueError(f"{path}: schema {data.get('schema')!r} is not a {schema_prefix}<n> record")
    return data


def iter_records(folder: Path, pattern: str = "*.yaml") -> Iterator[tuple[Path, dict]]:
    """Every record file in `folder` matching `pattern`, sorted by name, as (path, data)."""
    for path in sorted(Path(folder).glob(pattern)):
        if path.is_file() and ".tmp-" not in path.name:
            yield path, read_record(path)


# --- the store lock ---------------------------------------------------------------------------------------------

@dataclass
class _LockEntry:
    rlock: threading.RLock = field(default_factory=threading.RLock)
    handle: object = None
    depth: int = 0


_entries: dict[str, _LockEntry] = {}
_entries_guard = threading.Lock()


def _try_os_lock(handle) -> None:
    """One non-blocking attempt at the OS lock; OSError when another process holds it."""
    if os.name == "nt":
        import msvcrt

        handle.seek(_WIN_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _os_unlock(handle) -> None:
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(_WIN_LOCK_OFFSET)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    finally:
        handle.close()


def holder_of(lock_path: Path) -> dict | None:
    """The diagnostics the current or last holder wrote into the lock file, or None when unreadable."""
    try:
        with open(lock_path, "rb") as handle:
            raw = handle.read(_DIAG_SIZE)
        data = json.loads(raw.decode("utf-8").strip() or "null")
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _describe_holder(lock_path: Path) -> str:
    info = holder_of(lock_path)
    if not info:
        return "(holder unknown)"
    return f"(pid {info.get('pid')} on {info.get('host')} since {info.get('at')}, tool {info.get('tool')})"


def _write_diagnostics(handle, tool: str) -> None:
    info = {"pid": os.getpid(), "host": socket.gethostname(), "at": ids.now_iso(tool), "tool": tool}
    raw = json.dumps(info).encode("utf-8")[: _DIAG_SIZE - 1]
    try:
        handle.seek(0)
        handle.write(raw.ljust(_DIAG_SIZE - 1) + b"\n")
        handle.flush()
    except OSError:
        pass  # diagnostics only; the lock itself is what counts


def _acquire_os_lock(lock_path: Path, tool: str, timeout: float) -> object:
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    handle = os.fdopen(fd, "r+b", buffering=0)
    start = time.monotonic()
    while True:
        try:
            _try_os_lock(handle)
            break
        except OSError as exc:
            if exc.errno not in _CONTENTION:  # ENOLCK, EOPNOTSUPP, EINVAL...: waiting would only hide it
                handle.close()
                name = errno.errorcode.get(exc.errno or 0, f"errno {exc.errno}")
                raise _held(NO_LOCKS.format(lock=lock_path, error=name), lock_path, LockUnsupported) from None
            waited = time.monotonic() - start
            if waited >= timeout:
                holder = _describe_holder(lock_path)
                handle.close()
                raise _held(HELD_LOCK.format(lock=lock_path, holder=holder, waited=waited), lock_path) from None
            time.sleep(LOCK_POLL_S)
    _write_diagnostics(handle, tool)
    return handle


@contextmanager
def lock(store: Path, tool: str, timeout: float | None = None) -> Iterator[Path]:
    """Hold the store lock `<store>/.lock` for the `with` block; wait up to `timeout` (30 s), then exit 4.

    The lock file is created once and never deleted. Re-entrant for the thread that holds it.
    """
    wait = LOCK_TIMEOUT_S if timeout is None else timeout
    root = Path(store)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_NAME
    key = os.path.normcase(str(lock_path.resolve()))
    with _entries_guard:
        entry = _entries.setdefault(key, _LockEntry())
    if not entry.rlock.acquire(timeout=wait):
        raise _held(HELD_LOCK.format(lock=lock_path, holder="(another thread of this process)", waited=wait),
                    lock_path)
    try:
        if entry.depth == 0:
            entry.handle = _acquire_os_lock(lock_path, tool, wait)
        entry.depth += 1
        try:
            yield lock_path
        finally:
            entry.depth -= 1
            if entry.depth == 0:
                handle, entry.handle = entry.handle, None
                _os_unlock(handle)
    finally:
        entry.rlock.release()


# --- event logs -------------------------------------------------------------------------------------------------

def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def torn_path(path: Path) -> Path:
    """`<file>.torn`: where an append moves a torn last line of a JSONL log."""
    path = Path(path)
    return path.with_name(path.name + TORN_SUFFIX)


def _tail_start(fd: int, size: int) -> int:
    """The offset just past the last newline of the open file (0 when it has none), read backwards in blocks."""
    end = size
    while end > 0:
        start = max(0, end - _TAIL_BLOCK)
        os.lseek(fd, start, os.SEEK_SET)
        block = os.read(fd, end - start)
        at = block.rfind(b"\n")
        if at >= 0:
            return start + at + 1
        end = start
    return 0


def _parses(fragment: bytes) -> bool:
    try:
        json.loads(fragment.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return False
    return True


def append_bytes_locked(path: Path, data: bytes, *, jsonl: bool | None = None) -> None:
    """Append `data` in one write, starting on a fresh line. Call under `lock()`.

    When the file ends without a newline: in a JSONL log (`jsonl`, default: the `.jsonl` suffix) a last line
    that does not parse is a torn record; it is appended to `<file>.torn` and cut from the log before the write,
    so the log never keeps a broken line in its middle. A last line that parses, and any other file (a journal),
    keeps its text and only gets the newline.
    """
    path = Path(path)
    jsonl = path.suffix.lower() == ".jsonl" if jsonl is None else jsonl
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
    fd = _retry_permission(lambda: os.open(path, flags, 0o644), path)
    try:
        size = os.fstat(fd).st_size
        if size:
            os.lseek(fd, size - 1, os.SEEK_SET)
            if os.read(fd, 1) != b"\n":
                start = _tail_start(fd, size) if jsonl else size
                os.lseek(fd, start, os.SEEK_SET)
                fragment = os.read(fd, size - start) if jsonl else b""
                if jsonl and fragment.strip() and not _parses(fragment.strip()):
                    with open(torn_path(path), "ab") as torn:
                        torn.write(fragment.rstrip(b"\r") + b"\n")
                        torn.flush()
                        os.fsync(torn.fileno())
                    os.ftruncate(fd, start)
                    print(f"note: {path.name} ended in a torn line; it was moved to {torn_path(path).name}",
                          file=sys.stderr)
                else:
                    data = b"\n" + data
        _write_all(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def append_text(path: Path, text: str, *, tool: str, lock_root: Path | None = None,
                timeout: float | None = None) -> None:
    """Append UTF-8 text under the store lock of `lock_root` (default: the file's folder)."""
    path = Path(path)
    with lock(lock_root or path.parent, tool, timeout):
        append_bytes_locked(path, text.encode("utf-8"))


def event_line(event: Mapping[str, object], *, tool: str, source: str, v: int = 1) -> dict:
    """The event as it is stored: `at`, `v`, `source` first (filled in when absent), then the event's own keys."""
    rest = {k: val for k, val in event.items() if k not in ("at", "v", "source")}
    return {"at": event.get("at") or ids.now_iso(tool), "v": event.get("v", v),
            "source": event.get("source", source), **rest}


def append_jsonl(path: Path, event: Mapping[str, object], *, tool: str, source: str, v: int = 1,
                 lock_root: Path | None = None, timeout: float | None = None) -> dict:
    """Append one event as one JSON line under the store lock; returns the line as stored."""
    line = event_line(event, tool=tool, source=source, v=v)
    data = (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")
    path = Path(path)
    with lock(lock_root or path.parent, tool, timeout):
        append_bytes_locked(path, data)
    return line


@dataclass
class JsonlRead:
    """What a JSONL file held: the parsed records, the 1-based numbers of lines that did not parse, a torn tail,
    and `torn_file`, the `<file>.torn` sidecar when an earlier append moved torn lines there (else None)."""

    records: list[dict] = field(default_factory=list)
    bad_lines: list[int] = field(default_factory=list)
    torn_tail: bool = False
    torn_file: Path | None = None


def read_jsonl(path: Path) -> JsonlRead:
    """Parse a JSONL file, skipping blank lines and lines that are not a JSON object; a missing file is empty.

    A last line without its newline that does not parse is a torn tail (a writer stopped mid-line): it is skipped
    and flagged. Other unparseable lines are skipped and listed in `bad_lines` for the caller to report.
    """
    path = Path(path)
    result = JsonlRead()
    if torn_path(path).is_file():
        result.torn_file = torn_path(path)
    if not path.exists():
        return result
    raw = _retry_permission(path.read_bytes, path).removeprefix(b"\xef\xbb\xbf")  # tolerate an editor's BOM
    lines = raw.split(b"\n")
    for number, line in enumerate(lines, 1):
        text = line.strip()
        if not text:
            continue
        try:
            obj = json.loads(text.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            result.bad_lines.append(number)
            continue
        if not isinstance(obj, dict):
            result.bad_lines.append(number)
            continue
        result.records.append(obj)
    result.torn_tail = bool(result.bad_lines) and not raw.endswith(b"\n") and result.bad_lines[-1] == len(lines)
    return result


def latest_by_key(path: Path, key: str | Sequence[str] | Callable[[dict], object]) -> dict[object, dict]:
    """The last record per key, in file (append) order; records without the key and a torn tail are skipped.

    `key` is a field name, a sequence of field names (the key is then a tuple), or a function of the record.
    """
    if callable(key):
        keyfn = key
    elif isinstance(key, str):
        def keyfn(rec: dict) -> object:
            return rec[key]
    else:
        fields = tuple(key)

        def keyfn(rec: dict) -> object:
            return tuple(rec[f] for f in fields)

    latest: dict[object, dict] = {}
    for record in read_jsonl(path).records:
        try:
            value = keyfn(record)
            hash(value)
        except (KeyError, TypeError):
            continue
        latest[value] = record
    return latest


# --- generated views --------------------------------------------------------------------------------------------

def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list | tuple):
        text = ", ".join(_cell(v) for v in value)
    elif isinstance(value, dict):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return " ".join(text.split()).replace("|", "\\|")


def render_index_md(rows: Sequence[Mapping[str, object]], *, columns: Sequence[str], title: str, tool: str) -> str:
    """The INDEX.md text: title, the do-not-edit line, the count, one table row per record."""
    out = [f"# {title}", "", f"<!-- generated by `{tool} index`; do not edit: rebuilt from the records -->", "",
           f"{len(rows)} record{'' if len(rows) == 1 else 's'}.", ""]
    if rows:
        out.append("| " + " | ".join(columns) + " |")
        out.append("|" + "---|" * len(columns))
        for row in rows:
            out.append("| " + " | ".join(_cell(row.get(c)) for c in columns) + " |")
        out.append("")
    return "\n".join(out)


def write_index(folder: Path, rows: Iterable[Mapping[str, object]], *, columns: Sequence[str], title: str,
                tool: str, sort_key: str | None = None) -> tuple[Path, Path]:
    """Rebuild `folder/index.jsonl` and `folder/INDEX.md` from `rows`; returns both paths. Call under `lock()`.

    No timestamp goes into either file, so rebuilding unchanged records leaves them byte-identical.
    """
    ordered = list(rows)
    if sort_key is not None:
        ordered.sort(key=lambda r: str(r.get(sort_key, "")))
    folder = Path(folder)
    jsonl = "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in ordered)
    index = write_text(folder / "index.jsonl", jsonl)
    md = write_text(folder / "INDEX.md", render_index_md(ordered, columns=columns, title=title, tool=tool))
    return index, md
