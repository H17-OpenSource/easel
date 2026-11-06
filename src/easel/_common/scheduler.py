"""Wait for room before GPU work starts, through one interface: plan, ready, wait, start, check, requeue, done,
cancel, update.

Why it exists: a render or a training step needs free host RAM (the family's `host_ram_gb`) and free disk (the
`limits.disk_floor_gb` floor, 20 GB unless the instance sets another), and other work on the machine may need
them too. Every GPU verb goes through this interface, so whether the machine alone decides
(`scheduler.kind = "none"`) or an external scheduler does (`scheduler.kind = "command"`, configured in
`[scheduler]`) is one config line, never a code change.

The life of a job:  ticket = s.plan(request) -> s.wait(ticket) (granted) -> s.start(ticket) -> while it runs,
s.check(ticket) ("yield" -> the stop file appears; the job stops after its current unit and calls
s.requeue(ticket), then waits again) -> s.done(ticket) or s.cancel(ticket). `ready` asks once without waiting;
`update` changes the minutes or the RAM a planned job declares.

`NullScheduler` (kind "none"): waits until the free host RAM reaches the job's need and the disk floor holds,
printing one line every 30 s, for example `waiting: free 3.9 GB < 5.5 GB needed by family sdxl`. The need comes
from `families/<family>.yaml` `host_ram_gb.train` or `.eval` (the step); a null there means "not measured": the
wait falls back to `limits.host_ram_floor_gb` and says so. When free RAM cannot be measured, it refuses with
`cannot measure free RAM; set limits.skip_ram_check = true to skip` (fail closed; only the operator sets that
key). `wait(no_wait=True)` (the `--no-wait` flag) checks once and returns the waiting line instead of waiting.

`CommandScheduler` (kind "command"): runs the `scheduler.<verb>_command` templates of the instance TOML (written
per machine, never shipped). A template is split into words (double or single quotes group, no shell, no
backslash escapes), then `{ref} {title} {tool} {by} {family} {step} {gb} {minutes} {pausable} {stop_file}`
(and `{exit_code}` for done) are filled in. Exit codes of the external command: 0 ok (for `plan`: its last
output line is the job's ref; for `check`: a last line `yield` asks the job to stop: the stop file is created),
2 still waiting, 3 gone (cancelled or ended there). Anything else, a missing program or a hang means the
scheduler cannot be reached: then nothing starts (fail closed, exit 3) unless `scheduler.fail_open = true`, which
switches the job to the local RAM and disk check. `plan`, `wait`, `start` and `done` need a template; the others
are skipped when theirs is not set. Bookkeeping verbs (`check`, `requeue`, `done`, `cancel`, `update`) report a
failure in their outcome and never raise, so a job that already ran is never lost to a hiccup.

Exit codes of an outcome (`Outcome.code`): 0 ok or granted, 1 a bookkeeping step failed, 2 refused or gone,
3 still waiting. `SchedulerError` carries 2 (a refusal or a config problem) or 3 (cannot reach the scheduler).
Units: GB = 2**30 bytes, as in `_common.sysinfo`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from . import sysinfo

if TYPE_CHECKING:
    from .config import Config

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_WAITING = 3
ANNOUNCE_EVERY_S = 30.0
COMMAND_TIMEOUT_S = 120.0
WAIT_COMMAND_TIMEOUT_S = 900.0
CANNOT_MEASURE = "cannot measure free RAM; set limits.skip_ram_check = true to skip"
VERBS = ("plan", "ready", "wait", "start", "check", "requeue", "done", "cancel", "update")
REQUIRED_COMMANDS = ("plan", "wait", "start", "done")
STEPS = ("train", "eval")
DEFAULTS: dict[str, object] = {
    "scheduler.kind": "none",
    "scheduler.fail_open": False,
    "scheduler.poll_s": 5.0,
    "limits.host_ram_floor_gb": 2.0,
    "limits.disk_floor_gb": 20.0,  # neutral; an instance sets its own floor
    "limits.skip_ram_check": False,
}
_CODES = {"waiting": EXIT_WAITING, "gone": EXIT_REFUSED, "failed": EXIT_FAILED}
_COULD_NOT_RUN, _TIMED_OUT = -1, -2

Say = Callable[[str], None]


class SchedulerError(Exception):
    """A refusal (code 2) or an unreachable scheduler under fail-closed (code 3), with the lines to print."""

    def __init__(self, code: int, lines: list[str]) -> None:
        super().__init__("\n".join(lines))
        self.code = code
        self.lines = lines


def _say_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _gb(value: float) -> str:
    return f"{value:g}"


# --- what a job needs ---------------------------------------------------------------------------------------------

@dataclass
class Request:
    """What a job asks for. `step` picks `host_ram_gb.train` or `.eval` of the family; `host_ram_gb` overrides."""

    title: str
    tool: str
    family: str | None = None
    step: str = "eval"
    minutes: int = 15
    by: str = "operator"
    pausable: bool = False
    host_ram_gb: float | None = None
    disk_path: Path | None = None
    stop_file: Path | None = None


@dataclass
class Need:
    """The free RAM and disk a job waits for, and where the RAM number came from (`source`, for the lines)."""

    ram_gb: float
    source: str
    measure_ram: bool
    disk_floor_gb: float
    disk_path: Path
    notes: list[str] = field(default_factory=list)


@dataclass
class Ticket:
    """One planned job: its request, its need, the scheduler's ref, its stop file and where it stands."""

    request: Request
    need: Need
    ref: str | None = None
    stop_file: Path | None = None
    status: str = "planned"
    local_only: bool = False  # true after a fail-open fallback: every later verb uses the local check


@dataclass(frozen=True)
class Outcome:
    """The result of one verb: a status word and one line to print or to keep in the job's history."""

    verb: str
    status: str
    line: str
    ref: str | None = None

    @property
    def code(self) -> int:
        return _CODES.get(self.status, EXIT_OK)


def default_families_dir() -> Path:
    """`<repo>/families/` of the repository this module sits in (src layout), as `config.default_schema_path`."""
    return Path(__file__).resolve().parents[3] / "families"


def family_host_ram(family: str, step: str, families_dir: Path | None = None) -> float | None:
    """`host_ram_gb.<step>` of a family file (following `inherits` when the block is missing); None = not measured."""
    import yaml

    if step not in STEPS:
        raise SchedulerError(EXIT_REFUSED, [f"step must be train or eval, not {step!r}"])
    folder = default_families_dir() if families_dir is None else Path(families_dir)
    seen: list[str] = []
    current: object = family
    while isinstance(current, str) and current:
        if current in seen:
            raise SchedulerError(EXIT_REFUSED, [f"family {family}: inherits loops back to {current}"])
        seen.append(current)
        path = folder / f"{current}.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            raise SchedulerError(EXIT_REFUSED, [f"no family file families/{current}.yaml (family {family})"]) from None
        except (OSError, yaml.YAMLError) as exc:
            raise SchedulerError(EXIT_REFUSED, [f"families/{current}.yaml is not readable: {exc}"]) from exc
        block = data.get("host_ram_gb") if isinstance(data, dict) else None
        if isinstance(block, dict):
            value = block.get(step)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise SchedulerError(EXIT_REFUSED, [f"families/{current}.yaml: host_ram_gb.{step} is not a number"])
            return float(value)
        current = data.get("inherits") if isinstance(data, dict) else None
    return None


# --- the interface ------------------------------------------------------------------------------------------------

class Scheduler:
    """The verbs every scheduler has. Subclasses: NullScheduler (this machine alone), CommandScheduler."""

    kind = "abstract"

    def plan(self, request: Request) -> Ticket:
        raise NotImplementedError

    def ready(self, ticket: Ticket) -> Outcome:
        raise NotImplementedError

    def wait(self, ticket: Ticket, *, no_wait: bool = False, timeout_s: float | None = None) -> Outcome:
        raise NotImplementedError

    def start(self, ticket: Ticket) -> Outcome:
        raise NotImplementedError

    def check(self, ticket: Ticket) -> Outcome:
        raise NotImplementedError

    def requeue(self, ticket: Ticket) -> Outcome:
        raise NotImplementedError

    def done(self, ticket: Ticket, exit_code: int = 0) -> Outcome:
        raise NotImplementedError

    def cancel(self, ticket: Ticket) -> Outcome:
        raise NotImplementedError

    def update(self, ticket: Ticket, *, minutes: int | None = None, host_ram_gb: float | None = None) -> Outcome:
        raise NotImplementedError


def _clear(stop_file: Path | None) -> None:
    if stop_file is not None:
        try:
            Path(stop_file).unlink()
        except FileNotFoundError:
            pass


def _touch(stop_file: Path) -> None:
    stop_file = Path(stop_file)
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.touch()


class NullScheduler(Scheduler):
    """No external scheduler: wait for this machine's free RAM and disk, measured, never guessed."""

    kind = "none"

    def __init__(self, *, families_dir: Path | None = None, host_ram_floor_gb: float = 2.0,
                 disk_floor_gb: float = 20.0, skip_ram_check: bool = False, poll_s: float = 5.0,
                 stop_file: Path | None = None, default_disk_path: Path | None = None, say: Say | None = None,
                 free_ram: Callable[[], float] | None = None, free_disk: Callable[[Path], float] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> None:
        self.families_dir = families_dir
        self.host_ram_floor_gb = float(host_ram_floor_gb)
        self.disk_floor_gb = float(disk_floor_gb)
        self.skip_ram_check = bool(skip_ram_check)
        self.poll_s = float(poll_s)
        self.stop_file = Path(stop_file) if stop_file else None
        self.default_disk_path = Path(default_disk_path) if default_disk_path else Path.cwd()
        self.say = say or _say_stderr
        self.free_ram = free_ram or sysinfo.free_ram_gb
        self.free_disk = free_disk or sysinfo.disk_free_gb
        self.clock = clock
        self.sleep = sleep

    # what the job needs
    def need(self, request: Request) -> Need:
        """Resolve the RAM a request waits for (the job's own number, its family's, or the floor) and the disk."""
        notes: list[str] = []
        floor = self.host_ram_floor_gb
        if request.host_ram_gb is not None:
            ram, source = float(request.host_ram_gb), f"job {request.title}"
        elif request.family:
            measured = family_host_ram(request.family, request.step, self.families_dir)
            if measured is None:
                ram = floor
                source = (f"family {request.family} (host_ram_gb.{request.step} not measured: "
                          "limits.host_ram_floor_gb)")
                notes.append(f"family {request.family}: host_ram_gb.{request.step} is not measured; waiting for "
                             f"limits.host_ram_floor_gb = {_gb(floor)} GB instead")
            else:
                ram, source = measured, f"family {request.family}"
        else:
            ram, source = floor, "limits.host_ram_floor_gb"
            notes.append(f"no family given; waiting for limits.host_ram_floor_gb = {_gb(floor)} GB")
        if self.skip_ram_check:
            notes.append("RAM check skipped (limits.skip_ram_check = true)")
        return Need(ram_gb=ram, source=source, measure_ram=not self.skip_ram_check,
                    disk_floor_gb=self.disk_floor_gb, disk_path=Path(request.disk_path or self.default_disk_path),
                    notes=notes)

    def room(self, ticket: Ticket) -> tuple[bool, str]:
        """(True, the line for the job's history) when RAM and disk hold now, else (False, the waiting line)."""
        need = ticket.need
        parts = []
        if need.measure_ram:
            try:
                free = float(self.free_ram())
            except sysinfo.SysinfoUnavailable as exc:
                raise SchedulerError(EXIT_REFUSED, [CANNOT_MEASURE, f"(why: {exc})"]) from exc
            if free < need.ram_gb:
                return False, f"waiting: free {free:.1f} GB < {_gb(need.ram_gb)} GB needed by {need.source}"
            parts.append(f"free RAM {free:.1f} GB >= {_gb(need.ram_gb)} GB")
        try:
            disk = float(self.free_disk(need.disk_path))
        except sysinfo.SysinfoUnavailable as exc:
            raise SchedulerError(EXIT_REFUSED, [f"cannot measure free disk space; nothing starts (why: {exc})"]) from exc
        if disk < need.disk_floor_gb:
            return False, (f"waiting: free disk {disk:.1f} GB < {_gb(need.disk_floor_gb)} GB floor "
                           "(limits.disk_floor_gb)")
        parts.append(f"free disk {disk:.1f} GB >= {_gb(need.disk_floor_gb)} GB")
        return True, "; ".join(parts)

    # the verbs
    def plan(self, request: Request) -> Ticket:
        need = self.need(request)
        for note in need.notes:
            self.say(note)
        stop = request.stop_file or self.stop_file
        return Ticket(request=request, need=need, stop_file=Path(stop) if stop else None)

    def ready(self, ticket: Ticket) -> Outcome:
        ok, line = self.room(ticket)
        return Outcome("ready", "ready" if ok else "waiting", line, ticket.ref)

    def wait(self, ticket: Ticket, *, no_wait: bool = False, timeout_s: float | None = None) -> Outcome:
        began = self.clock()
        said_at: float | None = None
        while True:
            ok, line = self.room(ticket)
            if ok:
                ticket.status = "ready"
                return Outcome("wait", "granted", line, ticket.ref)
            now = self.clock()
            if said_at is None or now - said_at >= ANNOUNCE_EVERY_S:
                self.say(line)
                said_at = now
            if no_wait or (timeout_s is not None and now - began >= timeout_s):
                ticket.status = "waiting"
                return Outcome("wait", "waiting", line, ticket.ref)
            self.sleep(self.poll_s)

    def start(self, ticket: Ticket) -> Outcome:
        _clear(ticket.stop_file)  # a stop request from an earlier run must not stop this one
        ticket.status = "running"
        return Outcome("start", "started", f"started {ticket.request.title}", ticket.ref)

    def check(self, ticket: Ticket) -> Outcome:
        if ticket.stop_file is not None and Path(ticket.stop_file).exists():
            return Outcome("check", "yield", "a stop was asked: stop after the current unit, then requeue",
                           ticket.ref)
        return Outcome("check", "ok", "keep going", ticket.ref)

    def requeue(self, ticket: Ticket) -> Outcome:
        ticket.status = "waiting"
        return Outcome("requeue", "requeued", f"{ticket.request.title} waits again", ticket.ref)

    def done(self, ticket: Ticket, exit_code: int = 0) -> Outcome:
        _clear(ticket.stop_file)
        ticket.status = "done" if exit_code == 0 else "failed"
        return Outcome("done", "done", f"{ticket.request.title} ended with exit {exit_code}", ticket.ref)

    def cancel(self, ticket: Ticket) -> Outcome:
        _clear(ticket.stop_file)
        ticket.status = "cancelled"
        return Outcome("cancel", "cancelled", f"{ticket.request.title} cancelled", ticket.ref)

    def update(self, ticket: Ticket, *, minutes: int | None = None, host_ram_gb: float | None = None) -> Outcome:
        changes = {k: v for k, v in (("minutes", minutes), ("host_ram_gb", host_ram_gb)) if v is not None}
        ticket.request = replace(ticket.request, **changes)
        ticket.need = self.need(ticket.request)
        return Outcome("update", "updated", f"{ticket.request.title}: needs {_gb(ticket.need.ram_gb)} GB, "
                                            f"{ticket.request.minutes} min", ticket.ref)


# --- the external scheduler ---------------------------------------------------------------------------------------

def split_command(text: str) -> list[str]:
    """Words of a command template: whitespace separates, double or single quotes group; backslashes are literal."""
    words: list[str] = []
    current: list[str] = []
    quote: str | None = None
    started = False
    for ch in text:
        if quote:
            if ch == quote:
                quote = None
            else:
                current.append(ch)
        elif ch in "\"'":
            quote, started = ch, True
        elif ch.isspace():
            if started or current:
                words.append("".join(current))
                current, started = [], False
        else:
            current.append(ch)
    if quote:
        raise SchedulerError(EXIT_REFUSED, [f"unbalanced {quote} in the command template: {text}"])
    if started or current:
        words.append("".join(current))
    return words


def render(template: str, values: Mapping[str, str], key: str) -> list[str]:
    """The argv of one template with its placeholders filled; SchedulerError(2) for an unknown placeholder."""
    words = split_command(template)
    if not words:
        raise SchedulerError(EXIT_REFUSED, [f"{key} is empty"])
    try:
        return [word.format_map(values) for word in words]
    except KeyError as exc:
        known = " ".join("{" + k + "}" for k in sorted(values))
        line = f"{key} uses {{{exc.args[0]}}}, which is not known here; known: {known}"
        raise SchedulerError(EXIT_REFUSED, [line]) from None
    except (ValueError, IndexError) as exc:
        line = f"{key} is not a valid template ({exc}); write {{{{ for a literal {{"
        raise SchedulerError(EXIT_REFUSED, [line]) from None


Runner = Callable[..., subprocess.CompletedProcess]


class CommandScheduler(Scheduler):
    """A machine-wide scheduler driven through the `scheduler.<verb>_command` templates (exit codes 0/2/3)."""

    kind = "command"

    def __init__(self, commands: Mapping[str, str], *, local: NullScheduler, fail_open: bool = False,
                 poll_s: float = 5.0, say: Say | None = None, runner: Runner | None = None,
                 command_timeout_s: float = COMMAND_TIMEOUT_S, wait_timeout_s: float = WAIT_COMMAND_TIMEOUT_S) -> None:
        unknown = sorted(set(commands) - set(VERBS))
        if unknown:
            raise SchedulerError(EXIT_REFUSED, [f"no scheduler verb {v!r}; the verbs are {', '.join(VERBS)}"
                                                for v in unknown])
        missing = [f"scheduler.{v}_command" for v in REQUIRED_COMMANDS if not commands.get(v)]
        if missing:
            raise SchedulerError(EXIT_REFUSED, [f"scheduler.kind = \"command\" needs {', '.join(missing)} in the "
                                                "instance TOML"])
        self.commands = {k: v for k, v in commands.items() if v}
        self.local = local
        self.fail_open = bool(fail_open)
        self.poll_s = float(poll_s)
        self.say = say or local.say
        self.runner = runner or subprocess.run
        self.command_timeout_s = command_timeout_s
        self.wait_timeout_s = wait_timeout_s

    def _values(self, ticket: Ticket, **extra: str) -> dict[str, str]:
        request = ticket.request
        values = {"title": request.title, "tool": request.tool, "by": request.by, "family": request.family or "",
                  "step": request.step, "gb": _gb(ticket.need.ram_gb), "minutes": str(request.minutes),
                  "pausable": "true" if request.pausable else "false", "stop_file": str(ticket.stop_file or "")}
        if ticket.ref:
            values["ref"] = ticket.ref
        values.update(extra)
        return values

    def _call(self, verb: str, ticket: Ticket, **extra: str) -> tuple[int, str, str]:
        """(exit code or _COULD_NOT_RUN/_TIMED_OUT, the last output line, the error text) of one template."""
        argv = render(self.commands[verb], self._values(ticket, **extra), f"scheduler.{verb}_command")
        timeout = self.wait_timeout_s if verb == "wait" else self.command_timeout_s
        try:
            done = self.runner(argv, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               timeout=timeout, stdin=subprocess.DEVNULL,
                               creationflags=0x08000000 if os.name == "nt" else 0)  # CREATE_NO_WINDOW
        except subprocess.TimeoutExpired:
            return _TIMED_OUT, "", f"did not answer within {timeout:g} s"
        except (OSError, subprocess.SubprocessError) as exc:
            return _COULD_NOT_RUN, "", f"could not run ({type(exc).__name__}: {exc})"
        lines = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
        err = [line.strip() for line in (done.stderr or "").splitlines() if line.strip()]
        last = lines[-1] if lines else ""
        return done.returncode, last, (err[-1] if err else last)

    def _unreachable(self, verb: str, detail: str, ticket: Ticket) -> None:
        """Fail closed (SchedulerError 3), or under fail_open switch the ticket to the local check and say so."""
        line = f"scheduler unreachable: {verb} {detail}".rstrip()
        if not self.fail_open:
            raise SchedulerError(EXIT_WAITING, [line, "failing closed: nothing starts (scheduler.fail_open = false)"])
        self.say(f"{line}; scheduler.fail_open = true: using the local RAM and disk check instead")
        ticket.local_only = True

    @staticmethod
    def _exited(code: int, last: str, err: str) -> str:
        return f"exited {code}" + (f": {err}" if err else "") if code >= 0 else err

    # the gate verbs: fail closed
    def plan(self, request: Request) -> Ticket:
        ticket = self.local.plan(request)
        code, last, err = self._call("plan", ticket)
        if code == 0 and last:
            ticket.ref = last
            return ticket
        if code == 0:
            self._unreachable("plan", "printed no job ref", ticket)
        elif code in (EXIT_REFUSED, EXIT_WAITING):
            raise SchedulerError(EXIT_REFUSED, [f"the scheduler refused the plan for {request.title}: "
                                                f"{self._exited(code, last, err)}"])
        else:
            self._unreachable("plan", self._exited(code, last, err), ticket)
        return ticket

    def ready(self, ticket: Ticket) -> Outcome:
        if ticket.local_only:
            return self.local.ready(ticket)
        if "ready" not in self.commands:
            return Outcome("ready", "skipped", "scheduler.ready_command is not set; skipped", ticket.ref)
        code, last, err = self._call("ready", ticket)
        if code == 0:
            return Outcome("ready", "ready", last or "ready", ticket.ref)
        if code == EXIT_REFUSED:
            return Outcome("ready", "waiting", f"waiting: {last or 'the scheduler says wait'}", ticket.ref)
        if code == EXIT_WAITING:
            return Outcome("ready", "gone", f"the scheduler ended job {ticket.ref}: {last or err}", ticket.ref)
        self._unreachable("ready", self._exited(code, last, err), ticket)
        return self.local.ready(ticket)

    def wait(self, ticket: Ticket, *, no_wait: bool = False, timeout_s: float | None = None) -> Outcome:
        if ticket.local_only:
            return self.local.wait(ticket, no_wait=no_wait, timeout_s=timeout_s)
        began = self.local.clock()
        said_at: float | None = None
        while True:
            code, last, err = self._call("wait", ticket)
            if code == 0:
                ticket.status = "ready"
                return Outcome("wait", "granted", last or f"granted by the scheduler (job {ticket.ref})", ticket.ref)
            if code == EXIT_WAITING:
                ticket.status = "cancelled"
                return Outcome("wait", "gone", f"the scheduler ended job {ticket.ref}: {last or err}", ticket.ref)
            if code not in (EXIT_REFUSED, _TIMED_OUT):
                self._unreachable("wait", self._exited(code, last, err), ticket)
                return self.local.wait(ticket, no_wait=no_wait, timeout_s=timeout_s)
            said = last or err
            line = f"waiting: the scheduler has not granted job {ticket.ref} yet" + (f" ({said})" if said else "")
            now = self.local.clock()
            if said_at is None or now - said_at >= ANNOUNCE_EVERY_S:
                self.say(line)
                said_at = now
            if no_wait or (timeout_s is not None and now - began >= timeout_s):
                ticket.status = "waiting"
                return Outcome("wait", "waiting", line, ticket.ref)
            self.local.sleep(self.poll_s)

    def start(self, ticket: Ticket) -> Outcome:
        if ticket.local_only:
            return self.local.start(ticket)
        _clear(ticket.stop_file)
        code, last, err = self._call("start", ticket)
        if code == 0:
            ticket.status = "running"
            return Outcome("start", "started", last or f"started {ticket.request.title}", ticket.ref)
        if code == EXIT_WAITING:
            ticket.status = "cancelled"
            return Outcome("start", "gone", f"the scheduler ended job {ticket.ref}: {last or err}", ticket.ref)
        self._unreachable("start", self._exited(code, last, err), ticket)
        return self.local.start(ticket)

    # the bookkeeping verbs: report, never raise
    def _bookkeeping(self, verb: str, ticket: Ticket, ok_status: str, **extra: str) -> Outcome:
        if verb not in self.commands:
            return Outcome(verb, "skipped", f"scheduler.{verb}_command is not set; skipped", ticket.ref)
        try:
            code, last, err = self._call(verb, ticket, **extra)
        except SchedulerError as exc:
            return Outcome(verb, "failed", "; ".join(exc.lines), ticket.ref)
        if code == 0:
            return Outcome(verb, ok_status, last or ok_status, ticket.ref)
        return Outcome(verb, "failed", f"scheduler {verb} {self._exited(code, last, err)}", ticket.ref)

    def check(self, ticket: Ticket) -> Outcome:
        if ticket.local_only:
            return self.local.check(ticket)
        if "check" not in self.commands:
            return self.local.check(ticket)
        try:
            code, last, err = self._call("check", ticket)
        except SchedulerError as exc:
            return Outcome("check", "unknown", "; ".join(exc.lines), ticket.ref)
        if code == 0 and last.lower() == "yield":
            if ticket.stop_file is None:
                return Outcome("check", "yield", "the scheduler asks to yield; no stop file is configured, so the "
                                                 "job cannot be told", ticket.ref)
            _touch(ticket.stop_file)
            return Outcome("check", "yield", "the scheduler asks to yield: stop file written; stop after the "
                                             "current unit, then requeue", ticket.ref)
        if code == 0:
            return Outcome("check", "ok", last or "keep going", ticket.ref)
        return Outcome("check", "unknown", f"scheduler check {self._exited(code, last, err)}; keep going",
                       ticket.ref)

    def requeue(self, ticket: Ticket) -> Outcome:
        ticket.status = "waiting"
        return self.local.requeue(ticket) if ticket.local_only else self._bookkeeping("requeue", ticket, "requeued")

    def done(self, ticket: Ticket, exit_code: int = 0) -> Outcome:
        _clear(ticket.stop_file)
        ticket.status = "done" if exit_code == 0 else "failed"
        if ticket.local_only:
            return self.local.done(ticket, exit_code)
        return self._bookkeeping("done", ticket, "done", exit_code=str(exit_code))

    def cancel(self, ticket: Ticket) -> Outcome:
        _clear(ticket.stop_file)
        ticket.status = "cancelled"
        return self.local.cancel(ticket) if ticket.local_only else self._bookkeeping("cancel", ticket, "cancelled")

    def update(self, ticket: Ticket, *, minutes: int | None = None, host_ram_gb: float | None = None) -> Outcome:
        changes = {k: v for k, v in (("minutes", minutes), ("host_ram_gb", host_ram_gb)) if v is not None}
        ticket.request = replace(ticket.request, **changes)
        ticket.need = self.local.need(ticket.request)
        if ticket.local_only:
            return self.local.update(ticket)
        return self._bookkeeping("update", ticket, "updated")


# --- construction from the instance config ------------------------------------------------------------------------

def _value(cfg: Config, key: str) -> object:
    return cfg.get(key, DEFAULTS.get(key))


def from_config(cfg: Config, *, families_dir: Path | None = None, say: Say | None = None, **overrides: object
                ) -> Scheduler:
    """The scheduler a loaded `config.Config` names (`scheduler.kind`), with its limits and command templates.

    `overrides` are passed to NullScheduler (for example `free_ram=`, `free_disk=`, `clock=`, `sleep=`), plus
    `runner=` for CommandScheduler; tests use them, commands do not.
    """
    root = getattr(cfg, "root", None)
    stop_file = _value(cfg, "scheduler.stop_file")
    if isinstance(stop_file, str) and stop_file:
        stop_path = Path(stop_file).expanduser()
        stop_file = stop_path if stop_path.is_absolute() or root is None else Path(root) / stop_path
    else:
        stop_file = None
    runner = overrides.pop("runner", None)
    local = NullScheduler(
        families_dir=families_dir,
        host_ram_floor_gb=float(_value(cfg, "limits.host_ram_floor_gb")),
        disk_floor_gb=float(_value(cfg, "limits.disk_floor_gb")),
        skip_ram_check=bool(_value(cfg, "limits.skip_ram_check")),
        poll_s=float(_value(cfg, "scheduler.poll_s")),
        stop_file=stop_file, default_disk_path=root, say=say, **overrides)
    kind = _value(cfg, "scheduler.kind")
    if kind == "none":
        return local
    if kind == "command":
        commands = {verb: str(_value(cfg, f"scheduler.{verb}_command") or "") for verb in VERBS}
        return CommandScheduler(commands, local=local, fail_open=bool(_value(cfg, "scheduler.fail_open")),
                                poll_s=local.poll_s, say=say, runner=runner)
    raise SchedulerError(EXIT_REFUSED, [f"scheduler.kind = {kind!r} is not one of none, command"])


def add_wait_args(parser: argparse.ArgumentParser) -> None:
    """The `--no-wait` option of every verb that waits for room."""
    parser.add_argument("--no-wait", action="store_true",
                        help="check for room once; when there is none, print the waiting line and exit 3 instead "
                             "of waiting")
