"""Start a long-running process hidden and independent of the caller, record how, and stop it later.

Why it exists: a local server (ComfyUI, a review page) or a long batch started from an agent's shell must outlive
that shell. Measured on Windows 11: a child started with DETACHED_PROCESS died when the
agent's shell call ended, and again when the agent session restarted, even with CREATE_BREAKAWAY_FROM_JOB,
because the session's job object did not allow breakaway. A process created by the WMI service
(`Win32_Process.Create`) has the service as its parent, so nothing the agent's session does can end it. Hence
the order, which is the measured one and must not be swapped:
1. Windows: WMI `Win32_Process.Create`, called through PowerShell (itself started without a console window),
   with `ShowWindow = 0`; `cmd.exe` inside redirects the output into the log.
2. Windows fallback: DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP with CREATE_BREAKAWAY_FROM_JOB, and when the
   job refuses breakaway, the same without it.
3. POSIX: a child in a new session (`start_new_session=True`).

`start()` returns a `Started` (pid, method, detail) and writes two files: `<pid_file>` with the pid alone (easy
for any tool to read) and the sidecar `<pid_file>.json` with `{pid, method, detail, started_at, log}`, so a
status verb can say `started via wmi` or `started via detached`. `started_at` comes from the tool's clock
(`ids.now_iso(tool)`), so `<TOOL>_NOW` freezes it like every other time stamp; `tool` defaults to the package
this copy is loaded under.

On the WMI path the child does not inherit the caller's environment: it gets the user's default environment
plus `env` (and `PYTHONIOENCODING=utf-8`), set by `cmd.exe` before the command. Pass everything the child needs
in `env`. A command, log path or `env` value that `cmd.exe` cannot carry literally (a double quote, a line
break or a `%`) skips WMI and uses the detached start; the sidecar's `detail` says why.

    from <pkg>._common import background   # <pkg>: the package this copy lives in
    started = background.start([python, "worker.py", "--count", "4"], log, pid_file)
    print(started.status_line)          # started via wmi
    background.stop(pid_file)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import ids, store

# The tool this copy serves (easel or kiln): the package it is loaded under, read, never imported.
DEFAULT_TOOL = (__package__ or "").split(".")[0] or "tool"
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000
CREATE_NO_WINDOW = 0x08000000
POWERSHELL_TIMEOUT_S = 60
METHOD_WMI = "wmi"
METHOD_DETACHED = "detached"
_CMD_SPECIAL = set(' \t&|<>^()')
_CMD_UNSAFE = ('"', "\r", "\n", "%")

# PowerShell reads the command line and the folder from its own environment, so no quoting reaches its parser.
_WMI_SCRIPT = (
    "$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{ShowWindow=[uint16]0}; "
    "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
    "@{CommandLine=$env:BG_CMD; CurrentDirectory=$env:BG_CWD; ProcessStartupInformation=$si}; "
    "if ($r.ReturnValue -eq 0) { $r.ProcessId } else { 'ERR ' + $r.ReturnValue }"
)


@dataclass
class Started:
    """A started process: its pid, how it was started (`wmi` or `detached`) and the detail of that path."""

    pid: int
    method: str
    detail: str
    pid_file: Path
    log: Path
    process: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def status_line(self) -> str:
        return f"started via {self.method}"

    def poll(self) -> int | None:
        """None while the process runs, like Popen.poll(); an exit code (1 when unknown) once it has ended."""
        if self.process is not None:
            return self.process.poll()
        return None if pid_alive(self.pid) else 1


# --- small wrappers (tests replace these, so no test ever starts a real process by accident) ---------------------

def _run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess:
    return subprocess.run(argv, **kwargs)


def _spawn(argv: list[str], **kwargs: object) -> subprocess.Popen:
    return subprocess.Popen(argv, **kwargs)


def _no_window() -> int:
    return CREATE_NO_WINDOW if os.name == "nt" else 0


# --- the WMI path -------------------------------------------------------------------------------------------------

def _cmd_quote(arg: str) -> str:
    """One argument for a cmd.exe line: quoted when it holds a space or a cmd metacharacter (& | < > ^ ( ))."""
    if arg and not (_CMD_SPECIAL & set(arg)):
        return arg
    # The C runtime reads backslashes right before a closing quote as escapes, so those are doubled.
    doubled = "\\" * (len(arg) - len(arg.rstrip("\\")))
    return f'"{arg}{doubled}"'


def wmi_blocker(cmd: list[str], log: Path, env: dict[str, str] | None) -> str | None:
    """Why `cmd.exe` cannot carry this start literally, or None when it can."""
    texts = [*cmd, str(log), *(f"{k}={v}" for k, v in (env or {}).items())]
    for text in texts:
        for bad in _CMD_UNSAFE:
            if bad in text:
                shown = {"\r": "a line break", "\n": "a line break", '"': "a double quote", "%": "a %"}[bad]
                return f"wmi skipped: the command, log or env holds {shown}"
    return None


def wmi_command_line(cmd: list[str], log: Path, env: dict[str, str] | None) -> str:
    """The cmd.exe line WMI runs: set each variable, run the command, send stdout and stderr to the log."""
    sets = "".join(f'set "{k}={v}"&& ' for k, v in {"PYTHONIOENCODING": "utf-8", **(env or {})}.items())
    line = " ".join(_cmd_quote(a) for a in cmd)
    return f'cmd.exe /d /s /c "{sets}{line} > "{log}" 2>&1"'


def start_wmi(cmd: list[str], log: Path, cwd: Path | None, env: dict[str, str] | None) -> int | None:
    """Have the WMI service create the process (window hidden). The pid (of cmd.exe), or None when WMI refused."""
    script_env = {**os.environ, "BG_CMD": wmi_command_line(cmd, log, env), "BG_CWD": str(cwd or Path.cwd())}
    try:
        done = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _WMI_SCRIPT],
                    capture_output=True, text=True, env=script_env, timeout=POWERSHELL_TIMEOUT_S,
                    stdin=subprocess.DEVNULL, creationflags=_no_window())
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (done.stdout or "").strip().splitlines()
    last = lines[-1].strip() if lines else ""
    return int(last) if last.isdigit() and int(last) > 0 else None


# --- the detached path --------------------------------------------------------------------------------------------

def _start_detached(cmd: list[str], log: Path, cwd: Path | None, env: dict[str, str] | None,
                    platform: str) -> tuple[subprocess.Popen, str]:
    """A detached child writing into `log`: (process, detail)."""
    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "cwd": str(cwd) if cwd else None,
        "env": {**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})},
        "stderr": subprocess.STDOUT,
    }
    with open(log, "wb") as handle:
        kwargs["stdout"] = handle
        if platform == "win32":
            base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            try:
                return _spawn(cmd, creationflags=base | CREATE_BREAKAWAY_FROM_JOB, **kwargs), "breakaway"
            except OSError:  # the caller's job object does not allow breakaway
                return _spawn(cmd, creationflags=base, **kwargs), "no breakaway (the job refused it)"
        return _spawn(cmd, start_new_session=True, **kwargs), "new session"


# --- the public interface -----------------------------------------------------------------------------------------

def sidecar_path(pid_file: Path) -> Path:
    """`<pid_file>.json`: how the process was started."""
    pid_file = Path(pid_file)
    return pid_file.with_name(pid_file.name + ".json")


def start(cmd: list[str], log: Path, pid_file: Path, cwd: Path | None = None, env: dict[str, str] | None = None,
          *, platform: str | None = None, tool: str | None = None) -> Started:
    """Start `cmd` hidden and independent of the caller (WMI first on Windows); record pid, method and the time
    from `tool`'s clock (default: this package's tool)."""
    platform = sys.platform if platform is None else platform
    log, pid_file = Path(log), Path(pid_file)
    log.parent.mkdir(parents=True, exist_ok=True)
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(a) for a in cmd]
    env = {str(k): str(v) for k, v in (env or {}).items()}
    process = None
    pid = None
    detail = ""
    if platform == "win32":
        blocker = wmi_blocker(cmd, log, env)
        if blocker is None:
            pid = start_wmi(cmd, log, cwd, env)
            detail = "Win32_Process.Create, window hidden" if pid else "wmi refused; "
        else:
            detail = blocker + "; "
    if pid:
        method = METHOD_WMI
    else:
        process, how = _start_detached(cmd, log, cwd, env, platform)
        pid, method, detail = process.pid, METHOD_DETACHED, detail + how
    started_at = ids.now_iso(tool or DEFAULT_TOOL)
    store.write_text(pid_file, f"{pid}\n")
    store.write_text(sidecar_path(pid_file), json.dumps(
        {"pid": pid, "method": method, "detail": detail, "started_at": started_at, "log": log.name}, indent=1) + "\n")
    return Started(pid=pid, method=method, detail=detail, pid_file=pid_file, log=log, process=process)


def read(pid_file: Path) -> dict | None:
    """What `start()` recorded, plus `alive`; None when no pid file exists."""
    pid_file = Path(pid_file)
    try:
        pid_text = pid_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    info: dict = {"pid": int(pid_text) if pid_text.isdigit() else None, "method": None, "detail": None}
    try:
        side = json.loads(sidecar_path(pid_file).read_text(encoding="utf-8"))
        if isinstance(side, dict):
            info.update({k: side.get(k) for k in ("method", "detail", "started_at", "log")})
    except (OSError, ValueError):
        pass
    info["alive"] = bool(info["pid"]) and pid_alive(info["pid"])
    return info


def status_line(pid_file: Path) -> str:
    """`running (pid N, started via wmi)`, `not running (pid N ended)` or `not running`."""
    info = read(pid_file)
    if info is None or not info["pid"]:
        return "not running"
    if not info["alive"]:
        return f"not running (pid {info['pid']} ended)"
    via = f", started via {info['method']}" if info.get("method") else ""
    return f"running (pid {info['pid']}{via})"


def pid_alive(pid: int) -> bool:
    """True while a process with this pid exists (Windows: OpenProcess + GetExitCodeProcess; POSIX: kill 0)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p  # a HANDLE is pointer-sized
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists, it is someone else's
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def stop(pid_file: Path) -> int | None:
    """Stop the recorded process and its children, then remove the pid file and sidecar. The pid, or None.

    A pid file whose process has already ended is only cleaned up, so a reused pid is never killed by mistake
    when the process is gone; a live pid is trusted to be ours (the files are written only by `start()`).
    """
    pid_file = Path(pid_file)
    info = read(pid_file)
    if info is None:
        return None
    pid = info["pid"]
    if pid and info["alive"]:
        if os.name == "nt":
            _run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, stdin=subprocess.DEVNULL,
                 creationflags=_no_window())
        else:
            try:
                os.killpg(pid, signal.SIGTERM)  # started in its own session: the pid is the group id
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
    for path in (pid_file, sidecar_path(pid_file)):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return pid
