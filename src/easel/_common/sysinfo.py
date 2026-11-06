"""What this machine has free right now: host RAM, disk space, and where a virtual environment keeps its python.

Why it exists: GPU work starts only when the family's `host_ram_gb` and the disk floor hold (`_common.scheduler`),
so these numbers must be measured, never guessed. Every reader here returns a measured value or raises
`SysinfoUnavailable`; there is no fallback number. The scheduler turns that exception into the refusal
`cannot measure free RAM; set limits.skip_ram_check = true to skip` (fail closed).

How free RAM is read, per platform:
- Windows: `GlobalMemoryStatusEx` through ctypes (`ullAvailPhys`, the "available" figure Task Manager shows);
- Linux: `MemAvailable` in `/proc/meminfo` (the kernel's own estimate; kernels before 3.14 lack it -> raise);
- macOS: `vm_stat` (free + inactive + speculative pages times the page size) and `sysctl -n hw.memsize`;
- anything else: `SysinfoUnavailable`.

Units: "GB" in this module and in every message means 2**30 bytes, the unit the family files' `host_ram_gb`
numbers were measured in (Windows reports memory that way). Disk space uses the same unit, so a floor check is
never looser than the number the operator wrote.

The python of a virtual environment comes from `sysconfig`'s venv scheme (`Scripts/python.exe` on Windows,
`bin/python` elsewhere), so a config stores the folder (`venv_dir`) and never an interpreter path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

GB = 1 << 30
MEMINFO = Path("/proc/meminfo")
COMMAND_TIMEOUT_S = 10


class SysinfoUnavailable(RuntimeError):
    """A value could not be measured on this machine; the message says which and why."""


class Memory(NamedTuple):
    """Host RAM in GB (2**30 bytes): what is available to a new process, the total, and how it was read."""

    free_gb: float
    total_gb: float | None
    method: str


class Disk(NamedTuple):
    """Disk space in GB (2**30 bytes) on the volume of `checked` (the nearest existing folder of the path asked)."""

    free_gb: float
    total_gb: float
    checked: Path


Runner = Callable[[list[str]], str]


def _run_text(argv: list[str]) -> str:
    """stdout of a short command; SysinfoUnavailable when it is missing, fails or hangs."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as exc:
        raise SysinfoUnavailable(f"{argv[0]} could not run: {exc}") from exc
    if done.returncode != 0:
        raise SysinfoUnavailable(f"{argv[0]} exited {done.returncode}")
    return done.stdout


# --- Windows ------------------------------------------------------------------------------------------------------

def _windows_memory(kernel32: object | None = None) -> Memory:
    """GlobalMemoryStatusEx: ullAvailPhys and ullTotalPhys. `kernel32` may be a stand-in (tests)."""
    import ctypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    if kernel32 is None:
        try:
            kernel32 = ctypes.WinDLL("kernel32")
        except (AttributeError, OSError) as exc:
            raise SysinfoUnavailable(f"kernel32 is not loadable here: {exc}") from exc
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise SysinfoUnavailable("GlobalMemoryStatusEx failed")
    if status.ullTotalPhys == 0:
        raise SysinfoUnavailable("GlobalMemoryStatusEx reported no physical memory")
    return Memory(status.ullAvailPhys / GB, status.ullTotalPhys / GB, "GlobalMemoryStatusEx")


# --- Linux --------------------------------------------------------------------------------------------------------

def parse_meminfo(text: str) -> Memory:
    """`/proc/meminfo` text -> Memory from MemAvailable and MemTotal (both in kB)."""
    fields: dict[str, int] = {}
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        parts = rest.split()
        if sep and parts and parts[0].isdigit():
            scale = 1024 if parts[1:2] == ["kB"] else 1
            fields[name.strip()] = int(parts[0]) * scale
    if "MemAvailable" not in fields:
        raise SysinfoUnavailable("/proc/meminfo has no MemAvailable line (a kernel older than 3.14)")
    total = fields.get("MemTotal")
    return Memory(fields["MemAvailable"] / GB, total / GB if total else None, "/proc/meminfo MemAvailable")


def _linux_memory(meminfo: Path) -> Memory:
    try:
        text = Path(meminfo).read_text(encoding="ascii", errors="replace")
    except OSError as exc:
        raise SysinfoUnavailable(f"{meminfo} is not readable: {exc}") from exc
    return parse_meminfo(text)


# --- macOS --------------------------------------------------------------------------------------------------------

_PAGE_SIZE = re.compile(r"page size of (\d+) bytes")
_PAGES = re.compile(r"^Pages (free|inactive|speculative):\s+(\d+)\.?\s*$", re.MULTILINE)


def parse_vm_stat(text: str, page_size: int | None = None) -> int:
    """Bytes available from `vm_stat` output: (free + inactive + speculative pages) times the page size."""
    if page_size is None:
        match = _PAGE_SIZE.search(text)
        if not match:
            raise SysinfoUnavailable("vm_stat printed no page size")
        page_size = int(match.group(1))
    pages = {name: int(count) for name, count in _PAGES.findall(text)}
    if "free" not in pages:
        raise SysinfoUnavailable("vm_stat printed no 'Pages free' line")
    return sum(pages.values()) * page_size


def _macos_memory(run: Runner) -> Memory:
    available = parse_vm_stat(run(["vm_stat"]))
    total_text = run(["sysctl", "-n", "hw.memsize"]).strip()
    total = int(total_text) if total_text.isdigit() else None
    return Memory(available / GB, total / GB if total else None, "vm_stat")


# --- public readers -----------------------------------------------------------------------------------------------

def memory(platform: str | None = None, *, meminfo: Path = MEMINFO, run: Runner | None = None,
           kernel32: object | None = None) -> Memory:
    """Free and total host RAM on this platform, or SysinfoUnavailable (never a number that was not measured)."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return _windows_memory(kernel32)
    if platform.startswith("linux"):
        return _linux_memory(meminfo)
    if platform == "darwin":
        return _macos_memory(run or _run_text)
    raise SysinfoUnavailable(f"free RAM cannot be measured on platform {platform!r}")


def free_ram_gb(platform: str | None = None) -> float:
    """Free host RAM in GB (2**30 bytes); SysinfoUnavailable when it cannot be measured."""
    return memory(platform).free_gb


def _existing(path: Path) -> Path:
    """`path` or its nearest existing parent (a target folder often does not exist yet)."""
    path = Path(path).expanduser().absolute()
    for candidate in (path, *path.parents):
        if candidate.exists():
            return candidate
    raise SysinfoUnavailable(f"no existing folder on the way to {path}")


def disk(path: Path | str = ".") -> Disk:
    """Free and total space of the volume that holds `path` (or would hold it once created)."""
    checked = _existing(Path(path))
    try:
        usage = shutil.disk_usage(checked)
    except OSError as exc:
        raise SysinfoUnavailable(f"disk space of {checked} cannot be read: {exc}") from exc
    return Disk(usage.free / GB, usage.total / GB, checked)


def disk_free_gb(path: Path | str = ".") -> float:
    """Free disk space in GB (2**30 bytes) where `path` is or will be."""
    return disk(path).free_gb


# --- virtual environments -----------------------------------------------------------------------------------------

def _venv_scheme(os_name: str) -> str:
    preferred = "nt_venv" if os_name == "nt" else "posix_venv"
    if preferred in sysconfig.get_scheme_names():
        return preferred
    return "nt" if os_name == "nt" else "posix_prefix"


def venv_scripts_dir(venv_dir: Path | str, os_name: str | None = None) -> Path:
    """The folder of a venv's executables: `Scripts` on Windows, `bin` elsewhere (from sysconfig)."""
    os_name = os.name if os_name is None else os_name
    base = str(Path(venv_dir).expanduser())
    scripts = sysconfig.get_path("scripts", _venv_scheme(os_name), vars={"base": base, "platbase": base})
    return Path(scripts)


def venv_python(venv_dir: Path | str, os_name: str | None = None, *, must_exist: bool = False) -> Path:
    """The interpreter of the venv at `venv_dir`; with `must_exist`, FileNotFoundError when it is not there."""
    os_name = os.name if os_name is None else os_name
    python = venv_scripts_dir(venv_dir, os_name) / ("python.exe" if os_name == "nt" else "python")
    if must_exist and not python.is_file():
        raise FileNotFoundError(f"no python in the venv at {venv_dir} (looked for {python.name} in "
                                f"{python.parent.name}/); create it with python -m venv")
    return python
