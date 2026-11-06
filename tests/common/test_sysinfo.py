"""_common.sysinfo: free RAM on three platforms (or SysinfoUnavailable), disk free, the venv python via sysconfig."""

import ctypes
import sys
from pathlib import Path

import pytest

from easel._common import sysinfo

GiB = 1 << 30

MEMINFO = """MemTotal:       16384000 kB
MemFree:         1024000 kB
MemAvailable:    4194304 kB
Buffers:          100000 kB
Cached:          2000000 kB
"""

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               65536.
Pages active:                            200000.
Pages inactive:                           32768.
Pages speculative:                        32768.
Pages throttled:                              0.
Pages wired down:                        100000.
"""


def test_meminfo_uses_mem_available_in_kib():
    mem = sysinfo.parse_meminfo(MEMINFO)
    assert mem.free_gb == pytest.approx(4.0)  # 4194304 kB = 4 GiB
    assert mem.total_gb == pytest.approx(16384000 * 1024 / GiB)
    assert "MemAvailable" in mem.method


def test_meminfo_without_mem_available_is_not_guessed():
    old_kernel = "\n".join(line for line in MEMINFO.splitlines() if not line.startswith("MemAvailable"))
    with pytest.raises(sysinfo.SysinfoUnavailable, match="MemAvailable"):
        sysinfo.parse_meminfo(old_kernel)


def test_linux_reader_reads_the_file_and_fails_closed_when_missing(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO, encoding="ascii")
    assert sysinfo.memory("linux", meminfo=meminfo).free_gb == pytest.approx(4.0)
    with pytest.raises(sysinfo.SysinfoUnavailable, match="not readable"):
        sysinfo.memory("linux", meminfo=tmp_path / "absent")


def test_vm_stat_counts_free_inactive_and_speculative_pages():
    assert sysinfo.parse_vm_stat(VM_STAT) == (65536 + 32768 + 32768) * 16384
    assert sysinfo.parse_vm_stat(VM_STAT, page_size=4096) == (65536 + 32768 + 32768) * 4096
    with pytest.raises(sysinfo.SysinfoUnavailable, match="page size"):
        sysinfo.parse_vm_stat("Pages free: 10.\n")
    with pytest.raises(sysinfo.SysinfoUnavailable, match="Pages free"):
        sysinfo.parse_vm_stat("Mach Virtual Memory Statistics: (page size of 4096 bytes)\n")


def test_macos_reader_runs_vm_stat_and_sysctl():
    calls = []

    def run(argv):
        calls.append(argv)
        return VM_STAT if argv[0] == "vm_stat" else f"{32 * GiB}\n"

    mem = sysinfo.memory("darwin", run=run)
    assert mem.free_gb == pytest.approx((65536 + 32768 + 32768) * 16384 / GiB)
    assert mem.total_gb == pytest.approx(32.0)
    assert calls == [["vm_stat"], ["sysctl", "-n", "hw.memsize"]]


def test_macos_reader_fails_closed_when_vm_stat_fails():
    def run(argv):
        raise sysinfo.SysinfoUnavailable(f"{argv[0]} could not run")

    with pytest.raises(sysinfo.SysinfoUnavailable):
        sysinfo.memory("darwin", run=run)


def test_unknown_platform_raises_instead_of_returning_a_number():
    with pytest.raises(sysinfo.SysinfoUnavailable, match="cannot be measured"):
        sysinfo.memory("plan9")


class _FakeKernel32:
    """Stands in for kernel32: fills the MEMORYSTATUSEX it is handed, or fails."""

    def __init__(self, avail: int, total: int, ok: bool = True) -> None:
        self.avail, self.total, self.ok = avail, total, ok

    def GlobalMemoryStatusEx(self, pointer):  # noqa: N802 - the Win32 name
        status = pointer._obj
        assert status.dwLength == ctypes.sizeof(status)  # the caller must set dwLength first
        status.ullAvailPhys, status.ullTotalPhys = self.avail, self.total
        return 1 if self.ok else 0


def test_windows_reader_through_a_stand_in_kernel32():
    mem = sysinfo.memory("win32", kernel32=_FakeKernel32(avail=6 * GiB, total=32 * GiB))
    assert (mem.free_gb, mem.total_gb, mem.method) == (6.0, 32.0, "GlobalMemoryStatusEx")
    with pytest.raises(sysinfo.SysinfoUnavailable, match="failed"):
        sysinfo.memory("win32", kernel32=_FakeKernel32(1, 1, ok=False))
    with pytest.raises(sysinfo.SysinfoUnavailable, match="no physical memory"):
        sysinfo.memory("win32", kernel32=_FakeKernel32(0, 0))


@pytest.mark.skipif(not (sys.platform == "win32" or sys.platform.startswith("linux")),
                    reason="free RAM is read live only on Windows and Linux in this test")
def test_this_machine_reports_a_measured_number():
    mem = sysinfo.memory()
    assert 0 < mem.free_gb <= (mem.total_gb or float("inf"))
    assert sysinfo.free_ram_gb() > 0


def test_disk_free_walks_up_to_an_existing_folder(tmp_path):
    later = tmp_path / "not" / "made" / "yet"
    disk = sysinfo.disk(later)
    assert disk.checked == tmp_path.absolute()
    assert 0 < disk.free_gb <= disk.total_gb
    assert sysinfo.disk_free_gb(tmp_path) == pytest.approx(disk.free_gb, rel=0.05)


def test_venv_python_from_sysconfig_per_os(tmp_path):
    venv = tmp_path / "venv"
    assert sysinfo.venv_python(venv, "nt") == venv / "Scripts" / "python.exe"
    assert sysinfo.venv_python(venv, "posix") == venv / "bin" / "python"
    assert sysinfo.venv_scripts_dir(venv, "nt") == venv / "Scripts"


def test_venv_python_must_exist(tmp_path):
    venv = tmp_path / "venv"
    with pytest.raises(FileNotFoundError, match="python -m venv"):
        sysinfo.venv_python(venv, must_exist=True)
    python = sysinfo.venv_python(venv)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    assert sysinfo.venv_python(venv, must_exist=True) == python


def test_the_running_venv_resolves_to_this_interpreter():
    if sys.prefix == sys.base_prefix:
        pytest.skip("not running inside a virtual environment")
    found = sysinfo.venv_python(Path(sys.prefix), must_exist=True)
    assert found.samefile(sys.executable)
