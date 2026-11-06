"""_common.background: the hidden start order (WMI, detached with and without breakaway, new session) and the pid files.

Two kinds of test:
- offline: the process starters are replaced by recorders (`background._run`, `background._spawn`), so no
  process starts; these check the order, the flags, the cmd.exe line and the files written;
- `-m machine`: a real sleeper started from a child Python that exits at once must still be alive 2 s later
  (the measured failure this module exists for), then `stop()` ends it. Never run in CI.
"""

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from easel._common import background

SRC = Path(__file__).resolve().parents[2] / "src"


class FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self):
        return None


@pytest.fixture
def recorders(monkeypatch):
    """Replace both process starters; the test sets the answers. Any real start would be a bug."""
    calls = {"run": [], "spawn": []}
    answers = {"run": None, "spawn": []}

    def fake_run(argv, **kwargs):
        calls["run"].append((argv, kwargs))
        reply = answers["run"]
        if isinstance(reply, BaseException):
            raise reply
        return subprocess.CompletedProcess(argv, 0, reply or "", "")

    def fake_spawn(argv, **kwargs):
        calls["spawn"].append((argv, kwargs))
        reply = answers["spawn"].pop(0) if answers["spawn"] else FakeProcess(5150)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    monkeypatch.setattr(background, "_run", fake_run)
    monkeypatch.setattr(background, "_spawn", fake_spawn)
    return calls, answers


def test_cmd_line_for_wmi_sets_env_quotes_and_redirects():
    line = background.wmi_command_line(["py.exe", "-m", "http.server", "a&b", "with space", "end\\"],
                                       Path("logs") / "s.log", {"EASEL_HOME": "home dir"})
    log = str(Path("logs") / "s.log")
    assert line == ('cmd.exe /d /s /c "set "PYTHONIOENCODING=utf-8"&& set "EASEL_HOME=home dir"&& '
                    f'py.exe -m http.server "a&b" "with space" end\\ > "{log}" 2>&1"')
    assert background._cmd_quote("x(1)") == '"x(1)"'
    assert background._cmd_quote("dir\\ x\\") == '"dir\\ x\\\\"'
    assert background._cmd_quote("") == '""'


@pytest.mark.parametrize("cmd, env, needle", [
    (["py", 'say "hi"'], None, "a double quote"),
    (["py", "a\nb"], None, "a line break"),
    (["py", "50%"], None, "a %"),
    (["py"], {"X": "100%"}, "a %"),
])
def test_wmi_is_skipped_when_cmd_cannot_carry_the_line(cmd, env, needle):
    assert needle in background.wmi_blocker(cmd, Path("s.log"), env)


def test_wmi_blocker_passes_ordinary_lines():
    assert background.wmi_blocker(["py", "-m", "x", "--count", "4"], Path("s.log"), {"A": "b"}) is None


def test_windows_starts_through_wmi_first(tmp_path, recorders):
    calls, answers = recorders
    answers["run"] = "4242\n"
    pid_file = tmp_path / "run" / "review.pid"
    started = background.start(["py", "-m", "http.server"], tmp_path / "logs" / "review.log", pid_file,
                               cwd=tmp_path, env={"A": "b"}, platform="win32")
    assert (started.pid, started.method, started.status_line) == (4242, "wmi", "started via wmi")
    assert calls["spawn"] == []  # no detached fallback when WMI answered
    argv, kwargs = calls["run"][0]
    assert argv[0] == "powershell.exe" and "Win32_Process" in argv[-1] and "ShowWindow" in argv[-1]
    assert kwargs["env"]["BG_CMD"].startswith('cmd.exe /d /s /c "set "PYTHONIOENCODING=utf-8"&& set "A=b"&& py ')
    assert kwargs["env"]["BG_CWD"] == str(tmp_path)
    assert kwargs["creationflags"] == (background.CREATE_NO_WINDOW if os.name == "nt" else 0)
    assert pid_file.read_text(encoding="utf-8") == "4242\n"
    side = json.loads(background.sidecar_path(pid_file).read_text(encoding="utf-8"))
    assert side["method"] == "wmi" and side["pid"] == 4242 and side["log"] == "review.log"
    assert background.sidecar_path(pid_file).name == "review.pid.json"


def test_wmi_refused_then_detached_with_breakaway(tmp_path, recorders):
    calls, answers = recorders
    answers["run"] = "ERR 9\n"
    started = background.start(["py"], tmp_path / "s.log", tmp_path / "s.pid", platform="win32")
    assert (started.method, started.detail) == ("detached", "wmi refused; breakaway")
    flags = calls["spawn"][0][1]["creationflags"]
    assert flags == background.DETACHED_PROCESS | background.CREATE_NEW_PROCESS_GROUP | background.CREATE_BREAKAWAY_FROM_JOB


def test_breakaway_refused_falls_back_to_plain_detached(tmp_path, recorders):
    calls, answers = recorders
    answers["run"] = OSError("powershell missing")
    answers["spawn"] = [OSError("access denied: the job forbids breakaway"), FakeProcess(77)]
    started = background.start(["py"], tmp_path / "s.log", tmp_path / "s.pid", platform="win32")
    assert (started.pid, started.method) == (77, "detached")
    assert started.detail == "wmi refused; no breakaway (the job refused it)"
    assert [c[1]["creationflags"] for c in calls["spawn"]] == [
        background.DETACHED_PROCESS | background.CREATE_NEW_PROCESS_GROUP | background.CREATE_BREAKAWAY_FROM_JOB,
        background.DETACHED_PROCESS | background.CREATE_NEW_PROCESS_GROUP]
    assert json.loads(background.sidecar_path(tmp_path / "s.pid").read_text(encoding="utf-8"))["method"] == "detached"


def test_unsafe_line_skips_wmi_without_calling_it(tmp_path, recorders):
    calls, _answers = recorders
    started = background.start(["py", "50%"], tmp_path / "s.log", tmp_path / "s.pid", platform="win32")
    assert calls["run"] == [] and started.method == "detached"
    assert started.detail.startswith("wmi skipped: the command, log or env holds a %")


def test_posix_starts_in_a_new_session(tmp_path, recorders):
    calls, _answers = recorders
    started = background.start(["sleep", "1"], tmp_path / "s.log", tmp_path / "s.pid", env={"A": "b"},
                               platform="linux")
    assert calls["run"] == []
    kwargs = calls["spawn"][0][1]
    assert kwargs["start_new_session"] is True and "creationflags" not in kwargs
    assert kwargs["env"]["A"] == "b" and kwargs["env"]["PYTHONIOENCODING"] == "utf-8"
    assert (started.method, started.detail, started.poll()) == ("detached", "new session", None)


def test_started_at_follows_the_frozen_clock(tmp_path, recorders, frozen_clock, capsys):
    """The sidecar's time comes from the tool's clock, so a doc block or fixture showing it is reproducible."""
    _calls, answers = recorders
    answers["run"] = "4242\n"
    background.start(["py"], tmp_path / "s.log", tmp_path / "s.pid", platform="win32")
    side = json.loads(background.sidecar_path(tmp_path / "s.pid").read_text(encoding="utf-8"))
    assert side["started_at"] == "2030-01-01T00:00:00+00:00"
    background.start(["py"], tmp_path / "t.log", tmp_path / "t.pid", platform="win32", tool="demo")  # its own clock
    other = json.loads(background.sidecar_path(tmp_path / "t.pid").read_text(encoding="utf-8"))
    assert other["started_at"] != "2030-01-01T00:00:00+00:00"  # DEMO_NOW is not set: real time
    capsys.readouterr()


def test_status_line_and_read(tmp_path, monkeypatch, recorders):
    pid_file = tmp_path / "s.pid"
    assert background.status_line(pid_file) == "not running" and background.read(pid_file) is None
    _calls, answers = recorders
    answers["run"] = "4242\n"
    background.start(["py"], tmp_path / "s.log", pid_file, platform="win32")
    monkeypatch.setattr(background, "pid_alive", lambda pid: True)
    assert background.status_line(pid_file) == "running (pid 4242, started via wmi)"
    monkeypatch.setattr(background, "pid_alive", lambda pid: False)
    assert background.status_line(pid_file) == "not running (pid 4242 ended)"
    assert background.read(pid_file)["method"] == "wmi"


def test_stop_of_an_ended_process_only_cleans_up(tmp_path, monkeypatch, recorders):
    calls, _answers = recorders
    pid_file = tmp_path / "s.pid"
    pid_file.write_text("4242\n", encoding="utf-8")
    background.sidecar_path(pid_file).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(background, "pid_alive", lambda pid: False)
    assert background.stop(pid_file) == 4242
    assert not pid_file.exists() and not background.sidecar_path(pid_file).exists()
    assert calls["run"] == []  # nothing was killed
    assert background.stop(pid_file) is None


@pytest.mark.skipif(os.name != "nt", reason="the Windows stop path")
def test_stop_of_a_live_process_kills_its_tree_on_windows(tmp_path, monkeypatch, recorders):
    calls, _answers = recorders
    pid_file = tmp_path / "s.pid"
    pid_file.write_text("4242\n", encoding="utf-8")
    monkeypatch.setattr(background, "pid_alive", lambda pid: True)
    assert background.stop(pid_file) == 4242
    assert calls["run"][0][0] == ["taskkill", "/PID", "4242", "/T", "/F"]


@pytest.mark.skipif(os.name == "nt", reason="the POSIX stop path")
def test_stop_of_a_live_process_signals_its_group_on_posix(tmp_path, monkeypatch):
    sent = []
    pid_file = tmp_path / "s.pid"
    pid_file.write_text("4242\n", encoding="utf-8")
    monkeypatch.setattr(background, "pid_alive", lambda pid: True)
    monkeypatch.setattr(background.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    assert background.stop(pid_file) == 4242
    assert sent == [(4242, background.signal.SIGTERM)]


def test_pid_alive_for_this_process():
    assert background.pid_alive(os.getpid()) is True
    assert background.pid_alive(0) is False


# --- on a real machine ------------------------------------------------------------------------------------------

CHILD = textwrap.dedent("""
    import sys
    sys.path.insert(0, sys.argv[1])
    from easel._common import background
    started = background.start([sys.executable, "-c", "import time; time.sleep(60)"], sys.argv[2], sys.argv[3])
    print(started.pid, started.method)
""")


@pytest.mark.machine
def test_a_sleeper_outlives_the_child_that_started_it(tmp_path):
    pid_file = tmp_path / "sleeper.pid"
    child = subprocess.run([sys.executable, "-c", CHILD, str(SRC), str(tmp_path / "sleeper.log"), str(pid_file)],
                           capture_output=True, text=True, timeout=120)
    assert child.returncode == 0, child.stderr
    pid_text, method = child.stdout.split()
    if sys.platform == "win32":
        assert method == "wmi"  # the measured order: WMI first, detached only when WMI refuses
    time.sleep(2)
    try:
        assert background.pid_alive(int(pid_text)), "the sleeper died with the child that started it"
        assert background.status_line(pid_file).startswith(f"running (pid {pid_text}, started via {method}")
    finally:
        assert background.stop(pid_file) == int(pid_text)
    deadline = time.monotonic() + 10
    while background.pid_alive(int(pid_text)) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not background.pid_alive(int(pid_text))
