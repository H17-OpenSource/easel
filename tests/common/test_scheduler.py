"""_common.scheduler: the NullScheduler's RAM and disk wait, and the CommandScheduler's 0/2/3 contract.

No test waits in real time: a fake clock advances when the scheduler sleeps, and free RAM and disk come from
scripted readings. The external scheduler is a scripted stand-in, plus one real child Python for the argv path.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from easel._common import config, scheduler, sysinfo

REPO = Path(__file__).resolve().parents[2]


class Clock:
    """monotonic() and sleep() of a scheduler that never really sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def readings(*values: float):
    """A free-RAM reader that returns the values in turn, then repeats the last one."""
    queue = list(values)

    def read() -> float:
        return queue.pop(0) if len(queue) > 1 else queue[0]

    return read


@pytest.fixture
def families(tmp_path):
    folder = tmp_path / "families"
    folder.mkdir()
    files = {
        "base": "id: base\ninherits: null\nhost_ram_gb:\n  train: 5.5\n  eval: 8.5\n",
        "child": "id: child\ninherits: base\n",  # no host_ram_gb block: read from the parent
        "unmeasured": "id: unmeasured\ninherits: null\nhost_ram_gb:\n  train: null\n  eval: null\n",
        "loop_a": "id: loop_a\ninherits: loop_b\n",
        "loop_b": "id: loop_b\ninherits: loop_a\n",
        "wrong": "id: wrong\ninherits: null\nhost_ram_gb:\n  train: lots\n  eval: 1\n",
    }
    for name, text in files.items():
        (folder / f"{name}.yaml").write_text(text, encoding="utf-8")
    return folder


def null(families, *, ram=(9.0,), disk=(500.0,), lines=None, clock=None, **kwargs):
    clock = clock or Clock()
    return scheduler.NullScheduler(families_dir=families, free_ram=readings(*ram), free_disk=lambda _p: disk[0],
                                   say=(lines.append if lines is not None else lambda _l: None), clock=clock,
                                   sleep=clock.sleep, default_disk_path=families, **kwargs)


def request(family="base", step="eval", **kwargs):
    return scheduler.Request(title="rcp-20300101-001", tool="easel", family=family, step=step, **kwargs)


# --- the family files -------------------------------------------------------------------------------------------

def test_family_host_ram_reads_the_step_and_follows_inherits(families):
    assert scheduler.family_host_ram("base", "train", families) == 5.5
    assert scheduler.family_host_ram("base", "eval", families) == 8.5
    assert scheduler.family_host_ram("child", "eval", families) == 8.5
    assert scheduler.family_host_ram("unmeasured", "train", families) is None


@pytest.mark.parametrize("family, step, needle", [
    ("nosuch", "eval", "no family file families/nosuch.yaml"),
    ("loop_a", "eval", "inherits loops"),
    ("wrong", "train", "not a number"),
    ("base", "render", "step must be train or eval"),
])
def test_family_problems_are_refusals(families, family, step, needle):
    with pytest.raises(scheduler.SchedulerError, match=needle) as caught:
        scheduler.family_host_ram(family, step, families)
    assert caught.value.code == 2


def test_the_shipped_families_carry_the_numbers():
    assert scheduler.family_host_ram("sdxl", "train") == 5.5
    assert scheduler.family_host_ram("sdxl", "eval") == 8.5


# --- NullScheduler ----------------------------------------------------------------------------------------------

def test_wait_prints_one_line_every_30_s_until_ram_holds(families):
    lines: list[str] = []
    clock = Clock()
    s = null(families, ram=(3.9, 3.9, 3.9, 3.9, 3.9, 9.1), lines=lines, clock=clock, poll_s=10)
    ticket = s.plan(request(step="train"))
    outcome = s.wait(ticket)
    assert outcome.status == "granted" and outcome.code == 0
    assert outcome.line == "free RAM 9.1 GB >= 5.5 GB; free disk 500.0 GB >= 20 GB"  # the neutral 20 GB floor
    assert lines == ["waiting: free 3.9 GB < 5.5 GB needed by family base"] * 2  # at 0 s and 30 s, not 10 or 20
    assert clock.now == 50


def test_no_wait_checks_once_and_returns_the_waiting_line(families):
    lines: list[str] = []
    s = null(families, ram=(3.9,), lines=lines)
    outcome = s.wait(s.plan(request()), no_wait=True)
    assert (outcome.status, outcome.code) == ("waiting", 3)
    assert outcome.line == "waiting: free 3.9 GB < 8.5 GB needed by family base"
    assert lines == [outcome.line]


def test_timeout_gives_up_waiting(families):
    s = null(families, ram=(1.0,), poll_s=10)
    assert s.wait(s.plan(request()), timeout_s=25).status == "waiting"


def test_an_unmeasured_family_falls_back_to_the_floor_and_says_so(families):
    lines: list[str] = []
    s = null(families, ram=(1.5,), lines=lines, host_ram_floor_gb=2.0)
    ticket = s.plan(request(family="unmeasured"))
    assert ticket.need.ram_gb == 2.0
    assert lines[0] == ("family unmeasured: host_ram_gb.eval is not measured; waiting for "
                        "limits.host_ram_floor_gb = 2 GB instead")
    outcome = s.wait(ticket, no_wait=True)
    assert outcome.line == ("waiting: free 1.5 GB < 2 GB needed by family unmeasured "
                            "(host_ram_gb.eval not measured: limits.host_ram_floor_gb)")


def test_no_family_and_an_explicit_need(families):
    lines: list[str] = []
    s = null(families, lines=lines)
    assert s.plan(request(family=None)).need.ram_gb == 2.0
    assert "no family given" in lines[0]
    ticket = s.plan(request(host_ram_gb=12.2))
    assert (ticket.need.ram_gb, ticket.need.source) == (12.2, "job rcp-20300101-001")


def test_the_disk_floor_must_hold_too(families):
    s = null(families, ram=(20.0,), disk=(10.0,))
    outcome = s.wait(s.plan(request()), no_wait=True)
    assert outcome.line == "waiting: free disk 10.0 GB < 20 GB floor (limits.disk_floor_gb)"


def test_ram_that_cannot_be_measured_is_refused(families):
    def broken() -> float:
        raise sysinfo.SysinfoUnavailable("no reader for this platform")

    s = scheduler.NullScheduler(families_dir=families, free_ram=broken, free_disk=lambda _p: 500.0,
                                say=lambda _l: None)
    with pytest.raises(scheduler.SchedulerError) as caught:
        s.wait(s.plan(request()))
    assert caught.value.code == 2
    assert caught.value.lines[0] == "cannot measure free RAM; set limits.skip_ram_check = true to skip"


def test_skip_ram_check_starts_without_measuring(families):
    lines: list[str] = []

    def never() -> float:
        raise AssertionError("RAM must not be read")

    s = scheduler.NullScheduler(families_dir=families, free_ram=never, free_disk=lambda _p: 500.0,
                                say=lines.append, skip_ram_check=True)
    outcome = s.wait(s.plan(request()))
    assert outcome.status == "granted" and "RAM check skipped (limits.skip_ram_check = true)" in lines


def test_disk_that_cannot_be_measured_is_refused(families):
    def broken(_path):
        raise sysinfo.SysinfoUnavailable("unreadable volume")

    s = scheduler.NullScheduler(families_dir=families, free_ram=lambda: 20.0, free_disk=broken, say=lambda _l: None)
    with pytest.raises(scheduler.SchedulerError, match="cannot measure free disk space"):
        s.ready(s.plan(request()))


def test_the_life_of_a_job_with_a_stop_file(families, tmp_path):
    stop = tmp_path / "jobs" / "job-20300101-001.stop"
    s = null(families)
    ticket = s.plan(request(stop_file=stop))
    assert s.ready(ticket).status == "ready"
    stop.parent.mkdir()
    stop.write_text("left over from an earlier run")
    assert s.start(ticket).status == "started" and not stop.exists()  # a stale stop is cleared at start
    assert s.check(ticket).status == "ok"
    stop.touch()
    assert s.check(ticket).status == "yield"
    assert s.requeue(ticket).status == "requeued" and ticket.status == "waiting"
    updated = s.update(ticket, minutes=30, host_ram_gb=6.0)
    assert updated.status == "updated" and ticket.need.ram_gb == 6.0 and ticket.request.minutes == 30
    assert s.done(ticket, exit_code=0).status == "done" and not stop.exists()
    assert s.cancel(ticket).code == 0


# --- CommandScheduler -------------------------------------------------------------------------------------------

COMMANDS = {
    "plan": "sched plan --title {title} --gb {gb} --minutes {minutes} --by {by}",
    "ready": "sched ready {ref}",
    "wait": "sched wait {ref}",
    "start": "sched start {ref}",
    "check": "sched check {ref}",
    "done": "sched done {ref} --exit {exit_code}",
}


class FakeTool:
    """A scripted external scheduler: per verb, the (exit code, stdout) answers in turn; records every argv."""

    def __init__(self, **answers):
        self.answers = {verb: list(replies) for verb, replies in answers.items()}
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        replies = self.answers.get(argv[1], [(0, "")])
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        if isinstance(reply, BaseException):
            raise reply
        code, out = reply
        return subprocess.CompletedProcess(argv, code, out, "")


def command(families, tool, *, commands=COMMANDS, fail_open=False, lines=None, ram=(9.0,), clock=None):
    clock = clock or Clock()
    local = null(families, ram=ram, lines=lines, clock=clock)
    return scheduler.CommandScheduler(commands, local=local, fail_open=fail_open, runner=tool, poll_s=10)


def test_plan_passes_the_need_and_keeps_the_ref(families):
    tool = FakeTool(plan=[(0, "registered\n107\n")])
    s = command(families, tool)
    ticket = s.plan(request(step="train", minutes=40, by="demo-agent"))
    assert ticket.ref == "107"
    assert tool.calls[0] == ["sched", "plan", "--title", "rcp-20300101-001", "--gb", "5.5", "--minutes", "40",
                             "--by", "demo-agent"]


def test_wait_exit_2_waits_and_exit_0_grants(families):
    lines: list[str] = []
    tool = FakeTool(plan=[(0, "107")], wait=[(2, "queued"), (2, "queued"), (2, "queued"), (2, "queued"), (0, "granted")])
    s = command(families, tool, lines=lines)
    outcome = s.wait(s.plan(request()))
    assert (outcome.status, outcome.line) == ("granted", "granted")
    assert lines == ["waiting: the scheduler has not granted job 107 yet (queued)"] * 2  # 0 s and 30 s
    assert [c[1] for c in tool.calls].count("wait") == 5


def test_wait_exit_3_is_gone(families):
    s = command(families, FakeTool(plan=[(0, "107")], wait=[(3, "cancelled")]))
    outcome = s.wait(s.plan(request()))
    assert (outcome.status, outcome.code) == ("gone", 2)
    assert "cancelled" in outcome.line


def test_no_wait_asks_once(families):
    tool = FakeTool(plan=[(0, "107")], wait=[(2, "queued")])
    s = command(families, tool)
    outcome = s.wait(s.plan(request()), no_wait=True)
    assert (outcome.status, outcome.code) == ("waiting", 3)
    assert [c[1] for c in tool.calls] == ["plan", "wait"]


@pytest.mark.parametrize("reply", [(1, "Traceback"), (7, ""), OSError("no such program"),
                                   subprocess.TimeoutExpired("sched", 120)])
def test_an_unreachable_scheduler_fails_closed(families, reply):
    s = command(families, FakeTool(plan=[reply]))
    with pytest.raises(scheduler.SchedulerError) as caught:
        s.plan(request())
    assert caught.value.code == 3
    assert caught.value.lines[0].startswith("scheduler unreachable: plan")
    assert "failing closed" in caught.value.lines[1]


def test_fail_open_switches_to_the_local_check_and_says_so(families):
    lines: list[str] = []
    tool = FakeTool(plan=[(0, "107")], wait=[OSError("gone away")])
    s = command(families, tool, fail_open=True, lines=lines, ram=(3.0, 9.0))
    ticket = s.plan(request())
    outcome = s.wait(ticket)
    assert outcome.status == "granted" and ticket.local_only
    assert lines[0].startswith("scheduler unreachable: wait could not run")
    assert lines[0].endswith("scheduler.fail_open = true: using the local RAM and disk check instead")
    assert lines[1] == "waiting: free 3.0 GB < 8.5 GB needed by family base"
    calls_before = len(tool.calls)
    assert s.start(ticket).status == "started" and len(tool.calls) == calls_before  # later verbs stay local


def test_plan_refused_by_the_scheduler(families):
    s = command(families, FakeTool(plan=[(3, "over the limit")]))
    with pytest.raises(scheduler.SchedulerError, match="refused the plan") as caught:
        s.plan(request())
    assert caught.value.code == 2


def test_check_yield_writes_the_stop_file(families, tmp_path):
    stop = tmp_path / "jobs" / "job-20300101-001.stop"
    tool = FakeTool(plan=[(0, "107")], check=[(0, "running"), (0, "yield"), (1, "")])
    s = command(families, tool)
    ticket = s.plan(request(stop_file=stop))
    assert s.check(ticket).status == "ok" and not stop.exists()
    outcome = s.check(ticket)
    assert outcome.status == "yield" and stop.exists()
    unknown = s.check(ticket)
    assert unknown.status == "unknown" and unknown.code == 0  # a failed check never stops a running job
    assert ["sched", "check", "107"] in tool.calls


def test_bookkeeping_reports_failures_and_skips_unset_verbs(families):
    tool = FakeTool(plan=[(0, "107")], done=[(1, "")])
    s = command(families, tool)
    ticket = s.plan(request())
    done = s.done(ticket, exit_code=4)
    assert (done.status, done.code) == ("failed", 1)
    assert ["sched", "done", "107", "--exit", "4"] in tool.calls
    assert s.cancel(ticket).status == "skipped"
    assert s.requeue(ticket).status == "skipped"
    assert s.update(ticket, minutes=5).status == "skipped"


def test_ready_and_start_codes(families):
    tool = FakeTool(plan=[(0, "107")], ready=[(0, "queued"), (2, "busy"), (3, "ended")], start=[(0, ""), (3, "")])
    s = command(families, tool)
    ticket = s.plan(request())
    assert [s.ready(ticket).status for _ in range(3)] == ["ready", "waiting", "gone"]
    assert s.start(ticket).status == "started"
    assert s.start(ticket).status == "gone"


def test_required_templates_and_placeholders(families):
    local = null(families)
    with pytest.raises(scheduler.SchedulerError, match="scheduler.wait_command, scheduler.start_command"):
        scheduler.CommandScheduler({"plan": "x", "done": "y"}, local=local)
    with pytest.raises(scheduler.SchedulerError, match="no scheduler verb 'launch'"):
        scheduler.CommandScheduler({**COMMANDS, "launch": "x"}, local=local)
    bad = scheduler.CommandScheduler({**COMMANDS, "plan": "sched plan {ref}"}, local=local, runner=FakeTool())
    with pytest.raises(scheduler.SchedulerError, match=r"uses \{ref\}, which is not known here") as caught:
        bad.plan(request())
    assert caught.value.code == 2


def test_split_command_keeps_windows_paths_and_quotes():
    assert scheduler.split_command(r'"<tools dir>\py 3\python.exe" tool.py --note "a b" \x' + " 'c d'") == \
        [r"<tools dir>\py 3\python.exe", "tool.py", "--note", "a b", "\\x", "c d"]
    assert scheduler.split_command("a '' b") == ["a", "", "b"]
    with pytest.raises(scheduler.SchedulerError, match="unbalanced"):
        scheduler.split_command('a "b')
    assert scheduler.render("t --gb {gb} {{literal}}", {"gb": "5.5"}, "k") == ["t", "--gb", "5.5", "{literal}"]


def test_a_real_child_process_through_the_templates(families, tmp_path):
    script = tmp_path / "sched.py"
    script.write_text(textwrap.dedent("""
        import pathlib, sys
        verb, rest = sys.argv[1], sys.argv[2:]
        state = pathlib.Path(__file__).with_name("waits.txt")
        if verb == "plan":
            print("job", "ref follows"); print("job-7"); sys.exit(0)
        if verb == "wait":
            n = int(state.read_text()) if state.exists() else 0
            state.write_text(str(n + 1))
            print("queued" if n == 0 else "granted"); sys.exit(2 if n == 0 else 0)
        if verb == "check":
            print("yield"); sys.exit(0)
        print(verb, *rest); sys.exit(0)
    """), encoding="utf-8")
    base = f'"{sys.executable}" "{script}"'
    commands = {verb: f"{base} {verb} {{ref}}" for verb in ("wait", "start", "check", "done")}
    commands["plan"] = f"{base} plan --title {{title}}"
    lines: list[str] = []
    s = command(families, None, commands=commands, lines=lines)
    s.runner = subprocess.run
    stop = tmp_path / "stop"
    ticket = s.plan(request(stop_file=stop))
    assert ticket.ref == "job-7"
    assert s.wait(ticket).status == "granted"
    assert lines == ["waiting: the scheduler has not granted job job-7 yet (queued)"]
    assert s.start(ticket).line == "start job-7"
    assert s.check(ticket).status == "yield" and stop.exists()
    assert s.done(ticket).status == "done"


# --- from the instance config -----------------------------------------------------------------------------------

def write_toml(tmp_path, body: str) -> Path:
    path = tmp_path / "home" / "easel.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text('schema = "easel/config/1"\n' + body, encoding="utf-8")
    return path


def test_from_config_none_uses_the_limits(tmp_path, families):
    path = write_toml(tmp_path, '[limits]\nhost_ram_floor_gb = 3.0\ndisk_floor_gb = 50\n'
                                '[scheduler]\nkind = "none"\npoll_s = 2\nstop_file = "jobs/yield.stop"\n')
    cfg = config.load("easel", path)
    s = scheduler.from_config(cfg, families_dir=families, free_ram=lambda: 9.0, free_disk=lambda _p: 60.0)
    assert isinstance(s, scheduler.NullScheduler)
    assert (s.host_ram_floor_gb, s.disk_floor_gb, s.poll_s) == (3.0, 50.0, 2.0)
    assert s.stop_file == cfg.root / "jobs" / "yield.stop"
    assert s.default_disk_path == cfg.root
    assert s.wait(s.plan(request())).status == "granted"


def test_from_config_defaults_without_keys(tmp_path, families):
    s = scheduler.from_config(config.load("easel", write_toml(tmp_path, "")), families_dir=families)
    assert isinstance(s, scheduler.NullScheduler)
    assert (s.host_ram_floor_gb, s.disk_floor_gb, s.skip_ram_check) == (2.0, 20.0, False)


def test_from_config_command(tmp_path, families):
    body = "[scheduler]\nkind = \"command\"\nfail_open = true\n" + "".join(
        f"{verb}_command = '{template}'\n" for verb, template in COMMANDS.items())
    cfg = config.load("easel", write_toml(tmp_path, body))
    s = scheduler.from_config(cfg, families_dir=families, runner=FakeTool(plan=[(0, "9")]))
    assert isinstance(s, scheduler.CommandScheduler) and s.fail_open
    assert s.commands["plan"] == COMMANDS["plan"]
    assert s.plan(request()).ref == "9"


def test_from_config_command_without_templates_is_refused(tmp_path, families):
    cfg = config.load("easel", write_toml(tmp_path, '[scheduler]\nkind = "command"\n'))
    with pytest.raises(scheduler.SchedulerError, match="needs scheduler.plan_command"):
        scheduler.from_config(cfg, families_dir=families)


def test_no_wait_flag():
    import argparse

    parser = argparse.ArgumentParser()
    scheduler.add_wait_args(parser)
    assert parser.parse_args(["--no-wait"]).no_wait is True
    assert parser.parse_args([]).no_wait is False
