"""_common.store: atomic records, the PermissionError retry, the OS lock under contention, JSONL and the index."""

import errno
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from easel._common import store

PKG = "easel"
SRC = Path(__file__).resolve().parents[2] / "src"

# Child processes import the store from the source tree. Each prints one line when it is ready.
COUNTER_CHILD = """
import sys, time
from pathlib import Path
from {pkg}._common import store
root, n, who = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
deadline = time.monotonic() + 20
while not (root / "go").exists() and time.monotonic() < deadline:
    time.sleep(0.01)
for i in range(n):
    with store.lock(root, "{pkg}", timeout=20):
        counter = root / "counter.txt"
        value = int(counter.read_text()) if counter.exists() else 0
        time.sleep(0.003)  # widen the read-modify-write window: without the lock, updates get lost
        counter.write_text(str(value + 1))
    store.append_jsonl(root / "events.jsonl", {{"who": who, "i": i}}, tool="{pkg}", source="test", lock_root=root)
print("done", flush=True)
"""

LOCK_HOLDER = """
import os, sys, time
from pathlib import Path
from {pkg}._common import store
with store.lock(Path(sys.argv[1]), "{pkg}"):
    print("locked", os.getpid(), flush=True)
    time.sleep(60)
"""

FILE_HOLDER = """
import sys, time
handle = open(sys.argv[1], "rb")
print("open", flush=True)
time.sleep(float(sys.argv[2]))
handle.close()
"""


def spawn(code: str, *args: str) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    return subprocess.Popen([sys.executable, "-c", code.format(pkg=PKG), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)


# --- atomic records -----------------------------------------------------------------------------------------------

def test_write_text_is_atomic_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "sub" / "record.json"
    store.write_text(target, "one\n")
    store.write_text(target, "two\n")
    assert target.read_bytes() == b"two\n"
    assert [p.name for p in target.parent.iterdir()] == ["record.json"]


def test_write_record_puts_schema_first(tmp_path):
    yml = store.write_record(tmp_path / "r.yaml", "easel/resource/1", {"name": "demo_robot", "schema": "ignored"})
    jsn = store.write_record(tmp_path / "r.json", "easel/run/1", {"id": "run-20300101-001"})
    assert yml.read_text(encoding="utf-8").splitlines()[0] == "schema: easel/resource/1"
    assert list(json.loads(jsn.read_text(encoding="utf-8"))) == ["schema", "id"]
    assert store.read_record(yml, "easel/resource/")["name"] == "demo_robot"
    with pytest.raises(ValueError):
        store.read_record(jsn, "easel/resource/")
    with pytest.raises(ValueError):
        store.write_record(tmp_path / "r.txt", "easel/x/1", {})


def test_iter_records_skips_temp_files(tmp_path):
    store.write_record(tmp_path / "b.yaml", "easel/x/1", {"n": 2})
    store.write_record(tmp_path / "a.yaml", "easel/x/1", {"n": 1})
    (tmp_path / "c.yaml.tmp-123").write_text("schema: broken", encoding="utf-8")
    assert [data["n"] for _, data in store.iter_records(tmp_path)] == [1, 2]


def test_replace_while_open(tmp_path):
    """Another process holds the target open for 1 s (Windows refuses the replace meanwhile); the write lands."""
    target = tmp_path / "held.json"
    target.write_text("old\n", encoding="utf-8")
    holder = spawn(FILE_HOLDER, str(target), "1.0")
    try:
        assert holder.stdout.readline().strip() == "open", holder.stderr.read()
        started = time.monotonic()
        store.write_text(target, "new\n")
        elapsed = time.monotonic() - started
    finally:
        holder.wait(timeout=30)
    assert target.read_text(encoding="utf-8") == "new\n"
    assert not list(tmp_path.glob("*.tmp-*"))
    if os.name == "nt":
        assert elapsed >= 0.5, "the replace should have waited for the holder to close the file"


def test_replace_gives_up_with_exit_4_and_the_stuck_line(tmp_path, monkeypatch, capsys):
    target = tmp_path / "stuck.yaml"
    target.write_text("old\n", encoding="utf-8")
    calls = []

    def refuse(src, dst):
        calls.append(dst)
        raise PermissionError(13, "held")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(store.Held) as caught:
        store.write_text(target, "new\n", budget_s=0.3)
    assert caught.value.code == store.EXIT_HELD == 4
    assert 2 <= len(calls) <= store.REPLACE_ATTEMPTS
    err = capsys.readouterr().err
    assert "STUCK: another process holds" in err and "retry in a minute" in err
    assert target.read_text(encoding="utf-8") == "old\n"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_store_replace_retries_then_lands(tmp_path, monkeypatch):
    """`store.replace` is the retrying rename for a file written another way (a finished download)."""
    src, dst = tmp_path / "file.part", tmp_path / "file.bin"
    src.write_bytes(b"data")
    real, calls = os.replace, []

    def refuse_twice(a, b):
        calls.append(b)
        if len(calls) < 3:
            raise PermissionError(13, "held by a scanner")
        real(a, b)

    monkeypatch.setattr(os, "replace", refuse_twice)
    assert store.replace(src, dst) == dst
    assert dst.read_bytes() == b"data" and not src.exists() and len(calls) == 3


def test_held_is_not_swallowed_by_except_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "replace", lambda s, d: (_ for _ in ()).throw(PermissionError(13, "held")))
    with pytest.raises(SystemExit):
        try:
            store.write_text(tmp_path / "x.json", "{}", budget_s=0.1)
        except Exception:  # the point: a generic handler must not catch it
            pytest.fail("Held must not be an Exception")


# --- the lock -----------------------------------------------------------------------------------------------------

def test_two_processes_under_contention_lose_no_update(tmp_path):
    n = 15
    children = [spawn(COUNTER_CHILD, str(tmp_path), str(n), who) for who in ("a", "b")]
    (tmp_path / "go").write_text("", encoding="utf-8")
    for child in children:
        out, err = child.communicate(timeout=120)
        assert child.returncode == 0, err
        assert out.strip() == "done"
    assert int((tmp_path / "counter.txt").read_text()) == 2 * n
    read = store.read_jsonl(tmp_path / "events.jsonl")
    assert len(read.records) == 2 * n and read.bad_lines == [] and not read.torn_tail
    assert {(r["who"], r["i"]) for r in read.records} == {(w, i) for w in "ab" for i in range(n)}


def test_lock_holder_killed_next_writer_acquires_within_1s(tmp_path, capsys):
    holder = spawn(LOCK_HOLDER, str(tmp_path))
    try:
        word, pid = holder.stdout.readline().split()
        assert word == "locked", holder.stderr.read()
        with pytest.raises(store.Held) as caught, store.lock(tmp_path, PKG, timeout=0.3):
            pass
        assert caught.value.code == 4
        assert f"pid {pid} " in capsys.readouterr().err  # the diagnostics name the holder
        # TerminateProcess on Windows, SIGKILL elsewhere: no cleanup code runs. (A venv's python.exe on Windows
        # is a launcher; killing it also ends the interpreter it started, which is the process printing `pid`.)
        holder.kill()
        holder.wait(timeout=30)
        started = time.monotonic()
        with store.lock(tmp_path, PKG, timeout=1.0):
            elapsed = time.monotonic() - started
    finally:
        if holder.poll() is None:
            holder.kill()
            holder.wait(timeout=30)
    assert elapsed < 1.0
    assert (tmp_path / ".lock").exists()  # persistent: never deleted
    assert store.holder_of(tmp_path / ".lock")["pid"] == os.getpid()


@pytest.mark.parametrize("code", ["ENOLCK", "EOPNOTSUPP", "EINVAL"])
def test_a_folder_without_lock_support_fails_at_once(tmp_path, monkeypatch, capsys, code):
    """Not contention: no 30 s wait and no hunt for a holder that does not exist; exit 4 names the error."""
    number = getattr(errno, code)

    def unsupported(_handle):
        raise OSError(number, os.strerror(number))

    monkeypatch.setattr(store, "_try_os_lock", unsupported)
    started = time.monotonic()
    with pytest.raises(store.LockUnsupported) as caught, store.lock(tmp_path, PKG, timeout=20):
        pass
    assert time.monotonic() - started < 2.0
    assert caught.value.code == store.EXIT_HELD and isinstance(caught.value, store.Held)
    assert "does not support file locks" in caught.value.line and errno.errorcode[number] in caught.value.line
    assert "another process holds" not in capsys.readouterr().err


def test_contention_errors_are_still_waited_for(tmp_path, monkeypatch, capsys):
    attempts = []

    def busy_then_free(_handle):
        attempts.append(1)
        if len(attempts) < 3:
            raise BlockingIOError(errno.EAGAIN, "busy")

    monkeypatch.setattr(store, "_try_os_lock", busy_then_free)
    with store.lock(tmp_path, PKG, timeout=5):
        pass
    assert len(attempts) == 3
    capsys.readouterr()


def test_lock_is_reentrant_and_threads_take_turns(tmp_path, capsys):
    results = []
    with store.lock(tmp_path, PKG):
        with store.lock(tmp_path, PKG):  # the same thread again: no deadlock
            pass

        def other_thread():
            try:
                with store.lock(tmp_path, PKG, timeout=0.2):
                    results.append("acquired")
            except store.Held:
                results.append("held")

        t = threading.Thread(target=other_thread)
        t.start()
        t.join(timeout=10)
    assert results == ["held"]
    t2 = threading.Thread(target=other_thread)
    t2.start()
    t2.join(timeout=10)
    assert results == ["held", "acquired"]
    capsys.readouterr()


# --- event logs ---------------------------------------------------------------------------------------------------

def test_append_jsonl_stamps_at_v_source_first(tmp_path, frozen_clock):
    line = store.append_jsonl(tmp_path / "e.jsonl", {"run": "run-20300101-001", "verdict": "keep"},
                              tool=PKG, source="review")
    assert list(line)[:3] == ["at", "v", "source"]
    assert line["at"] == "2030-01-01T00:00:00+00:00" and line["v"] == 1 and line["source"] == "review"
    stored = json.loads((tmp_path / "e.jsonl").read_text(encoding="utf-8"))
    assert stored == line


def test_torn_tail_is_skipped_and_the_next_append_starts_clean(tmp_path, capsys):
    log = tmp_path / "reviews.jsonl"
    log.write_bytes(b'{"k": "a", "n": 1}\n{"k": "b", "n": 2}\n{"k": "a", "n": 3')  # a writer died mid-line
    read = store.read_jsonl(log)
    assert [r["n"] for r in read.records] == [1, 2]
    assert read.torn_tail and read.bad_lines == [3] and read.torn_file is None
    assert store.latest_by_key(log, "k") == {"a": {"k": "a", "n": 1}, "b": {"k": "b", "n": 2}}
    store.append_jsonl(log, {"k": "a", "n": 4}, tool=PKG, source="test")
    again = store.read_jsonl(log)
    assert [r["n"] for r in again.records] == [1, 2, 4]
    assert again.bad_lines == [] and not again.torn_tail  # the log is clean: no broken line in its middle
    assert store.latest_by_key(log, "k")["a"]["n"] == 4
    # The fragment is kept, findable, beside the log, and the append said so on stderr (by file name only).
    assert again.torn_file == store.torn_path(log) == tmp_path / "reviews.jsonl.torn"
    assert again.torn_file.read_bytes() == b'{"k": "a", "n": 3\n'
    assert "reviews.jsonl ended in a torn line" in capsys.readouterr().err


def test_a_second_torn_tail_is_appended_to_the_same_sidecar(tmp_path, capsys):
    log = tmp_path / "e.jsonl"
    for fragment in (b'{"x": 1', b'{"y": 2'):
        with open(log, "ab") as handle:
            handle.write(b'{"ok": true}\n' + fragment)
        store.append_jsonl(log, {"z": 3}, tool=PKG, source="test")
    assert store.torn_path(log).read_bytes() == b'{"x": 1\n{"y": 2\n'
    assert store.read_jsonl(log).bad_lines == []
    capsys.readouterr()


def test_a_last_line_that_parses_or_a_non_jsonl_file_is_kept(tmp_path):
    """Only a JSONL record that does not parse is torn; a complete record or a journal line just gets its newline."""
    log = tmp_path / "e.jsonl"
    log.write_bytes(b'{"n": 1}\n{"n": 2}')  # complete, only the newline is missing
    store.append_jsonl(log, {"n": 3}, tool=PKG, source="test")
    assert [r["n"] for r in store.read_jsonl(log).records] == [1, 2, 3]
    notes = tmp_path / "2030-01-01.md"
    notes.write_bytes(b"# Journal 2030-01-01\n\n- 00:00 typed by hand without a newline")
    store.append_bytes_locked(notes, b"- 00:01 next\n")
    assert notes.read_bytes().endswith(b"without a newline\n- 00:01 next\n")
    assert not store.torn_path(log).exists() and not store.torn_path(notes).exists()


def test_a_torn_tail_longer_than_one_read_block(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "_TAIL_BLOCK", 4)
    log = tmp_path / "e.jsonl"
    log.write_bytes(b'{"n": 1}\n' + b'{"long": "' + b"x" * 40)
    store.append_jsonl(log, {"n": 2}, tool=PKG, source="test")
    assert [r["n"] for r in store.read_jsonl(log).records] == [1, 2]
    assert store.torn_path(log).read_bytes() == b'{"long": "' + b"x" * 40 + b"\n"
    capsys.readouterr()


def test_latest_by_key_composite_and_missing_keys(tmp_path):
    log = tmp_path / "e.jsonl"
    rows = [{"run": "r1", "n": 1, "v": "keep"}, {"run": "r1", "n": 2, "v": "drop"}, {"run": "r1", "n": 1, "v": "drop"},
            {"other": True}]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows) + "\n\n", encoding="utf-8")
    latest = store.latest_by_key(log, ("run", "n"))
    assert latest == {("r1", 1): rows[2], ("r1", 2): rows[1]}
    assert store.latest_by_key(log, lambda r: r.get("v")) == {"drop": rows[2], "keep": rows[0], None: rows[3]}
    assert store.read_jsonl(tmp_path / "missing.jsonl").records == []


def test_append_text_completes_a_line_without_newline(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("# head", encoding="utf-8")
    store.append_text(path, "- one\n", tool=PKG)
    assert path.read_text(encoding="utf-8") == "# head\n- one\n"


# --- generated views ----------------------------------------------------------------------------------------------

def test_write_index_is_deterministic_and_marked_generated(tmp_path):
    rows = [{"sha16": "bbbb", "name": "two|pipes", "tags": ["x", "y"], "self_trained": True},
            {"sha16": "aaaa", "name": "demo_robot\nlora", "tags": [], "self_trained": False}]
    jsonl, md = store.write_index(tmp_path, rows, columns=["sha16", "name", "tags", "self_trained"],
                                  title="Resources", tool=PKG, sort_key="sha16")
    first = md.read_bytes()
    text = first.decode("utf-8")
    assert text.startswith("# Resources\n")
    assert "do not edit" in text and "`easel index`" in text and "2 records." in text
    assert "| aaaa | demo_robot lora |  | no |" in text
    assert "| bbbb | two\\|pipes | x, y | yes |" in text
    assert [json.loads(line)["sha16"] for line in jsonl.read_text(encoding="utf-8").splitlines()] == ["aaaa", "bbbb"]
    store.write_index(tmp_path, rows, columns=["sha16", "name", "tags", "self_trained"], title="Resources", tool=PKG,
                      sort_key="sha16")
    assert md.read_bytes() == first


def test_readers_tolerate_a_bom(tmp_path):
    (tmp_path / "e.jsonl").write_bytes(b'\xef\xbb\xbf{"k": 1}\n')
    assert store.read_jsonl(tmp_path / "e.jsonl").records == [{"k": 1}]
    (tmp_path / "r.json").write_bytes(b'\xef\xbb\xbf{"schema": "easel/x/1"}')
    assert store.read_record(tmp_path / "r.json", "easel/x/")["schema"] == "easel/x/1"
