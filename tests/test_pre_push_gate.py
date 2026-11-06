"""The pre-push gate and its installer, run against a scratch clone in tmp_path; no remote is ever contacted.

The gate is invoked the way git invokes a pre-push hook: the remote's name as an argument and one
`<local ref> <local sha> <remote ref> <remote sha>` line per pushed ref on stdin. Every value that would trip
the scanner (the planted word, the e-mail address, the placeholder) is built here at run time from fragments,
so this file itself stays clean under the scan it tests.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "tools" / "pre-push-gate"
INSTALLER = REPO / "tools" / "install-hooks.py"
ZEROS = "0" * 40
AT = chr(64)
EMAIL = "builder" + AT + "example.invalid"
IDENTITY = f"Demo Builder <{EMAIL}>"
OTHER_IDENTITY = "Somebody Else <someone" + AT + "example.invalid>"
PLANTED = "zq" + "xv" + "-planted-" + "word"
PLACEHOLDER = "<" + "org" + ">"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not on PATH")


@pytest.fixture
def env(tmp_path: Path) -> dict[str, str]:
    """git isolated from the machine's own configuration, with the demo identity."""
    config = tmp_path / "gitconfig"
    config.write_text("", encoding="utf-8")
    return {**os.environ, "GIT_CONFIG_GLOBAL": str(config), "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "Demo Builder", "GIT_AUTHOR_EMAIL": EMAIL,
            "GIT_COMMITTER_NAME": "Demo Builder", "GIT_COMMITTER_EMAIL": EMAIL,
            "GIT_AUTHOR_DATE": "2030-01-01T00:00:00+00:00", "GIT_COMMITTER_DATE": "2030-01-01T00:00:00+00:00"}


@pytest.fixture
def words(tmp_path: Path) -> Path:
    path = tmp_path / "private-words.txt"
    path.write_text("# synthetic test entries\n" + PLANTED + "\n", encoding="utf-8")
    return path


@pytest.fixture
def clone(tmp_path: Path, env: dict[str, str]) -> Path:
    repo = tmp_path / "clone"
    repo.mkdir()
    git(repo, env, "-c", "init.defaultBranch=main", "init", "-q")
    return repo


def git(repo: Path, env: dict[str, str], *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=repo, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def commit(repo: Path, env: dict[str, str], files: dict[str, str | bytes], message: str = "demo change") -> str:
    for rel, content in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    git(repo, env, "add", "-A")
    git(repo, env, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
    return git(repo, env, "rev-parse", "HEAD")


def gate(repo: Path, env: dict[str, str], stdin: str, *options: str, words: Path | None = None,
         identity: str | None = IDENTITY) -> subprocess.CompletedProcess[str]:
    args = [sys.executable, str(GATE), "--no-gitleaks", *options]
    if words is not None:
        args += ["--words", str(words)]
    if identity is not None:
        args += ["--identity", identity]
    return subprocess.run([*args, "origin"], cwd=repo, env=env, input=stdin, capture_output=True, text=True,
                          timeout=120)


def line(local: str, remote: str = ZEROS) -> str:
    return f"refs/heads/main {local} refs/heads/main {remote}\n"


def assert_quiet(done: subprocess.CompletedProcess[str]) -> None:
    """The gate prints counts only: never the planted word or the e-mail address."""
    shown = (done.stdout + done.stderr).lower()
    assert PLANTED not in shown
    assert EMAIL not in shown


def last_line(done: subprocess.CompletedProcess[str]) -> str:
    return done.stdout.strip().splitlines()[-1]


def test_clean_range_passes(clone: Path, env: dict[str, str], words: Path) -> None:
    sha = commit(clone, env, {"README.md": "# demo_robot\n\nclone from https://github.com/H17-OpenSource/easel\n"})
    done = gate(clone, env, line(sha), words=words)
    assert done.returncode == 0, done.stdout + done.stderr
    assert last_line(done) == "pre-push gate: 0 hit(s) in 1 commit(s) and tag(s) on 1 ref(s); push allowed"
    assert "gitleaks skipped" in done.stderr
    log = (clone / ".git" / "pre-push-gate.log").read_text(encoding="utf-8")
    assert "result=allowed" in log and "gitleaks=skipped" in log


def test_planted_word_in_an_added_line_is_refused(clone: Path, env: dict[str, str], words: Path) -> None:
    base = commit(clone, env, {"README.md": "# demo_robot\n"})
    head = commit(clone, env, {"docs/notes.md": f"intro\nthe {PLANTED.upper()} sits here\n"})
    done = gate(clone, env, line(head, base), "--explain", words=words)
    assert done.returncode == 1, done.stdout + done.stderr
    assert "push refused" in last_line(done)
    assert f"{head[:12]} docs/notes.md:2: private:synthetic test entries 1" in done.stdout
    assert_quiet(done)
    # the clean commit below the range is not checked again
    assert "1 of 1 commit(s)" in last_line(done)


def test_planted_word_in_a_message_is_refused(clone: Path, env: dict[str, str], words: Path) -> None:
    sha = commit(clone, env, {"README.md": "# demo_robot\n"}, message=f"demo change\n\nsee {PLANTED}")
    done = gate(clone, env, line(sha), words=words)
    assert done.returncode == 1
    assert f"{sha[:12]} message: private:synthetic test entries 1" in done.stdout
    assert_quiet(done)


def test_public_patterns_apply_and_tests_are_never_excused(clone: Path, env: dict[str, str], words: Path) -> None:
    drive = "C" + ":" + "\\" + "Users" + "\\" + "example"
    sha = commit(clone, env, {"README.md": "see the docs\n", "tests/test_demo.py": f"# {PLACEHOLDER}\n",
                              "docs/setup.md": f"open {drive}\n"})
    done = gate(clone, env, line(sha), words=words)
    assert done.returncode == 1
    assert f"{sha[:12]} tests/test_demo.py: placeholder 1" in done.stdout
    assert f"{sha[:12]} docs/setup.md: drive-path 1" in done.stdout
    assert "README.md" not in done.stdout  # a clean file outside tests/ is not reported


def test_identity_must_be_allowed(clone: Path, env: dict[str, str], words: Path) -> None:
    sha = commit(clone, env, {"README.md": "# demo_robot\n"})
    done = gate(clone, env, line(sha), words=words, identity=OTHER_IDENTITY)
    assert done.returncode == 1
    assert f"{sha[:12]} author: identity-not-allowed 1" in done.stdout
    assert f"{sha[:12]} committer: identity-not-allowed 1" in done.stdout
    assert_quiet(done)


@pytest.mark.parametrize("role", ["author", "committer"])
def test_a_date_offset_other_than_utc_is_refused(clone: Path, env: dict[str, str], words: Path, role: str) -> None:
    """A local offset tells the author's time zone; the publish route commits in UTC (+0000)."""
    local = {**env, f"GIT_{role.upper()}_DATE": "2030-01-01T05:30:00+05:30"}
    sha = commit(clone, local, {"README.md": "# demo_robot\n"})
    done = gate(clone, env, line(sha), words=words)
    assert done.returncode == 1, done.stdout + done.stderr
    assert f"{sha[:12]} {role}: date-not-utc 1" in done.stdout
    other = "committer" if role == "author" else "author"
    assert f"{sha[:12]} {other}:" not in done.stdout
    assert "TZ=UTC" in done.stdout and "push refused" in last_line(done)
    assert_quiet(done)


def test_a_tagger_date_offset_other_than_utc_is_refused(clone: Path, env: dict[str, str], words: Path) -> None:
    commit(clone, env, {"README.md": "# demo_robot\n"})
    git(clone, {**env, "GIT_COMMITTER_DATE": "2030-01-01T00:00:00-07:00"}, "tag", "-a", "v0.0.2", "-m", "release")
    tag = git(clone, env, "rev-parse", "v0.0.2")
    done = gate(clone, env, f"refs/tags/v0.0.2 {tag} refs/tags/v0.0.2 {ZEROS}\n", words=words)
    assert done.returncode == 1, done.stdout + done.stderr
    assert f"{tag[:12]} tagger: date-not-utc 1" in done.stdout


def test_trailer_email_is_excused_in_messages_only(clone: Path, env: dict[str, str], words: Path) -> None:
    helper = "helper" + AT + "example.invalid"
    sha = commit(clone, env, {"README.md": "# demo_robot\n"}, message=f"demo change\n\nCo-Authored-By: H <{helper}>")
    refused = gate(clone, env, line(sha), words=words)
    assert refused.returncode == 1 and f"{sha[:12]} message: email 1" in refused.stdout
    allowed = gate(clone, env, line(sha), "--trailer-email", helper, words=words)
    assert allowed.returncode == 0, allowed.stdout


def test_binary_files_are_refused(clone: Path, env: dict[str, str], words: Path) -> None:
    sha = commit(clone, env, {"data/blob.dat": b"\x00\x01\x02demo\x00"})
    done = gate(clone, env, line(sha), words=words)
    assert done.returncode == 1
    assert f"{sha[:12]} data/blob.dat: binary-file 1" in done.stdout


def test_annotated_tag_message_is_checked(clone: Path, env: dict[str, str], words: Path) -> None:
    sha = commit(clone, env, {"README.md": "# demo_robot\n"})
    git(clone, env, "tag", "-a", "v0.0.1", "-m", f"release {PLANTED}")
    tag = git(clone, env, "rev-parse", "v0.0.1")
    done = gate(clone, env, f"refs/tags/v0.0.1 {tag} refs/tags/v0.0.1 {ZEROS}\n", words=words)
    assert done.returncode == 1
    assert f"{tag[:12]} tag message: private:synthetic test entries 1" in done.stdout
    assert sha[:12] not in done.stdout


def test_deletions_and_empty_input_send_nothing(clone: Path, env: dict[str, str], words: Path) -> None:
    commit(clone, env, {"README.md": f"{PLANTED}\n"})
    done = gate(clone, env, f"(delete) {ZEROS} refs/heads/old {'1' * 40}\n", words=words)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "0 hit(s) in 0 commit(s)" in last_line(done)
    assert gate(clone, env, "", words=words).returncode == 0


@pytest.mark.parametrize("problem", ["missing list", "list inside the clone", "no identity", "bad stdin"])
def test_the_gate_fails_closed(clone: Path, env: dict[str, str], words: Path, tmp_path: Path, problem: str) -> None:
    sha = commit(clone, env, {"README.md": "# demo_robot\n"})
    kwargs: dict = {"words": words}
    stdin = line(sha)
    if problem == "missing list":
        kwargs["words"] = tmp_path / "no-such-list.txt"
    elif problem == "list inside the clone":
        inside = clone / "words.txt"
        inside.write_text(PLANTED + "\n", encoding="utf-8")
        kwargs["words"] = inside
    elif problem == "no identity":
        kwargs["identity"] = None
    else:
        stdin = "not a ref line\n"
    done = gate(clone, env, stdin, **kwargs)
    assert done.returncode == 2, done.stdout + done.stderr
    assert done.stderr.startswith("gate cannot run:")
    assert_quiet(done)


def install(clone: Path, env: dict[str, str], words: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(INSTALLER), "--repo", str(clone), "--words", str(words),
                           "--identity", IDENTITY, "--no-gitleaks", *extra], env=env, capture_output=True, text=True,
                          timeout=60)


def test_installer_writes_the_hook_with_the_absolute_interpreter(clone: Path, env: dict[str, str],
                                                                  words: Path) -> None:
    done = install(clone, env, words)
    assert done.returncode == 0, done.stderr
    hook = clone / ".git" / "hooks" / "pre-push"
    text = hook.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert Path(os.path.abspath(sys.executable)).as_posix() in text
    assert GATE.resolve().as_posix() in text
    assert "py -3" in text and "python3" in text and "interpreter missing" in text
    assert "\r" not in text
    assert install(clone, env, words).returncode == 0  # its own hook is replaced in place


def test_installer_keeps_a_foreign_hook_unless_forced(clone: Path, env: dict[str, str], words: Path) -> None:
    hooks = clone / ".git" / "hooks"
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "pre-push").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    refused = install(clone, env, words)
    assert refused.returncode == 2 and "--force" in refused.stderr
    assert (hooks / "pre-push").read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n"
    forced = install(clone, env, words, "--force")
    assert forced.returncode == 0, forced.stderr
    assert (hooks / "pre-push.replaced-by-gate").read_text(encoding="utf-8") == "#!/bin/sh\nexit 0\n"


@pytest.mark.parametrize("problem", ["missing list", "no identity"])
def test_installer_refuses_what_the_gate_could_not_run(clone: Path, env: dict[str, str], tmp_path: Path,
                                                       problem: str) -> None:
    args = [sys.executable, str(INSTALLER), "--repo", str(clone)]
    if problem == "missing list":
        args += ["--words", str(tmp_path / "no-such-list.txt"), "--identity", IDENTITY]
    else:
        listed = tmp_path / "list.txt"
        listed.write_text("# empty\n", encoding="utf-8")
        args += ["--words", str(listed)]
    done = subprocess.run(args, env=env, capture_output=True, text=True, timeout=60)
    assert done.returncode == 2, done.stdout + done.stderr
    assert not (clone / ".git" / "hooks" / "pre-push").exists()


@pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX sh on PATH to run the hook script")
def test_installed_hook_runs_the_gate(clone: Path, env: dict[str, str], words: Path) -> None:
    clean = commit(clone, env, {"README.md": "# demo_robot\n"})
    dirty = commit(clone, env, {"docs/notes.md": f"{PLANTED}\n"})
    assert install(clone, env, words).returncode == 0
    hook = (clone / ".git" / "hooks" / "pre-push").as_posix()

    def run(stdin: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["sh", hook, "origin", "unused-url"], cwd=clone, env=env, input=stdin,
                              capture_output=True, text=True, timeout=120)

    allowed = run(line(clean))
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    refused = run(line(dirty, clean))
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert_quiet(refused)
