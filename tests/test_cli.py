"""The CLI skeleton: --help and --version work, and every verb in COMMANDS is either built or says it is planned."""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from easel import cli

PKG = "easel"
SRC = Path(__file__).resolve().parents[1] / "src"


def run(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    """Call cli.main in-process; argparse's SystemExit (from --help, --version, usage errors) becomes a code."""
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    """`python -m easel ...` from the source tree, the way an uninstalled checkout runs it."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run([sys.executable, "-m", PKG, *args], capture_output=True, text=True, env=env, timeout=60)


def test_help_lists_every_verb(capsys):
    code, out = run(["--help"], capsys)
    assert code == 0
    assert f"usage: {PKG}" in out
    for cmd in cli.COMMANDS:
        assert cmd.verb in out


def test_version(capsys):
    code, out = run(["--version"], capsys)
    assert code == 0
    assert out.strip() == f"{PKG} {cli.version()}"


def test_python_m_help_and_version():
    helped = run_module("--help")
    assert helped.returncode == 0, helped.stderr
    assert f"usage: {PKG}" in helped.stdout
    versioned = run_module("--version")
    assert versioned.returncode == 0, versioned.stderr
    assert versioned.stdout.startswith(f"{PKG} ")


def test_no_verb_prints_help_and_exits_2(capsys):
    code, out = run([], capsys)
    assert code == 2
    assert f"usage: {PKG}" in out


def test_table_is_well_formed():
    verbs = [cmd.verb for cmd in cli.COMMANDS]
    assert len(verbs) == len(set(verbs)), "a verb appears twice in COMMANDS"
    for cmd in cli.COMMANDS:
        assert cmd.module.startswith(f"{PKG}."), cmd
        assert cmd.help and "\n" not in cmd.help, cmd
        assert cmd.planned and " " not in cmd.planned, cmd


@pytest.mark.parametrize("cmd", cli.COMMANDS, ids=lambda cmd: cmd.verb)
def test_every_verb_answers_help_or_planned(cmd, capsys):
    code, out = run([cmd.verb, "--help"], capsys)
    if cli.module_exists(cmd.module):
        assert code == 0, out
        assert f"usage: {PKG} {cmd.verb}" in out
    else:
        assert code == 2, out
        assert out.strip() == f"planned: {cmd.planned} ({cmd.help})"


def test_help_before_a_planned_verb_is_top_level_help(capsys):
    planned = [cmd for cmd in cli.COMMANDS if not cli.module_exists(cmd.module)]
    if not planned:
        pytest.skip("every verb is built")
    code, out = run(["--help", planned[0].verb], capsys)
    assert code == 0
    assert f"usage: {PKG}" in out


def test_no_verb_module_fails_to_import():
    built = [cmd for cmd in cli.COMMANDS if cli.module_exists(cmd.module)]
    for cmd in built:
        module = importlib.import_module(cmd.module)
        assert callable(getattr(module, "register", None)), f"{cmd.module} must expose register(subparsers)"


def test_a_built_verb_dispatches(tmp_path, monkeypatch, capsys):
    """A throwaway verb module on sys.path proves the lazy import, the handler's exit code and the no-subverb path."""
    (tmp_path / "zz_demo_verb.py").write_text(
        "def register(subparsers):\n"
        "    p = subparsers.add_parser('zz-demo', help='demo verb')\n"
        "    inner = p.add_subparsers(dest='action')\n"
        "    hello = inner.add_parser('hello')\n"
        "    hello.set_defaults(func=lambda args: print('hello from', args.tool) or 3)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    monkeypatch.setattr(cli, "COMMANDS", (*cli.COMMANDS, cli.Command("zz-demo", "zz_demo_verb", "demo verb", "T0")))
    code, out = run(["zz-demo", "hello"], capsys)
    assert (code, out.strip()) == (3, f"hello from {PKG}")
    code, out = run(["zz-demo"], capsys)
    assert code == 2
    assert f"usage: {PKG} zz-demo" in out
