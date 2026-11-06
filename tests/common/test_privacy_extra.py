"""The optional extra-patterns file: the id shapes of a user's own records, kept outside every repository.

The shipped public-patterns.yaml holds generic shapes only. Each test here writes its own extra-patterns file into
tmp_path with made-up shapes (an upper-case prefix and digits, a dated image id, a prefixed hex marker), so no
real record's scheme appears in this file. The needles are built at run time from fragments, as in
test_privacy.py, and the scan of `tests/` stays clean.
"""

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = next(d.name for d in sorted((REPO / "src").iterdir()) if (d / "_common").is_dir())
privacy = importlib.import_module(f"{PKG}._common.privacy")
cli = importlib.import_module(f"{PKG}.cli")
DATA = REPO / "data" / "privacy"
BS = chr(92)

# Made-up shapes in the format of the public file; each regex keeps a one-character class in the middle of its
# prefix (`Z[Q]`), the same trick the public file uses so a pattern never matches its own line.
EXTRA = "\n".join([
    'schema: "privacy/extra-patterns/1"',
    "patterns:",
    "  - id: demo-old-run-id",
    "    what: a made-up old run id, two capitals, a dash and four digits",
    "    regex: '" + BS + "bZ[Q]-" + BS + "d{4}" + BS + "b'",
    "  - id: demo-image-id",
    "    what: a made-up dated image id",
    "    regex: '" + BS + "bP[X]I-" + BS + "d{8}-" + BS + "d{4}" + BS + "b'",
    "  - id: demo-artifact-marker",
    "    what: a made-up source prefix and sixteen hex digits",
    "    regex: '" + BS + "bq[z]:[0-9a-f]{16}" + BS + "b'",
    "  - id: demo-forbidden-name",
    "    what: a made-up file name that must never be committed",
    "    target: path",
    "    regex: '(?:^|/)secret-n[o]tes" + BS + ".md$'",
    "",
])

NEEDLES = {
    "demo-old-run-id": "see " + "ZQ" + "-" + "0" * 3 + "7" + " here",
    "demo-image-id": "PXI" + "-" + "2030" + "0101" + "-" + "0001",
    "demo-artifact-marker": "qz" + ":" + "0" * 16,
}


def put(folder: Path, name: str, text: str) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.fixture
def extra_file(tmp_path) -> Path:
    return put(tmp_path / "private", "extra-patterns.yaml", EXTRA)


@pytest.fixture(scope="module")
def public():
    return privacy.load_public_patterns(DATA / "public-patterns.yaml")


def ids(result) -> set[str]:
    return {hit.pattern for hit in result.hits}


def run_cli(argv: list[str], capsys) -> tuple[int, str]:
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    captured = capsys.readouterr()
    return code, captured.out + captured.err


@pytest.mark.parametrize("pattern_id", sorted(NEEDLES))
def test_each_extra_shape_is_found_and_the_public_list_alone_misses_it(pattern_id, tmp_path, public, extra_file):
    path = put(tmp_path / "proj", "notes.md", NEEDLES[pattern_id])
    extra = privacy.load_extra_patterns(extra_file, public)
    assert ids(privacy.scan([path], public)) == set()
    assert ids(privacy.scan([path], [*public, *extra])) == {pattern_id}


def test_an_extra_path_pattern_matches_names_only(tmp_path, public, extra_file):
    extra = privacy.load_extra_patterns(extra_file, public)
    named = put(tmp_path / "proj", "docs/secret-notes.md", "fine")
    mentioned = put(tmp_path / "proj", "docs/readme.md", "never commit secret-notes.md")
    assert ids(privacy.scan([named], [*public, *extra])) == {"demo-forbidden-name"}
    assert ids(privacy.scan([mentioned], [*public, *extra])) == set()


def test_the_shipped_public_list_is_generic(public):
    """The workspace-specific id schemes moved out of the shipped file; the generic real-year id stays."""
    shipped = {p.id for p in public}
    assert "real-year-id" in shipped
    assert not any(name.startswith("legacy-") for name in shipped) and "spec-id" not in shipped


def test_an_extra_id_that_repeats_a_public_one_is_refused(tmp_path, public):
    clash = put(tmp_path, "clash.yaml", "patterns:\n  - id: email\n    regex: 'zz[q]'\n")
    with pytest.raises(privacy.PrivacyError, match="repeated"):
        privacy.load_extra_patterns(clash, public)


def test_an_unreadable_or_empty_extra_file_fails_closed(tmp_path, public):
    with pytest.raises(privacy.PrivacyError):
        privacy.load_extra_patterns(tmp_path / "missing.yaml", public)
    with pytest.raises(privacy.PrivacyError, match="non-empty"):
        privacy.load_extra_patterns(put(tmp_path, "empty.yaml", "patterns: []\n"), public)


def test_run_scan_names_the_extra_patterns_in_its_summary(tmp_path, extra_file, capsys):
    path = put(tmp_path / "proj", "notes.md", NEEDLES["demo-old-run-id"])
    code = privacy.run_scan([path], patterns_file=DATA / "public-patterns.yaml", extra_file=extra_file)
    out = capsys.readouterr().out
    assert code == 1 and "demo-old-run-id 1" in out
    last = out.strip().splitlines()[-1]
    assert last.startswith("privacy scan, public patterns + extra patterns only: 1 hit(s)")
    assert "4 extra pattern(s)" in last


def test_cli_extra_patterns_flag_works_with_public(tmp_path, extra_file, capsys):
    path = put(tmp_path / "proj", "notes.md", NEEDLES["demo-image-id"])
    code, out = run_cli(["privacy", "scan", "--public", str(path)], capsys)
    assert code == 0
    code, out = run_cli(["privacy", "scan", "--public", "--extra-patterns", str(extra_file), str(path)], capsys)
    assert code == 1 and "demo-image-id 1" in out and NEEDLES["demo-image-id"] not in out


@pytest.fixture
def instance(tmp_path, monkeypatch, extra_file):
    """An instance whose TOML names a word list and the extra patterns through a ${VAR}; no repo schema."""
    home = tmp_path / "home"
    words = put(tmp_path / "private", "words.txt", "# people\nExample Person\n")
    put(home, f"{PKG}.toml", f'[privacy]\nwordlist = "{words.as_posix()}"\n'
                             'extra_patterns = "${DEMO_PRIVATE_DIR}/extra-patterns.yaml"\n')
    monkeypatch.setenv("DEMO_PRIVATE_DIR", extra_file.parent.as_posix())
    config = importlib.import_module(f"{PKG}._common.config")
    monkeypatch.setattr(config, "default_schema_path", lambda: tmp_path / "no-schema.json")
    for var in (f"{PKG.upper()}_PRIVACY__WORDLIST", f"{PKG.upper()}_PRIVACY__EXTRA_PATTERNS"):
        monkeypatch.delenv(var, raising=False)
    return home


def test_the_configured_extra_patterns_are_read_without_public(tmp_path, instance, capsys):
    path = put(tmp_path / "proj", "notes.md", NEEDLES["demo-artifact-marker"])
    code, out = run_cli(["privacy", "scan", "--workspace", str(instance), str(path)], capsys)
    assert code == 1 and "demo-artifact-marker 1" in out
    assert "public patterns + extra patterns + private list" in out
    code, out = run_cli(["privacy", "scan", "--public", "--workspace", str(instance), str(path)], capsys)
    assert code == 0  # --public reads no instance config: neither the list nor the extra patterns


def test_a_configured_extra_file_whose_variable_is_unset_fails_closed(tmp_path, instance, monkeypatch, capsys):
    monkeypatch.delenv("DEMO_PRIVATE_DIR")
    path = put(tmp_path / "proj", "notes.md", "fine")
    code, out = run_cli(["privacy", "scan", "--workspace", str(instance), str(path)], capsys)
    assert code == 2 and "extra_patterns" in out and "DEMO_PRIVATE_DIR" in out


def test_privacy_check_tool_takes_extra_patterns(tmp_path, extra_file):
    path = put(tmp_path / "proj", "notes.md", NEEDLES["demo-old-run-id"])
    tool = REPO / "tools" / "privacy_check.py"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    plain = subprocess.run([sys.executable, str(tool), "--public", str(path)], capture_output=True, text=True,
                           env=env, timeout=120)
    assert plain.returncode == 0, plain.stdout + plain.stderr
    done = subprocess.run([sys.executable, str(tool), "--public", "--extra-patterns", str(extra_file), str(path)],
                          capture_output=True, text=True, env=env, timeout=120)
    assert done.returncode == 1 and "demo-old-run-id 1" in done.stdout, done.stdout + done.stderr
