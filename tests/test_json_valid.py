"""Every `.json` file the repository would commit parses with `json.load` (CI runs the same check).

The file list is git's view (tracked plus untracked-but-not-ignored), so generated folders such as
tests/fixtures/_gen/ are left out; without git the tree is walked with the same folders skipped.
"""

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", "_gen", "build", "dist", "node_modules"}


def repo_files(root: Path, suffix: str) -> list[Path]:
    """The files under `root` with `suffix` that git would commit (or a filtered walk without git)."""
    try:
        done = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                              capture_output=True, timeout=60, check=True)
        names = [n for n in done.stdout.decode("utf-8", "replace").split("\0") if n.endswith(suffix)]
        return sorted(root / n for n in names if (root / n).is_file())
    except (OSError, subprocess.SubprocessError):
        return sorted(p for p in root.rglob(f"*{suffix}") if p.is_file()
                      and not set(p.relative_to(root).parts) & SKIP
                      and not any(part.startswith(".venv") or part.endswith(".egg-info") for part in p.parts))


def invalid_json(paths: list[Path], root: Path) -> list[str]:
    """`<path>: <error>` for every file that json.load refuses."""
    bad = []
    for path in paths:
        try:
            with path.open(encoding="utf-8") as handle:
                json.load(handle)
        except (ValueError, UnicodeDecodeError) as exc:
            bad.append(f"{path.relative_to(root).as_posix()}: {exc}")
    return bad


def test_the_repository_has_json_files():
    assert repo_files(REPO, ".json"), "expected at least the schemas"


def test_every_json_file_parses():
    assert invalid_json(repo_files(REPO, ".json"), REPO) == []


@pytest.mark.parametrize("text, ok", [('{"a": 1}', True), ('{"a": 1,}', False), ("", False), ("[1, 2", False)])
def test_the_check_refuses_broken_json(tmp_path, text, ok):
    path = tmp_path / "x.json"
    path.write_text(text, encoding="utf-8")
    assert (invalid_json([path], tmp_path) == []) is ok
