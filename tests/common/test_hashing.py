"""_common.hashing, and its consumer tools/sync_common.py (MANIFEST.sha256 and the family shared-key check)."""

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from easel._common import hashing

REPO = Path(__file__).resolve().parents[2]
SYNC = REPO / "tools" / "sync_common.py"
ABC = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # sha256("abc"), the FIPS test vector


def test_known_vector_and_text():
    assert hashing.sha256_bytes(b"abc") == ABC
    assert hashing.sha256_text("abc") == ABC


def test_file_hash_in_small_chunks_matches_bytes(tmp_path):
    data = bytes(range(256)) * 41
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert hashing.sha256_file(path, chunk=7) == hashing.sha256_bytes(data) == hashing.sha256_file(path)


def test_canonical_json_ignores_key_order_and_keeps_text():
    a = {"b": 1, "a": {"y": [1, 2], "x": "demo_robot é"}}
    b = {"a": {"x": "demo_robot é", "y": [1, 2]}, "b": 1}
    assert hashing.canonical_json(a) == hashing.canonical_json(b)
    assert hashing.canonical_json(a).decode("utf-8") == '{"a":{"x":"demo_robot é","y":[1,2]},"b":1}'
    assert hashing.sha256_json(a) == hashing.sha256_json(b)


def test_short_verify_and_shape(tmp_path):
    path = tmp_path / "abc.txt"
    path.write_bytes(b"abc")
    assert hashing.short(ABC.upper(), 16) == ABC[:16]
    assert hashing.verify_file(path, ABC) and hashing.verify_file(path, ABC[:12].upper())
    assert not hashing.verify_file(path, "0" * 64)
    with pytest.raises(ValueError):
        hashing.verify_file(path, ABC[:8])
    assert hashing.is_sha256(ABC) and not hashing.is_sha256(ABC[:-1]) and not hashing.is_sha256("g" * 64)


# --- tools/sync_common.py on a synthetic pair of checkouts -----------------------------------------------------


def _sync_names() -> tuple[str, str]:
    """(source-of-truth tool, its copy) as the script itself names them, so this file needs no tool literals."""
    spec = importlib.util.spec_from_file_location("sync_common_under_test", SYNC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOURCE_OF_TRUTH, module.OTHER[module.SOURCE_OF_TRUTH]


SOURCE, COPY = _sync_names()

FAMILY = """schema: {tool}/family/1
id: demo
name: Demo family
licence_summary: {{status: unverified}}
host_ram_gb: {{train: 5.5, eval: 8.5}}
{block}: {{steps: {steps}}}
"""


def family(tool: str, steps: int = 30) -> bytes:
    return FAMILY.format(tool=tool, block="inference" if tool == SOURCE else "training", steps=steps).encode("utf-8")


def common_of(repo: Path) -> Path:
    return repo / "src" / repo.name / "_common"


def make_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two minimal checkouts side by side: the source (with a _common) and the copy (without), one family each."""
    for tool in (SOURCE, COPY):
        repo = tmp_path / tool
        pkg = repo / "src" / tool
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_bytes(b"")
        (pkg / "cli.py").write_bytes(b"")
        (repo / "tools").mkdir()
        shutil.copyfile(SYNC, repo / "tools" / "sync_common.py")
        (repo / "families").mkdir()
        (repo / "families" / "demo.yaml").write_bytes(family(tool))
    common = common_of(tmp_path / SOURCE)
    common.mkdir()
    # bytes, not write_text: text mode on Windows would already write CRLF
    (common / "__init__.py").write_bytes(b'"""shared"""\n')
    (common / "store.py").write_bytes(b"X = 1\n")
    (common / "ids.py").write_bytes(b"Y = 2\n")
    return tmp_path / SOURCE, tmp_path / COPY


def sync(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(repo / "tools" / "sync_common.py"), *args], cwd=repo,
                          capture_output=True, text=True, timeout=60)


def test_sync_then_check_is_identical_in_both(tmp_path):
    src, dst = make_pair(tmp_path)
    before = sync(src, "--check")
    assert before.returncode == 1 and "MANIFEST.sha256 is missing" in before.stdout
    done = sync(src, "--to", f"../{COPY}")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (common_of(dst) / "store.py").read_bytes() == (common_of(src) / "store.py").read_bytes()
    assert (common_of(dst) / "MANIFEST.sha256").read_bytes() == (common_of(src) / "MANIFEST.sha256").read_bytes()
    for repo in (src, dst):
        checked = sync(repo, "--check")
        assert (checked.returncode, checked.stdout.strip()) == (0, "identical"), checked.stdout
    assert "removed" not in done.stdout


def test_check_names_each_kind_of_difference(tmp_path):
    src, dst = make_pair(tmp_path)
    assert sync(src, "--to", f"../{COPY}").returncode == 0
    (common_of(dst) / "store.py").write_bytes(b"X = 2\n")  # someone edited the copy by hand
    (dst / "families" / "demo.yaml").write_bytes(family(COPY).replace(b"8.5", b"9.0"))
    out = sync(dst, "--check").stdout
    assert f"differ: {COPY}: _common/store.py differs from MANIFEST.sha256" in out
    assert f"differ: _common/store.py differs between {COPY} and {SOURCE}" in out
    assert "differ: families/demo.yaml: shared keys differ: host_ram_gb" in out
    assert sync(src, "--check").returncode == 1


def test_inference_and_training_blocks_are_not_shared(tmp_path):
    src, dst = make_pair(tmp_path)
    (dst / "families" / "demo.yaml").write_bytes(family(COPY, steps=2000))
    assert sync(src, "--to", f"../{COPY}").returncode == 0
    assert sync(src, "--check").stdout.strip() == "identical"


def test_crlf_checkout_still_matches(tmp_path):
    src, dst = make_pair(tmp_path)
    assert sync(src, "--to", f"../{COPY}").returncode == 0
    target = common_of(dst) / "ids.py"
    target.write_bytes(target.read_bytes().replace(b"\n", b"\r\n"))
    assert sync(dst, "--check").stdout.strip() == "identical"


def test_only_partial_sync_and_refusals(tmp_path):
    src, dst = make_pair(tmp_path)
    partial = sync(src, "--to", f"../{COPY}", "--only", "store")
    assert partial.returncode == 0, partial.stdout
    assert sorted(p.name for p in common_of(dst).iterdir()) == ["MANIFEST.sha256", "__init__.py", "store.py"]
    assert f"differ: {SOURCE}: _common/ids.py is not in MANIFEST.sha256" in sync(src, "--check").stdout
    assert sync(dst, "--check").returncode == 1  # ids.py is only in the source
    assert sync(src, "--to", f"../{COPY}", "--only", "nosuch").returncode == 2
    assert sync(dst, "--to", f"../{SOURCE}").returncode == 2  # the copy never syncs back
    assert sync(dst, "--write-manifest").returncode == 2


def test_full_sync_removes_files_the_source_dropped(tmp_path):
    src, dst = make_pair(tmp_path)
    assert sync(src, "--to", f"../{COPY}").returncode == 0
    (common_of(src) / "ids.py").unlink()
    again = sync(src, "--to", f"../{COPY}")
    assert "removed ids.py" in again.stdout
    assert not (common_of(dst) / "ids.py").exists()
    assert sync(dst, "--check").stdout.strip() == "identical"


def test_check_without_a_sibling_checks_the_manifest_only(tmp_path):
    src, dst = make_pair(tmp_path)
    assert sync(src, "--write-manifest").returncode == 0
    shutil.rmtree(dst)
    checked = sync(src, "--check")
    assert checked.returncode == 0
    assert checked.stdout.splitlines() == [
        f"note: no sibling checkout at ../{COPY}; compared this repo's MANIFEST.sha256 only", "identical"]
