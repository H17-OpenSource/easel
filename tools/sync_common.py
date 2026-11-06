"""Copy easel's `_common` into kiln byte-identically, and check that both copies and the family files agree.

usage, from the easel checkout (the source of truth, ADR-0001):
  python tools/sync_common.py --to ../kiln [--only store,config]   copy, then write MANIFEST.sha256 in both
  python tools/sync_common.py --check [--sibling ../kiln]          verify: prints `identical` or the differences
  python tools/sync_common.py --write-manifest                     rewrite this repo's MANIFEST from its files
In the kiln checkout the same script (the same bytes) runs `--check` on kiln's own copy, and against easel when
easel sits beside it; `--to` and `--write-manifest` refuse there, because kiln's `_common` is never edited.

`--to` copies every file of `_common` (or, with `--only`, the named modules plus `__init__.py`), removes files
the target has and the source does not (full sync only), then writes the same MANIFEST.sha256, listing the
target's resulting files, into both repos.

`--check` compares:
1. this repo's `_common/MANIFEST.sha256` with its `_common` files (changed, missing or unlisted);
2. when the sibling repo exists (default `../<other tool>`): its MANIFEST and its `_common` files with this repo's;
3. the shared keys of `families/*.yaml` in both repos: every top-level key except `schema`, `inference` and
   `training`.
Hashes ignore CRLF versus LF, so a checkout with Windows line endings matches a manifest written elsewhere.

exit codes: 0 done or identical; 1 differences; 2 usage error or refusal.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_OF_TRUTH = "easel"
OTHER = {"easel": "kiln", "kiln": "easel"}
MANIFEST = "MANIFEST.sha256"
NOT_SHARED = {"schema", "inference", "training"}
_MISSING = object()


def package_of(repo: Path) -> str | None:
    """The one package under `<repo>/src/` that has a cli.py (easel or kiln), or None."""
    src = repo / "src"
    if not src.is_dir():
        return None
    found = [d.name for d in sorted(src.iterdir()) if (d / "__init__.py").is_file() and (d / "cli.py").is_file()]
    return found[0] if len(found) == 1 else None


def shown(path: Path) -> str:
    """A path as printed: relative to this repo, so no output line carries an absolute path."""
    try:
        return Path(os.path.relpath(path, REPO)).as_posix()
    except ValueError:  # another drive on Windows
        return path.name


def digest(path: Path) -> str:
    """sha256 of the file with CRLF read as LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def common_files(common: Path) -> dict[str, Path]:
    """Every shared file under `_common` by its posix path; caches, temp files, locks and the manifest left out."""
    out: dict[str, Path] = {}
    if not common.is_dir():
        return out
    for path in sorted(common.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(common).as_posix()
        if rel == MANIFEST or path.suffix in (".pyc", ".pyo") or ".tmp-" in path.name or path.name == ".lock":
            continue
        out[rel] = path
    return out


def manifest_text(files: dict[str, Path]) -> str:
    """`<sha256>  <path>` per file, sorted by path (the sha256sum format)."""
    return "".join(f"{digest(files[rel])}  {rel}\n" for rel in sorted(files))


def read_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    """(path -> sha256, problem lines) from a MANIFEST.sha256 file."""
    entries: dict[str, str] = {}
    problems: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            problems.append(f"{shown(path)}:{number}: not a `<sha256>  <path>` line")
            continue
        entries[parts[1].strip()] = parts[0].lower()
    return entries, problems


def write_file(path: Path, text: str) -> None:
    """Write through a temp file and a rename, so a reader never sees half a manifest."""
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)


# --- --to ------------------------------------------------------------------------------------------------------

def sync(pkg: str, to: Path, only: str | None) -> int:
    if pkg != SOURCE_OF_TRUTH:
        print(f"refused: {pkg}'s _common is a copy; run the sync from the {SOURCE_OF_TRUTH} checkout")
        return 2
    target_pkg = package_of(to)
    if target_pkg is None or target_pkg == pkg:
        print(f"refused: {shown(to)} is not a checkout of another tool (no src/<package>/cli.py)")
        return 2
    source = REPO / "src" / pkg / "_common"
    target = to / "src" / target_pkg / "_common"
    files = common_files(source)
    if not files:
        print(f"refused: {shown(source)} has no files")
        return 2
    if only:
        selected: dict[str, Path] = {}
        for name in (n.strip() for n in only.split(",")):
            if not name:
                continue
            rel = name if "." in name.rsplit("/", 1)[-1] else f"{name}.py"
            if rel not in files:
                print(f"refused: {rel} is not in {shown(source)}")
                return 2
            selected[rel] = files[rel]
        if "__init__.py" in files:
            selected.setdefault("__init__.py", files["__init__.py"])
    else:
        selected = files
    for rel in sorted(selected):
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file() and dest.read_bytes() == selected[rel].read_bytes():
            print(f"unchanged {rel}")
            continue
        shutil.copyfile(selected[rel], dest)
        print(f"copied {rel}")
    if not only:
        for rel in sorted(set(common_files(target)) - set(files)):
            (target / rel).unlink()
            print(f"removed {rel}: not in {pkg}")
    text = manifest_text(common_files(target))
    write_file(target / MANIFEST, text)
    write_file(source / MANIFEST, text)
    print(f"wrote {MANIFEST} in {pkg} and {target_pkg}: {text.count(chr(10))} files")
    return 0


# --- --check ---------------------------------------------------------------------------------------------------

def manifest_diffs(pkg: str, common: Path) -> list[str]:
    path = common / MANIFEST
    if not path.is_file():
        return [f"{pkg}: _common/{MANIFEST} is missing; run `python tools/sync_common.py --to ../kiln` from easel"]
    listed, diffs = read_manifest(path)
    present = common_files(common)
    for rel in sorted(set(listed) | set(present)):
        if rel not in present:
            diffs.append(f"{pkg}: _common/{rel} is listed in {MANIFEST} but missing")
        elif rel not in listed:
            diffs.append(f"{pkg}: _common/{rel} is not in {MANIFEST}")
        elif digest(present[rel]) != listed[rel]:
            diffs.append(f"{pkg}: _common/{rel} differs from {MANIFEST}")
    return diffs


def copy_diffs(pkg: str, mine: Path, other: str, theirs: Path) -> list[str]:
    a, b = common_files(mine), common_files(theirs)
    diffs = []
    for rel in sorted(set(a) | set(b)):
        if rel not in b:
            diffs.append(f"_common/{rel} is only in {pkg}")
        elif rel not in a:
            diffs.append(f"_common/{rel} is only in {other}")
        elif digest(a[rel]) != digest(b[rel]):
            diffs.append(f"_common/{rel} differs between {pkg} and {other}")
    ma, mb = mine / MANIFEST, theirs / MANIFEST
    if ma.is_file() and mb.is_file() and digest(ma) != digest(mb):
        diffs.append(f"_common/{MANIFEST} differs between {pkg} and {other}")
    return diffs


def shared_keys(data: object) -> dict:
    return {k: v for k, v in data.items() if k not in NOT_SHARED} if isinstance(data, dict) else {}


def family_diffs(pkg: str, mine: Path, other: str, theirs: Path) -> list[str]:
    try:
        import yaml
    except ImportError:
        return ["families: pyyaml is missing, so the family files were not compared"]
    a = {p.stem: p for p in sorted(mine.glob("*.yaml"))}
    b = {p.stem: p for p in sorted(theirs.glob("*.yaml"))}
    diffs = []
    for fid in sorted(set(a) | set(b)):
        if fid not in b:
            diffs.append(f"families/{fid}.yaml is only in {pkg}")
            continue
        if fid not in a:
            diffs.append(f"families/{fid}.yaml is only in {other}")
            continue
        loaded = []
        for name, path in ((pkg, a[fid]), (other, b[fid])):
            try:
                loaded.append(shared_keys(yaml.safe_load(path.read_text(encoding="utf-8"))))
            except (OSError, yaml.YAMLError):
                diffs.append(f"families/{fid}.yaml is not readable YAML in {name}")
        if len(loaded) == 2 and loaded[0] != loaded[1]:
            keys = sorted(k for k in set(loaded[0]) | set(loaded[1])
                          if loaded[0].get(k, _MISSING) != loaded[1].get(k, _MISSING))
            diffs.append(f"families/{fid}.yaml: shared keys differ: {', '.join(keys)}")
    return diffs


def check(pkg: str, sibling: Path | None) -> int:
    common = REPO / "src" / pkg / "_common"
    diffs = manifest_diffs(pkg, common)
    other = OTHER.get(pkg, "")
    sib = sibling if sibling is not None else REPO.parent / other
    sib_pkg = package_of(sib) if sib.is_dir() else None
    if sib_pkg is None:
        print(f"note: no sibling checkout at {shown(sib)}; compared this repo's {MANIFEST} only")
    else:
        diffs += copy_diffs(pkg, common, sib_pkg, sib / "src" / sib_pkg / "_common")
        diffs += family_diffs(pkg, REPO / "families", sib_pkg, sib / "families")
    if diffs:
        for line in diffs:
            print(f"differ: {line}")
        return 1
    print("identical")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--to", type=Path, help="copy this repo's _common into the checkout at this path")
    mode.add_argument("--check", action="store_true", help="verify the manifest, the sibling copy and the families")
    mode.add_argument("--write-manifest", action="store_true", help="rewrite this repo's manifest from its files")
    parser.add_argument("--only", help="with --to: comma-separated module names, for example store,config")
    parser.add_argument("--sibling", type=Path, help="with --check: the other checkout; default ../<other tool>")
    args = parser.parse_args(argv)
    pkg = package_of(REPO)
    if pkg is None:
        print(f"refused: {shown(REPO)} has no src/<package>/cli.py")
        return 2
    if args.only and args.to is None:
        parser.error("--only needs --to")
    if args.sibling and not args.check:
        parser.error("--sibling needs --check")
    if args.to is not None:
        to = args.to if args.to.is_absolute() else (Path.cwd() / args.to)
        return sync(pkg, to.resolve(), args.only)
    if args.write_manifest:
        if pkg != SOURCE_OF_TRUTH:
            print(f"refused: {pkg}'s manifest comes from the sync; run the sync from the {SOURCE_OF_TRUTH} checkout")
            return 2
        common = REPO / "src" / pkg / "_common"
        text = manifest_text(common_files(common))
        write_file(common / MANIFEST, text)
        print(f"wrote {MANIFEST} in {pkg}: {text.count(chr(10))} files")
        return 0
    sibling = None
    if args.sibling is not None:
        sibling = (args.sibling if args.sibling.is_absolute() else Path.cwd() / args.sibling).resolve()
    return check(pkg, sibling)


if __name__ == "__main__":
    sys.exit(main())
