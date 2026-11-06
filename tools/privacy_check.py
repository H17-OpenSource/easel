"""The privacy scan for CI and hooks: this repository's public patterns and allowlist, plus an optional private list.

usage:
  python tools/privacy_check.py --public [PATH ...]              public patterns + allowlist (what CI runs)
  python tools/privacy_check.py --words <file> [PATH ...]        the same plus a private word list
  add --extra-patterns <file> for an extra-patterns file (the id shapes of your own records), kept outside
  every repository, and --explain to print file:line and the pattern id of every hit

PATH defaults to the repository root. It never reads an instance config: CI has none, and a hook must say which
private files it uses. It prints counts per pattern and file, never the matched text, and exits 0 clean, 1 on any
hit, 2 when it cannot run (fail closed). The scanner is `<pkg>._common.privacy` of this checkout; the file is the
same in both repositories.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "privacy"
PACKAGES = ("easel", "kiln")  # the vendored scanner is byte-identical in both packages


def scanner():
    """This checkout's scanner module, else the other package's identical copy; None when neither imports."""
    src = REPO / "src"
    own = [d.name for d in sorted(src.iterdir()) if (d / "__init__.py").is_file()] if src.is_dir() else []
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for name in [*own, *(p for p in PACKAGES if p not in own)]:
        try:
            module = importlib.import_module(f"{name}._common.privacy")
        except ModuleNotFoundError:
            continue
        if name not in own:
            print(f"note: using the scanner vendored in {name}; run tools/sync_common.py", file=sys.stderr)
        return module
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="privacy_check.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", type=Path, metavar="PATH", help="files or folders; default: the repository root")
    parser.add_argument("--public", action="store_true", help="public patterns and the allowlist")
    parser.add_argument("--words", type=Path, metavar="FILE", help="also a private word list kept outside the repository")
    parser.add_argument("--extra-patterns", dest="extra_patterns", type=Path, metavar="FILE",
                        help="also an extra-patterns file (public-patterns format) kept outside the repository")
    parser.add_argument("--explain", action="store_true", help="print file:line and the pattern id of every hit")
    args = parser.parse_args(argv)
    if not args.public and args.words is None:
        print("privacy_check: choose --public, --words <file>, or both", file=sys.stderr)
        return 2
    privacy = scanner()
    if privacy is None:
        print("privacy_check: the scanner module is missing from src/; install the checkout or run "
              "tools/sync_common.py", file=sys.stderr)
        return 2
    extra = {}
    if args.extra_patterns is not None:
        if not hasattr(privacy, "load_extra_patterns"):
            print("privacy_check: this checkout's scanner predates --extra-patterns; run tools/sync_common.py",
                  file=sys.stderr)
            return 2
        extra = {"extra_file": args.extra_patterns}
    allow = DATA / "allow.yaml"
    return privacy.run_scan(args.paths or [REPO], patterns_file=DATA / "public-patterns.yaml",
                            allow_file=allow if allow.is_file() else None, words_file=args.words,
                            explain=args.explain, **extra)


if __name__ == "__main__":
    sys.exit(main())
