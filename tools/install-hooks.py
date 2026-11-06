"""Install the pre-push privacy gate (tools/pre-push-gate) as this clone's git `pre-push` hook.

usage:
  python tools/install-hooks.py --words <private list> --identity "Name <e-mail>" [--identity ...]
                                [--trailer-email <e-mail>] [--no-gitleaks] [--repo <clone>] [--force] [--dry-run]

It writes `pre-push` into the folder `git rev-parse --git-path hooks` names (`.git/hooks`, or `core.hooksPath`):
a short POSIX shell script that runs the gate with the absolute path of the interpreter running this installer,
then `py -3`, then `python3`. When none of them starts, the hook prints `gate cannot run: interpreter missing`
and exits 1, so the push is refused. The hook holds absolute paths and your identity; that is fine, because hooks
live inside the git folder and are never committed. Run the installer again after moving the clone, the
interpreter or the private list; it replaces its own hook and refuses to overwrite anyone else's without --force
(the old hook is then kept beside it as `pre-push.replaced-by-gate`).

The private list defaults to `[privacy] wordlist` of the instance config or `<TOOL>_PRIVACY__WORDLIST`; it is
read once now, so a hook is never installed with a list the gate cannot read. At least one --identity is
required: the name and e-mail your public commits carry (for example your GitHub noreply address).

Exit 0: written, or printed with --dry-run. 2: refused, with the reason on stderr.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GATE = REPO / "tools" / "pre-push-gate"
PACKAGES = ("easel", "kiln")
MARKER = "# managed-by: tools/install-hooks.py, the pre-push privacy gate"
KEPT_NAME = "pre-push.replaced-by-gate"
IDENTITY = re.compile(r"^\s*[^<>]*?\S[^<>]*?\s*<[^<>\s]+>\s*$")
FALLBACKS = ("py -3", "python3")


class Refused(Exception):
    """The installer refuses; main prints `install-hooks: <reason>` and exits 2."""


def scanner():
    """This checkout's scanner module (for reading the private list once), else None."""
    src = REPO / "src"
    own = [d.name for d in sorted(src.iterdir()) if (d / "__init__.py").is_file()] if src.is_dir() else []
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    for name in [*own, *(p for p in PACKAGES if p not in own)]:
        try:
            return importlib.import_module(f"{name}._common.privacy"), name
        except ModuleNotFoundError:
            continue
    return None, ""


def git_text(top: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=top, capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        raise Refused(f"git could not run: {exc.__class__.__name__}") from exc
    if done.returncode != 0:
        raise Refused(f"`git {args[0]}` failed; is {top.name or top} a git clone?")
    return done.stdout.decode("utf-8", "replace").strip()


def resolve_words(given: Path | None, privacy, pkg: str) -> Path:
    if given is not None:
        return Path(given).expanduser().resolve()
    configured = getattr(privacy, "configured_wordlist", None)
    if callable(configured):
        path, why = configured(pkg)
        if path is not None:
            return Path(path).expanduser().resolve()
        raise Refused(f"no private word list: {why}; pass --words <file>")
    raise Refused("no private word list; pass --words <file>")


def check_words(words: Path, privacy, top: Path) -> None:
    """Fail now rather than at the first push: the list must exist, lie outside the clone and parse."""
    if not words.is_file():
        raise Refused("the private word list is missing or not a file; fix the path given to --words")
    if words.is_relative_to(top.resolve()):
        raise Refused("the private word list lies inside the repository; keep it outside every clone")
    if privacy is not None:
        try:
            privacy.load_private_words(words)
        except privacy.PrivacyError as exc:
            raise Refused(str(exc)) from exc


def posix(path: Path) -> str:
    return path.resolve().as_posix()


def hook_text(python: str, gate: str, options: list[str]) -> str:
    """The hook script. Every value is quoted for sh; the probes read nothing from git's stdin."""
    q = shlex.quote
    opts = " ".join(q(o) for o in options)
    lines = [
        "#!/bin/sh",
        MARKER,
        f"# Written {date.today().isoformat()}. Local to this clone and never committed; to change it, run",
        "# python tools/install-hooks.py --help",
        f"gate={q(gate)}",
        'if [ ! -f "$gate" ]; then',
        '  echo "gate cannot run: the gate script is missing; run tools/install-hooks.py again" >&2',
        "  exit 1",
        "fi",
        f"if {q(python)} -c 'import sys' </dev/null >/dev/null 2>&1; then",
        f'  exec {q(python)} "$gate" {opts} -- "$@"',
        "fi",
    ]
    for fallback in FALLBACKS:
        lines += [f"if {fallback} -c 'import sys' </dev/null >/dev/null 2>&1; then",
                  f'  exec {fallback} "$gate" {opts} -- "$@"',
                  "fi"]
    tried = ", ".join(["the installer's interpreter", *FALLBACKS])
    lines += [f'echo "gate cannot run: interpreter missing; tried {tried}. Run tools/install-hooks.py again" >&2',
              "exit 1", ""]
    return "\n".join(lines)


def install(args: argparse.Namespace) -> int:
    top = Path(git_text(args.repo or REPO, "rev-parse", "--show-toplevel"))
    hooks = (top / git_text(top, "rev-parse", "--git-path", "hooks")).resolve()
    privacy, pkg = scanner()
    if privacy is None:
        raise Refused("the scanner module is missing from src/; install the checkout or run tools/sync_common.py")
    if not GATE.is_file():
        raise Refused("tools/pre-push-gate is missing from this checkout")
    words = resolve_words(args.words, privacy, pkg)
    check_words(words, privacy, top)
    if not args.identity:
        raise Refused("pass at least one --identity \"Name <e-mail>\": the identity your pushed commits carry")
    for value in args.identity:
        if not IDENTITY.match(value):
            raise Refused("--identity needs the form \"Name <e-mail>\"")
    try:
        import yaml  # noqa: F401  the scanner reads its pattern files with it
    except ImportError as exc:
        raise Refused("this interpreter has no pyyaml; run the installer with the clone's environment") from exc
    options = ["--words", posix(words)]
    for value in args.identity:
        options += ["--identity", value.strip()]
    for email in args.trailer_email:
        options += ["--trailer-email", email.strip()]
    if args.no_gitleaks:
        options.append("--no-gitleaks")
    # the interpreter path as it is called, not resolved: a venv's python is often a link to the base one
    text = hook_text(Path(os.path.abspath(sys.executable)).as_posix(), posix(GATE), options)
    if args.dry_run:
        sys.stdout.write(text)
        return 0
    target = hooks / "pre-push"
    if target.exists():
        current = target.read_text(encoding="utf-8", errors="replace")
        if MARKER not in current:
            if not args.force:
                raise Refused(f"{target.name} exists and was not written by this installer; read it, then rerun "
                              f"with --force to keep it as {KEPT_NAME} and install the gate")
            kept = hooks / KEPT_NAME
            if kept.exists():
                raise Refused(f"{KEPT_NAME} already exists; move it away first, nothing was changed")
            os.replace(target, kept)
    hooks.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"pre-push.tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.chmod(tmp, 0o755)
    os.replace(tmp, target)
    gitleaks = "off" if args.no_gitleaks else "when on PATH"
    print(f"installed the pre-push gate in {top.name}: private list read, {len(args.identity)} identity value(s), "
          f"gitleaks {gitleaks}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="install-hooks.py", description=__doc__.split("\n\n")[0])
    parser.add_argument("--words", type=Path, metavar="FILE",
                        help="the private word list, outside every repository; default: the configured one")
    parser.add_argument("--identity", action="append", default=[], metavar="'NAME <E-MAIL>'",
                        help="an identity allowed as author and committer of pushed commits; repeat for several")
    parser.add_argument("--trailer-email", action="append", default=[], metavar="E-MAIL",
                        help="an e-mail address excused in commit messages only, for example a co-author trailer")
    parser.add_argument("--no-gitleaks", action="store_true", help="tell the gate to skip gitleaks; it warns each time")
    parser.add_argument("--repo", type=Path, metavar="DIR", help="the clone to install into; default: this checkout")
    parser.add_argument("--force", action="store_true",
                        help="replace a pre-push hook this installer did not write, keeping it beside the new one")
    parser.add_argument("--dry-run", action="store_true", help="print the hook instead of writing it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return install(args)
    except Refused as exc:
        print(f"install-hooks: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
