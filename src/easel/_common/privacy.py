"""The privacy scanner: count private or machine-specific values in files before they leave the machine.

The lists (plan, privacy export, "The scanner"):
- the public patterns, `data/privacy/public-patterns.yaml`: generic structural shapes only (paths, e-mail
  addresses, real-year ids, tokens, placeholders, file types); shipped, and all that CI uses (`--public`);
- an optional extra-patterns file in the same format, kept outside every repository, for the id shapes of the
  user's own records (`[privacy] extra_patterns`, or `--extra-patterns <file>`); its ids must not repeat a
  public one;
- an optional private word list kept outside every repository: one literal per line, matched
  case-insensitively as a substring after NFKC; a `re:` line is a regex; `#` lines are headings; values below
  the `# --- harvested ---` marker were appended by `--harvest-sql`;
- the allowlist, `data/privacy/allow.yaml`: exact texts with a reason, never applied to paths under tests/.

How a file is read: text as UTF-8 or UTF-16 (BOM or NUL-interleaved); every line is also JSON-unescaped again
and again (which opens nested JSON strings at any depth; `\\n`, `\\t`, `\\r`, `\\b` and `\\f` become a space, so a
value after an escaped line break keeps its word border) and URL-decoded, and each form is checked as written
and NFKC-normalised and casefolded. A PNG is read through its tEXt/zTXt/iTXt chunks only, never the pixels; a
safetensors file through its JSON header only, never the tensors; other binary files are counted, not read. A
chunk or header that claims more bytes than the file holds counts as unscannable; nothing is allocated for it.
Inside a git work tree the scan covers the files git would commit (tracked, plus untracked files that are not
ignored); elsewhere it walks the folder and skips `.git`, `_gen` and tool caches. Each file's
repository-relative path is checked too.

What it prints: counts per pattern and file, never the matched text; `--explain` adds `file:line pattern` per
hit. Exit 0 clean, 1 any hit, 2 a usage error or a list it cannot read: it fails closed, never open.

The core takes explicit arguments (`load_public_patterns`, `load_private_words`, `load_allowlist`, `scan`,
`run_scan`, `harvest_sql`), so CI and a pre-push hook use it without any configuration. Only the CLI asks the
config for the defaults, and only without `--public`: `[privacy] wordlist` and `[privacy] extra_patterns`,
overridable as `<TOOL>_PRIVACY__WORDLIST` and `<TOOL>_PRIVACY__EXTRA_PATTERNS`.

Example: `<tool> privacy scan --public .` in a checkout prints `privacy scan, public patterns only: 0 hit(s) in
N file(s) ...` and exits 0, or one count line per file with hits and exits 1.
"""

from __future__ import annotations

import argparse
import codecs
import fnmatch
import json
import os
import re
import struct
import subprocess
import sys
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, TextIO
from urllib.parse import unquote_plus

EXIT_OK = 0
EXIT_HITS = 1
EXIT_USAGE = 2

HARVEST_MARKER = "# --- harvested ---"
PATTERNS_FILE = "public-patterns.yaml"
ALLOW_FILE = "allow.yaml"
UNSCANNABLE = "unscannable"
PATTERN_KEYS = ("id", "what", "regex", "target")
ALLOW_KEYS = ("text", "why", "paths", "patterns")

MAX_TEXT_BYTES = 64 * 2**20  # a text file larger than this is reported as unscannable, never skipped quietly
MAX_HEADER_BYTES = 100 * 2**20  # the largest safetensors header read
MAX_CHUNK_TEXT = 16 * 2**20  # the largest PNG text chunk, compressed or inflated
MAX_DECODE_ROUNDS = 10  # JSON-unescape and URL-decode rounds per line (nested strings, double encoding)
HEADING_WIDTH = 48
GATE_PREFIX = 6  # characters of each private literal in the prefix gate (see _prefix_gate)

SKIP_DIRS = frozenset({".git", "_gen", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".tox",
                       ".nox", "node_modules"})
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
PNG_TEXT_CHUNKS = (b"tEXt", b"zTXt", b"iTXt")

_JSON_ESCAPE = re.compile(r'\\(?:u([0-9a-fA-F]{4})|([\\/"])|([bfnrt]))')
_SURROGATE = re.compile("[\ud800-\udfff]")
# Invisible format characters (soft hyphen, zero-width and direction marks, BOM), removed before comparing.
_FORMAT_RANGES = ((0x00AD, 0x00AD), (0x061C, 0x061C), (0x180E, 0x180E), (0x200B, 0x200F), (0x202A, 0x202E),
                  (0x2060, 0x2064), (0x2066, 0x206F), (0xFEFF, 0xFEFF))
_FORMAT_CHARS = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _FORMAT_RANGES) + "]")
_SQL_COMMENT = re.compile(r"--[^\n]*|/\*.*?\*/", re.S)


class PrivacyError(Exception):
    """A list that cannot be read or a request that cannot run; the CLI prints it and exits 2."""


class _Unscannable(Exception):
    """A file that cannot be read completely; it counts as a hit so a scan never passes by skipping it."""


# --- normalisation ------------------------------------------------------------------------------------------------

def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def fold(text: str) -> str:
    """NFKC, casefold, NFKC again, without invisible format characters: the form every comparison uses."""
    return _FORMAT_CHARS.sub("", _nfkc(_nfkc(text).casefold()))


def _json_unescape(text: str) -> str:
    """One round of JSON string unescaping: `\\\\`, `\\/`, `\\"`, `\\uXXXX` (surrogate pairs joined), and the
    control escapes `\\b \\f \\n \\r \\t` as one space each, so the form stays one line and the word after an
    escaped line break keeps its border (a path or token right after a captured log's line break still hits)."""

    def one(match: re.Match[str]) -> str:
        if match.group(1):
            return chr(int(match.group(1), 16))
        return match.group(2) or " "

    out = _JSON_ESCAPE.sub(one, text)
    if _SURROGATE.search(out):
        out = out.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    return out


def decoded_forms(line: str) -> list[str]:
    """The line as written, then each new form from repeated JSON unescaping and URL decoding."""
    forms = [line]
    current = line
    for _ in range(MAX_DECODE_ROUNDS):
        nxt = _json_unescape(current) if "\\" in current else current
        if nxt == current and ("%" in current or "+" in current):
            nxt = unquote_plus(current)
        if nxt == current:
            break
        forms.append(nxt)
        current = nxt
    return forms


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))


def views(line: str) -> tuple[list[str], list[str]]:
    """(forms with their case kept, casefolded forms) of one line; every pattern runs over these."""
    if line.isascii() and not any(c in line for c in "\\%+"):
        return [line], [line.lower()]
    forms = decoded_forms(line)
    kept = _unique([*forms, *(_nfkc(f) for f in forms)])
    return kept, _unique(fold(f) for f in forms)


# --- the lists ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    """One public pattern or private entry. `id` is what the output prints; the regex text never is."""

    id: str
    regex: re.Pattern[str]
    what: str = ""
    target: str = "content"  # content: every line of a file; path: the file's repository-relative path only
    private: bool = False
    gate: str = ""  # private entries: a casefolded text every match contains (the literal, or a plain re: word)
    literal: bool = False  # a private literal: a substring of the casefolded forms; `re:` entries see every form
    line: int = 0  # the line of a private entry in its list, for --explain


@dataclass(frozen=True)
class Allow:
    """An allowlist entry: hits whose matched span lies inside an occurrence of `text` are excused."""

    text: str
    why: str
    kept: re.Pattern[str]
    folded: re.Pattern[str]
    paths: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()

    def applies_to_path(self, rel: str) -> bool:
        return not self.paths or any(fnmatch.fnmatchcase(rel, glob) for glob in self.paths)

    def applies_to(self, pattern: Pattern) -> bool:
        return not self.patterns or ("private" if pattern.private else pattern.id) in self.patterns


def _compile(text: str, flags: int, where: str) -> re.Pattern[str]:
    try:
        regex = re.compile(text, flags)
    except re.error as exc:
        raise PrivacyError(f"{where}: not a valid regular expression: {exc.msg}") from exc
    if regex.search(""):
        raise PrivacyError(f"{where}: the regular expression matches empty text, so it would hit every line")
    return regex


def _read_yaml(path: Path, what: str) -> object:
    import yaml  # a core dependency; imported here so `--help` stays fast

    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise PrivacyError(f"cannot read the {what} {path}: {exc.__class__.__name__}") from exc


def load_public_patterns(path: Path) -> list[Pattern]:
    """The shipped structural patterns (see the comments in public-patterns.yaml for the format)."""
    return _load_patterns(path, "public patterns")


def load_extra_patterns(path: Path, public: Iterable[Pattern] = ()) -> list[Pattern]:
    """An extra-patterns file (the public file's format, kept outside every repository) read like the public one.

    It holds shapes that only the user's own records have (old id schemes, artifact markers), so the shipped list
    stays generic. An id that repeats one of `public` is refused: every printed id must name one pattern.
    """
    taken = {p.id for p in public}
    return _load_patterns(path, "extra patterns", taken)


def _load_patterns(path: Path, what: str, taken: Iterable[str] = ()) -> list[Pattern]:
    data = _read_yaml(path, what)
    rows = data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not rows:
        raise PrivacyError(f"{path}: expected a non-empty `patterns:` list")
    out: list[Pattern] = []
    seen: set[str] = set(taken)
    for n, row in enumerate(rows, 1):
        where = f"{Path(path).name} pattern {n}"
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("regex"), str):
            raise PrivacyError(f"{where}: needs an `id` and a `regex` string")
        unknown = sorted(str(k) for k in row if k not in PATTERN_KEYS)
        if unknown:  # also catches a bare YAML `on:`, which YAML 1.1 reads as the boolean true
            raise PrivacyError(f"{where}: unknown key {', '.join(unknown)}; the keys are {', '.join(PATTERN_KEYS)}")
        pid, target = row["id"].strip(), row.get("target", "content")
        if not pid or pid in seen or pid.startswith("private") or pid == UNSCANNABLE:
            raise PrivacyError(f"{where}: the id is empty, repeated (here or in the public patterns) or reserved")
        if target not in ("content", "path"):
            raise PrivacyError(f"{where}: `target` is content or path")
        seen.add(pid)
        out.append(Pattern(id=pid, regex=_compile(row["regex"], 0, where), what=str(row.get("what", "")),
                           target=target))
    return out


def _heading(comment: str) -> str:
    """A short category name from a `#` heading line of the private list."""
    text = comment.lstrip("#").strip()
    text = re.split(r"\s\(|;|:\s", text, maxsplit=1)[0].strip(" .,-")
    return text[:HEADING_WIDTH].rstrip() or "private list"


def _is_marker(line: str) -> bool:
    return line.replace(" ", "") == HARVEST_MARKER.replace(" ", "")


def load_private_words(path: Path) -> list[Pattern]:
    """A private word list. Each entry is labelled `private:<heading>` after the nearest `#` line above it."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise PrivacyError(f"cannot read the private word list: {exc.__class__.__name__}; fix the path or the "
                           f"encoding, UTF-8 expected") from exc
    raw_entries: list[tuple[int, str, str]] = []
    heading, harvested = "private list", False
    for n, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            if _is_marker(line):
                harvested, heading = True, "harvested"
            elif not harvested:
                heading = _heading(line)
            continue
        raw_entries.append((n, heading, line))
    entries: list[Pattern] = []
    for n, heading, line in raw_entries:
        where = f"private word list line {n}"
        literal = not line.startswith("re:")
        if literal:
            gate = fold(line)
            if not gate:
                continue
            regex = re.compile(re.escape(gate))
        else:
            regex, gate = _compile(line[3:], re.IGNORECASE, where), _plain_word(line[3:])
        entries.append(Pattern(id=heading, regex=regex, private=True, gate=gate, literal=literal, line=n))
    # A heading that itself contains a listed value would print that value: name it by number instead.
    matcher = Matcher.build(entries)
    names: dict[str, str] = {}
    for k, heading in enumerate(dict.fromkeys(e.id for e in entries), 1):
        leaks = bool(line_hits(heading, matcher))
        names[heading] = f"section {k}" if leaks else heading
    return [Pattern(id=f"private:{names[e.id]}", regex=e.regex, private=True, gate=e.gate, literal=e.literal,
                    line=e.line) for e in entries]


def load_allowlist(path: Path) -> list[Allow]:
    """The allowlist (see the comments in allow.yaml for the format)."""
    data = _read_yaml(path, "allowlist")
    rows = data.get("allow") if isinstance(data, dict) else None
    if rows is None:
        return []
    if not isinstance(rows, list):
        raise PrivacyError(f"{path}: `allow:` is a list")
    out: list[Allow] = []
    for n, row in enumerate(rows, 1):
        where = f"{Path(path).name} entry {n}"
        if not isinstance(row, dict) or not str(row.get("text") or "").strip() or not str(row.get("why") or "").strip():
            raise PrivacyError(f"{where}: needs a non-empty `text` and a `why`")
        unknown = sorted(str(k) for k in row if k not in ALLOW_KEYS)
        if unknown:
            raise PrivacyError(f"{where}: unknown key {', '.join(unknown)}; the keys are {', '.join(ALLOW_KEYS)}")
        paths, patterns = row.get("paths") or [], row.get("patterns") or []
        if not isinstance(paths, list) or not isinstance(patterns, list):
            raise PrivacyError(f"{where}: `paths` and `patterns` are lists")
        text = str(row["text"])
        out.append(Allow(text=text, why=str(row["why"]), kept=re.compile(re.escape(_nfkc(text)), re.IGNORECASE),
                         folded=re.compile(re.escape(fold(text))), paths=tuple(map(str, paths)),
                         patterns=tuple(map(str, patterns))))
    return out


def default_data_dir() -> Path | None:
    """`<repo>/data/privacy` of the checkout this module runs from (src layout), or None when absent."""
    here = Path(__file__).resolve()
    if len(here.parents) < 4:
        return None
    folder = here.parents[3] / "data" / "privacy"
    return folder if (folder / PATTERNS_FILE).is_file() else None


# --- which files ------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """A file to scan: where it is, its repository-relative path and the name the output shows."""

    path: Path
    rel: str
    display: str
    in_tests: bool


def _is_work_tree(folder: Path) -> bool:
    dot = folder / ".git"
    return dot.is_file() or (dot.is_dir() and (dot / "HEAD").is_file())


def _absolute(path: Path) -> Path:
    """An absolute path without following symlinks (a linked file keeps its place in the tree)."""
    return Path(os.path.abspath(path))


def _work_tree_of(path: Path) -> Path | None:
    start = _absolute(path if path.is_dir() else path.parent)
    for folder in (start, *start.parents):
        if _is_work_tree(folder):
            return folder
    return None


def _skip_dir(name: str) -> bool:
    return name in SKIP_DIRS or name.startswith(".venv") or name.endswith(".egg-info")


def _target(path: Path, top: Path | None, base: Path, shown_base: str) -> Target:
    if top is not None:
        rel = _absolute(path).relative_to(top).as_posix()
        return Target(path, rel, f"{top.name}/{rel}", rel.split("/")[0] == "tests")
    rel = path.relative_to(base).as_posix()
    shown = f"{shown_base.rstrip('/')}/{rel}" if shown_base else rel
    return Target(path, rel, shown, rel.split("/")[0] == "tests" or _absolute(base).name == "tests")


def _git_files(folder: Path) -> list[str] | None:
    """Paths under `folder` that git would commit (tracked, plus untracked and not ignored), or None if git fails."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # a pure read: never refresh or lock the index
    try:
        done = subprocess.run(["git", "-C", str(folder), "ls-files", "-z", "--cached", "--others",
                               "--exclude-standard"], capture_output=True, timeout=120, env=env)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    names = [n.decode("utf-8", "replace") for n in done.stdout.split(b"\0") if n]
    return sorted(dict.fromkeys(names))


def _walk(root: Path, shown: str, err: TextIO) -> Iterator[Target]:
    top = _work_tree_of(root)
    if top is not None:
        listed = _git_files(root)
        if listed is not None:
            for rel in listed:
                path = root / rel
                if path.is_file():
                    yield _target(path, top, root, shown)
            return
        print(f"note: git could not list {shown}; walking it instead, ignored files included", file=err)
    for dirpath, dirnames, filenames in os.walk(root):
        folder = Path(dirpath)
        nested = [folder / d for d in sorted(dirnames) if not _skip_dir(d) and _is_work_tree(folder / d)]
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d) and folder / d not in nested)
        for name in sorted(filenames):
            yield _target(folder / name, top, root, shown)
        for sub in nested:
            yield from _walk(sub, f"{shown.rstrip('/')}/{sub.relative_to(root).as_posix()}", err)


def iter_targets(paths: Sequence[Path | str], err: TextIO | None = None) -> Iterator[Target]:
    """Every file to scan under `paths`; a file named explicitly is scanned even when git ignores it."""
    err = sys.stderr if err is None else err
    for given in paths:
        path = Path(given)
        shown = Path(given).as_posix()
        if path.is_file():
            top = _work_tree_of(path)
            yield _target(path, top, path.parent, "") if top is not None else Target(path, path.name, shown,
                                                                                     path.parent.name == "tests")
        elif path.is_dir():
            yield from _walk(path, shown, err)
        else:
            raise PrivacyError(f"no such file or folder: {shown}")


# --- reading one file ------------------------------------------------------------------------------------------

def _text_encoding(head: bytes) -> str | None:
    """The codec of a text file from its first bytes, or None for binary."""
    if head.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if head.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return "utf-32"
    if head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    sample = head[: len(head) - len(head) % 2]
    pairs = len(sample) // 2
    if pairs >= 2:
        even, odd = sample[0::2].count(0), sample[1::2].count(0)
        if odd >= 0.3 * pairs and even <= 0.05 * pairs:
            return "utf-16-le"
        if even >= 0.3 * pairs and odd <= 0.05 * pairs:
            return "utf-16-be"
    return None if b"\0" in head else "utf-8"


def _looks_safetensors(path: Path, head: bytes, size: int) -> bool:
    if path.suffix.lower() == ".safetensors":
        return True
    if len(head) < 9 or head[8:9] != b"{":
        return False
    (length,) = struct.unpack("<Q", head[:8])
    return 0 < length <= size - 8


def _text_lines(path: Path, encoding: str, size: int) -> Iterator[tuple[str, str]]:
    if size > MAX_TEXT_BYTES:
        raise _Unscannable("too large")
    raw = path.read_bytes()
    if encoding == "utf-8":
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("cp1252", "replace")
    else:
        text = raw.decode(encoding, "replace")
    for n, line in enumerate(text.split("\n"), 1):
        yield str(n), line.rstrip("\r")


def _inflate(data: bytes) -> bytes:
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(data, MAX_CHUNK_TEXT)
    except zlib.error as exc:
        raise _Unscannable("a compressed PNG text chunk is corrupt") from exc
    if inflater.unconsumed_tail:
        raise _Unscannable("a PNG text chunk inflates beyond the size limit")
    return out


def _png_text(kind: bytes, data: bytes) -> tuple[str, str]:
    """(keyword line, text) of one tEXt, zTXt or iTXt chunk."""
    key, _, rest = data.partition(b"\0")
    keyword = key.decode("latin-1")
    if kind == b"tEXt":
        return keyword, rest.decode("latin-1")
    if kind == b"zTXt":
        return keyword, _inflate(rest[1:]).decode("latin-1")
    compressed = rest[:1] == b"\x01"
    language, _, rest = rest[2:].partition(b"\0")
    translated, _, body = rest.partition(b"\0")
    if compressed:
        body = _inflate(body)
    keyline = " ".join(p for p in (keyword, language.decode("latin-1"), translated.decode("utf-8", "replace")) if p)
    return keyline, body.decode("utf-8", "replace")


def _read_declared(handle: BinaryIO, length: int, size: int, what: str) -> bytes:
    """`length` bytes a header declared, refused before any allocation when the file cannot hold them."""
    if length > size - handle.tell():
        raise _Unscannable(f"{what} claims more bytes than the file holds (a truncated or damaged file)")
    data = handle.read(length)
    if len(data) < length:
        raise _Unscannable(f"{what} ends early (the file changed while it was read)")
    return data


def _png_lines(path: Path) -> Iterator[tuple[str, str]]:
    """The text chunks of a PNG, chunk by chunk; image data is skipped with a seek, never read."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.read(len(PNG_MAGIC))
        n = 0
        while True:
            head = handle.read(8)
            if len(head) < 8:
                return
            length, kind = struct.unpack(">I4s", head)
            if kind == b"IEND":
                return
            if kind not in PNG_TEXT_CHUNKS:
                handle.seek(length + 4, os.SEEK_CUR)
                continue
            if length > MAX_CHUNK_TEXT:
                raise _Unscannable("a PNG text chunk is larger than the size limit")
            data = _read_declared(handle, length, size, "a PNG text chunk")
            handle.seek(4, os.SEEK_CUR)
            n += 1
            keyline, text = _png_text(kind, data)
            yield f"png-text-{n}:key", keyline
            for i, line in enumerate(text.splitlines(), 1):
                yield f"png-text-{n}:{i}", line


def _safetensors_lines(path: Path) -> Iterator[tuple[str, str]]:
    """The JSON header of a safetensors file, pretty-printed one key per line; the tensors are never read."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(8)
        if len(head) < 8:
            raise _Unscannable("a safetensors file without a header")
        (length,) = struct.unpack("<Q", head)
        if length > MAX_HEADER_BYTES:
            raise _Unscannable("a safetensors header larger than the size limit")
        raw = _read_declared(handle, length, size, "a safetensors header")
    text = raw.decode("utf-8", "replace")
    try:
        lines = json.dumps(json.loads(text), indent=1, ensure_ascii=False).splitlines()
    except ValueError:
        lines = text.splitlines()
    for n, line in enumerate(lines, 1):
        yield f"header:{n}", line


def _file_lines(path: Path) -> tuple[str, Iterator[tuple[str, str]]]:
    """(kind, (location, line) pairs) of one file; kind is text, png, safetensors or binary."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(8192)
    if head.startswith(PNG_MAGIC):
        return "png", _png_lines(path)
    encoding = _text_encoding(head)
    if encoding is None and _looks_safetensors(path, head, size):
        return "safetensors", _safetensors_lines(path)
    if encoding is None:
        return "binary", iter(())
    return "text", _text_lines(path, encoding, size)


# --- matching ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Hit:
    """One pattern hitting one place in one file. It never holds the matched text."""

    file: str
    location: str  # a line number, `png-text-N:line`, `header:line`, `name` (the path) or `file`
    pattern: str
    list_line: int = 0  # the private list line of the entry, for --explain


def _excused(span: tuple[int, int], view: str, allows: Sequence[Allow]) -> bool:
    start, end = span
    for allow in allows:
        for regex in (allow.kept, allow.folded):
            for occurrence in regex.finditer(view):
                if occurrence.start() <= start and end <= occurrence.end():
                    return True
    return False


def _hits_in(pattern: Pattern, texts: Sequence[str], allows: Sequence[Allow]) -> bool:
    for text in texts:
        if pattern.literal and pattern.gate not in text:
            continue
        for match in pattern.regex.finditer(text):
            if not allows or not _excused(match.span(), text, allows):
                return True
    return False


# The shape of a `re:` entry that is a plain word: optionally wrapped in a scoped-flags group such as `(?-i:...)`,
# optionally bordered by `\b` or by a lookbehind/lookahead character class, around literal characters only.
_SCOPED_FLAGS = re.compile(r"\(\?[a-zA-Z]*-?[a-zA-Z]*:(.*)\)", re.S)
_BORDER_BEFORE = re.compile(r"^(?:\\b|\(\?<!\[[^\]]*\]\))")
_BORDER_AFTER = re.compile(r"(?:\\b|\(\?!\[[^\]]*\]\))$")
_LITERAL_CHARS = re.compile(r"(?:[^\\.^$*+?{}\[\]|()]|\\[^A-Za-z0-9])+")


def _plain_word(source: str) -> str:
    """The casefolded word every match of a `re:` entry contains when the entry is a plain word, else "".

    `\\bword\\b`, `(?<![...])word(?![...])` and the same inside `(?-i:...)` qualify; anything with a real regex
    operator does not and is run on every line instead.
    """
    wrapped = _SCOPED_FLAGS.fullmatch(source)
    body = wrapped.group(1) if wrapped else source
    body = _BORDER_AFTER.sub("", _BORDER_BEFORE.sub("", body, count=1), count=1)
    if not _LITERAL_CHARS.fullmatch(body):
        return ""
    return fold(re.sub(r"\\(.)", r"\1", body))


def _prefix_gate(prefixes: Iterable[str]) -> re.Pattern[str] | None:
    """One regex whose matches start wherever any of `prefixes` occurs (a trie inside a lookahead, so it is fast).

    It has no false negatives: an entry whose gate word occurs in a text has its prefix at the same place. Lines
    where it finds nothing skip the gated entries entirely, which keeps a list of thousands of entries cheap.
    """
    trie: dict = {}
    for prefix in prefixes:
        node = trie
        for ch in prefix:
            node = node.setdefault(ch, {})
        node[""] = {}
    if not trie:
        return None

    def build(node: dict) -> str:
        branches = [re.escape(ch) + build(child) for ch, child in sorted(node.items()) if ch]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else "(?:" + "|".join(branches) + ")"
        return f"(?:{body})?" if "" in node else body

    return re.compile(f"(?={build(trie)})")


@dataclass(frozen=True)
class Matcher:
    """The patterns of one scan, prepared once.

    Patterns and `re:` entries see every form of a line, case kept and casefolded (an entry may scope its own
    case rule, for example `(?-i:...)`); private literals see the casefolded forms, since the list is
    case-insensitive after NFKC. Private entries with a `gate` word run only on lines where the prefix gate finds
    one of those words in a casefolded form; the other `re:` entries (real regexes) run on every line.
    """

    public: tuple[Pattern, ...]  # public patterns with target: content
    path_only: tuple[Pattern, ...]  # public patterns with target: path
    private: tuple[Pattern, ...]  # private regexes without a gate word
    by_prefix: dict[str, tuple[Pattern, ...]]  # private literals and plain-word regexes, by their gate prefix
    prefix_lengths: tuple[int, ...]
    gate: re.Pattern[str] | None

    @classmethod
    def build(cls, patterns: Iterable[Pattern]) -> Matcher:
        patterns = list(patterns)
        by_prefix: dict[str, list[Pattern]] = {}
        for p in patterns:
            if p.private and p.gate:
                by_prefix.setdefault(p.gate[:GATE_PREFIX], []).append(p)
        return cls(public=tuple(p for p in patterns if not p.private and p.target == "content"),
                   path_only=tuple(p for p in patterns if not p.private and p.target == "path"),
                   private=tuple(p for p in patterns if p.private and not p.gate),
                   by_prefix={k: tuple(v) for k, v in by_prefix.items()},
                   prefix_lengths=tuple(sorted({len(k) for k in by_prefix})), gate=_prefix_gate(by_prefix))

    def gated(self, texts: Sequence[str]) -> list[Pattern]:
        """The gated entries whose prefix occurs in one of `texts`, in list order; the rest cannot match."""
        if self.gate is None:
            return []
        out: dict[int, Pattern] = {}
        for text in texts:
            for start in self.gate.finditer(text):
                at = start.start()
                for length in self.prefix_lengths:
                    for p in self.by_prefix.get(text[at:at + length], ()):
                        out[id(p)] = p
        return sorted(out.values(), key=lambda p: p.line)


def line_hits(line: str, matcher: Matcher, allows: Sequence[Allow] = (), *, path: bool = False) -> list[Pattern]:
    """The patterns that hit one line (one per id), after the allowlist; the core of the scan.

    `path=True` marks the line as a file's repository-relative path, which the `target: path` patterns also see.
    """
    kept, folded = views(line)
    found: dict[str, Pattern] = {}

    def check(pattern: Pattern, texts: Sequence[str]) -> None:
        if pattern.id not in found and _hits_in(pattern, texts, [a for a in allows if a.applies_to(pattern)]):
            found[pattern.id] = pattern

    for pattern in (*matcher.public, *matcher.path_only) if path else matcher.public:
        check(pattern, [*kept, *folded])
    for pattern in matcher.private:
        check(pattern, [*kept, *folded])
    for pattern in matcher.gated(folded):
        check(pattern, folded if pattern.literal else [*kept, *folded])
    return list(found.values())


@dataclass
class ScanResult:
    """The outcome of a scan: hits (never matched text), files read and binary files counted but not read."""

    hits: list[Hit] = field(default_factory=list)
    files: int = 0
    binary: list[str] = field(default_factory=list)

    def by_file(self) -> dict[str, Counter[str]]:
        out: dict[str, Counter[str]] = {}
        for hit in self.hits:
            out.setdefault(hit.file, Counter())[hit.pattern] += 1
        return out

    def totals(self) -> Counter[str]:
        return Counter(hit.pattern for hit in self.hits)


def scan_target(target: Target, matcher: Matcher, allows: Sequence[Allow], result: ScanResult) -> None:
    """Scan one file into `result`: its path against every pattern, its content against the content patterns."""
    file_allows = [] if target.in_tests else [a for a in allows if a.applies_to_path(target.rel)]

    def record(location: str, line: str, path: bool = False) -> None:
        for pattern in line_hits(line, matcher, file_allows, path=path):
            result.hits.append(Hit(target.display, location, pattern.id, pattern.line))

    record("name", target.rel, path=True)
    result.files += 1
    try:
        kind, lines = _file_lines(target.path)
        if kind == "binary":
            result.binary.append(target.display)
            return
        for location, line in lines:
            if line:
                record(location, line)
    except (_Unscannable, OSError) as exc:
        detail = str(exc) if isinstance(exc, _Unscannable) else "cannot be read"
        print(f"note: {target.display}: {detail}; counted as {UNSCANNABLE}", file=sys.stderr)
        result.hits.append(Hit(target.display, "file", UNSCANNABLE))


def scan(paths: Sequence[Path | str], patterns: Sequence[Pattern], allows: Sequence[Allow] = (), *,
         skip: Iterable[Path | None] = (), err: TextIO | None = None) -> ScanResult:
    """Scan files and folders with explicit lists; `skip` names files never to scan (the private list itself)."""
    skipped = {Path(s).resolve() for s in skip if s is not None}
    matcher = Matcher.build(patterns)
    result = ScanResult()
    for target in iter_targets(paths, err):
        if target.path.resolve() in skipped:
            continue
        scan_target(target, matcher, allows, result)
    return result


# --- output -----------------------------------------------------------------------------------------------------

def _counts(counter: Counter[str]) -> str:
    return ", ".join(f"{name} {count}" for name, count in sorted(counter.items()))


def print_report(result: ScanResult, out: TextIO, *, explain: bool, mode: str, sizes: tuple[int, int, int],
                 extra: int | None = None) -> None:
    """Counts per file and category, then one summary line (always the last line). Never the matched text.

    `sizes` is (public patterns, private entries, allowlist entries); `extra` the extra patterns, when a file of
    them was loaded (the summary names them only then, so a public-only line reads as before).
    """
    by_file = result.by_file()
    for name in sorted(by_file):
        out.write(f"  {name}: {_counts(by_file[name])}\n")
        if explain:
            for hit in (h for h in result.hits if h.file == name):
                extra = f" (list line {hit.list_line})" if hit.list_line else ""
                out.write(f"    {hit.file}:{hit.location} {hit.pattern}{extra}\n")
    if by_file:
        out.write(f"by category: {_counts(result.totals())}\n")
    if result.binary:
        out.write(f"note: {len(result.binary)} binary file(s) counted but not read: no text, PNG or safetensors "
                  f"header{'' if explain else '; --explain lists them'}\n")
        if explain:
            for name in result.binary:
                out.write(f"    {name}\n")
    public, private, allow = sizes
    files_with = f"{len(by_file)} of {result.files}" if by_file else str(result.files)
    extras = "" if extra is None else f", {extra} extra pattern(s)"
    out.write(f"privacy scan, {mode}: {len(result.hits)} hit(s) in {files_with} file(s); {public} public "
              f"pattern(s){extras}, {private} private entr{'y' if private == 1 else 'ies'}, {allow} allowlist "
              f"entr{'y' if allow == 1 else 'ies'}\n")


def run_scan(paths: Sequence[Path | str], *, patterns_file: Path, allow_file: Path | None = None,
             words_file: Path | None = None, extra_file: Path | None = None, explain: bool = False,
             out: TextIO | None = None, err: TextIO | None = None) -> int:
    """Load the lists, scan, print the report; the exit code is 0 clean, 1 hits, 2 a list or path problem."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    try:
        public = load_public_patterns(patterns_file)
        extra = load_extra_patterns(extra_file, public) if extra_file is not None else []
        private = load_private_words(words_file) if words_file is not None else []
        allows = load_allowlist(allow_file) if allow_file is not None else []
        result = scan(paths, [*public, *extra, *private], allows, skip=[words_file, extra_file], err=err)
    except PrivacyError as exc:
        print(f"privacy scan: {exc}", file=err)
        return EXIT_USAGE
    mode = "public patterns" + (" + extra patterns" if extra_file is not None else "")
    mode += " + private list" if words_file is not None else " only"
    print_report(result, out, explain=explain, mode=mode, sizes=(len(public), len(private), len(allows)),
                 extra=len(extra) if extra_file is not None else None)
    return EXIT_HITS if result.hits else EXIT_OK


# --- the harvest: the user's own SELECT statements append values to the private list ----------------------------

def split_statements(sql: str) -> list[str]:
    """The statements of an SQL file, split at semicolons outside strings and comments."""
    import sqlite3

    out: list[str] = []
    buffer = ""
    parts = sql.split(";")
    for i, part in enumerate(parts):
        buffer += part + (";" if i < len(parts) - 1 else "")
        if sqlite3.complete_statement(buffer) or i == len(parts) - 1:
            if _SQL_COMMENT.sub("", buffer).strip(" \t\r\n;"):
                out.append(buffer.strip())
            buffer = ""
    return out


def _is_select(statement: str) -> bool:
    words = _SQL_COMMENT.sub(" ", statement).split()
    return bool(words) and words[0].upper() in ("SELECT", "WITH")


def _read_only(action: int, *_: object) -> int:
    """sqlite authorizer: reading and functions only; every write, attach or pragma is denied."""
    import sqlite3

    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 33)}
    return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY


def _cell_values(cell: object) -> list[str]:
    if cell is None:
        return []
    if isinstance(cell, bytes):
        try:
            cell = cell.decode("utf-8")
        except UnicodeDecodeError:
            return []
    return [line.strip() for line in str(cell).splitlines() if line.strip()]


def harvested_values(sql_file: Path, db: Path) -> tuple[list[str], int]:
    """(every string the SELECT statements return, statement count). The database is opened read-only."""
    import sqlite3

    try:
        statements = split_statements(Path(sql_file).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError) as exc:
        raise PrivacyError(f"cannot read the SQL file {sql_file}: {exc.__class__.__name__}") from exc
    if not statements:
        raise PrivacyError(f"{sql_file}: no statement found")
    for n, statement in enumerate(statements, 1):
        if not _is_select(statement):
            raise PrivacyError(f"{sql_file}: statement {n} is not a SELECT; only SELECT or WITH ... SELECT runs")
    if not Path(db).is_file():
        raise PrivacyError(f"no database file at {db}")
    uri = Path(db).resolve().as_uri() + "?mode=ro"  # read-only; never immutable=1, which would skip the WAL
    values: list[str] = []
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise PrivacyError(f"cannot open {db} read-only: {exc}") from exc
    try:
        conn.set_authorizer(_read_only)
        for n, statement in enumerate(statements, 1):
            try:
                for row in conn.execute(statement):
                    for cell in row:
                        values.extend(_cell_values(cell))
            except sqlite3.Error as exc:
                raise PrivacyError(f"{sql_file}: statement {n} failed: {exc.__class__.__name__}: {exc}") from exc
    finally:
        conn.close()
    return values, len(statements)


def _write_text(path: Path, text: str) -> None:
    """Atomic replace through store.py (imported only here, so the scanner itself needs no store)."""
    from . import store

    store.write_text(path, text)


def append_harvested(words_file: Path, values: Iterable[str], *, min_length: int = 3) -> tuple[int, int, int]:
    """Append new values below the harvested marker: (added, already listed, too short). Values never print."""
    path = Path(words_file)
    if not path.is_file():
        raise PrivacyError("the private word list does not exist; create it from "
                           "data/privacy/private-words.example.txt first")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise PrivacyError(f"cannot read the private word list: {exc.__class__.__name__}") from exc
    lines = text.splitlines()
    known = {fold(line.strip()) for line in lines if line.strip() and not line.strip().startswith("#")}
    added: list[str] = []
    present = short = 0
    for value in values:
        folded = fold(value)
        if len(folded) < min_length:
            short += 1
            continue
        escaped = "re:" + re.escape(folded)
        if folded in known or fold(escaped) in known:
            present += 1
            continue
        known.add(folded)
        plain = value == value.strip() and not value.startswith("#") and value[:3].lower() != "re:"
        added.append(value if plain else escaped)
    if added:
        body = text if not text or text.endswith("\n") else text + "\n"
        if not any(_is_marker(line.strip()) for line in lines):
            body += f"\n{HARVEST_MARKER}\n"
        _write_text(path, body + "".join(f"{v}\n" for v in added))
    return len(added), present, short


def harvest_sql(sql_file: Path, db: Path, words_file: Path, *, min_length: int = 3,
                out: TextIO | None = None) -> int:
    """Run the user's SELECT statements read-only and append each returned string to the private list."""
    out = sys.stdout if out is None else out
    values, statements = harvested_values(sql_file, db)
    added, present, short = append_harvested(words_file, values, min_length=min_length)
    out.write(f"harvest: {added} new value(s) from {statements} statement(s) appended below the harvested marker; "
              f"{present} already listed, {short} shorter than {min_length} characters skipped\n")
    return EXIT_OK


# --- configuration and the CLI ----------------------------------------------------------------------------------

def _tool() -> str:
    return __name__.split(".")[0]


def _configured(tool: str, key: str, workspace: str | None) -> tuple[Path | None, str, list[str]]:
    """(the path `[privacy] <key>` names, why there is none, the config's lines when the key is set but broken).

    The instance config wins when it loads. When it cannot load (no instance yet), or the key's `${VAR}` is
    unset, the variable `<TOOL>_PRIVACY__<KEY>` is read directly, so a hook or a shell that sets only the
    variable still gets its file.
    """
    var = f"{tool.upper()}_PRIVACY__{key.upper()}"
    why = "this install has no config loader"
    broken: list[str] = []
    try:
        from . import config
    except ImportError:
        config = None
    load = getattr(config, "load", None)
    if callable(load):
        error = getattr(config, "ConfigError", Exception)
        try:
            cfg = load(tool, workspace, strict=False)
        except error as exc:
            why = "; ".join(getattr(exc, "lines", None) or [str(exc)])
        else:
            try:
                if cfg.get(f"privacy.{key}", None):
                    return Path(cfg.path(f"privacy.{key}")), "", []
                why = f"[privacy] {key} is not set"
            except error as exc:  # set, but its ${VAR} is unset or it cannot be placed
                broken = list(getattr(exc, "lines", None) or [str(exc)])
                why = "; ".join(broken)
    raw = os.environ.get(var, "").strip()
    if raw:
        return Path(raw).expanduser(), "", []
    return None, f"{why}, and {var} is not set", broken


def configured_wordlist(tool: str, workspace: str | None = None) -> tuple[Path | None, str]:
    """(the configured private list, why there is none): `[privacy] wordlist` or `<TOOL>_PRIVACY__WORDLIST`."""
    path, why, _ = _configured(tool, "wordlist", workspace)
    return path, why


def configured_extra_patterns(tool: str, workspace: str | None = None) -> Path | None:
    """The configured extra-patterns file (`[privacy] extra_patterns` or `<TOOL>_PRIVACY__EXTRA_PATTERNS`), or
    None when none is configured. A key that is set but cannot be resolved (an unset `${VAR}`) is a PrivacyError:
    a scan never runs quietly without patterns the instance asked for."""
    path, _, broken = _configured(tool, "extra_patterns", workspace)
    if path is None and broken:
        raise PrivacyError("[privacy] extra_patterns is set but cannot be resolved: " + "; ".join(broken))
    return path


def _cmd_scan(args: argparse.Namespace) -> int:
    tool = getattr(args, "tool", None) or _tool()
    err = sys.stderr
    data_dir = default_data_dir()
    patterns = args.patterns or (data_dir / PATTERNS_FILE if data_dir else None)
    if patterns is None:
        print("privacy scan: the public patterns file was not found next to this checkout; pass --patterns <file>",
              file=err)
        return EXIT_USAGE
    allow = args.allow
    if allow is None and (Path(patterns).parent / ALLOW_FILE).is_file():
        allow = Path(patterns).parent / ALLOW_FILE
    if args.db is not None and args.harvest_sql is None:
        print("privacy scan: --db goes with --harvest-sql", file=err)
        return EXIT_USAGE
    words = args.words
    if words is None and not args.public:
        words, why = configured_wordlist(tool, args.workspace)
        if words is None:
            print(f"privacy scan: no private word list: {why}. Pass --words <file>, set [privacy] wordlist, or scan "
                  f"with --public for the public patterns only", file=err)
            return EXIT_USAGE
    extra = args.extra_patterns
    if extra is None and not args.public:
        try:
            extra = configured_extra_patterns(tool, args.workspace)
        except PrivacyError as exc:
            print(f"privacy scan: {exc}", file=err)
            return EXIT_USAGE
    if args.harvest_sql is not None:
        if words is None or args.db is None:
            print("privacy scan: --harvest-sql needs --db <file> and a private word list to append to: --words "
                  "<file> or [privacy] wordlist", file=err)
            return EXIT_USAGE
        try:
            harvest_sql(args.harvest_sql, args.db, words, min_length=args.min_length)
        except PrivacyError as exc:
            print(f"privacy scan: {exc}", file=err)
            return EXIT_USAGE
        if not args.paths:
            return EXIT_OK
    return run_scan(args.paths or [Path(".")], patterns_file=Path(patterns), allow_file=allow, words_file=words,
                    extra_file=extra, explain=args.explain)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `<tool> privacy scan` to the CLI."""
    parser = subparsers.add_parser(
        "privacy", help="scan files for private values before they leave the machine",
        description="Count private or machine-specific values in files before they leave the machine. Prints "
                    "counts per pattern and file, never the matched text.")
    actions = parser.add_subparsers(dest="privacy_action", metavar="<action>", title="actions")
    scan_parser = actions.add_parser(
        "scan", help="count public-pattern and private-list hits in files and folders; exit 1 on any hit",
        description="Scan files and folders: inside a git work tree the files git would commit, elsewhere every "
                    "file except .git, _gen and tool caches. Text is read as UTF-8 or UTF-16, JSON-unescaped, "
                    "URL-decoded and NFKC-casefolded; PNG text chunks and safetensors headers are read, pixels "
                    "and tensors never. Exit 0 clean, 1 any hit, 2 a usage error or an unreadable list.",
        epilog="The private list defaults to [privacy] wordlist of the instance config, overridable with the "
               "<TOOL>_PRIVACY__WORDLIST variable; the extra patterns to [privacy] extra_patterns "
               "(<TOOL>_PRIVACY__EXTRA_PATTERNS). --public reads no instance config; --words and --extra-patterns "
               "name files explicitly, also together with --public.")
    scan_parser.add_argument("paths", nargs="*", type=Path, metavar="PATH",
                             help="files or folders to scan; default: the current folder, or none with --harvest-sql")
    scan_parser.add_argument("--public", action="store_true",
                             help="public patterns and the allowlist only, without the configured private list and "
                                  "extra patterns")
    scan_parser.add_argument("--words", type=Path, metavar="FILE",
                             help="a private word list to use: one literal per line, re: lines are regexes")
    scan_parser.add_argument("--extra-patterns", dest="extra_patterns", type=Path, metavar="FILE",
                             help="an extra-patterns file in the public file's format (the id shapes of your own "
                                  "records), kept outside every repository")
    scan_parser.add_argument("--patterns", type=Path, metavar="FILE",
                             help="the public patterns file; default: data/privacy/public-patterns.yaml of this checkout")
    scan_parser.add_argument("--allow", type=Path, metavar="FILE",
                             help="the allowlist; default: allow.yaml next to the patterns file")
    scan_parser.add_argument("--explain", action="store_true",
                             help="also print file:line and the pattern id of every hit, never the matched text")
    scan_parser.add_argument("--harvest-sql", type=Path, metavar="FILE",
                             help="run the SELECT statements of this SQL file read-only against --db and append "
                                  "each returned string below the harvested marker of the private list")
    scan_parser.add_argument("--db", type=Path, metavar="FILE", help="the SQLite database for --harvest-sql")
    scan_parser.add_argument("--min-length", type=int, default=3, metavar="N",
                             help="harvested values shorter than N characters are skipped; default 3")
    scan_parser.add_argument("--workspace", metavar="PATH", default=None,
                             help="the instance folder or its TOML file, for the configured private list")
    scan_parser.set_defaults(func=_cmd_scan)
