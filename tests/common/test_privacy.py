"""The privacy scanner on synthetic hazards.

Every needle is built at run time from fragments (a backslash is `chr(92)`, the at sign `chr(64)`), so this file's
own text never holds a path, an address, an id or a placeholder, and the repo-wide scan of `tests/` stays clean.
"""

import importlib
import json
import shutil
import sqlite3
import struct
import zlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PKG = next(d.name for d in sorted((REPO / "src").iterdir()) if (d / "_common").is_dir())
privacy = importlib.import_module(f"{PKG}._common.privacy")
cli = importlib.import_module(f"{PKG}.cli")
DATA = REPO / "data" / "privacy"
BS = chr(92)
AT = chr(64)


def drive_path() -> str:
    return "C" + ":" + BS + "Users" + BS + "example" + BS


def email() -> str:
    return "someone" + AT + "example.invalid"


@pytest.fixture(scope="module")
def public():
    return privacy.load_public_patterns(DATA / "public-patterns.yaml")


@pytest.fixture(scope="module")
def allows():
    return privacy.load_allowlist(DATA / "allow.yaml")


def put(folder: Path, name: str, data: str | bytes) -> Path:
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def ids(result) -> set[str]:
    return {hit.pattern for hit in result.hits}


def words_file(folder: Path, text: str) -> Path:
    return put(folder, "words.txt", text)


def run_cli(argv: list[str], capsys) -> tuple[int, str]:
    try:
        code = cli.main(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def png_bytes(text_chunks: list[tuple[bytes, bytes]]) -> bytes:
    """A 1x1 PNG with the given extra chunks before the image data; built by hand so any chunk type works."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    body = b"".join(chunk(kind, data) for kind, data in text_chunks)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + body + chunk(b"IDAT", zlib.compress(b"\x00\x00"))
            + chunk(b"IEND", b""))


def safetensors_bytes(metadata: dict, tensor: bytes) -> bytes:
    header = {"__metadata__": metadata, "w": {"dtype": "U8", "shape": [len(tensor)], "data_offsets": [0, len(tensor)]}}
    raw = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw + tensor


# --- hazard classes ---------------------------------------------------------------------------------------------

HAZARDS = {
    "drive path": (lambda: ("path = " + drive_path()).encode("utf-8"), {"drive-path"}),
    "json-escaped drive path": (lambda: json.dumps({"path": drive_path()}).encode("utf-8"),
                                {"drive-path", "drive-path-escaped"}),
    "utf-16 with bom": (lambda: ("see " + drive_path()).encode("utf-16"), {"drive-path"}),
    "utf-16 without bom": (lambda: ("see " + drive_path() + " now").encode("utf-16-le"), {"drive-path"}),
    "percent-encoded folder in a file url": (
        lambda: ("file" + ":" + "/" * 3 + "srv/My" + "%" + "20" + "Folder/notes.txt").encode("utf-8"),
        {"file-url", "percent-encoded-separator"}),
    "url-encoded address": (lambda: ("mailto=someone" + "%" + "40" + "example.invalid").encode("utf-8"), {"email"}),
    "nested json string": (
        lambda: json.dumps({"payload": '{"mail": "someone' + BS + 'u0040example.invalid"}'}).encode("utf-8"),
        {"email"}),
    "e-mail": (lambda: ("contact " + email()).encode("utf-8"), {"email"}),
    # A captured log in a JSON record: the escaped line break must not glue its letter onto the next word.
    "escaped newline before a drive path": (
        lambda: json.dumps({"output": "Traceback:\n" + drive_path() + "x"}).encode("utf-8"), {"drive-path"}),
    "escaped tab before a token": (
        lambda: json.dumps({"output": "token:\t" + "hf" + "_" + "B" * 24}).encode("utf-8"), {"hf-token"}),
    "escaped carriage return and form feed": (
        lambda: json.dumps({"log": "a\r" + "hf" + "_" + "C" * 24 + "\fb"}).encode("utf-8"), {"hf-token"}),
    "real-year id": (lambda: ("run-" + "2026" + "0101" + "-demo").encode("utf-8"), {"real-year-id"}),
    "numbered folder": (lambda: ("12" + "_" + "assets" + "/x.yaml").encode("utf-8"), {"numbered-folder"}),
    "copyright placeholder": (lambda: ("Copyright 2030 " + "<COPYRIGHT" + " HOLDER>").encode("utf-8"), {"placeholder"}),
    "insert placeholder": (lambda: ("[INS" + "ERT your name]").encode("utf-8"), {"placeholder"}),
    "handle placeholder": (lambda: ("* @" + "<han" + "dle>").encode("utf-8"), {"placeholder"}),
    "network share": (lambda: ("see " + BS * 2 + "server" + BS + "share").encode("utf-8"), {"unc-path"}),
    "users folder": (lambda: ("/" + "Users" + "/" + "someone" + "/x").encode("utf-8"), {"users-folder"}),
    "home folder": (lambda: ("/" + "home" + "/" + "someone" + "/x").encode("utf-8"), {"home-folder"}),
    "app data folder": (lambda: ("x/" + "App" + "Data" + "/y").encode("utf-8"), {"app-data-folder"}),
    "temp variable": (lambda: ("%" + "TEMP" + "%" + " folder").encode("utf-8"), {"profile-or-temp-variable"}),
    "pip file reference": (lambda: ("demo " + AT + " file" + ":" + "x").encode("utf-8"), {"pip-file-reference"}),
    "cloud-sync brand": (lambda: ("One" + "Drive" + " folder").encode("utf-8"), {"cloud-sync-brand"}),
    "secret assignment": (lambda: ("api" + "_key = " + "a" * 20).encode("utf-8"), {"secret-assignment"}),
    "hf token": (lambda: ("hf" + "_" + "A" * 24).encode("utf-8"), {"hf-token"}),
    "civitai page": (lambda: ("civitai" + ".com/models/" + "1").encode("utf-8"), {"civitai-page-url"}),
    "civitai private api": (lambda: ("civitai" + ".com/api/" + "trpc/x").encode("utf-8"), {"civitai-private-api"}),
    "frozen clock output": (lambda: ("clock frozen (" + "EASEL" + "_NOW set)").encode("utf-8"), {"frozen-clock-output"}),
    "detector import": (lambda: ("import " + "ultra" + "lytics").encode("utf-8"), {"agpl-import"}),
    "png text chunk": (lambda: png_bytes([(b"tEXt", b"parameters\x00made by " + email().encode())]), {"email"}),
    "png compressed text chunk": (
        lambda: png_bytes([(b"zTXt", b"prompt\x00\x00" + zlib.compress(drive_path().encode("latin-1")))]),
        {"drive-path"}),
    "png international text chunk": (
        lambda: png_bytes([(b"iTXt", b"workflow\x00\x01\x00en\x00\x00" + zlib.compress(email().encode()))]),
        {"email"}),
}


@pytest.mark.parametrize("name", sorted(HAZARDS))
def test_hazard_class_is_found(name, tmp_path, public):
    make, expected = HAZARDS[name]
    path = put(tmp_path, "sample.png" if name.startswith("png") else "sample.txt", make())
    found = ids(privacy.scan([path], public))
    found.discard("media-file")  # the .png name itself is a path hit; the test is about the content
    assert found == expected


PATH_NEEDLES = {"notes.db": "forbidden-file-type", "desktop.ini": "desktop-ini", "direct_url.json": "install-metadata-file",
                "lora.safetensors": "weights-file", "out.webp": "media-file"}


def test_every_shipped_pattern_has_a_needle(public):
    """A new pattern in public-patterns.yaml needs a synthetic needle here (HAZARDS or PATH_NEEDLES)."""
    covered = set().union(*(expected for _, expected in HAZARDS.values())) | set(PATH_NEEDLES.values())
    assert {p.id for p in public} == covered


@pytest.mark.parametrize("name", sorted(PATH_NEEDLES))
def test_path_needle_is_found(name, tmp_path, public):
    assert ids(privacy.scan([put(tmp_path / "proj", name, b"")], public)) == {PATH_NEEDLES[name]}


def test_nested_string_needs_more_than_one_unescape():
    line = json.dumps({"payload": '{"mail": "someone' + BS + 'u0040example.invalid"}'})
    assert AT not in line
    forms = privacy.decoded_forms(line)
    assert AT not in forms[1] and any(AT in form for form in forms[2:])


def test_control_escapes_decode_as_spaces():
    """`\\n \\t \\r \\b \\f` become one space each, so a form stays one line and every word keeps its border."""
    line = json.dumps({"x": "a\nb\tc\rd\be\ff"})
    assert privacy.decoded_forms(line)[1] == '{"x": "a b c d e f"}'
    assert privacy.decoded_forms(BS * 2 + "n")[1] == BS + "n"  # an escaped backslash before n is not a newline


def test_bordered_private_word_after_an_escaped_newline(tmp_path):
    words = privacy.load_private_words(words_file(tmp_path, "# words\nre:" + BS + "bzorbax" + BS + "b\n"))
    record = json.dumps({"caption": "seen:\nzorbax at noon"})
    assert ids(privacy.scan([put(tmp_path, "rows.jsonl", record)], words)) == {"private:words"}


def test_png_chunk_that_claims_more_than_the_file_is_unscannable(tmp_path, public):
    """A 4 GiB text-chunk length in a tiny file is refused before anything is read or allocated."""
    lying = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", privacy.MAX_CHUNK_TEXT) + b"tEXt" + b"key\x00short"
    path = put(tmp_path, "lying.png", lying)
    assert ids(privacy.scan([path], public)) - {"media-file"} == {privacy.UNSCANNABLE}


def test_safetensors_header_that_claims_more_than_the_file_is_unscannable(tmp_path, public):
    path = put(tmp_path, "lying.safetensors", struct.pack("<Q", 64 * 2**20) + b'{"__metadata__": {}}')
    assert ids(privacy.scan([path], public)) == {privacy.UNSCANNABLE, "weights-file"}


def test_safetensors_header_is_read_and_tensors_never(tmp_path, public):
    nested = json.dumps({drive_path() + "data": {"n_repeats": 1}})
    path = put(tmp_path, "model.safetensors", safetensors_bytes({"ss_dataset_dirs": nested}, email().encode()))
    found = ids(privacy.scan([path], public))
    assert "drive-path" in found
    assert "email" not in found  # the address sits in the tensor bytes, which are never read
    assert "weights-file" in found  # and the file name alone says it must not be committed


def test_png_pixels_and_other_chunks_are_never_read(tmp_path, public):
    path = put(tmp_path, "pixels.png", png_bytes([(b"prVt", email().encode())]))
    assert ids(privacy.scan([path], public)) == {"media-file"}


def test_binary_files_are_counted_not_read(tmp_path, public):
    path = put(tmp_path, "blob.dat", b"\x00\x01\x02" + email().encode() + b"\x00")
    result = privacy.scan([path], public)
    assert result.hits == [] and len(result.binary) == 1


def test_oversized_text_counts_as_unscannable(tmp_path, public, monkeypatch):
    monkeypatch.setattr(privacy, "MAX_TEXT_BYTES", 10)
    path = put(tmp_path, "big.txt", "x" * 20)
    assert ids(privacy.scan([path], public)) == {privacy.UNSCANNABLE}


def test_path_patterns_apply_to_names(tmp_path, public):
    put(tmp_path / "proj", "notes.db", b"")
    put(tmp_path / "proj", "desktop.ini", b"")
    put(tmp_path / "proj", "docs/readme.md", "fine")
    result = privacy.scan([tmp_path / "proj"], public)
    assert {(h.file.split("/")[-1], h.pattern) for h in result.hits} == {
        ("notes.db", "forbidden-file-type"), ("desktop.ini", "desktop-ini")}


def test_path_patterns_ignore_content(tmp_path, public):
    """A .gitignore names forbidden types on purpose; only a committed file of that type is a hit."""
    assert {p.id for p in public if p.target == "path"} >= {"desktop-ini", "forbidden-file-type", "media-file"}
    ignore = "\n".join(["*.db", "*.safetensors", "*.png", "desktop" + ".ini", "outputs/001.png"])
    assert privacy.scan([put(tmp_path, "ignore-list.txt", ignore)], public).hits == []


def test_pattern_file_rejects_unknown_keys(tmp_path):
    bad = put(tmp_path, "patterns.yaml", "patterns:\n  - id: x\n    regex: 'abc'\n    on: path\n")
    with pytest.raises(privacy.PrivacyError, match="unknown key"):
        privacy.load_public_patterns(bad)


# --- what must stay clean ---------------------------------------------------------------------------------------

def test_demo_ids_are_clean(tmp_path, public, allows):
    text = ("rcp-20300101-001 run-20300101-001 job-20300101-001 snap-20300101 dec-9001 img-0123456789ab "
            "demo_robot dmrbt 0123456789abcdef.yaml 2030-01-01T00:00:00+00:00 ~/easel-home <home> <models_root>")
    assert privacy.scan([put(tmp_path, "demo.md", text)], public, allows).hits == []


def test_shipped_readme_is_clean(public, allows):
    result = privacy.scan([REPO / "README.md"], public, allows)
    assert [(h.location, h.pattern) for h in result.hits] == []


def test_readme_install_block_is_clean(tmp_path, public, allows):
    block = ("git clone https://github.com/H17-OpenSource/" + PKG + " && cd " + PKG + "\n"
             "python -m venv .venv && .venv/Scripts/pip install -e \".[dev]\"\n"
             f"{PKG} init ~/{PKG}-home\n{PKG} config set comfyui.url http://127.0.0.1:8188\n")
    assert privacy.scan([put(tmp_path, "README.md", block)], public, allows).hits == []


def test_privacy_sources_are_plain_ascii():
    """No invisible or direction-changing character hides in the scanner, its lists or this test."""
    paths = [Path(privacy.__file__), Path(__file__), *sorted(DATA.iterdir())]
    assert [p.name for p in paths if not p.read_bytes().isascii()] == []


def test_privacy_files_and_this_test_are_clean(public, allows):
    paths = [DATA, Path(privacy.__file__), Path(__file__)]
    result = privacy.scan(paths, public, allows)
    assert [(h.file, h.location, h.pattern) for h in result.hits] == []


# --- the private list and the allowlist -------------------------------------------------------------------------

def test_synthetic_private_list(tmp_path, public, allows):
    words = privacy.load_private_words(words_file(tmp_path, "# people\nExample Person\n# ports\nre:127\\.0\\.0\\.1:\\d+\n"))
    fullwidth = "".join(chr(ord(c) + 0xFEE0) for c in "Example")
    lines = ["met Example Person here", "127.0.0.1:8188 is the public default", "the host listens on 127.0.0.1:65123",
             fullwidth + " Person", "Exam" + chr(0x200B) + "ple Person", "Example" + "%" + "20" + "Person"]
    result = privacy.scan([put(tmp_path, "notes.md", "\n".join(lines))], [*public, *words], allows)
    found = {(h.location, h.pattern) for h in result.hits}
    assert found == {("1", "private:people"), ("3", "private:ports"), ("4", "private:people"),
                     ("5", "private:people"), ("6", "private:people"), ("6", "percent-encoded-separator")}


def test_regex_entries_keep_their_own_case_rule_and_borders(tmp_path):
    """A harvested id may be scoped case-sensitive with lookaround borders; it must still hit, and only exactly."""
    border = "[0-9A-Za-z-]"
    scoped = "re:(?-i:(?<!" + border + ")DEMO" + BS + "-7(?!" + border + "))"
    plain = "re:" + BS + "bzorb" + BS + "b"
    words = privacy.load_private_words(words_file(tmp_path, f"# ids\n{scoped}\n# words\n{plain}\n"))
    assert all(w.gate for w in words)  # both are plain words, so both sit behind the prefix gate
    lines = ["see DEMO-7 here", "see demo-7 here", "see DEMO-77 here", "a zorb b", "zorbax", "a ZORB b"]
    result = privacy.scan([put(tmp_path, "notes.md", "\n".join(lines))], words)
    assert {(h.location, h.pattern) for h in result.hits} == {("1", "private:ids"), ("4", "private:words"),
                                                              ("6", "private:words")}


def test_plain_word_shapes():
    border = "[0-9A-Za-z-]"
    assert privacy._plain_word(BS + "bAbc" + BS + "b") == "abc"
    assert privacy._plain_word("(?<!" + border + ")x" + BS + ".y(?!" + border + ")") == "x.y"
    assert privacy._plain_word("(?-i:(?<!" + border + ")Q" + BS + "-1(?!" + border + "))") == "q-1"
    assert privacy._plain_word("(?:a|b)") == "" and privacy._plain_word(BS + "d+") == ""


def test_heading_that_holds_a_value_is_not_printed(tmp_path):
    words = privacy.load_private_words(words_file(tmp_path, "# about zorblax\nzorblax\n# --- harvested ---\nqwerty1\n"))
    assert [w.id for w in words] == ["private:section 1", "private:harvested"]


def test_private_list_errors_fail_closed(tmp_path):
    with pytest.raises(privacy.PrivacyError, match="line 2"):
        privacy.load_private_words(words_file(tmp_path, "ok\nre:(unclosed\n"))
    with pytest.raises(privacy.PrivacyError, match="empty text"):
        privacy.load_private_words(words_file(tmp_path, "re:x*\n"))
    with pytest.raises(privacy.PrivacyError):
        privacy.load_private_words(tmp_path / "missing.txt")


def test_allowlist_never_applies_under_tests(tmp_path, public):
    placeholder = "clone https://github.com/" + "<" + "org>/demo"
    entry = "allow:\n  - text: \"" + "<" + "org>\"\n    why: a synthetic entry for this test\n"
    local = privacy.load_allowlist(put(tmp_path, "allow.yaml", entry))
    put(tmp_path / "proj", "docs/install.md", placeholder)
    put(tmp_path / "proj", "tests/install.md", placeholder)
    result = privacy.scan([tmp_path / "proj"], public, local)
    assert {(h.file.split("/proj/")[-1], h.pattern) for h in result.hits} == {("tests/install.md", "placeholder")}


def test_git_internals_generated_and_cache_folders_are_skipped(tmp_path, public):
    for name in (".git/config", "tests/fixtures/_gen/a.txt", "__pycache__/a.txt", ".venv/lib/a.txt",
                 "src/demo.egg-info/PKG-INFO"):
        put(tmp_path / "proj", name, "contact " + email())
    put(tmp_path / "proj", "docs/a.txt", "contact " + email())
    result = privacy.scan([tmp_path / "proj"], public)
    assert {h.file.split("/proj/")[-1] for h in result.hits} == {"docs/a.txt"}


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs git and a checkout")
def test_git_listing_is_what_could_be_committed():
    listed = privacy._git_files(REPO)
    assert listed is not None and "README.md" in listed
    assert not any(name.startswith((".pytest_cache/", ".ruff_cache/")) or "/__pycache__/" in name for name in listed)


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(), reason="needs git and a checkout")
def test_a_relative_subfolder_keeps_repository_paths(monkeypatch):
    """`scan .` from inside src/ still names files from the repository root, so path-scoped rules hold."""
    monkeypatch.chdir(REPO / "src")
    rels = {t.rel for t in privacy.iter_targets(["."])}
    assert f"src/{PKG}/cli.py" in rels and all(rel.startswith("src/") for rel in rels)


# --- the command line -------------------------------------------------------------------------------------------

def test_cli_prints_counts_and_lines_never_the_text(tmp_path, capsys):
    path = put(tmp_path, "notes.txt", "first\nsecond " + drive_path() + "\nthird " + email() + "\n")
    code, out = run_cli(["privacy", "scan", "--public", "--explain", str(path)], capsys)
    assert code == 1
    assert email() not in out and drive_path() not in out and "example" not in out
    assert ":2 drive-path" in out and ":3 email" in out
    assert "notes.txt: drive-path 1, email 1" in out
    assert out.strip().splitlines()[-1].startswith("privacy scan, public patterns only: 2 hit(s) in 1 of 1 file(s)")


def test_cli_clean_scan_exits_0(tmp_path, capsys):
    code, out = run_cli(["privacy", "scan", "--public", str(put(tmp_path, "ok.md", "nothing to see"))], capsys)
    assert code == 0 and "0 hit(s) in 1 file(s)" in out


def test_cli_words_adds_the_private_list(tmp_path, capsys):
    words = words_file(tmp_path, "# people\nExample Person\n")
    path = put(tmp_path, "notes.md", "met example person")
    code, out = run_cli(["privacy", "scan", "--public", "--words", str(words), str(path)], capsys)
    assert code == 1 and "private:people 1" in out and "example person" not in out.lower()


@pytest.fixture
def no_instance(tmp_path, monkeypatch):
    """No instance config anywhere: an empty working folder and an empty user folder."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(home)
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.delenv("PRIVATE_WORDS_FILE", raising=False)
    return home


def test_cli_default_list_comes_from_the_config_variable(tmp_path, capsys, monkeypatch, no_instance):
    words = words_file(tmp_path, "# people\nExample Person\n")
    monkeypatch.setenv(f"{PKG.upper()}_PRIVACY__WORDLIST", str(words))
    code, out = run_cli(["privacy", "scan", str(put(tmp_path, "notes.md", "Example Person"))], capsys)
    assert code == 1 and "private:people 1" in out


def test_cli_without_any_list_refuses(tmp_path, capsys, no_instance):
    code, out = run_cli(["privacy", "scan", str(put(tmp_path, "notes.md", "fine"))], capsys)
    assert code == 2 and "no private word list" in out


# --- the harvest ------------------------------------------------------------------------------------------------

def make_db(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE people (name TEXT, ref INTEGER)")
    conn.executemany("INSERT INTO people VALUES (?, ?)",
                     [("Example Person", 9000000001), ("Zed Quill", None), ("ab", None), (None, 9000000002)])
    conn.commit()
    conn.close()
    return path


def rows(db: Path) -> int:
    conn = sqlite3.connect(db)
    try:
        return conn.execute("SELECT count(*) FROM people").fetchone()[0]
    finally:
        conn.close()


def test_harvest_appends_below_the_marker_and_prints_no_value(tmp_path, capsys):
    db = make_db(tmp_path / "demo.db")
    sql = put(tmp_path, "harvest.sql", "-- names; and refs\nSELECT name FROM people;\nSELECT ref FROM people WHERE ref IS NOT NULL")
    words = words_file(tmp_path, "# people\nExample Person\n")
    code, out = run_cli(["privacy", "scan", "--words", str(words), "--harvest-sql", str(sql), "--db", str(db)], capsys)
    assert code == 0, out
    assert "Zed" not in out and "9000000001" not in out
    assert "harvest: 3 new value(s) from 2 statement(s)" in out and "1 already listed" in out and "1 shorter" in out
    text = words.read_text(encoding="utf-8")
    assert text.index(privacy.HARVEST_MARKER) < text.index("Zed Quill")
    assert text.count("Example Person") == 1 and "9000000002" in text
    target = put(tmp_path, "notes.md", "zed quill")
    assert ids(privacy.scan([target], privacy.load_private_words(words))) == {"private:harvested"}
    code, out = run_cli(["privacy", "scan", "--words", str(words), "--harvest-sql", str(sql), "--db", str(db)], capsys)
    assert code == 0 and "harvest: 0 new value(s)" in out


@pytest.mark.parametrize("statement", ["DELETE FROM people;", "WITH x AS (SELECT 1) DELETE FROM people;",
                                       "SELECT 1; DROP TABLE people;"])
def test_harvest_runs_reads_only(tmp_path, capsys, statement):
    db = make_db(tmp_path / "demo.db")
    words = words_file(tmp_path, "# people\n")
    sql = put(tmp_path, "bad.sql", statement)
    code, out = run_cli(["privacy", "scan", "--words", str(words), "--harvest-sql", str(sql), "--db", str(db)], capsys)
    assert code == 2, out
    assert rows(db) == 4
    assert words.read_text(encoding="utf-8") == "# people\n"
