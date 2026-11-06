"""Civitai is used through its public API only: every Civitai URL in src/, docs/ and the root docs is under
`/api/v1/` or `/api/download/`.

A link to a model, image, post or user page would point at a person's account; the private tRPC endpoints are
outside the site's terms for automation. A bare host (no path) is allowed, so a client may keep the host in one
constant. Planted URLs in the self-test are assembled at run time so this file holds none.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HOST = "civitai" + r"\.com"
URL = re.compile(r"(?i)(?:https?://)?(?:[a-z0-9-]+\.)*" + HOST + r"(?P<path>/[^\s'\"`)<>\]}]*)?")
ALLOWED = re.compile(r"/api/(?:v1|download)(?:/|$)")
SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".html", ".js", ".css"}


def scanned_files(root: Path) -> list[Path]:
    """Text files under src/ and docs/, and the markdown files at the root."""
    files = sorted(root.glob("*.md"))
    for folder in ("src", "docs"):
        base = root / folder
        if base.is_dir():
            files += sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in SUFFIXES
                            and "__pycache__" not in p.parts)
    return files


def bad_urls(path: Path, root: Path) -> list[str]:
    """`<file>:<line>: <path part>` for every Civitai URL whose path is not one of the public API endpoints."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        for m in URL.finditer(line):
            part = m.group("path") or ""
            if part in ("", "/") or ALLOWED.match(part):
                continue
            out.append(f"{path.relative_to(root).as_posix()}:{n}: {part}")
    return out


def test_only_public_api_urls_in_src_and_docs():
    found = [hit for path in scanned_files(REPO) for hit in bad_urls(path, REPO)]
    assert found == []


def test_the_check_tells_api_urls_from_pages(tmp_path):
    host = "civitai" + ".com"
    lines = [
        f"https://{host}/api/v1/models?query=demo",
        f"https://{host}/api/download/models/9000000001",
        f"base = 'https://{host}'",
        f"https://{host}/api/v1",
        f"https://{host}/" + "models/9000000001",
        f"https://{host}/" + "user/demo_author",
        f"{host}/api/" + "trpc/demo.getInfinite",
        f"https://image.{host}/demo/1.png",
    ]
    (tmp_path / "docs").mkdir()
    page = tmp_path / "docs" / "demo.md"
    page.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert [hit.split(": ", 1)[0] for hit in bad_urls(page, tmp_path)] == [
        "docs/demo.md:5", "docs/demo.md:6", "docs/demo.md:7", "docs/demo.md:8"]
    assert scanned_files(tmp_path) == [page]
