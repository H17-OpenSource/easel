"""_common.hf on hand-written answers from an injected opener: URLs, info, files, card licence, search, downloads.

No test reaches the network (`no_network`); the opener answers from a table keyed by URL path and records every
request, so the `hub` fixture's teardown asserts that each URL the module built is a public API record URL or a
file `resolve` URL on huggingface.co. The repo is the hand-written `demo_author/demo-robot-lora`.
"""

import email.message
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
import urllib.response
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit

import jsonschema
import pytest

from easel._common import civitai, config, hf

pytestmark = pytest.mark.usefixtures("no_network")

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "demo_author/demo-robot-lora"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
LORA_BYTES = bytes((i * 13 + 5) % 253 for i in range(20_000))
LORA_SHA = hashlib.sha256(LORA_BYTES).hexdigest()
CONFIG_BYTES = b'{"demo": true}\n'
TOKEN = "demo-hf-token-not-a-secret"
EMPTY_BLOB = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"  # git's id of the empty blob, a fixed vector
_RESOURCE_SCHEMA = json.loads((ROOT / "schemas" / "resource.schema.json").read_text(encoding="utf-8"))
LICENCE_SCHEMA = jsonschema.Draft202012Validator({"$defs": _RESOURCE_SCHEMA["$defs"], "$ref": "#/$defs/licence"})
PUBLIC_SHAPE = re.compile(r"^https://huggingface\.co/(?:api/models(?:/|\?)|[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/resolve/)")


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def info_record(**over: object) -> dict:
    record = {
        "id": REPO_ID,
        "modelId": REPO_ID,
        "author": "demo_author",
        "sha": COMMIT,
        "private": False,
        "gated": False,
        "disabled": False,
        "lastModified": "2030-01-01T00:00:00.000Z",
        "pipeline_tag": "text-to-image",
        "library_name": "diffusers",
        "tags": ["lora", "license:apache-2.0", "base_model:adapter:demo_author/demo-base"],
        "cardData": {"license": "apache-2.0", "base_model": "demo_author/demo-base"},
        "siblings": [
            {"rfilename": "demo_robot.safetensors", "size": len(LORA_BYTES), "blobId": "f" * 40,
             "lfs": {"sha256": LORA_SHA.upper(), "size": len(LORA_BYTES), "pointerSize": 134}},
            {"rfilename": "config.json", "size": len(CONFIG_BYTES), "blobId": blob_id(CONFIG_BYTES)},
            {"rfilename": "notes/unlisted.txt", "size": 3},
        ],
    }
    record.update(over)
    return record


class Hub:
    """The opener the module calls: answers from `routes` (path -> (status, headers, body) or a function)."""

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.requests: list[urllib.request.Request] = []

    def __call__(self, req: urllib.request.Request, timeout: float | None = None):
        self.requests.append(req)
        path = urlsplit(req.full_url).path
        route = self.routes.get(path)
        if route is None:
            status, headers, body = 404, {"Content-Type": "application/json"}, b'{"error": "Repository not found"}'
        elif callable(route):
            status, headers, body = route(req)
        else:
            status, headers, body = route
        message = email.message.Message()
        for key, value in headers.items():
            message[key] = value
        if status >= 400:
            raise urllib.error.HTTPError(req.full_url, status, "demo", message, io.BytesIO(body))
        return urllib.response.addinfourl(io.BytesIO(body), message, req.full_url, status)

    def urls(self) -> list[str]:
        return [r.full_url for r in self.requests]


def js(obj: object, status: int = 200) -> tuple[int, dict, bytes]:
    return status, {"Content-Type": "application/json"}, json.dumps(obj).encode("utf-8")


def served(data: bytes, content_type: str = "application/octet-stream"):
    """A route serving `data`, honouring a `Range: bytes=N-` request with 206."""

    def route(req: urllib.request.Request) -> tuple[int, dict, bytes]:
        wanted = req.get_header("Range")
        if wanted:
            start = int(re.fullmatch(r"bytes=(\d+)-", wanted).group(1))
            return 206, {"Content-Type": content_type,
                         "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}",
                         "Content-Length": str(len(data) - start)}, data[start:]
        return 200, {"Content-Type": content_type, "Content-Length": str(len(data))}, data

    return route


INFO_PATH = f"/api/models/{REPO_ID}"
LORA_PATH = f"/{REPO_ID}/resolve/{COMMIT}/demo_robot.safetensors"
CONFIG_PATH = f"/{REPO_ID}/resolve/{COMMIT}/config.json"


@pytest.fixture
def hub():
    h = Hub()
    yield h
    for url in h.urls():
        assert url.startswith(hf.PUBLIC_PREFIXES) and PUBLIC_SHAPE.match(url), url


@pytest.fixture
def client(hub, tmp_path):
    slept: list[float] = []
    made = hf.Client(opener=hub, cache_dir=tmp_path / "cache", disk_floor_gb=0, sleep=slept.append,
                     monotonic=lambda: 0.0)
    made.slept = slept
    return made


# --- URLs --------------------------------------------------------------------------------------------------------

def test_url_builders_stay_on_the_public_api():
    assert hf.info_url(REPO_ID) == f"https://huggingface.co/api/models/{REPO_ID}?blobs=true"
    pr = hf.info_url(REPO_ID, "refs/pr/1")  # the revision is one path segment: its slashes are encoded
    assert pr == f"https://huggingface.co/api/models/{REPO_ID}/revision/{quote('refs/pr/1', safe='')}?blobs=true"
    assert "refs/pr" not in pr
    spaced = hf.resolve_url(REPO_ID, "unet/demo model.safetensors", COMMIT)  # folders kept, each part quoted
    assert spaced == f"https://huggingface.co/{REPO_ID}/resolve/{COMMIT}/unet/{quote('demo model.safetensors')}"
    assert " " not in spaced
    assert hf.blob_url(REPO_ID, "LICENSE.md", COMMIT) == f"https://huggingface.co/{REPO_ID}/blob/{COMMIT}/LICENSE.md"
    query = parse_qs(urlsplit(hf.search_url("demo robot", limit=7, filters=["lora"], sort="downloads")).query)
    assert query == {"limit": ["7"], "search": ["demo robot"], "filter": ["lora"], "sort": ["downloads"]}
    for url in (hf.info_url(REPO_ID), hf.resolve_url(REPO_ID, "a.bin"), hf.search_url()):
        assert PUBLIC_SHAPE.match(url), url


@pytest.mark.parametrize("call", [
    lambda: hf.info_url("../escape"),
    lambda: hf.info_url("a/b/c"),
    lambda: hf.info_url("demo--author/x"),
    lambda: hf.info_url(""),
    lambda: hf.info_url("demo author/x"),
    lambda: hf.resolve_url(REPO_ID, "../x.bin"),
    lambda: hf.resolve_url(REPO_ID, "/abs.bin"),
    lambda: hf.resolve_url(REPO_ID, "a\\b.bin"),
    lambda: hf.resolve_url(REPO_ID, "a.bin", "main branch"),
    lambda: hf.search_url(limit=0),
    lambda: hf.Client().http.check_url("https://example.invalid/api/models"),
])
def test_bad_ids_paths_and_hosts_are_refused(call):
    with pytest.raises(hf.Refused) as caught:
        call()
    assert caught.value.code == 2


# --- info, files, licence ----------------------------------------------------------------------------------------

def test_summary_files_and_licence(hub, client, frozen_clock):
    hub.routes[INFO_PATH] = js(info_record())
    summary = client.summary(REPO_ID)
    assert (summary["site"], summary["model_id"], summary["revision"], summary["type"]) == (
        "huggingface", REPO_ID, COMMIT, "lora")
    assert summary["name"] == "demo-robot-lora" and summary["base_model"] == "demo_author/demo-base"
    assert summary["rating"] == "unknown"
    assert "author" not in summary
    licence = summary["licence"]
    assert (licence["status"], licence["name"], licence["spdx"], licence["gated"], licence["source"]) == (
        "unverified", "apache-2.0", "Apache-2.0", False, "hf-card")
    assert licence["checked_at"] == "2030-01-01"
    assert all(licence[k] is None for k in ("commercial_use", "credit_required", "derivatives_allowed",
                                            "relicense_allowed", "redistributable"))
    assert "read the licence text" in licence["note"]
    LICENCE_SCHEMA.validate(licence)
    lora, conf, unlisted = summary["files"]
    assert (lora["path"], lora["sha256"], lora["git_sha1"], lora["lfs"], lora["bytes"]) == (
        "demo_robot.safetensors", LORA_SHA, None, True, len(LORA_BYTES))
    assert (conf["sha256"], conf["git_sha1"], conf["lfs"]) == (None, blob_id(CONFIG_BYTES), False)
    assert (unlisted["sha256"], unlisted["git_sha1"]) == (None, None)
    assert hub.requests[0].get_header("Accept") == "application/json"
    assert client.files(REPO_ID) == summary["files"]
    assert len(hub.requests) == 1  # the second read came from the cache


@pytest.mark.parametrize("over, expected", [
    ({"cardData": {"license": "other", "license_name": "demo-open-licence", "license_link": "LICENSE.md"}},
     {"name": "demo-open-licence", "spdx": None, "url": f"https://huggingface.co/{REPO_ID}/blob/{COMMIT}/LICENSE.md",
      "status": "unverified"}),
    ({"cardData": {"license": "openrail++", "license_link": "https://example.invalid/demo-licence"}},
     {"name": "openrail++", "spdx": None, "url": "https://example.invalid/demo-licence", "status": "unverified"}),
    ({"cardData": {"license": ["mit", "cc-by-4.0"]}},
     {"name": "mit, cc-by-4.0", "spdx": None, "status": "unverified"}),
    ({"cardData": {}, "tags": ["license:cc-by-nc-4.0"]},
     {"name": "cc-by-nc-4.0", "spdx": "CC-BY-NC-4.0", "status": "unverified"}),
    ({"cardData": None, "tags": []}, {"name": None, "spdx": None, "url": None, "status": "unknown"}),
    ({"gated": "manual"}, {"gated": True}),
    ({"gated": "auto"}, {"gated": True}),
    ({"gated": None}, {"gated": None}),
])
def test_licence_variants(over, expected):
    licence = hf.licence_from(info_record(**over), checked_at="2030-01-01")
    assert {k: licence[k] for k in expected} == expected
    LICENCE_SCHEMA.validate(licence)


def test_notes_say_what_is_unclear():
    several = hf.licence_from(info_record(cardData={"license": ["mit", "cc-by-4.0"]}))
    assert "several licences" in several["note"]
    manual = hf.licence_from(info_record(gated="manual"))
    assert "granted by hand" in manual["note"]
    bad_link = hf.licence_from(info_record(cardData={"license": "other", "license_link": "../outside"}))
    assert bad_link["url"] is None and "not a usable path" in bad_link["note"]


def test_rating_and_type_hints():
    assert hf.rating_from(info_record(tags=["lora", hf.NOT_FOR_ALL_AUDIENCES])) == "explicit"
    assert hf.rating_from(info_record()) == "unknown"
    assert hf.resource_type(info_record(tags=["textual_inversion"])) == "embedding"
    assert hf.resource_type(info_record(tags=["controlnet"])) == "controlnet"
    assert hf.resource_type(info_record(tags=[])) == "checkpoint"
    assert hf.resource_type(info_record(tags=[], pipeline_tag="image-to-text")) == "other"


def test_git_blob_sha1_matches_git(tmp_path):
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert hf.git_blob_sha1(empty) == EMPTY_BLOB
    conf = tmp_path / "config.json"
    conf.write_bytes(CONFIG_BYTES)
    assert hf.git_blob_sha1(conf) == blob_id(CONFIG_BYTES)


# --- search ------------------------------------------------------------------------------------------------------

def test_search_summarizes_each_hit(hub, client):
    hits = [{"id": REPO_ID, "tags": ["lora", "license:mit"], "pipeline_tag": "text-to-image", "downloads": 3},
            {"id": "demo_author/demo-upscaler", "tags": ["license:apache-2.0"]}]
    hub.routes["/api/models"] = js(hits)
    found = client.search("demo robot", filters=["lora"], limit=5)
    assert [h["model_id"] for h in found] == [REPO_ID, "demo_author/demo-upscaler"]
    assert found[0]["licence"]["spdx"] == "MIT" and found[0]["files"] == []
    assert parse_qs(urlsplit(hub.urls()[0]).query) == {"limit": ["5"], "search": ["demo robot"], "filter": ["lora"]}
    hub.routes["/api/models"] = js({"not": "a list"})
    with pytest.raises(hf.NetError):
        client.search("demo other")


# --- downloads ---------------------------------------------------------------------------------------------------

def test_an_lfs_file_is_verified_by_sha256_and_pinned_to_the_commit(hub, client, tmp_path):
    hub.routes[INFO_PATH] = js(info_record())
    hub.routes[LORA_PATH] = served(LORA_BYTES)
    result = client.download(REPO_ID, "demo_robot.safetensors", tmp_path / "models")
    assert result.path.read_bytes() == LORA_BYTES
    assert (result.checked, result.sha256) == ("sha256", LORA_SHA)
    assert urlsplit(hub.urls()[-1]).path == LORA_PATH  # the commit the checksums came from, not "main"


def test_a_git_file_is_verified_by_its_blob_id(hub, client, tmp_path):
    hub.routes[INFO_PATH] = js(info_record())
    hub.routes[CONFIG_PATH] = served(CONFIG_BYTES, "application/json")  # a .json file is a file, not a message
    result = client.download(REPO_ID, "config.json", tmp_path / "models")
    assert result.path.read_bytes() == CONFIG_BYTES
    assert result.checked == "git blob sha1"
    assert result.sha256 == hashlib.sha256(CONFIG_BYTES).hexdigest()


def test_a_download_resumes_from_the_part(hub, client, tmp_path):
    hub.routes[INFO_PATH] = js(info_record())
    hub.routes[LORA_PATH] = served(LORA_BYTES)
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "demo_robot.safetensors.part").write_bytes(LORA_BYTES[:4096])
    result = client.download(REPO_ID, "demo_robot.safetensors", folder)
    assert result.resumed_from == 4096
    assert hub.requests[-1].get_header("Range") == "bytes=4096-"
    assert result.path.read_bytes() == LORA_BYTES


def test_a_mismatch_leaves_no_file(hub, client, tmp_path):
    hub.routes[INFO_PATH] = js(info_record())
    hub.routes[LORA_PATH] = served(LORA_BYTES[::-1])
    folder = tmp_path / "models"
    with pytest.raises(hf.ShaMismatch):
        client.download(REPO_ID, "demo_robot.safetensors", folder)
    assert list(folder.iterdir()) == []


def test_missing_files_and_checksums_are_refused(hub, client, tmp_path):
    hub.routes[INFO_PATH] = js(info_record())
    with pytest.raises(hf.NotFound):
        client.download(REPO_ID, "absent.safetensors", tmp_path)
    with pytest.raises(hf.Refused) as caught:
        client.download(REPO_ID, "notes/unlisted.txt", tmp_path)
    assert "no checksum" in caught.value.lines[0]


def test_a_gated_repo_is_login_required(hub, client, tmp_path):
    hub.routes[INFO_PATH] = js({"error": "Access to model is restricted"}, 401)
    with pytest.raises(hf.LoginRequired) as caught:
        client.info(REPO_ID)
    assert caught.value.code == 2
    assert "HF_TOKEN" in caught.value.lines[0] and "never logs in" in caught.value.lines[0]
    hub.routes[INFO_PATH] = js(info_record(gated="manual"))
    hub.routes[LORA_PATH] = (200, {"Content-Type": "text/html"}, b"<!doctype html><p>accept the terms")
    with pytest.raises(hf.LoginRequired):
        client.download(REPO_ID, "demo_robot.safetensors", tmp_path / "models", fresh=True)


def test_the_exceptions_are_civitais():
    assert hf.NetError is civitai.NetError and hf.LoginRequired is civitai.LoginRequired


# --- the token ---------------------------------------------------------------------------------------------------

def test_the_token_comes_from_hf_token_and_is_never_logged(hub, tmp_path, monkeypatch):
    (tmp_path / "easel.toml").write_text('schema = "easel/config/1"\n[limits]\ndisk_floor_gb = 0\n', encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    c = hf.Client.from_config(config.load("easel", tmp_path, env={}), opener=hub, sleep=lambda _s: None)
    assert c.http.has_token and c.http.cache.folder == tmp_path / "cache" / "huggingface"
    hub.routes[INFO_PATH] = js(info_record())
    hub.routes[LORA_PATH] = served(LORA_BYTES)
    c.download(REPO_ID, "demo_robot.safetensors", tmp_path / "models")
    with pytest.raises(hf.NotFound) as missing:
        c.info("demo_author/absent")
    for req in hub.requests:
        assert req.unredirected_hdrs.get("Authorization") == f"Bearer {TOKEN}"  # dropped by urllib on a redirect
        assert "Authorization" not in req.headers
        assert TOKEN not in req.full_url
    assert TOKEN not in repr(c) and TOKEN not in " ".join(missing.value.lines) and TOKEN not in str(missing.value)
    for cached in (tmp_path / "cache").rglob("*.json"):
        assert TOKEN not in cached.read_text(encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN")
    assert not hf.Client.from_config(config.load("easel", tmp_path, env={})).http.has_token
    named = hf.Client.from_config(config.load("easel", tmp_path, env={}), env={"DEMO_HF_TOKEN": TOKEN},
                                  requests_per_s=4)
    assert not named.http.has_token  # HF_TOKEN is the default name; another name must be configured
    (tmp_path / "easel.toml").write_text('schema = "easel/config/1"\n[search]\nhf_token_env = "DEMO_HF_TOKEN"\n',
                                         encoding="utf-8")
    renamed = hf.Client.from_config(config.load("easel", tmp_path, env={}), env={"DEMO_HF_TOKEN": TOKEN})
    assert renamed.http.has_token
