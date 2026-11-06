"""_common.civitai on hand-written answers served from 127.0.0.1: URLs, rate limit, backoff, cache, licence, download.

Every answer here is written by hand (ids >= 9000000000, the uploader `demo_author`, the description
`<p>demo</p>`); no test reaches the network (`no_network`). A recording opener sends the module's real public-API
URLs to a local server instead, so every test also asserts (in the `client` fixture's teardown) that each URL
the module built starts with /api/v1/ or /api/download/. The fake clock records sleeps, so no test waits.
"""

import hashlib
import json
import re
import threading
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import jsonschema
import pytest

from easel._common import civitai, config, store

pytestmark = pytest.mark.usefixtures("no_network")

REPO = Path(__file__).resolve().parents[2]
HOST = "https://civitai.com/"
MODEL_ID = 9000000001
VERSION_ID = 9000000002
FILE_ID = 9000000003
FILE_BYTES = bytes((i * 7 + 3) % 251 for i in range(40_000))
FILE_SHA = hashlib.sha256(FILE_BYTES).hexdigest()
FILE_NAME = "demo_robot.safetensors"
TOKEN = "demo-token-not-a-secret"
GIB = 1 << 30
_RESOURCE_SCHEMA = json.loads((REPO / "schemas" / "resource.schema.json").read_text(encoding="utf-8"))
LICENCE_SCHEMA = jsonschema.Draft202012Validator({"$defs": _RESOURCE_SCHEMA["$defs"], "$ref": "#/$defs/licence"})


# --- hand-written records ----------------------------------------------------------------------------------------

def version_record(**over: object) -> dict:
    record = {
        "id": VERSION_ID,
        "modelId": MODEL_ID,
        "name": "v1.0",
        "baseModel": "Illustrious",
        "trainedWords": ["dmrbt"],
        "availability": "Public",
        "publishedAt": "2030-01-01T00:00:00.000Z",
        "nsfwLevel": 1,
        "description": "<p>demo</p>",
        "images": [{"url": "https://example.invalid/preview.png", "nsfwLevel": 1}],
        "files": [{
            "id": FILE_ID,
            "name": FILE_NAME,
            "sizeKB": len(FILE_BYTES) / 1024,
            "type": "Model",
            "primary": True,
            "metadata": {"format": "SafeTensor", "fp": "fp16", "size": "pruned"},
            "hashes": {"SHA256": FILE_SHA.upper()},
            "downloadUrl": f"{civitai.DOWNLOAD_ROOT}models/{VERSION_ID}",
        }],
    }
    record.update(over)
    return record


def model_record(**over: object) -> dict:
    record = {
        "id": MODEL_ID,
        "name": "Demo Robot LoRA",
        "type": "LORA",
        "nsfw": False,
        "nsfwLevel": 1,
        "description": "<p>demo</p>",
        "creator": {"username": "demo_author", "image": "https://example.invalid/avatar.png"},
        "allowNoCredit": False,
        "allowCommercialUse": ["Image", "RentCivit"],
        "allowDerivatives": True,
        "allowDifferentLicense": False,
        "tags": ["robot"],
        "modelVersions": [version_record()],
    }
    record.update(over)
    return record


def file_summary(**over: object) -> dict:
    summary = civitai.summarize_file(version_record()["files"][0], VERSION_ID)
    summary.update(over)
    return summary


# --- a local stand-in for the site -------------------------------------------------------------------------------

@dataclass
class Reply:
    status: int = 200
    headers: dict = field(default_factory=dict)
    body: bytes = b""
    send_only: int | None = None  # send this many body bytes after a full Content-Length, then close


def js(obj: object, status: int = 200, **headers: str) -> Reply:
    return Reply(status, {"Content-Type": "application/json", **headers}, json.dumps(obj).encode("utf-8"))


def file_reply(data: bytes = FILE_BYTES, *, honour_range: bool = True):
    """A route serving `data`, with HTTP Range support unless `honour_range` is false."""

    def route(headers: dict) -> Reply:
        wanted = headers.get("Range")
        if honour_range and wanted:
            start = int(re.fullmatch(r"bytes=(\d+)-", wanted).group(1))
            if start >= len(data):
                return Reply(416, {"Content-Range": f"bytes */{len(data)}"})
            return Reply(206, {"Content-Type": "application/octet-stream",
                               "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}, data[start:])
        return Reply(200, {"Content-Type": "application/octet-stream"}, data)

    return route


class Site:
    """Routes by URL path: a Reply, a list of Replies used in turn (the last one repeats), or a function."""

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.requests: list[tuple[str, dict]] = []
        self.url = ""

    def reply_for(self, path: str, headers: dict) -> Reply:
        route = self.routes.get(urlsplit(path).path)
        if route is None:
            return js({"error": "not mocked"}, 404)
        if callable(route):
            return route(headers)
        if isinstance(route, list):
            return route.pop(0) if len(route) > 1 else route[0]
        return route

    def paths(self) -> list[str]:
        return [urlsplit(p).path for p, _ in self.requests]


@pytest.fixture
def site():
    s = Site()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            headers = self.headers  # case-insensitive: urllib sends `User-agent`, a test may ask for `User-Agent`
            s.requests.append((self.path, headers))
            reply = s.reply_for(self.path, headers)
            self.send_response(reply.status)
            for key, value in reply.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body if reply.send_only is None else reply.body[: reply.send_only])

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    s.url = f"http://127.0.0.1:{server.server_address[1]}/"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield s
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class Recorder:
    """The opener the module calls: records the URL it built, then sends the request to the local site."""

    def __init__(self, site: Site) -> None:
        self.site = site
        self.urls: list[str] = []
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, req: urllib.request.Request, timeout: float | None = None):
        self.urls.append(req.full_url)
        assert req.full_url.startswith(HOST)
        local = urllib.request.Request(self.site.url + req.full_url[len(HOST):], headers=dict(req.headers),
                                       method=req.get_method())
        for key, value in req.unredirected_hdrs.items():
            local.add_unredirected_header(key, value)
        return self._opener.open(local, timeout=timeout)


class FakeClock:
    """monotonic, time and sleep for the module: sleeping only advances the clock and records the wait."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.t

    def time(self) -> float:
        return 1_900_000_000.0 + self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.t += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(site, clock, tmp_path):
    """A factory of clients wired to the local site; on teardown every URL they built must be public-API."""
    made: list[civitai.Client] = []

    def make(**kw: object) -> civitai.Client:
        kw.setdefault("cache_dir", tmp_path / "cache")
        kw.setdefault("disk_floor_gb", 0)
        made_client = civitai.Client(opener=Recorder(site), sleep=clock.sleep, monotonic=clock.monotonic,
                                     clock=clock.time, **kw)
        made.append(made_client)
        return made_client

    yield make
    for c in made:
        for url in c.http._open.urls:
            assert url.startswith(civitai.PUBLIC_PREFIXES), url


def built_urls(c: civitai.Client) -> list[str]:
    return c.http._open.urls


# --- URLs --------------------------------------------------------------------------------------------------------

def test_every_url_builder_stays_under_the_public_prefixes():
    urls = [
        civitai.model_url(MODEL_ID),
        civitai.version_url(VERSION_ID),
        civitai.by_hash_url(FILE_SHA.upper()),
        civitai.download_url(VERSION_ID),
        civitai.search_url("demo robot", type_="lora", limit=5, cursor="c-2", sort="Newest",
                           base_models=["Illustrious"]),
        civitai.search_url(),
        file_summary()["download_url"],
    ]
    assert civitai.PUBLIC_PREFIXES == ("https://civitai.com/api/v1/", "https://civitai.com/api/download/")
    for url in urls:
        assert url.startswith(civitai.PUBLIC_PREFIXES), url
    assert civitai.by_hash_url(FILE_SHA.upper()).endswith(FILE_SHA)  # always lower case


def test_urls_outside_the_public_api_are_refused():
    http = civitai.Client().http
    internal = HOST + "api/" + "tr" + "pc/" + "demo.getInfinite"  # the site's own endpoints, built at run time
    page = HOST + "mod" + "els/" + str(MODEL_ID)
    for url in (internal, page, "http://civitai.com/api/v1/models", "https://example.invalid/api/v1/models", None):
        with pytest.raises(civitai.Refused) as caught:
            http.check_url(url)
        assert caught.value.code == 2


def test_the_module_source_names_no_other_civitai_url():
    text = (REPO / "src" / "easel" / "_common" / "civitai.py").read_text(encoding="utf-8")
    shapes = re.findall(r"civitai\.com/[^\s\"'`)]*", text)
    assert shapes, "the module should name its two public prefixes"
    for shape in shapes:
        assert shape.startswith(("civitai.com/api/v1/", "civitai.com/api/download/")), shape
    assert ("tr" + "pc") not in text.lower()


@pytest.mark.parametrize("call", [
    lambda: civitai.model_url("abc"),
    lambda: civitai.model_url(0),
    lambda: civitai.version_url(-3),
    lambda: civitai.by_hash_url("not-a-hash"),
    lambda: civitai.search_url(limit=0),
    lambda: civitai.search_url(limit=101),
    lambda: civitai.site_type("bogus"),
    lambda: civitai.safe_file_name("../escape.safetensors"),
    lambda: civitai.safe_file_name("sub/dir.safetensors"),
    lambda: civitai.safe_file_name("a:b.safetensors"),
    lambda: civitai.safe_file_name(".hidden"),
    lambda: civitai.safe_file_name(""),
])
def test_bad_ids_types_and_names_are_refused(call):
    with pytest.raises(civitai.Refused) as caught:
        call()
    assert caught.value.code == 2


def test_type_mapping_both_ways():
    assert civitai.site_type("lora") == "LORA"
    assert civitai.site_type("embedding") == "TextualInversion"
    assert civitai.site_type("LoCon") == "LoCon"
    assert civitai.resource_type("DoRA") == "lycoris"
    assert civitai.resource_type("Wildcards") == "other"


# --- records, licence, rating ------------------------------------------------------------------------------------

def test_model_summary_keeps_registry_facts_only(site, client, frozen_clock):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record())
    c = client()
    summary = c.summary(MODEL_ID)
    assert (summary["site"], summary["model_id"], summary["type"], summary["site_type"]) == (
        "civitai", str(MODEL_ID), "lora", "LORA")
    assert summary["rating"] == "sfw"
    assert summary["api_url"] == f"{civitai.API_ROOT}models/{MODEL_ID}"
    version = summary["versions"][0]
    assert version["trigger_words"] == ["dmrbt"]
    assert version["base_model"] == "Illustrious"
    assert version["api_url"] == f"{civitai.API_ROOT}model-versions/{VERSION_ID}"
    primary = civitai.primary_file(version)
    assert primary["sha256"] == FILE_SHA
    assert primary["bytes"] == len(FILE_BYTES)
    assert primary["download_url"].startswith(civitai.DOWNLOAD_ROOT)
    assert summary["licence"]["checked_at"] == "2030-01-01"
    LICENCE_SCHEMA.validate(summary["licence"])
    text = json.dumps(summary)
    for dropped in ("demo_author", "<p>demo</p>", "preview.png", "avatar.png", '"creator":', '"description":',
                    '"images":', '"tags":'):
        assert dropped not in text, dropped
    path, headers = site.requests[0]
    assert headers["Accept"] == "application/json"
    assert headers["User-Agent"].startswith("easel/")
    assert "Authorization" not in headers  # no token configured, none sent


@pytest.mark.parametrize("flags, expected", [
    ({"allowCommercialUse": ["Image", "RentCivit"]}, "images"),
    ({"allowCommercialUse": "Sell"}, "sell"),
    ({"allowCommercialUse": ["Image", "Rent", "Sell"]}, "sell"),
    ({"allowCommercialUse": ["Rent"]}, "rent"),
    ({"allowCommercialUse": []}, "none"),
    ({"allowCommercialUse": "None"}, "none"),
    ({"allowCommercialUse": None}, None),
])
def test_commercial_use_is_the_highest_rung_granted(flags, expected):
    licence = civitai.licence_from({"allowDerivatives": True, **flags}, checked_at="2030-01-01")
    assert licence["commercial_use"] == expected
    LICENCE_SCHEMA.validate(licence)


def test_licence_flags_map_to_the_resource_block():
    licence = civitai.licence_from(model_record(), version_record(), checked_at="2030-01-01")
    assert licence == {
        "status": "unverified", "name": None, "spdx": None, "url": None,
        "commercial_use": "images", "credit_required": True, "derivatives_allowed": True, "relicense_allowed": False,
        "redistributable": None, "gated": False, "source": "civitai-api", "checked_at": "2030-01-01",
        "note": "RentCivit: renting the model is allowed only through the site's own generator",
    }
    LICENCE_SCHEMA.validate(licence)
    free = civitai.licence_from({"allowNoCredit": True, "allowDifferentLicense": True})
    assert (free["credit_required"], free["relicense_allowed"], free["commercial_use"]) == (False, True, None)


def test_no_flags_is_unknown_and_early_access_is_gated():
    unknown = civitai.licence_from({})
    assert unknown["status"] == "unknown"
    assert all(unknown[k] is None for k in ("commercial_use", "credit_required", "derivatives_allowed",
                                            "relicense_allowed", "gated", "redistributable"))
    early = civitai.licence_from(model_record(), version_record(availability="EarlyAccess"))
    assert (early["gated"], early["redistributable"]) == (True, False)
    assert "EarlyAccess" in early["note"]
    odd = civitai.licence_from({"allowDerivatives": "yes", "allowCommercialUse": ["Image", "Lease"]})
    assert odd["derivatives_allowed"] is None and odd["commercial_use"] == "images"
    assert "Lease" in odd["note"] and "allowDerivatives" in odd["note"]
    for block in (unknown, early, odd):
        LICENCE_SCHEMA.validate(block)


@pytest.mark.parametrize("record, rating", [
    ({"nsfwLevel": 1}, "sfw"),
    ({"nsfwLevel": 2}, "suggestive"),
    ({"nsfwLevel": 3}, "suggestive"),
    ({"nsfwLevel": 4}, "explicit"),
    ({"nsfwLevel": 31}, "explicit"),
    ({"nsfw": False}, "sfw"),
    ({"nsfw": True}, "unknown"),
    ({"nsfw": True, "nsfwLevel": 1}, "unknown"),
    ({}, "unknown"),
    ({"nsfwLevel": True}, "unknown"),
])
def test_rating_is_never_sfw_without_evidence(record, rating):
    assert civitai.rating_from(record) == rating


# --- search ------------------------------------------------------------------------------------------------------

def test_search_sends_the_query_and_follows_only_its_own_cursor(site, client):
    answer = {"items": [model_record()],
              "metadata": {"nextCursor": "demo-cursor-2", "nextPage": "https://example.invalid/next"}}
    site.routes["/api/v1/models"] = js(answer)
    c = client()
    page = c.search("demo robot", type_="lora", limit=5)
    assert [item["model_id"] for item in page.items] == [str(MODEL_ID)]
    assert page.next_cursor == "demo-cursor-2"
    query = parse_qs(urlsplit(built_urls(c)[0]).query)
    assert query == {"limit": ["5"], "query": ["demo robot"], "types": ["LORA"], "nsfw": ["false"]}
    c.search("demo robot", type_="lora", limit=5, cursor=page.next_cursor)
    assert parse_qs(urlsplit(built_urls(c)[1]).query)["cursor"] == ["demo-cursor-2"]
    assert site.paths() == ["/api/v1/models", "/api/v1/models"]  # never the answer's own next-page URL
    assert "demo_author" not in json.dumps(page.items)


def test_search_answer_must_be_an_object(site, client):
    site.routes["/api/v1/models"] = js(["not", "an", "object"])
    with pytest.raises(civitai.NetError):
        client().search("demo")


# --- rate limit, backoff, cache ----------------------------------------------------------------------------------

def test_at_most_one_request_per_second(site, client, clock):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record())
    site.routes[f"/api/v1/model-versions/{VERSION_ID}"] = js(version_record())
    site.routes[f"/api/v1/model-versions/by-hash/{FILE_SHA}"] = js(version_record())
    c = client(cache_hours=0)
    c.model(MODEL_ID)
    c.version(VERSION_ID)
    c.version_by_hash(FILE_SHA)
    assert clock.slept == [1.0, 1.0]
    clock.slept.clear()
    faster = client(cache_hours=0, requests_per_s=4)
    faster.model(MODEL_ID)
    faster.model(MODEL_ID)
    assert clock.slept == [0.25]


def test_429_waits_the_retry_after_then_succeeds(site, client, clock):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = [
        js({"error": "slow down"}, 429, **{"Retry-After": "7"}),
        js({"error": "slow down"}, 503, **{"Retry-After": "3"}),
        js(model_record()),
    ]
    record = client().model(MODEL_ID)
    assert record["id"] == MODEL_ID
    assert clock.slept == [7.0, 3.0]
    assert len(site.requests) == 3


def test_429_without_retry_after_backs_off_then_gives_up(site, client, clock):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js({"error": "slow down"}, 429)
    with pytest.raises(civitai.RateLimited) as caught:
        client().model(MODEL_ID)
    assert caught.value.code == 3
    assert clock.slept == [5.0, 10.0, 20.0, 40.0]
    assert len(site.requests) == 1 + civitai.MAX_RETRIES
    assert "do not loop" in caught.value.lines[0]


def test_429_asking_for_a_long_wait_gives_up_at_once(site, client, clock):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js({"error": "slow down"}, 429, **{"Retry-After": "3600"})
    with pytest.raises(civitai.RateLimited) as caught:
        client().model(MODEL_ID)
    assert clock.slept == []
    assert "3600 s" in caught.value.lines[0]


def test_retry_after_reads_seconds_and_http_dates():
    when = 1_900_000_000.0
    date = "Sun, 17 Mar 2030 17:46:40 GMT"  # = when
    assert civitai.retry_after_s("12") == 12.0
    assert civitai.retry_after_s(date, clock=lambda: when - 30) == pytest.approx(30.0)
    assert civitai.retry_after_s(date, clock=lambda: when + 30) == 0.0
    assert civitai.retry_after_s("soon") is None
    assert civitai.retry_after_s(None) is None


def test_answers_are_cached_for_24_hours(site, client, clock, tmp_path):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record())
    c = client()
    c.model(MODEL_ID)
    c.model(MODEL_ID)
    assert len(site.requests) == 1
    clock.t += 24 * 3600 - 2
    c.model(MODEL_ID)
    assert len(site.requests) == 1
    clock.t += 5
    c.model(MODEL_ID)
    assert len(site.requests) == 2
    c.model(MODEL_ID, fresh=True)
    assert len(site.requests) == 3
    cached = list((tmp_path / "cache").glob("*.json"))
    assert len(cached) == 1 and re.fullmatch(r"[0-9a-f]{32}\.json", cached[0].name)
    cached[0].write_text("{ torn", encoding="utf-8")  # a damaged entry is a miss, never an error
    c.model(MODEL_ID)
    assert len(site.requests) == 4


def test_http_errors_map_to_exit_codes(site, client):
    site.routes["/api/v1/models/9000000098"] = js({"error": "login"}, 401)
    c = client()
    with pytest.raises(civitai.NotFound) as missing:
        c.model(9000000099)
    with pytest.raises(civitai.LoginRequired) as login:
        c.model(9000000098)
    site.routes["/api/v1/models/9000000097"] = Reply(200, {"Content-Type": "text/html"}, b"<!doctype html><p>busy")
    with pytest.raises(civitai.NetError) as not_json:
        c.model(9000000097)
    assert (missing.value.code, login.value.code, not_json.value.code) == (2, 2, 1)
    assert "never logs in" in login.value.lines[0]


def test_token_goes_in_an_unredirected_header_only(site, client, tmp_path):
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record())
    site.routes[f"/api/download/models/{VERSION_ID}"] = Reply(302, {"Location": "/files/demo_robot.bin"})
    site.routes["/files/demo_robot.bin"] = file_reply()
    c = client(token=TOKEN)
    c.model(MODEL_ID)
    c.download(file_summary(), tmp_path / "models")
    with pytest.raises(civitai.NotFound) as missing:
        c.model(9000000099)
    by_path = {urlsplit(p).path: h for p, h in site.requests}
    assert by_path[f"/api/v1/models/{MODEL_ID}"]["Authorization"] == f"Bearer {TOKEN}"
    assert by_path[f"/api/download/models/{VERSION_ID}"]["Authorization"] == f"Bearer {TOKEN}"
    assert "Authorization" not in by_path["/files/demo_robot.bin"]  # the file host never sees the token
    assert all(TOKEN not in url for url in built_urls(c))
    assert TOKEN not in repr(c) and "token=set" in repr(c)
    assert TOKEN not in str(missing.value) and TOKEN not in " ".join(missing.value.lines)
    for cached in (tmp_path / "cache").glob("*"):
        assert TOKEN not in cached.read_text(encoding="utf-8")


def test_from_config_reads_search_limits_and_the_instance_cache(tmp_path):
    (tmp_path / "easel.toml").write_text(
        'schema = "easel/config/1"\n[search]\ncivitai_token_env = "DEMO_CIVITAI_TOKEN"\ncache_hours = 12\n'
        'requests_per_s = 2\n[limits]\ndisk_floor_gb = 5\n', encoding="utf-8")
    cfg = config.load("easel", tmp_path, env={})
    c = civitai.Client.from_config(cfg, env={"DEMO_CIVITAI_TOKEN": TOKEN})
    assert c.http.has_token and TOKEN not in repr(c)
    assert c.http.cache.folder == tmp_path / "cache" / "civitai"
    assert c.http.cache.seconds == 12 * 3600
    assert c.http.limiter.interval == 0.5
    assert c.disk_floor_gb == 5.0
    (tmp_path / "easel.toml").write_text('schema = "easel/config/1"\n', encoding="utf-8")
    plain = civitai.Client.from_config(config.load("easel", tmp_path, env={}),
                                       env={civitai.DEFAULT_TOKEN_ENV: TOKEN})
    assert plain.http.has_token
    assert (plain.http.cache.seconds, plain.http.limiter.interval, plain.disk_floor_gb) == (24 * 3600, 1.0, 20.0)
    assert civitai.DEFAULT_DISK_FLOOR_GB == 20.0  # a neutral shipped default; an instance sets its own floor
    assert not civitai.Client.from_config(config.load("easel", tmp_path, env={}), env={}).http.has_token


# --- downloads ---------------------------------------------------------------------------------------------------

DOWNLOAD_PATH = f"/api/download/models/{VERSION_ID}"


def test_download_verifies_the_sha256_and_renames_into_place(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    seen: list[tuple[int, int | None]] = []
    result = client().download(file_summary(), tmp_path / "models", progress=lambda got, total: seen.append((got, total)))
    assert result.path == tmp_path / "models" / FILE_NAME
    assert result.path.read_bytes() == FILE_BYTES
    assert (result.sha256, result.checked, result.bytes, result.resumed_from) == (FILE_SHA, "sha256", len(FILE_BYTES), 0)
    assert not (tmp_path / "models" / (FILE_NAME + ".part")).exists()
    assert seen[-1] == (len(FILE_BYTES), len(FILE_BYTES))


def test_download_resumes_with_a_range_request(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / (FILE_NAME + ".part")).write_bytes(FILE_BYTES[:1000])
    result = client().download(file_summary(), folder)
    assert result.resumed_from == 1000
    assert site.requests[0][1]["Range"] == "bytes=1000-"
    assert result.path.read_bytes() == FILE_BYTES


def test_a_server_ignoring_the_range_restarts_from_zero(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply(honour_range=False)
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / (FILE_NAME + ".part")).write_bytes(b"\x00" * 1000)
    result = client().download(file_summary(), folder)
    assert result.resumed_from == 0
    assert result.path.read_bytes() == FILE_BYTES
    assert any("ignored the resume range" in n for n in result.notes)


def test_a_dropped_connection_keeps_the_part_and_the_next_run_resumes(site, client, tmp_path):
    folder = tmp_path / "models"
    site.routes[DOWNLOAD_PATH] = Reply(200, {"Content-Type": "application/octet-stream"}, FILE_BYTES, send_only=15_000)
    c = client()
    with pytest.raises(civitai.NetError) as caught:
        c.download(file_summary(), folder)
    assert "resume" in caught.value.lines[0]
    assert (folder / (FILE_NAME + ".part")).read_bytes() == FILE_BYTES[:15_000]
    site.routes[DOWNLOAD_PATH] = file_reply()
    result = c.download(file_summary(), folder)
    assert result.resumed_from == 15_000
    assert result.path.read_bytes() == FILE_BYTES


def test_a_complete_part_answered_416_is_finished(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / (FILE_NAME + ".part")).write_bytes(FILE_BYTES)
    result = client().download(file_summary(), folder)
    assert result.resumed_from == len(FILE_BYTES)
    assert result.path.read_bytes() == FILE_BYTES


def test_a_login_page_is_refused_and_never_saved(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = Reply(302, {"Location": "/login?next=demo"})
    site.routes["/login"] = Reply(200, {"Content-Type": "text/html; charset=utf-8"},
                                  b"<!DOCTYPE html><html><body>Sign in</body></html>")
    folder = tmp_path / "models"
    with pytest.raises(civitai.LoginRequired) as caught:
        client().download(file_summary(), folder)
    assert caught.value.code == 2
    assert "never logs in" in caught.value.lines[0] and "rescan the model registry" in caught.value.lines[0]
    assert not (folder / FILE_NAME).exists() and not (folder / (FILE_NAME + ".part")).exists()


def test_the_login_hint_names_no_tool_and_a_caller_may_give_its_own(site, client, tmp_path):
    """This module serves both tools, so its default fix line names no command; the caller knows its own."""
    site.routes[DOWNLOAD_PATH] = Reply(403, {"Content-Type": "application/json"}, b'{"error": "login"}')
    with pytest.raises(civitai.LoginRequired) as caught:
        client().download(file_summary(), tmp_path / "models")
    line = caught.value.lines[0]
    assert line.endswith(civitai.LOGIN_HINT) and "`" not in line  # no command of either tool is quoted
    own = client(login_hint=": then run the demo rescan")
    with pytest.raises(civitai.LoginRequired) as caught:
        own.download(file_summary(), tmp_path / "models")
    assert caught.value.lines[0].endswith("this tool never logs in: then run the demo rescan")


def test_the_final_rename_retries_and_then_exits_4(site, client, tmp_path, monkeypatch, capsys):
    """An antivirus holding the fresh file: the rename waits like every atomic write, then the STUCK path."""
    site.routes[DOWNLOAD_PATH] = file_reply()
    calls = []

    def refuse(src, dst):
        calls.append(dst)
        raise PermissionError(13, "held by a scanner")

    monkeypatch.setattr(civitai.os, "replace", refuse)
    monkeypatch.setattr(store, "REPLACE_BUDGET_S", 0.2)
    with pytest.raises(store.Held) as caught:
        client().download(file_summary(), tmp_path / "models")
    assert caught.value.code == 4 and len(calls) >= 2
    assert "STUCK: another process holds" in capsys.readouterr().err
    assert (tmp_path / "models" / (FILE_NAME + ".part")).read_bytes() == FILE_BYTES  # kept: the next run resumes


def test_a_web_page_without_an_html_type_is_still_sniffed(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = Reply(200, {"Content-Type": "application/octet-stream"},
                                       b"\n  <html><body>Sign in</body></html>")
    with pytest.raises(civitai.LoginRequired):
        client().download(file_summary(), tmp_path / "models")


def test_a_stale_web_page_part_is_discarded(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / (FILE_NAME + ".part")).write_bytes(b"<!doctype html><title>Sign in</title>")
    result = client().download(file_summary(), folder)
    assert result.resumed_from == 0 and "Range" not in site.requests[0][1]
    assert result.path.read_bytes() == FILE_BYTES
    assert any("web page" in n for n in result.notes)


def test_a_json_message_instead_of_the_file_is_an_error(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = js({"error": "demo file not available"})
    with pytest.raises(civitai.NetError) as caught:
        client().download(file_summary(), tmp_path / "models")
    assert "demo file not available" in caught.value.lines[0]


def test_a_sha256_mismatch_deletes_the_part(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply(FILE_BYTES[::-1])
    folder = tmp_path / "models"
    with pytest.raises(civitai.ShaMismatch) as caught:
        client().download(file_summary(), folder)
    assert caught.value.code == 1
    assert not (folder / FILE_NAME).exists() and not (folder / (FILE_NAME + ".part")).exists()


def test_the_disk_floor_refuses_before_the_first_byte(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    folder = tmp_path / "models"
    with pytest.raises(civitai.DiskFloor) as caught:
        client(disk_floor_gb=100, disk_free=lambda _p: 50 * GIB).download(file_summary(), folder)
    assert caught.value.code == 3
    assert "limits.disk_floor_gb" in caught.value.lines[0]
    assert not (folder / (FILE_NAME + ".part")).exists()
    roomy = client(disk_floor_gb=100, disk_free=lambda _p: 200 * GIB).download(file_summary(), folder)
    assert roomy.path.read_bytes() == FILE_BYTES


def test_an_existing_file_is_never_overwritten(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / FILE_NAME).write_bytes(FILE_BYTES)
    c = client()
    again = c.download(file_summary(), folder)
    assert again.already_present and again.sha256 == FILE_SHA
    other = folder / "other"
    other.mkdir()
    (other / FILE_NAME).write_bytes(b"someone else's file")
    with pytest.raises(civitai.AlreadyExists):
        c.download(file_summary(), other)
    assert (other / FILE_NAME).read_bytes() == b"someone else's file"
    assert site.requests == []  # neither case asked the site


def test_a_file_without_a_checksum_needs_an_explicit_yes(site, client, tmp_path):
    site.routes[DOWNLOAD_PATH] = file_reply()
    c = client()
    with pytest.raises(civitai.Refused):
        c.download(file_summary(sha256=None), tmp_path / "models")
    result = c.download(file_summary(sha256=None), tmp_path / "models", allow_unverified=True)
    assert result.checked == "none" and result.sha256 == FILE_SHA


def test_a_download_url_outside_the_download_root_is_not_used():
    odd = dict(version_record()["files"][0], downloadUrl="https://example.invalid/file", primary=False)
    assert civitai.summarize_file(odd, VERSION_ID)["download_url"] is None
    primary = dict(odd, primary=True)
    assert civitai.summarize_file(primary, VERSION_ID)["download_url"] == civitai.download_url(VERSION_ID)
    with pytest.raises(civitai.Refused):
        civitai.Client().download(civitai.summarize_file(odd, VERSION_ID), Path("unused"))
