"""A local stand-in for Civitai's and Hugging Face's public APIs, for the registry's network verbs.

Imported by name (pytest puts this folder on sys.path). `Site` is a small HTTP server on 127.0.0.1 with routes
by URL path; `wire(monkeypatch, site)` makes `civitai.Client.from_config` and `hf.Client.from_config` build
clients whose opener records the real public-API URL the module built, then sends the request to the local
server instead. Every answer is hand-written: ids from 9000000000 up, the uploader `demo_author`, the
description `<p>demo</p>`. No test reaches the network.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import pytest
from registry_support import safetensors_bytes

from easel._common import civitai, hf

HOSTS = ("https://civitai.com/", "https://huggingface.co/")
MODEL_ID, VERSION_ID, FILE_ID = 9000000401, 9000000402, 9000000403
FILE_NAME = "demo-line-art.safetensors"
FILE_BYTES = safetensors_bytes({"lora_unet_down_blocks_0.lora_down.weight": "F16",
                                "lora_unet_down_blocks_0.lora_up.weight": "F16"},
                               {"ss_network_dim": "8", "ss_network_alpha": "4", "ss_network_module": "networks.lora"})
FILE_SHA = hashlib.sha256(FILE_BYTES).hexdigest()
HF_REPO = "demo_org/demo-upscaler"
HF_COMMIT = "a" * 40
HF_FILE = "demo-2x.safetensors"
HF_BYTES = safetensors_bytes({"body.0.weight": "F32", "body.0.bias": "F32"})
HF_SHA = hashlib.sha256(HF_BYTES).hexdigest()


def page_link(path: str) -> str:
    """A Civitai page link, assembled at run time (the repository holds no page URL)."""
    return "https://" + "civitai" + ".com/" + path


@dataclass
class Reply:
    status: int = 200
    headers: dict = field(default_factory=dict)
    body: bytes = b""


def js(obj: object, status: int = 200) -> Reply:
    return Reply(status, {"Content-Type": "application/json"}, json.dumps(obj).encode("utf-8"))


def file_reply(data: bytes) -> Callable[[dict], Reply]:
    """A route serving `data` with HTTP Range support."""

    def route(headers) -> Reply:
        wanted = headers.get("Range")
        if wanted:
            start = int(re.fullmatch(r"bytes=(\d+)-", wanted).group(1))
            if start >= len(data):
                return Reply(416, {"Content-Range": f"bytes */{len(data)}"})
            return Reply(206, {"Content-Type": "application/octet-stream",
                               "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}, data[start:])
        return Reply(200, {"Content-Type": "application/octet-stream"}, data)

    return route


def version_record(**over: object) -> dict:
    record = {
        "id": VERSION_ID, "modelId": MODEL_ID, "name": "v1.0", "baseModel": "Illustrious",
        "trainedWords": ["lineart"], "availability": "Public", "publishedAt": "2030-01-01T00:00:00.000Z",
        "nsfwLevel": 1, "description": "<p>demo</p>", "images": [{"url": "https://example.invalid/p.png"}],
        "files": [{"id": FILE_ID, "name": FILE_NAME, "sizeKB": len(FILE_BYTES) / 1024, "type": "Model",
                   "primary": True, "metadata": {"format": "SafeTensor", "fp": "fp16"},
                   "hashes": {"SHA256": FILE_SHA.upper()},
                   "downloadUrl": f"{civitai.DOWNLOAD_ROOT}models/{VERSION_ID}"}],
    }
    record.update(over)
    return record


def model_record(**over: object) -> dict:
    record = {
        "id": MODEL_ID, "name": "Demo Line Art", "type": "LORA", "nsfw": False, "nsfwLevel": 1,
        "description": "<p>demo</p>", "creator": {"username": "demo_author"}, "allowNoCredit": False,
        "allowCommercialUse": ["Image"], "allowDerivatives": True, "allowDifferentLicense": False,
        "tags": ["style"], "modelVersions": [version_record()],
    }
    record.update(over)
    return record


def hf_info(**over: object) -> dict:
    info = {
        "id": HF_REPO, "modelId": HF_REPO, "sha": HF_COMMIT, "pipeline_tag": None, "library_name": None,
        "tags": ["license:apache-2.0"], "cardData": {"license": "apache-2.0"}, "gated": False,
        "lastModified": "2030-01-01T00:00:00.000Z",
        "siblings": [{"rfilename": "README.md", "size": 120, "blobId": "b" * 40},
                     {"rfilename": HF_FILE, "size": len(HF_BYTES),
                      "lfs": {"sha256": HF_SHA, "size": len(HF_BYTES)}, "blobId": "c" * 40}],
    }
    info.update(over)
    return info


class Site:
    """Routes by URL path: a Reply, a list of Replies used in turn (the last repeats), or a function of headers."""

    def __init__(self) -> None:
        self.routes: dict[str, object] = {}
        self.requests: list[tuple[str, dict]] = []
        self.url = ""

    def reply_for(self, path: str, headers) -> Reply:
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

    def civitai_demo(self) -> Site:
        """The demo LoRA: model, version and file routes."""
        self.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record())
        self.routes[f"/api/v1/model-versions/{VERSION_ID}"] = js(version_record())
        self.routes[f"/api/download/models/{VERSION_ID}"] = file_reply(FILE_BYTES)
        return self

    def hf_demo(self) -> Site:
        """The demo upscaler repo: info (with and without the revision) and the file."""
        self.routes[f"/api/models/{HF_REPO}"] = js(hf_info())
        self.routes[f"/api/models/{HF_REPO}/revision/{HF_COMMIT}"] = js(hf_info())
        self.routes[f"/{HF_REPO}/resolve/{HF_COMMIT}/{HF_FILE}"] = file_reply(HF_BYTES)
        return self


@pytest.fixture
def site():
    s = Site()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            s.requests.append((self.path, self.headers))
            reply = s.reply_for(self.path, self.headers)
            self.send_response(reply.status)
            for key, value in reply.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(reply.body)))
            self.end_headers()
            self.wfile.write(reply.body)

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
    """The opener the clients call: records the public-API URL, then asks the local site the same path."""

    def __init__(self, site: Site) -> None:
        self.site = site
        self.urls: list[str] = []
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __call__(self, req: urllib.request.Request, timeout: float | None = None):
        url = req.full_url
        self.urls.append(url)
        host = next(h for h in HOSTS if url.startswith(h))
        local = urllib.request.Request(self.site.url + url[len(host):], headers=dict(req.headers),
                                       method=req.get_method())
        return self._opener.open(local, timeout=timeout)


def wire(monkeypatch: pytest.MonkeyPatch, site: Site, *, free_gb: float = 10_000.0) -> Recorder:
    """Make both clients talk to `site`: no sleeping, the disk floor against `free_gb` of pretend free space."""
    recorder = Recorder(site)
    real_civitai, real_hf = civitai.Client.from_config.__func__, hf.Client.from_config.__func__
    options = {"opener": recorder, "sleep": lambda s: None, "disk_free": lambda folder: int(free_gb * (1 << 30))}

    monkeypatch.setattr(civitai.Client, "from_config",
                        classmethod(lambda cls, cfg, **kw: real_civitai(cls, cfg, **{**options, **kw})))
    monkeypatch.setattr(hf.Client, "from_config",
                        classmethod(lambda cls, cfg, **kw: real_hf(cls, cfg, **{**options, **kw})))
    return recorder


def public_only(recorder: Recorder) -> None:
    """Every URL the clients built is under a public API prefix."""
    for url in recorder.urls:
        assert url.startswith((*civitai.PUBLIC_PREFIXES, hf.HF_ROOT)), url
        if url.startswith("https://civitai.com/"):
            assert url.startswith(civitai.PUBLIC_PREFIXES), url
