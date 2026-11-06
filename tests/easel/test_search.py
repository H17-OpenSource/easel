"""`easel search civitai|hf|vet` on hand-written API answers: no network, no real ids.

The answers are synthetic (ids from 9000000000 up, `demo_author`, `<p>demo</p>`); the URLs are assembled at run
time. A fake opener serves them to the real `_common.civitai` / `_common.hf` clients, so the rate limit, the URL
checks and the record summaries all run; `sleep` is a no-op so a backoff never waits.
"""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest

from easel import search
from easel._common import civitai, hf
from easel.cli import main
from easel.registry.shipped import schema_errors

HOST = "civitai" + ".com"
API = f"https://{HOST}/api/v1/"
DOWNLOAD = f"https://{HOST}/api/download/models/"
HF_API = "https://" + "huggingface.co/api/models/"
GIB = 1 << 30
SHA_A = "a1" * 32
SHA_B = "b2" * 32


class Response:
    """Just enough of an HTTP response for the client: read, status, headers, a context manager."""

    def __init__(self, body: object, status: int = 200) -> None:
        self._data = io.BytesIO(json.dumps(body).encode("utf-8"))
        self.status = status
        self.headers = {"Content-Type": "application/json"}

    def read(self, n: int = -1) -> bytes:
        return self._data.read(n)

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


class Site:
    """A fake opener: URL (without query, or with) -> body; every request is recorded."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def __call__(self, request, timeout: float = 0) -> Response:
        url = request.full_url
        self.asked.append(url)
        for key in (url, url.split("?", 1)[0]):
            if key in self.answers:
                body = self.answers[key]
                if isinstance(body, int):
                    raise urllib.error.HTTPError(url, body, "synthetic", {}, None)
                return Response(body)
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)


def _version(vid: int, mid: int, name: str, file_name: str, sha: str, *, kb: float = 65536.0,
             availability: str = "Public", level: int = 1) -> dict:
    return {"id": vid, "modelId": mid, "name": name, "baseModel": "Illustrious", "availability": availability,
            "nsfwLevel": level, "trainedWords": [], "images": [{"url": "https://example.invalid/demo.png"}],
            "files": [{"id": vid + 1, "name": file_name, "sizeKB": kb, "primary": True, "type": "Model",
                       "hashes": {"SHA256": sha.upper()}, "metadata": {"format": "SafeTensor", "fp": "fp16"},
                       "downloadUrl": f"{DOWNLOAD}{vid}"}]}


def _model(mid: int, name: str, mtype: str, versions: list[dict], *, licence: bool = True, **extra) -> dict:
    model = {"id": mid, "name": name, "type": mtype, "nsfw": False, "nsfwLevel": 1, "poi": False,
             "tags": ["upscaler"], "creator": {"username": "demo_author"}, "description": "<p>demo</p>",
             "modelVersions": versions, **extra}
    if licence:
        model.update(allowCommercialUse=["Image"], allowNoCredit=False, allowDerivatives=True,
                     allowDifferentLicense=False)
    return model


def _answers() -> dict[str, object]:
    ok_v = _version(9000000402, 9000000401, "v1", "demo-2x-upscaler.safetensors", SHA_A)
    unclear_v = _version(9000000502, 9000000501, "v1", "demo-line-upscaler.safetensors", SHA_B)
    return {
        f"{API}model-versions/9000000402": ok_v,
        f"{API}models/9000000401": _model(9000000401, "Demo 2x Upscaler", "Upscaler", [ok_v]),
        f"{API}model-versions/9000000502": unclear_v,
        f"{API}models/9000000501": _model(9000000501, "Demo Line Upscaler", "Upscaler", [unclear_v], licence=False),
    }


def _clients(site: Site, *, free_gb: float = 500.0) -> tuple[civitai.Client, hf.Client]:
    kw = dict(tool="easel", opener=site, sleep=lambda s: None, disk_free=lambda p: int(free_gb * GIB),
              disk_floor_gb=100.0)
    return civitai.Client(**kw), hf.Client(**kw)


@pytest.fixture
def site(monkeypatch, no_network):
    fake = Site(_answers())
    monkeypatch.setattr(search, "make_clients", lambda cfg: _clients(fake))
    return fake


def _rows(home: Path) -> list[dict]:
    path = home / "candidates" / "2030-01-01.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_vet_writes_one_ok_and_one_licence_unclear_row(demo_workspace, frozen_clock, site, capsys):
    before = len(_rows(demo_workspace))
    code = main(["search", "vet", f"{API}model-versions/9000000402", f"{API}model-versions/9000000502",
                 "--query", "upscaler"])
    out = capsys.readouterr().out
    assert code == 0
    rows = _rows(demo_workspace)[before:]
    assert [r["class"] for r in rows] == ["ok", "licence-unclear"]
    assert all(schema_errors("candidates-entry", r) == [] for r in rows)
    ok, unclear = rows
    assert ok["url"] == f"{API}model-versions/9000000402" and ok["model_id"] == "9000000401"
    assert ok["type"] == "upscaler" and ok["family"] == "illustrious" and ok["sha256"] == SHA_A
    assert ok["licence"]["commercial_use"] == "images" and ok["licence"]["credit_required"] is True
    assert ok["query"] == "upscaler" and ok["bytes"] == 64 * 1024 * 1024
    assert unclear["licence"]["status"] == "unknown"
    assert out.splitlines()[0].startswith("ok | Demo 2x Upscaler (upscaler, illustrious")
    assert any(line.startswith("licence-unclear | Demo Line Upscaler") for line in out.splitlines())
    assert "wrote 2 rows to candidates/2030-01-01.jsonl" in out
    assert "demo_author" not in json.dumps(rows) and "<p>" not in json.dumps(rows)


def _vet_one(site: Site, url: str, *, free_gb: float = 500.0, max_file_gb: float | None = None) -> dict:
    from easel.registry import classify

    civ, hug = _clients(site, free_gb=free_gb)
    limits = search.Limits(folder=Path("."), floor_gb=100.0, max_file_gb=max_file_gb, disk_free=civ.disk_free)
    return search.vet(url, civitai_client=civ, hf_client=hug, markers=classify.load_markers(None), limits=limits)


def test_the_other_three_classes(no_network):
    gated_v = _version(9000000602, 9000000601, "v1", "demo-early.safetensors", SHA_A, availability="EarlyAccess")
    big_v = _version(9000000702, 9000000701, "v1", "demo-big.safetensors", SHA_A, kb=8 * 1024 * 1024)
    cue_v = _version(9000000802, 9000000801, "v1", "demo-style.safetensors", SHA_A)
    site = Site({
        f"{API}model-versions/9000000602": gated_v,
        f"{API}models/9000000601": _model(9000000601, "Demo Early", "LORA", [gated_v]),
        f"{API}model-versions/9000000702": big_v,
        f"{API}models/9000000701": _model(9000000701, "Demo Big", "Checkpoint", [big_v]),
        f"{API}model-versions/9000000802": cue_v,
        f"{API}models/9000000801": _model(9000000801, "Demo NSFW Style", "LORA", [cue_v]),
    })
    assert _vet_one(site, f"{API}model-versions/9000000602")["class"] == "gated"
    big = _vet_one(site, f"{API}model-versions/9000000702", free_gb=105.0)
    assert big["class"] == "too-big" and "under the 100 GB floor" in " ".join(big["reasons"])
    assert _vet_one(site, f"{API}model-versions/9000000702", max_file_gb=4.0)["class"] == "too-big"
    conflict = _vet_one(site, f"{API}model-versions/9000000802")
    assert conflict["class"] == "rating-conflict" and conflict["rating"] == "sfw"


def test_a_real_person_flag_is_a_conflict_and_a_page_link_is_understood(no_network):
    v = _version(9000000902, 9000000901, "v1", "demo-person.safetensors", SHA_A)
    site = Site({f"{API}models/9000000901": _model(9000000901, "Demo Person", "LORA", [v], poi=True)})
    page = f"https://{HOST}/" + "models/9000000901/demo?modelVersionId=9000000902"
    row = _vet_one(site, page)
    assert row["class"] == "rating-conflict" and row["version_id"] == "9000000902"
    assert any("real person" in r for r in row["reasons"])
    assert row["url"] == f"{API}model-versions/9000000902"


def test_hugging_face_vet(no_network):
    base = {"id": "demo-org/demo-lora", "sha": "c" * 40, "tags": ["lora", "license:apache-2.0"],
            "pipeline_tag": "text-to-image", "cardData": {"license": "apache-2.0", "base_model": "sdxl-base"},
            "siblings": [{"rfilename": "README.md", "size": 10, "blobId": "d" * 40},
                         {"rfilename": "demo_lora.safetensors", "size": 5 * 1024 * 1024,
                          "lfs": {"sha256": SHA_B, "size": 5 * 1024 * 1024}}]}
    no_licence = {**base, "id": "demo-org/demo-plain", "cardData": {}, "tags": ["lora"]}
    gated = {**base, "id": "demo-org/demo-gated", "gated": "auto"}
    site = Site({f"{HF_API}{m['id']}": m for m in (base, no_licence, gated)})
    ok = _vet_one(site, "https://" + "huggingface.co/demo-org/demo-lora")
    assert ok["class"] == "ok" and ok["site"] == "huggingface" and ok["type"] == "lora"
    assert ok["file_name"] == "demo_lora.safetensors" and ok["sha256"] == SHA_B and ok["licence"]["spdx"] == "Apache-2.0"
    assert schema_errors("candidates-entry", ok) == []
    assert _vet_one(site, f"{HF_API}demo-org/demo-plain")["class"] == "licence-unclear"
    assert _vet_one(site, f"{HF_API}demo-org/demo-gated")["class"] == "gated"


def test_vet_refuses_a_foreign_url_before_any_request(demo_workspace, site, capsys):
    assert main(["search", "vet", "https://example.invalid/models/1"]) == 2
    assert site.asked == []
    assert "Civitai and Hugging Face" in capsys.readouterr().err


def test_rate_limit_is_exit_3(demo_workspace, monkeypatch, no_network, capsys):
    fake = Site({f"{API}model-versions/9000000402": 429})
    monkeypatch.setattr(search, "make_clients", lambda cfg: _clients(fake))
    assert main(["search", "vet", f"{API}model-versions/9000000402"]) == 3
    assert "slow down" in capsys.readouterr().err
    assert len(fake.asked) == civitai.MAX_RETRIES + 1


def test_search_civitai_lists_the_url_to_vet(demo_workspace, monkeypatch, no_network, capsys):
    answers = _answers()
    hit = answers[f"{API}models/9000000401"]
    fake = Site({f"{API}models": {"items": [hit], "metadata": {}}})
    monkeypatch.setattr(search, "make_clients", lambda cfg: _clients(fake))
    assert main(["search", "civitai", "--query", "upscaler", "--type", "upscaler"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"{API}model-versions/9000000402 | Demo 2x Upscaler | upscaler | Illustrious")
    assert "query=upscaler" in fake.asked[0] and "types=Upscaler" in fake.asked[0] and "nsfw=false" in fake.asked[0]
    assert lines[-1] == "next: easel search vet <url>"


def test_search_hf_lists_hits(demo_workspace, monkeypatch, no_network, capsys):
    hits = [{"id": "demo-org/demo-lora", "tags": ["lora", "license:mit"], "pipeline_tag": "text-to-image"}]
    fake = Site({f"{HF_API[:-1]}": hits})
    monkeypatch.setattr(search, "make_clients", lambda cfg: _clients(fake))
    assert main(["search", "hf", "--query", "demo", "--type", "lora"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{HF_API}demo-org/demo-lora")
    assert "filter=lora" in fake.asked[0]


def test_parse_url_shapes():
    assert search.parse_url(f"{API}model-versions/by-hash/{SHA_A}") == search.Target("civitai", "hash", SHA_A)
    assert search.parse_url(f"{DOWNLOAD}9000000402") == search.Target("civitai", "version", "9000000402")
    assert search.parse_url(f"{API}models/9000000401").kind == "model"
    with pytest.raises(search.SearchError):
        search.parse_url(f"https://{HOST}/" + "user/demo_author")
    assert search.family_for("Pony") == "pony" and search.family_for("SD 1.5") == "sd15"
    assert search.family_for("Other") is None
