"""`easel registry download <pending>`: verified download, registration, scan; resume, mismatch, floor, login wall."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli
from remote_support import (
    FILE_BYTES,
    FILE_NAME,
    FILE_SHA,
    HF_BYTES,
    HF_COMMIT,
    HF_FILE,
    HF_REPO,
    HF_SHA,
    MODEL_ID,
    VERSION_ID,
    Reply,
    file_reply,
    js,
    public_only,
    site,  # noqa: F401  (the local stand-in site fixture)
    version_record,
    wire,
)

from easel._common import civitai
from easel.registry.shipped import schema_errors

pytestmark = pytest.mark.usefixtures("no_network")
PENDING = f"civitai-{MODEL_ID}-{VERSION_ID}"
HF_PENDING = f"huggingface-demo_org--demo-upscaler-{HF_COMMIT[:12]}"
ADD_URL = f"{civitai.API_ROOT}model-versions/{VERSION_ID}"


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _record(home: Path, sha16: str) -> dict:
    return yaml.safe_load((home / "registry" / "resources" / f"{sha16}.yaml").read_text(encoding="utf-8"))


def _changelog(home: Path) -> list[dict]:
    return [json.loads(line) for line in
            (home / "registry" / "changelog.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def _added(home: Path, site, monkeypatch, capsys, *, free_gb: float = 10_000.0):  # noqa: F811
    recorder = wire(monkeypatch, site.civitai_demo(), free_gb=free_gb)
    code, _out, err = run_cli(["registry", "add", ADD_URL], capsys)
    assert code == 0, err
    return recorder


def test_download_verifies_registers_and_scans(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    recorder = _added(home, site, monkeypatch, capsys)
    code, out, err = run_cli(["registry", "download", PENDING], capsys)
    assert code == 0, err
    sha16 = FILE_SHA[:16]
    target = home / "models" / "loras" / FILE_NAME
    assert target.read_bytes() == FILE_BYTES
    assert not (home / "models" / "loras" / f"{FILE_NAME}.part").exists()
    assert not (home / "registry" / "pending" / f"{PENDING}.yaml").exists()
    record = _record(home, sha16)
    assert schema_errors("resource", record) == []
    assert record["identity"]["path"] == f"loras/{FILE_NAME}" and record["identity"]["family"] == "illustrious"
    assert record["integrity"]["sha256"] == FILE_SHA and record["integrity"]["bytes"] == len(FILE_BYTES)
    assert record["integrity"]["dtypes"] == ["F16"]  # the scan read the header
    assert record["lora"]["network_dim"] == 8 and record["lora"]["trigger_words"] == ["lineart"]
    assert record["provenance"]["status"] == "verified"
    assert record["provenance"]["downloaded_at"] == "2030-01-01T00:00:00+00:00"
    assert record["lifecycle"]["install_status"] == "active"
    assert record["licence"]["commercial_use"] == "images"
    lines = out.splitlines()
    assert lines[0] == f"downloaded <models_root>/loras/{FILE_NAME} ({len(FILE_BYTES)} bytes, sha256 verified)"
    assert f"registered {sha16} {FILE_NAME} (lora, family illustrious); removed registry/pending/{PENDING}.yaml" in lines
    assert lines[-1] == f"next: easel registry show {sha16}"
    download = [c for c in _changelog(home) if c["action"] == "download"]
    assert download and download[-1]["target"] == {"kind": "resource", "id": sha16}
    index = (home / "registry" / "index.jsonl").read_text(encoding="utf-8")
    assert sha16 in index
    # the checksum is asked for again at download time, never answered from the cache
    assert site.paths().count(f"/api/v1/model-versions/{VERSION_ID}") == 2
    public_only(recorder)
    code, out, _err = run_cli(["registry", "show", sha16], capsys)
    assert code == 0 and "licence: unverified" in out


def test_an_interrupted_download_resumes(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    _added(home, site, monkeypatch, capsys)
    part = home / "models" / "loras" / f"{FILE_NAME}.part"
    part.parent.mkdir(parents=True, exist_ok=True)
    part.write_bytes(FILE_BYTES[:100])
    code, out, err = run_cli(["registry", "download", PENDING], capsys)
    assert code == 0, err
    assert "resumed at byte 100" in out.splitlines()[0]
    ranges = [h.get("Range") for p, h in site.requests if p.startswith("/api/download/")]
    assert ranges == ["bytes=100-"]
    assert (home / "models" / "loras" / FILE_NAME).read_bytes() == FILE_BYTES


def test_a_checksum_mismatch_deletes_the_part_and_keeps_the_pending_entry(home: Path, site, monkeypatch,  # noqa: F811
                                                                         capsys) -> None:
    _added(home, site, monkeypatch, capsys)
    site.routes[f"/api/download/models/{VERSION_ID}"] = file_reply(b"\x00" * len(FILE_BYTES))
    code, _out, err = run_cli(["registry", "download", PENDING], capsys)
    assert code == 1 and "sha256 mismatch" in err
    folder = home / "models" / "loras"
    assert not (folder / FILE_NAME).exists() and not (folder / f"{FILE_NAME}.part").exists()
    assert (home / "registry" / "pending" / f"{PENDING}.yaml").is_file()
    assert not (home / "registry" / "resources" / f"{FILE_SHA[:16]}.yaml").exists()


def test_the_disk_floor_refuses_before_a_byte_is_written(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    _added(home, site, monkeypatch, capsys, free_gb=10.0)  # the demo's floor is 20 GB
    code, _out, err = run_cli(["registry", "download", PENDING], capsys)
    assert code == 3 and "limits.disk_floor_gb" in err
    assert not (home / "models" / "loras" / FILE_NAME).exists()
    assert (home / "registry" / "pending" / f"{PENDING}.yaml").is_file()


def test_a_login_page_is_refused(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    _added(home, site, monkeypatch, capsys)
    site.routes[f"/api/download/models/{VERSION_ID}"] = Reply(200, {"Content-Type": "text/html"},
                                                              b"<!DOCTYPE html><html>sign in</html>")
    code, _out, err = run_cli(["registry", "download", PENDING], capsys)
    assert code == 2 and "never logs in" in err
    assert (home / "registry" / "pending" / f"{PENDING}.yaml").is_file()


def test_bytes_already_registered_are_merged_not_downloaded(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    _added(home, site, monkeypatch, capsys)
    robot = _record(home, "cf56db17414bcb55")
    upscaler = _record(home, "99134ef8d78fb1bb")
    known = version_record()
    known["files"][0]["hashes"] = {"SHA256": upscaler["integrity"]["sha256"]}
    site.routes[f"/api/v1/model-versions/{VERSION_ID}"] = js(known)
    code, out, err = run_cli(["registry", "download", PENDING], capsys)
    assert code == 0, err
    assert out.startswith("already registered: 99134ef8d78fb1bb demo-4x.safetensors has these bytes; nothing downloaded")
    assert not any(p.startswith("/api/download/") for p in site.paths())
    assert not (home / "registry" / "pending" / f"{PENDING}.yaml").exists()
    after = _record(home, "99134ef8d78fb1bb")
    assert after["provenance"]["site"] == "huggingface"  # a recorded source is never replaced by a merge
    assert after["licence"] == upscaler["licence"]
    assert _record(home, "cf56db17414bcb55") == robot


def test_download_from_hugging_face(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    recorder = wire(monkeypatch, site.hf_demo())
    code, _out, err = run_cli(["registry", "add", f"https://huggingface.co/{HF_REPO}", "--type", "upscaler"], capsys)
    assert code == 0, err
    code, out, err = run_cli(["registry", "download", HF_PENDING], capsys)
    assert code == 0, err
    assert (home / "models" / "upscale_models" / HF_FILE).read_bytes() == HF_BYTES
    record = _record(home, HF_SHA[:16])
    assert record["identity"]["type"] == "upscaler" and record["provenance"]["site"] == "huggingface"
    assert record["provenance"]["upstream_revision"] == HF_COMMIT
    assert out.splitlines()[0] == f"downloaded <models_root>/upscale_models/{HF_FILE} ({len(HF_BYTES)} bytes, " \
                                  "sha256 verified)"
    public_only(recorder)


def test_without_a_name_download_lists_the_pending_entries(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    code, out, _err = run_cli(["registry", "download"], capsys)
    assert code == 0 and out.strip() == "0 pending entries in registry/pending/; `easel registry add <url>` adds one"
    _added(home, site, monkeypatch, capsys)
    code, out, _err = run_cli(["registry", "download"], capsys)
    assert code == 0 and out.splitlines() == [PENDING, "1 pending entry in registry/pending/; next: easel registry "
                                                       "download <pending>"]


def test_an_unknown_pending_name_is_exit_2(home: Path, capsys) -> None:
    code, _out, err = run_cli(["registry", "download", "civitai-9000000001-9000000002"], capsys)
    assert code == 2 and "no pending entry" in err and "nothing is pending" in err


def test_a_pending_entry_that_is_not_pending_is_refused(home: Path, capsys) -> None:
    folder = home / "registry" / "pending"
    folder.mkdir(parents=True)
    record = _record(home, "cf56db17414bcb55")
    (folder / "civitai-1-2.yaml").write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")
    code, _out, err = run_cli(["registry", "download", "civitai-1-2"], capsys)
    assert code == 2 and "not a valid pending entry" in err
