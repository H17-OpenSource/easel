"""`easel registry register-upstream <sha16>`: a source found by hash, compared, or given; refusals (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from registry_support import empty_registry, header_reader, run_cli
from remote_support import (
    MODEL_ID,
    VERSION_ID,
    hf_info,
    js,
    model_record,
    public_only,
    site,  # noqa: F401  (the local stand-in site fixture)
    version_record,
    wire,
)

from easel._common import civitai
from easel.registry.shipped import schema_errors

pytestmark = pytest.mark.usefixtures("no_network")
INK, ROBOT, UPSCALER = "60d12866cf6377b0", "cf56db17414bcb55", "99134ef8d78fb1bb"
REPO = "demo_org/demo-ink"


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch, capsys) -> Path:
    """The demo with a freshly scanned registry: every record's source unknown, as for files copied in by hand."""
    header_reader(monkeypatch)
    empty_registry(demo_workspace)
    code, _out, err = run_cli(["registry", "scan", "--hash", "--quiet"], capsys)
    assert code == 0, err
    return demo_workspace


def _record(home: Path, sha16: str) -> dict:
    return yaml.safe_load((home / "registry" / "resources" / f"{sha16}.yaml").read_text(encoding="utf-8"))


def _by_hash(site, sha256: str) -> None:  # noqa: F811
    version = version_record()
    version["files"][0]["hashes"] = {"SHA256": sha256.upper()}
    site.routes[f"/api/v1/model-versions/by-hash/{sha256}"] = js(version)
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record(allowCommercialUse="None"))


def test_civitai_by_hash_fills_the_source_and_the_licence(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    before = _record(home, INK)
    assert before["provenance"]["source_class"] == "unknown" and before["licence"]["status"] == "unknown"
    _by_hash(site, before["integrity"]["sha256"])
    recorder = wire(monkeypatch, site)
    code, out, err = run_cli(["registry", "register-upstream", INK, "--civitai"], capsys)
    assert code == 0, err
    assert out.splitlines()[0].startswith(f"registered the source of {INK} demo-style-ink.safetensors: civitai model "
                                          f"{MODEL_ID} version {VERSION_ID}")
    after = _record(home, INK)
    assert schema_errors("resource", after) == []
    prov = after["provenance"]
    assert (prov["source_class"], prov["site"], prov["status"], prov["model_id"], prov["version_id"]) == (
        "downloaded", "civitai", "verified", str(MODEL_ID), str(VERSION_ID))
    assert prov["url"] == f"{civitai.API_ROOT}model-versions/{VERSION_ID}"
    assert after["licence"]["commercial_use"] == "none" and after["licence"]["source"] == "civitai-api"
    assert after["identity"]["path"] == before["identity"]["path"]  # the scan-owned fields stay
    last = json.loads((home / "registry" / "changelog.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["action"] == "update" and last["target"] == {"kind": "resource", "id": INK}
    assert "provenance.site" in last["changes"]
    public_only(recorder)
    # a second scan keeps what the API said
    assert run_cli(["registry", "scan", "--quiet"], capsys)[0] == 0
    assert _record(home, INK)["provenance"]["site"] == "civitai"
    # and the audit now flags the binding term
    code, out, _err = run_cli(["registry", "audit", "--only", "licence-terms"], capsys)
    assert f"{INK} | lora | sdxl | demo-style-ink.safetensors | commercial_use: none" in out


def test_a_recorded_source_is_kept_unless_forced(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    sha = _record(home, INK)["integrity"]["sha256"]
    _by_hash(site, sha)
    wire(monkeypatch, site)
    code, _out, err = run_cli(["registry", "register-upstream", INK, "--url", "https://example.invalid/ink",
                               "--site", "github"], capsys)
    assert code == 0, err
    assert _record(home, INK)["provenance"]["status"] == "partial"
    code, out, _err = run_cli(["registry", "register-upstream", INK, "--civitai"], capsys)
    assert code == 0 and _record(home, INK)["provenance"]["site"] == "github"
    assert "licence" in out  # the licence was still unknown, so the API's terms are filled in
    code, _out, _err = run_cli(["registry", "register-upstream", INK, "--civitai", "--force"], capsys)
    assert code == 0 and _record(home, INK)["provenance"]["site"] == "civitai"


def test_hugging_face_compares_the_checksum(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    sha = _record(home, UPSCALER)["integrity"]["sha256"]
    info = hf_info(id=REPO, modelId=REPO, siblings=[{"rfilename": "x2.safetensors", "size": 10,
                                                     "lfs": {"sha256": sha, "size": 10}},
                                                    {"rfilename": "other.safetensors", "size": 10,
                                                     "lfs": {"sha256": "e" * 64, "size": 10}}])
    site.routes[f"/api/models/{REPO}"] = js(info)
    wire(monkeypatch, site)
    code, _out, err = run_cli(["registry", "register-upstream", UPSCALER, "--hf", REPO, "--path",
                               "other.safetensors"], capsys)
    assert code == 1 and "HASH MISMATCH" in err
    assert _record(home, UPSCALER)["provenance"]["site"] is None
    code, out, err = run_cli(["registry", "register-upstream", UPSCALER, "--hf", REPO, "--path",
                              "x2.safetensors"], capsys)
    assert code == 0, err
    prov = _record(home, UPSCALER)["provenance"]
    assert (prov["site"], prov["status"], prov["model_id"]) == ("huggingface", "verified", REPO)
    assert prov["download_url"] == f"https://huggingface.co/{REPO}/resolve/{'a' * 40}/x2.safetensors"
    assert _record(home, UPSCALER)["licence"]["spdx"] == "Apache-2.0"


def test_a_forced_mismatch_is_recorded_as_partial(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    info = hf_info(id=REPO, modelId=REPO, siblings=[{"rfilename": "other.safetensors", "size": 10,
                                                     "lfs": {"sha256": "e" * 64, "size": 10}}])
    site.routes[f"/api/models/{REPO}"] = js(info)
    wire(monkeypatch, site)
    code, out, _err = run_cli(["registry", "register-upstream", UPSCALER, "--hf", REPO, "--path",
                               "other.safetensors", "--force"], capsys)
    assert code == 0 and "FORCED" in out
    assert _record(home, UPSCALER)["provenance"]["status"] == "partial"


def test_a_plain_url_names_its_site_and_the_licence_by_hand(home: Path, capsys) -> None:
    code, out, err = run_cli(["registry", "register-upstream", INK, "--url", "https://example.invalid/demo-ink",
                              "--site", "github", "--licence-name", "Demo Open Licence", "--author", "demo_author"],
                             capsys)
    assert code == 0, err
    rec = _record(home, INK)
    assert (rec["provenance"]["site"], rec["provenance"]["status"], rec["provenance"]["author"]) == (
        "github", "partial", "demo_author")
    assert (rec["licence"]["name"], rec["licence"]["status"], rec["licence"]["source"]) == (
        "Demo Open Licence", "unverified", "manual")


def test_refusals(home: Path, capsys) -> None:
    code, _out, err = run_cli(["registry", "register-upstream", INK, "--url", "https://example.invalid/x"], capsys)
    assert code == 2 and "--url needs --site" in err
    code, _out, err = run_cli(["registry", "register-upstream", INK, "--hf", REPO], capsys)
    assert code == 2 and "--hf needs --path" in err
    code, _out, err = run_cli(["registry", "register-upstream", INK, "--civitai", "--path", "x"], capsys)
    assert code == 2 and "--path and --revision go with --hf" in err
    code, _out, _err = run_cli(["registry", "register-upstream", INK], capsys)
    assert code == 2  # one source is required
    code, _out, err = run_cli(["registry", "register-upstream", "0000000000000000", "--civitai"], capsys)
    assert code == 2 and "no resource" in err


def test_a_self_trained_resource_is_refused(demo_workspace: Path, frozen_clock, monkeypatch, capsys) -> None:
    header_reader(monkeypatch)
    code, _out, err = run_cli(["registry", "register-upstream", ROBOT, "--civitai"], capsys)
    assert code == 2 and "self-trained" in err
