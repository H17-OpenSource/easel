"""`easel registry add <url>`: the link forms, the pending entry it writes, and its refusals (offline, local site)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli
from remote_support import (
    FILE_NAME,
    HF_COMMIT,
    HF_FILE,
    HF_REPO,
    MODEL_ID,
    VERSION_ID,
    hf_info,
    js,
    model_record,
    page_link,
    public_only,
    site,  # noqa: F401  (the local stand-in site fixture)
    version_record,
    wire,
)

from easel._common import civitai
from easel.registry import remote
from easel.registry.shipped import schema_errors

pytestmark = pytest.mark.usefixtures("no_network")
PENDING = f"civitai-{MODEL_ID}-{VERSION_ID}"
HF_PENDING = f"huggingface-demo_org--demo-upscaler-{HF_COMMIT[:12]}"


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _pending(home: Path, name: str) -> dict:
    return yaml.safe_load((home / "registry" / "pending" / f"{name}.yaml").read_text(encoding="utf-8"))


# --- links ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize(("link", "expected"), [
    (page_link(f"models/{MODEL_ID}?modelVersionId={VERSION_ID}"), ("civitai", str(MODEL_ID), str(VERSION_ID))),
    (page_link(f"models/{MODEL_ID}/demo-line-art"), ("civitai", str(MODEL_ID), None)),
    (f"{civitai.API_ROOT}models/{MODEL_ID}", ("civitai", str(MODEL_ID), None)),
    (f"{civitai.API_ROOT}model-versions/{VERSION_ID}", ("civitai", None, str(VERSION_ID))),
    (f"{civitai.DOWNLOAD_ROOT}models/{VERSION_ID}", ("civitai", None, str(VERSION_ID))),
    (f"https://huggingface.co/{HF_REPO}", ("huggingface", HF_REPO, None)),
    (f"https://huggingface.co/{HF_REPO}/blob/main/{HF_FILE}", ("huggingface", HF_REPO, None)),
    (f"https://huggingface.co/api/models/{HF_REPO}", ("huggingface", HF_REPO, None)),
])
def test_link_forms(link: str, expected: tuple) -> None:
    target = remote.parse_link(link)
    assert (target.site, target.model_id, target.version_id) == expected


def test_a_file_link_names_revision_and_path() -> None:
    target = remote.parse_link(f"https://huggingface.co/{HF_REPO}/resolve/{HF_COMMIT}/sub/{HF_FILE}")
    assert (target.revision, target.path) == (HF_COMMIT, f"sub/{HF_FILE}")


@pytest.mark.parametrize("link", ["not a link", "https://example.invalid/models/9000000401",
                                  page_link("models/abc"), page_link("user/demo_author"),
                                  "https://huggingface.co/datasets/demo_org/demo", "ftp://huggingface.co/x/y"])
def test_links_that_are_refused(link: str) -> None:
    with pytest.raises(remote.LinkError) as caught:
        remote.parse_link(link)
    assert caught.value.code == 2


def test_family_guess_needs_an_existing_family_file() -> None:
    known = {"sdxl", "illustrious", "pony", "sd15", "flux1"}
    assert remote.family_for("Illustrious", known) == "illustrious"
    assert remote.family_for("SDXL 1.0", known) == "sdxl"
    assert remote.family_for("SD 1.5", known) == "sd15"
    assert remote.family_for("Flux.1 D", known) == "flux1"
    assert remote.family_for("stabilityai/stable-diffusion-xl-base-1.0", known) == "sdxl"
    assert remote.family_for("Demo Unknown 9", known) is None
    assert remote.family_for(None, known) is None


def test_stricter_rating_is_never_sfw_on_a_doubt() -> None:
    assert remote.stricter(["sfw", "sfw"]) == "sfw"
    assert remote.stricter(["sfw", "unknown"]) == "unknown"
    assert remote.stricter(["suggestive", "sfw"]) == "suggestive"
    assert remote.stricter([]) == "unknown"


# --- civitai ----------------------------------------------------------------------------------------------------

def test_add_writes_a_valid_pending_entry(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    recorder = wire(monkeypatch, site.civitai_demo())
    code, out, err = run_cli(["registry", "add", page_link(f"models/{MODEL_ID}?modelVersionId={VERSION_ID}")], capsys)
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0] == f"pending registry/pending/{PENDING}.yaml"
    assert lines[-1] == f"next: easel registry download {PENDING}"
    record = _pending(home, PENDING)
    assert schema_errors("resource", record) == []
    assert record["id"] is None and record["integrity"]["sha256"] is None
    assert record["lifecycle"]["install_status"] == "pending"
    assert record["identity"] == {"type": "lora", "file_name": FILE_NAME, "path": None, "family": "illustrious",
                                  "base_model_label": "Illustrious", "title": "Demo Line Art v1.0"}
    assert record["provenance"]["url"] == f"{civitai.API_ROOT}model-versions/{VERSION_ID}"
    assert record["provenance"]["download_url"] == f"{civitai.DOWNLOAD_ROOT}models/{VERSION_ID}"
    assert record["provenance"]["author"] is None  # the uploader is never kept
    lic = record["licence"]
    assert (lic["status"], lic["commercial_use"], lic["credit_required"], lic["derivatives_allowed"],
            lic["relicense_allowed"], lic["gated"], lic["source"], lic["checked_at"]) == (
        "unverified", "images", True, True, False, False, "civitai-api", "2030-01-01")
    assert record["semantics"]["content"] == {"rating": "sfw", "class": "sfw", "basis": "source"}
    assert record["lora"]["trigger_words"] == ["lineart"]
    assert "description" not in json.dumps(record["provenance"])
    changelog = (home / "registry" / "changelog.jsonl").read_text(encoding="utf-8").splitlines()
    last = json.loads(changelog[-1])
    assert (last["action"], last["target"]) == ("add", {"kind": "resource", "id": f"pending/{PENDING}"})
    assert site.paths() == [f"/api/v1/model-versions/{VERSION_ID}", f"/api/v1/models/{MODEL_ID}"]
    public_only(recorder)


def test_the_candidates_url_works_too_and_a_second_add_is_refused(home: Path, site, monkeypatch,  # noqa: F811
                                                                   capsys) -> None:
    wire(monkeypatch, site.civitai_demo())
    url = f"{civitai.API_ROOT}model-versions/{VERSION_ID}"
    assert run_cli(["registry", "add", url], capsys)[0] == 0
    code, _out, err = run_cli(["registry", "add", url], capsys)
    assert code == 2 and f"already pending: registry/pending/{PENDING}.yaml" in err
    code, out, _err = run_cli(["registry", "add", url, "--replace"], capsys)
    assert code == 0 and out.startswith(f"pending registry/pending/{PENDING}.yaml")


def test_a_model_link_takes_the_newest_version(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    wire(monkeypatch, site.civitai_demo())
    code, out, _err = run_cli(["registry", "add", f"{civitai.API_ROOT}models/{MODEL_ID}"], capsys)
    assert code == 0 and out.startswith(f"pending registry/pending/{PENDING}.yaml")


def test_no_published_checksum_is_refused(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    site.civitai_demo()
    bare = version_record()
    del bare["files"][0]["hashes"]
    site.routes[f"/api/v1/model-versions/{VERSION_ID}"] = js(bare)
    wire(monkeypatch, site)
    code, _out, err = run_cli(["registry", "add", f"{civitai.API_ROOT}model-versions/{VERSION_ID}"], capsys)
    assert code == 2 and "publishes no sha256" in err
    assert not (home / "registry" / "pending").exists()


def test_a_type_without_a_folder_needs_type(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    site.civitai_demo()
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record(type="Poses"))
    wire(monkeypatch, site)
    url = f"{civitai.API_ROOT}model-versions/{VERSION_ID}"
    code, _out, err = run_cli(["registry", "add", url], capsys)
    assert code == 2 and "give --type" in err
    code, _out, err = run_cli(["registry", "add", url, "--type", "lora"], capsys)
    assert code == 0, err


def test_bytes_the_registry_already_has_are_not_added(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    ink = yaml.safe_load((home / "registry" / "resources" / "60d12866cf6377b0.yaml").read_text(encoding="utf-8"))
    site.civitai_demo()
    known = version_record()
    known["files"][0]["hashes"] = {"SHA256": ink["integrity"]["sha256"]}
    site.routes[f"/api/v1/model-versions/{VERSION_ID}"] = js(known)
    wire(monkeypatch, site)
    code, out, _err = run_cli(["registry", "add", f"{civitai.API_ROOT}model-versions/{VERSION_ID}"], capsys)
    assert code == 0 and out.startswith("already registered: 60d12866cf6377b0 ")
    assert "register-upstream 60d12866cf6377b0 --civitai" in out
    assert not (home / "registry" / "pending").exists()


def test_a_missing_model_is_exit_2_and_a_rate_limit_exit_3(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    wire(monkeypatch, site)
    code, _out, err = run_cli(["registry", "add", f"{civitai.API_ROOT}models/{MODEL_ID}"], capsys)
    assert code == 2 and "not found (HTTP 404)" in err
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js({"error": "slow down"}, 429)
    code, _out, err = run_cli(["registry", "add", f"{civitai.API_ROOT}models/{MODEL_ID}"], capsys)
    assert code == 3 and "slow down" in err


def test_a_site_sfw_rating_against_a_cue_word_is_a_conflict(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    markers = yaml.safe_load((Path(__file__).resolve().parents[2] / "data" / "content_markers.yaml")
                             .read_text(encoding="utf-8"))
    cue = str(markers["non_sfw_cues"][0])
    site.civitai_demo()
    site.routes[f"/api/v1/models/{MODEL_ID}"] = js(model_record(name=f"Demo {cue}"))
    wire(monkeypatch, site)
    code, out, err = run_cli(["registry", "add", f"{civitai.API_ROOT}model-versions/{VERSION_ID}"], capsys)
    assert code == 0
    assert f"RATING CONFLICT {PENDING}" in err
    assert "(title withheld: class" not in out  # the site said sfw; the conflict is for the operator to settle


# --- hugging face -----------------------------------------------------------------------------------------------

def test_add_from_hugging_face(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    recorder = wire(monkeypatch, site.hf_demo())
    code, _out, err = run_cli(["registry", "add", f"https://huggingface.co/{HF_REPO}"], capsys)
    assert code == 2 and "give --type" in err  # an upscaler repo carries no tag easel can read a type from
    code, out, err = run_cli(["registry", "add", f"https://huggingface.co/{HF_REPO}", "--type", "upscaler"], capsys)
    assert code == 0, err
    assert out.splitlines()[0] == f"pending registry/pending/{HF_PENDING}.yaml"
    record = _pending(home, HF_PENDING)
    assert schema_errors("resource", record) == []
    assert record["identity"]["type"] == "upscaler" and record["identity"]["file_name"] == HF_FILE
    assert record["provenance"]["download_url"] == f"https://huggingface.co/{HF_REPO}/resolve/{HF_COMMIT}/{HF_FILE}"
    assert record["provenance"]["upstream_revision"] == HF_COMMIT
    assert (record["licence"]["name"], record["licence"]["spdx"], record["licence"]["source"]) == (
        "apache-2.0", "Apache-2.0", "hf-card")
    assert record["semantics"]["content"]["class"] == "unknown"  # the site never declares a repo sfw
    assert "licence unknown" not in err
    public_only(recorder)


def test_a_repo_with_two_model_files_needs_file(home: Path, site, monkeypatch, capsys) -> None:  # noqa: F811
    site.hf_demo()
    info = hf_info()
    info["siblings"].append({"rfilename": "demo-4x.safetensors", "size": 64, "lfs": {"sha256": "d" * 64, "size": 64}})
    site.routes[f"/api/models/{HF_REPO}"] = js(info)
    wire(monkeypatch, site)
    link = f"https://huggingface.co/{HF_REPO}"
    code, _out, err = run_cli(["registry", "add", link, "--type", "upscaler"], capsys)
    assert code == 2 and "holds 2 model files" in err and "give --file" in err
    code, out, _err = run_cli(["registry", "add", link, "--type", "upscaler", "--file", HF_FILE], capsys)
    assert code == 0 and out.splitlines()[0] == f"pending registry/pending/{HF_PENDING}--demo-2x.yaml"


def test_the_pending_name_shape() -> None:
    assert remote.pending_name("civitai", "9000000401", "9000000402") == "civitai-9000000401-9000000402"
    assert remote.pending_name("huggingface", "Demo_Org/Demo-Upscaler", HF_COMMIT[:12], "demo-2x") == \
        f"huggingface-demo_org--demo-upscaler-{HF_COMMIT[:12]}--demo-2x"
