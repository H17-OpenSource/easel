"""`easel registry audit`: every check on planted records, the filters, JSON, --check, and names withheld."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli

from easel.registry import audit as auditing
from easel.registry import resources as res

ANIME_XL, BASE_XL, ROBOT, INK, VAE, UPSCALER = ("bbbabeef2c0d57fe", "85306657134cbde8", "cf56db17414bcb55",
                                                "60d12866cf6377b0", "b4d12aa2b3121a5e", "99134ef8d78fb1bb")
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _path(home: Path, sha16: str) -> Path:
    return home / "registry" / "resources" / f"{sha16}.yaml"


def _edit(home: Path, sha16: str, **dotted: object) -> None:
    rec = yaml.safe_load(_path(home, sha16).read_text(encoding="utf-8"))
    for key, value in dotted.items():
        res.set_field(rec, key.replace("__", "."), value)
    _path(home, sha16).write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")


def _plant(home: Path, base: str, sha_hex: str, **dotted: object) -> str:
    """A copy of `base`'s record under a new id, with fields changed (no file behind it unless one is placed)."""
    rec = yaml.safe_load(_path(home, base).read_text(encoding="utf-8"))
    rec = copy.deepcopy(rec)
    sha256 = sha_hex * (64 // len(sha_hex))
    rec["id"], rec["integrity"]["sha256"] = sha256[:16], sha256
    for key, value in dotted.items():
        res.set_field(rec, key.replace("__", "."), value)
    _path(home, rec["id"]).write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")
    return rec["id"]


def _non_sfw_cue() -> str:
    data = yaml.safe_load((REPO / "data" / "content_markers.yaml").read_text(encoding="utf-8"))
    return str(data["non_sfw_cues"][0])


def _sections(out: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current = None
    for line in out.splitlines()[1:]:
        if line.startswith("## "):
            current = line[3:].split(" ", 1)[0]
            sections[current] = []
        elif current:
            sections[current].append(line)
    return sections


def test_the_demo_before_a_standards_build(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "audit"], capsys)
    assert code == 0
    assert out.splitlines()[0] == ("registry audit: 3 finding(s) over 6 resource(s) and 0 pending: no-usage 3")
    rows = _sections(out)["no-usage"]
    assert {r.split(" | ")[0] for r in rows} == {ANIME_XL, BASE_XL, INK}  # the robot LoRA has a manual weight
    assert all(r.endswith("no usage card yet (`easel standards build`)") for r in rows)
    assert run_cli(["standards", "build"], capsys)[0] == 0
    code, out, _err = run_cli(["registry", "audit"], capsys)
    assert out.splitlines()[0] == "registry audit: 1 finding(s) over 6 resource(s) and 0 pending: no-usage 1"
    assert _sections(out)["no-usage"] == [f"{BASE_XL} | checkpoint | sdxl | demo-base-xl.safetensors | "
                                          "only family priors"]


def test_every_check_on_planted_records(home: Path, capsys) -> None:
    no_category = _plant(home, INK, "1a", semantics__category=None, provenance__url=None,
                         provenance__download_url=None, provenance__source_class="unknown", provenance__site=None,
                         identity__path="loras/demo-planted.safetensors", identity__file_name="demo-planted.safetensors")
    trained = _plant(home, ROBOT, "2b", provenance__trained_by=None, identity__path="loras/demo-own.safetensors",
                     identity__file_name="demo-own.safetensors")
    unknown = _plant(home, ANIME_XL, "3c", semantics__content={"rating": "unknown", "class": "unknown",
                                                               "basis": "default"},
                     identity__family=None, identity__path="checkpoints/demo-unrated.safetensors",
                     identity__file_name="demo-unrated.safetensors")
    conflict = _plant(home, VAE, "4d", identity__file_name=f"demo-{_non_sfw_cue()}-vae.safetensors",
                      identity__path="vae/demo-planted.safetensors")
    _edit(home, ANIME_XL, licence__commercial_use="none", licence__derivatives_allowed=False)
    _edit(home, BASE_XL, provenance__url=None, provenance__download_url=None, licence__status="unknown")
    (home / "models" / "upscale_models" / "demo-4x.safetensors").unlink()
    pending = home / "registry" / "pending"
    pending.mkdir(parents=True)
    rec = yaml.safe_load(_path(home, INK).read_text(encoding="utf-8"))
    rec.update(id=None)
    rec["integrity"]["sha256"] = None
    rec["identity"]["path"] = None
    rec["lifecycle"]["install_status"] = "pending"
    (pending / "civitai-9000000901-9000000902.yaml").write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")

    code, out, _err = run_cli(["registry", "audit"], capsys)
    assert code == 0
    found = {check: {row.split(" | ")[0] for row in rows} for check, rows in _sections(out).items()}
    assert found["licence-terms"] == {ANIME_XL}
    assert found["undocumented"] == {BASE_XL, no_category}  # ROBOT and the planted own LoRA: self-trained, exempt
    assert found["unattributed"] == {no_category, trained}
    assert found["no-category"] == {no_category}
    assert found["rating-unknown"] == {unknown}
    assert found["no-family"] == {unknown}
    assert found["rating-conflict"] == {conflict}
    assert UPSCALER in found["missing-file"] and no_category in found["missing-file"]
    assert found["pending"] == {"civitai-9000000901-9000000902"}
    rows = _sections(out)
    assert any("commercial_use: none" in r and "derivatives not allowed" in r for r in rows["licence-terms"])
    assert any(r.startswith(f"{BASE_XL} |") and r.endswith("no source link, licence unknown")
               for r in rows["undocumented"])
    assert any(r.startswith(trained) and "self-trained, but no training record" in r for r in rows["unattributed"])
    # a resource that is not sfw is named by id only
    assert any(r.startswith(f"{unknown} | checkpoint | - | (name withheld: class unknown)")
               for r in rows["rating-unknown"])


def test_only_json_and_check(home: Path, capsys) -> None:
    _edit(home, ANIME_XL, licence__commercial_use="none")
    code, out, _err = run_cli(["registry", "audit", "--only", "licence-terms"], capsys)
    assert code == 0 and list(_sections(out)) == ["licence-terms"]
    code, out, _err = run_cli(["registry", "audit", "--only", "licence-terms", "--json"], capsys)
    data = json.loads(out)
    assert data["counts"] == {"licence-terms": 1}
    assert data["findings"][0]["id"] == ANIME_XL and set(data["fixes"]) == set(auditing.CHECKS)
    assert run_cli(["registry", "audit", "--only", "licence-terms", "--check"], capsys)[0] == 1
    assert run_cli(["registry", "audit", "--only", "pending", "--check"], capsys)[0] == 0
    code, _out, err = run_cli(["registry", "audit", "--only", "no-such-check"], capsys)
    assert code == 2 and "unknown check(s): no-such-check" in err


def test_undocumented_and_unattributed_rules() -> None:
    base = {"identity": {"type": "lora"}, "provenance": {"source_class": "downloaded", "url": None,
                                                         "download_url": "https://example.invalid/f", "trained_by": None},
            "integrity": {"sha256": "a" * 64}, "licence": {"status": "unverified"}}
    assert auditing.undocumented(base) == [] and auditing.unattributed(base) is None
    gone = copy.deepcopy(base)
    gone["provenance"]["download_url"] = None
    assert auditing.undocumented(gone) == ["no source link"]
    assert auditing.unattributed(gone) == "no training record and no source link (source class downloaded)"
    own = copy.deepcopy(gone)
    own["provenance"]["source_class"] = "self-trained"
    assert auditing.undocumented(own) == []
    own["provenance"]["trained_by"] = {"tool": "kiln", "project": "demo_robot", "run": "run-20300102-001"}
    assert auditing.unattributed(own) is None
