"""`easel registry scan [--hash]` on the demo instance: records, index, changelog, and the scan's safety rules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from registry_support import (
    empty_registry,
    header_reader,
    model_sha16,
    run_cli,
    safetensors_bytes,
)

from easel.registry import scan as scanner
from easel.registry.shipped import schema_errors


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _records(home: Path) -> dict[str, dict]:
    folder = home / "registry" / "resources"
    return {p.stem: yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.yaml"))
            if ".usage" not in p.name}


def _changelog(home: Path) -> list[dict]:
    path = home / "registry" / "changelog.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_scan_hash_writes_six_records_and_the_index(home: Path, capsys) -> None:
    empty_registry(home)
    code, out, err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0, out + err
    expected = model_sha16()
    records = _records(home)
    assert sorted(records) == sorted(expected.values())
    for rel, sha16 in expected.items():
        rec = records[sha16]
        assert not schema_errors("resource", rec), (sha16, schema_errors("resource", rec))
        assert rec["id"] == sha16 and rec["identity"]["path"] == rel
        assert rec["integrity"]["sha256"].startswith(sha16)
        assert f"new {sha16} {rel}" in out
    assert (home / "registry" / "INDEX.md").is_file() and (home / "registry" / "index.jsonl").is_file()
    index = [json.loads(line) for line in (home / "registry" / "index.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(row["id"] for row in index) == sorted(expected.values())
    assert "6 records." in (home / "registry" / "INDEX.md").read_text(encoding="utf-8")
    assert "6 new, 0 changed, 0 unchanged, 0 missing" in out
    assert "unknown licence: 6" in out


def test_fresh_records_say_only_what_the_header_shows(home: Path, capsys) -> None:
    empty_registry(home)
    assert run_cli(["registry", "scan", "--hash"], capsys)[0] == 0
    sha = model_sha16()
    records = _records(home)
    types = {rel: records[s]["identity"]["type"] for rel, s in sha.items()}
    assert types == {"checkpoints/demo-base-xl.safetensors": "checkpoint",
                     "checkpoints/demo-anime-xl.safetensors": "checkpoint",
                     "loras/demo-style-ink.safetensors": "lora", "loras/demo_robot-v1.safetensors": "lora",
                     "vae/demo-xl-vae.safetensors": "vae", "upscale_models/demo-4x.safetensors": "upscaler"}
    ink = records[sha["loras/demo-style-ink.safetensors"]]
    assert ink["identity"]["family"] == "sdxl" and ink["lora"]["trained_on_family"] == "sdxl"
    assert ink["lora"]["network_dim"] == 16 and ink["lora"]["network_alpha"] == 8.0
    assert ink["lora"]["trigger_words"] == []  # never guessed from training metadata
    assert ink["semantics"]["content"] == {"rating": "unknown", "class": "unknown", "basis": "default"}
    assert ink["licence"]["status"] == "unknown"
    assert records[sha["upscale_models/demo-4x.safetensors"]]["identity"]["family"] is None
    robot = records[sha["loras/demo_robot-v1.safetensors"]]
    # kohya metadata in the file is no proof of self-training: only overrides.yaml says so
    assert robot["provenance"]["source_class"] == "self-trained"
    assert robot["semantics"]["category"] == "character"
    others = [r for s, r in records.items() if s != robot["id"]]
    assert all(r["provenance"]["source_class"] == "unknown" for r in others)


def test_scan_changelog_lines_follow_the_schema(home: Path, capsys) -> None:
    before = len(_changelog(home))
    empty_registry(home)
    assert run_cli(["registry", "scan", "--hash"], capsys)[0] == 0
    lines = _changelog(home)
    assert before == 7 and len(lines) == 7  # 6 new files + 1 override applied (the old log went with the records)
    for line in lines:
        assert next(iter(line)) == "schema"
        assert not schema_errors("changelog-event", line), schema_errors("changelog-event", line)
    sets = [line for line in lines if line["action"] == "set"]
    assert len(sets) == 1 and sets[0]["source"] == "operator" and sets[0]["target"]["kind"] == "override"
    assert sets[0]["changes"]["provenance.source_class"] == {"from": "unknown", "to": "self-trained"}


def test_scan_without_hash_registers_nothing_new(home: Path, capsys) -> None:
    empty_registry(home)
    code, out, _err = run_cli(["registry", "scan"], capsys)
    assert code == 0
    assert _records(home) == {}
    assert out.count("not hashed yet") == 6
    assert "next: easel registry scan --hash" in out
    assert "no such resource in the registry yet" in out  # the override waits for its file


def test_rescan_keeps_what_only_a_person_or_an_api_knows(home: Path, capsys) -> None:
    original = _records(home)
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    assert "0 new, 0 changed, 6 unchanged" in out
    assert "0 hashed" in out  # same size and older than hashed_at: not read again
    after = _records(home)
    for sha16, old in original.items():
        new = after[sha16]
        for block in ("provenance", "licence", "semantics", "when_to_use", "usage", "relations"):
            assert new[block] == old[block], (sha16, block)
        assert new["identity"] == old["identity"], sha16  # the demo's illustrious families survive
        assert new["lifecycle"]["scanned_at"] == "2030-01-01T00:00:00+00:00"
        assert new["lifecycle"]["updated_at"] == old["lifecycle"]["updated_at"]


def test_a_second_scan_changes_no_byte(home: Path, capsys) -> None:
    assert run_cli(["registry", "scan", "--hash"], capsys)[0] == 0
    folder = home / "registry"
    snapshot = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
                and p.name != ".lock"}
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0 and "wrote 0 record(s)" in out
    again = {p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()
             and p.name != ".lock"}
    assert again == snapshot


def test_rehash_reads_every_file(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "scan", "--hash", "--rehash"], capsys)
    assert code == 0 and "6 hashed" in out and "6 unchanged" in out


def test_a_file_that_goes_away_turns_missing_and_comes_back(home: Path, capsys, tmp_path: Path) -> None:
    rel = "vae/demo-xl-vae.safetensors"
    sha16 = model_sha16()[rel]
    parked = tmp_path / "parked.safetensors"
    shutil.move(home / "models" / rel, parked)
    code, out, _err = run_cli(["registry", "scan"], capsys)
    assert code == 0 and f"missing {sha16} {rel}" in out
    assert _records(home)[sha16]["lifecycle"]["install_status"] == "missing"
    shutil.move(parked, home / "models" / rel)
    code, out, _err = run_cli(["registry", "scan"], capsys)
    assert code == 0
    assert _records(home)[sha16]["lifecycle"]["install_status"] == "active"
    actions = [(line["target"]["id"], line["changes"].get("lifecycle.install_status")) for line in _changelog(home)[-2:]]
    assert actions == [(sha16, {"from": "active", "to": "missing"}), (sha16, {"from": "missing", "to": "active"})]


def test_a_moved_file_keeps_its_record(home: Path, capsys) -> None:
    rel, moved = "loras/demo-style-ink.safetensors", "loras/styles/demo-style-ink.safetensors"
    sha16 = model_sha16()[rel]
    (home / "models" / "loras" / "styles").mkdir()
    shutil.move(home / "models" / rel, home / "models" / moved)
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0 and f"changed {sha16} {moved}: identity.path" in out
    rec = _records(home)[sha16]
    assert rec["identity"]["path"] == moved and rec["lifecycle"]["install_status"] == "active"
    assert rec["provenance"]["author"] == "demo_author"  # the record moved with the file


def test_the_same_bytes_twice_are_one_resource(home: Path, capsys) -> None:
    rel = "checkpoints/demo-base-xl.safetensors"
    shutil.copyfile(home / "models" / rel, home / "models" / "checkpoints" / "copy-of-base.safetensors")
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    assert len(_records(home)) == 6
    assert f"duplicate checkpoints/copy-of-base.safetensors: the same bytes as {rel}" in out
    assert _records(home)[model_sha16()[rel]]["identity"]["path"] == rel


def test_a_lycoris_in_the_lora_folder_is_typed_by_its_header(home: Path, capsys) -> None:
    data = safetensors_bytes({"lora_unet_down_blocks_0.hada_w1_a": "F16", "lora_unet_down_blocks_0.hada_w1_b": "F16"},
                             {"ss_network_module": "lycoris.kohya", "ss_base_model_version": "sdxl_base_v1-0"})
    (home / "models" / "loras" / "demo-loha.safetensors").write_bytes(data)
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    new = [r for r in _records(home).values() if r["identity"]["file_name"] == "demo-loha.safetensors"]
    assert len(new) == 1 and new[0]["identity"]["type"] == "lycoris"
    assert new[0]["lora"]["lora_type"] == "loha"


def test_a_file_nobody_can_type_is_reported_not_guessed(home: Path, capsys) -> None:
    (home / "models" / "misc").mkdir()
    (home / "models" / "misc" / "mystery.safetensors").write_bytes(safetensors_bytes({"weight": "F32"}))
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    assert "skipped misc/mystery.safetensors: not in a known folder" in out
    assert len(_records(home)) == 6


def test_a_placeholder_is_neither_read_nor_marked_missing(home: Path, capsys, monkeypatch) -> None:
    real = scanner._placeholder

    def fake(stat):
        return stat.st_size == 294 or real(stat)  # the upscaler's size: pretend it is online-only

    monkeypatch.setattr(scanner, "_placeholder", fake)
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    assert "skipped upscale_models/demo-4x.safetensors: an online-only placeholder" in out
    assert _records(home)[model_sha16()["upscale_models/demo-4x.safetensors"]]["lifecycle"]["install_status"] == "active"


def test_a_bad_override_value_is_refused_and_reported(home: Path, capsys) -> None:
    path = home / "registry" / "overrides.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["resources"]["cf56db17414bcb55"]["set"]["semantics.content.rating"] = "great"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    code, _out, err = run_cli(["registry", "scan"], capsys)
    assert code == 1
    assert "override cf56db17414bcb55 semantics.content.rating: refused" in err
    rec = _records(home)["cf56db17414bcb55"]
    assert rec["semantics"]["content"]["rating"] == "sfw" and rec["provenance"]["source_class"] == "self-trained"


def test_a_sfw_record_with_a_non_sfw_cue_is_a_rating_conflict(home: Path, capsys) -> None:
    rel = "loras/demo-style-ink.safetensors"
    sha16 = model_sha16()[rel]
    shutil.move(home / "models" / rel, home / "models" / "loras" / "demo-style-ink-nsfw.safetensors")
    code, out, _err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    assert f"RATING CONFLICT {sha16} loras/demo-style-ink-nsfw.safetensors: rated sfw (basis source)" in out
    assert "rating conflicts: 1" in out
    assert _records(home)[sha16]["semantics"]["content"]["class"] == "sfw"  # the operator decides, not the scan


def test_scan_refuses_without_the_models_folder(home: Path, capsys) -> None:
    shutil.rmtree(home / "models")
    code, _out, err = run_cli(["registry", "scan"], capsys)
    assert code == 2 and "models folder not found: <models_root>" in err


def test_scan_output_names_no_absolute_path(home: Path, capsys) -> None:
    empty_registry(home)
    code, out, err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0
    assert str(home) not in out and str(home) not in err
    assert str(home.as_posix()) not in out
