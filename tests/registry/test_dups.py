"""`easel registry dups`, and the scan over `paths.extra_model_roots` (the roots dups walks too)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli, safetensors_bytes

from easel._common import hashing
from easel.registry import dups as duplicates

INK, UPSCALER = "60d12866cf6377b0", "99134ef8d78fb1bb"


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _extra_root(home: Path, name: str = "more") -> Path:
    """Point paths.extra_model_roots at <home>/<name> (written into the demo's easel.toml)."""
    toml = home / "easel.toml"
    text = toml.read_text(encoding="utf-8")
    text = text.replace('models_root = "models"\n', f'models_root = "models"\nextra_model_roots = ["{name}"]\n')
    toml.write_text(text, encoding="utf-8")
    (home / name).mkdir(exist_ok=True)
    return home / name


def test_no_duplicates_in_the_demo(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "dups"], capsys)
    assert code == 0
    assert out.strip() == "dups: no two files hold the same bytes among 6 file(s) under 1 model root(s)"


def test_a_copy_is_probable_then_confirmed_with_full(home: Path, capsys) -> None:
    src = home / "models" / "loras" / "demo-style-ink.safetensors"
    shutil.copyfile(src, home / "models" / "loras" / "demo-style-ink (copy).safetensors")
    code, out, err = run_cli(["registry", "dups"], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith("~") and lines[0].endswith("| probable (8 sampled chunks match; --full confirms)")
    assert "<models_root>/loras/demo-style-ink (copy).safetensors | <models_root>/loras/demo-style-ink.safetensors" \
        in lines[0]
    assert "sampling" in err
    code, out, _err = run_cli(["registry", "dups", "--full"], capsys)
    first = out.splitlines()[0]
    assert first.startswith(f"{INK} | 2 files | {src.stat().st_size} bytes each | ")
    assert first.endswith("| confirmed (sha256)")
    assert out.splitlines()[-1].startswith("dups: 1 group(s) (1 confirmed, 0 probable), 1 extra file(s), ")
    code, out, _err = run_cli(["registry", "dups", "--full", "--json"], capsys)
    groups = json.loads(out)
    assert groups == [{"sha256": hashing.sha256_file(src), "sample": None, "confirmed": True,
                       "bytes": src.stat().st_size, "files": ["<models_root>/loras/demo-style-ink (copy).safetensors",
                                                              "<models_root>/loras/demo-style-ink.safetensors"]}]


def test_same_size_other_bytes_is_no_group(home: Path, capsys) -> None:
    src = home / "models" / "loras" / "demo-style-ink.safetensors"
    data = bytearray(src.read_bytes())
    data[-1] ^= 0xFF
    (home / "models" / "loras" / "demo-ink-other.safetensors").write_bytes(bytes(data))
    code, out, _err = run_cli(["registry", "dups"], capsys)
    assert code == 0 and out.startswith("dups: no two files hold the same bytes among 7 file(s)")


def test_known_hashes_are_not_read_again(home: Path) -> None:
    from easel._common import config
    from easel.registry import resources as res

    cfg = config.load("easel", home, strict=True)
    reg = res.open_registry(cfg, "easel")
    records = res.load_records(reg).records
    report = duplicates.find(reg, records)
    assert report.sampled == 0 and report.hashed == 0 and report.groups == []


def test_scan_walks_the_extra_roots_after_models_root(home: Path, capsys) -> None:
    more = _extra_root(home)
    (more / "loras").mkdir()
    extra = safetensors_bytes({"lora_unet_mid.lora_down.weight": "F16"}, {"ss_network_dim": "4"})
    (more / "loras" / "demo-extra.safetensors").write_bytes(extra)
    shutil.copyfile(home / "models" / "upscale_models" / "demo-4x.safetensors",
                    more / "upscale_models.tmp")  # not a model suffix: ignored
    (more / "upscale_models").mkdir()
    (more / "upscale_models" / "demo-4x.safetensors").write_bytes(b"different bytes, same name")
    code, out, err = run_cli(["registry", "scan", "--hash"], capsys)
    assert code == 0, err
    sha16 = hashing.sha256_bytes(extra)[:16]
    assert any(line.startswith(f"new {sha16} loras/demo-extra.safetensors") for line in out.splitlines())
    assert "shadowed <extra_model_roots[0]>/upscale_models/demo-4x.safetensors: an earlier model root has the " \
           "same name" in out
    record = yaml.safe_load((home / "registry" / "resources" / f"{sha16}.yaml").read_text(encoding="utf-8"))
    assert record["identity"]["path"] == "loras/demo-extra.safetensors"
    upscaler = yaml.safe_load((home / "registry" / "resources" / f"{UPSCALER}.yaml").read_text(encoding="utf-8"))
    assert upscaler["lifecycle"]["install_status"] == "active"  # the models_root copy keeps its record
    # explain finds the extra root's file for the header
    code, out, _err = run_cli(["registry", "explain", sha16], capsys)
    assert code == 0 and "header: safetensors header read" in out
    # a second scan changes nothing; removing the extra root's file turns its record missing
    assert run_cli(["registry", "scan"], capsys)[0] == 0
    (more / "loras" / "demo-extra.safetensors").unlink()
    code, out, _err = run_cli(["registry", "scan"], capsys)
    assert code == 0 and f"missing {sha16} loras/demo-extra.safetensors" in out


def test_a_missing_extra_root_is_reported_and_skipped(home: Path, capsys) -> None:
    more = _extra_root(home, "gone")
    more.rmdir()
    code, out, _err = run_cli(["registry", "scan"], capsys)
    assert code == 0 and "skipped <extra_model_roots[0]> (paths.extra_model_roots): not a folder" in out
    code, _out, err = run_cli(["registry", "dups"], capsys)
    assert code == 0 and "skipped <extra_model_roots[0]>" in err


def test_dups_finds_a_copy_under_an_extra_root(home: Path, capsys) -> None:
    more = _extra_root(home)
    (more / "loras").mkdir()
    shutil.copyfile(home / "models" / "loras" / "demo-style-ink.safetensors",
                    more / "loras" / "demo-style-ink.safetensors")
    code, out, _err = run_cli(["registry", "dups", "--full"], capsys)
    assert code == 0
    assert out.splitlines()[0].endswith("| <models_root>/loras/demo-style-ink.safetensors | "
                                        "<extra_model_roots[0]>/loras/demo-style-ink.safetensors | confirmed (sha256)")
    assert "under 2 model root(s)" in out.splitlines()[-1]
