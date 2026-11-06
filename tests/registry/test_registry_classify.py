"""The classifier and the shipped registry data: types, families, cue words, folders, compat rules."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from registry_support import REPO, fixtures_module

from easel.registry import classify as cls
from easel.registry import compat
from easel.registry.shipped import DataError, schema_errors


def _header(tensors: list[str], **meta: str) -> cls.Header:
    return cls.Header(tensors={name: {"dtype": "F16", "shape": [1]} for name in tensors}, metadata=meta)


def _demo_headers() -> dict[str, cls.Header]:
    fx = fixtures_module()
    return {fx.model_relpath(i): _header(list(tensors), **meta) for i, (_f, _n, tensors, meta) in enumerate(fx.MODELS)}


# --- type and family ---------------------------------------------------------------------------------------------------

def test_the_demo_headers_give_their_types() -> None:
    types = {rel: cls.type_from_header(h).value for rel, h in _demo_headers().items()}
    assert types == {"checkpoints/demo-base-xl.safetensors": "checkpoint",
                     "checkpoints/demo-anime-xl.safetensors": "checkpoint",
                     "loras/demo-style-ink.safetensors": "lora", "loras/demo_robot-v1.safetensors": "lora",
                     "vae/demo-xl-vae.safetensors": "vae", "upscale_models/demo-4x.safetensors": "upscaler"}


@pytest.mark.parametrize(("tensors", "meta", "expected"), [
    (["lora_unet_x.hada_w1_a"], {}, "lycoris"),
    (["lora_unet_x.lora_down.weight"], {"ss_network_module": "lycoris.kohya"}, "lycoris"),
    (["transformer.blocks.0.lora_A.weight"], {}, "lora"),
    (["control_model.input_hint_block.0.weight"], {}, "controlnet"),
    (["model.diffusion_model.input_blocks.0.0.weight"], {}, "unet"),
    (["double_blocks.0.img_attn.qkv.weight"], {}, "unet"),
    (["text_model.encoder.layers.0.mlp.fc1.weight"], {}, "text_encoder"),
    (["vision_model.encoder.layers.0.mlp.fc1.weight"], {}, "clip_vision"),
    (["clip_l", "clip_g"], {}, "embedding"),
    (["weight"], {}, None),
])
def test_type_from_header(tensors: list[str], meta: dict, expected: str | None) -> None:
    found = cls.type_from_header(_header(tensors, **meta))
    assert (found.value if found else None) == expected


@pytest.mark.parametrize(("meta", "file_name", "tensors", "expected"), [
    ({"modelspec.architecture": "stable-diffusion-xl-v1-base"}, "x.safetensors", [], "sdxl"),
    ({"modelspec.architecture": "flux-1-dev"}, "x.safetensors", [], "flux1"),
    ({"modelspec.architecture": "stable-diffusion-xl-v1-base", "modelspec.title": "Some Illustrious Merge"},
     "x.safetensors", [], "illustrious"),
    ({"ss_base_model_version": "sdxl_base_v1-0", "ss_sd_model_name": "noobaiXL_v11.safetensors"}, "x.safetensors",
     [], "noobai"),
    ({"ss_base_model_version": "sdxl_base_v1-0"}, "demo-style-ink.safetensors", [], "sdxl"),
    ({"ss_base_model_version": "sd_v1"}, "x.safetensors", [], "sd15"),
    ({}, "ponyDiffusionStyle.safetensors", [], "pony"),
    ({}, "ponytail-helper.safetensors", ["lora_te1_x.lora_down.weight"], "sd15"),   # 'ponytail' is not pony
    ({}, "x.safetensors", ["lora_te2_x.lora_down.weight", "lora_te1_x.lora_down.weight"], "sdxl"),
    ({}, "demo-xl-vae.safetensors", [], "sdxl"),
    ({}, "pixel-art.safetensors", [], None),                                        # 'pixel' holds no 'xl'
])
def test_family_from_header(meta: dict, file_name: str, tensors: list[str], expected: str | None) -> None:
    found = cls.family_from_header(_header(tensors, **meta), file_name)
    assert (found.value if found else None) == expected


def test_family_without_a_header_comes_from_the_file_name_only() -> None:
    assert cls.family_from_header(None, "something_illustrious_v2.ckpt").value == "illustrious"
    assert cls.family_from_header(None, "model.ckpt") is None


def test_integrity_facts_and_quantization() -> None:
    header = cls.Header(tensors={"a": {"dtype": "BF16", "shape": [2, 3]}, "b": {"dtype": "BF16", "shape": [4]},
                                 "c": {"dtype": "F32", "shape": [1]}}, metadata={})
    facts = cls.integrity_facts(header)
    assert facts == {"dtypes": ["BF16", "F32"], "tensor_count": 3, "param_count": 11, "precision": "bf16"}
    assert cls.quantization(Path("demo-q4_k_m.gguf")) == "GGUF-Q4_K_M"
    assert cls.quantization(Path("demo.safetensors")) == "none"
    assert cls.quantization(Path("demo-nf4.safetensors")) == "nf4"
    assert cls.quantization(Path("demo.ckpt")) is None


def test_lora_facts_never_guess_trigger_words() -> None:
    header = _header(["lora_unet_x.lora_down.weight"], ss_network_dim="32", ss_network_alpha="16",
                     ss_network_module="networks.lora", ss_clip_skip="None")
    facts = cls.lora_facts(header, "sdxl")
    assert facts["network_dim"] == 32 and facts["network_alpha"] == 16.0 and facts["lora_type"] == "standard"
    assert facts["trigger_words"] == [] and facts["clip_skip"] is None and facts["trained_on_family"] == "sdxl"


def test_split_header_accepts_both_shapes() -> None:
    raw = {"__metadata__": {"k": "v"}, "t": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]}}
    assert cls.split_header(raw) == cls.split_header({"tensors": {"t": raw["t"]}, "metadata": {"k": "v"}})


# --- folders -----------------------------------------------------------------------------------------------------------

def test_shipped_folders_type_every_demo_file() -> None:
    folders = cls.load_folders()
    for rel in _demo_headers():
        assert folders.type_of(rel) is not None, rel
    assert folders.type_of("loras/sub/x.safetensors").value == "lora"       # lycoris shares loras/; lora is first
    assert folders.type_of("Clip/x.safetensors").value == "text_encoder"     # case does not matter
    assert folders.type_of("unet/x.safetensors").value == "unet"
    assert folders.type_of("ultralytics/bbox/x.pt").value == "detector"
    assert folders.type_of("misc/x.safetensors") is None
    assert folders.type_of("x.safetensors") is None
    assert folders.folder_for("upscaler") == "upscale_models"


def test_shipped_folders_cover_every_resource_type() -> None:
    schema_types = set(yaml.safe_load((REPO / "schemas" / "examples" / "folders.example.yaml").read_text())
                       ["type_folders"]) | {"clip", "clip_vision", "ipadapter", "detector", "sam", "style_model",
                                            "model_patch"}
    folders = cls.load_folders()
    assert schema_types <= set(folders.type_folders)
    for comfy in folders.type_folders.values():
        assert comfy in folders.comfyui_folders, comfy


def test_an_instance_folders_file_replaces_the_shipped_one(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "folders.yaml").write_text(
        "schema: easel/folders/1\nbase_path: '{models_root}'\ncomfyui_folders:\n  loras: [Lora]\n"
        "type_folders:\n  lora: loras\n", encoding="utf-8")
    folders = cls.load_folders(tmp_path)
    assert folders.source == "<home>/data/folders.yaml"
    assert folders.type_of("Lora/x.safetensors").value == "lora"
    assert folders.type_of("checkpoints/x.safetensors") is None
    (tmp_path / "data" / "folders.yaml").write_text("schema: easel/folders/1\nbase_path: nowhere\n", encoding="utf-8")
    with pytest.raises(DataError):
        cls.load_folders(tmp_path)


# --- cue words ---------------------------------------------------------------------------------------------------------

def test_words_split_names_the_way_people_write_them() -> None:
    assert cls.words("demo_robot-v1") == ("demo", "robot", "v1")
    assert cls.words("NSFWLora") == ("nsfw", "lora")
    assert cls.words("myStyleXL") == ("my", "style", "xl")
    assert cls.words("R-18 art") == ("r", "18", "art")


def test_rating_is_default_deny_and_non_sfw_wins() -> None:
    markers = cls.load_markers()
    assert cls.rating_from_markers(["demo-style-ink"], markers)[0] == \
        {"rating": "unknown", "class": "unknown", "basis": "default"}
    assert cls.rating_from_markers(["demo-sfw-pack"], markers)[0]["class"] == "sfw"
    assert cls.rating_from_markers(["demo-sfw-nsfw-pack"], markers)[0]["class"] == "nsfw"
    assert cls.rating_from_markers(["not safe for work"], markers)[0]["class"] == "nsfw"
    assert cls.rating_from_markers(["nsfwish"], markers)[0]["class"] == "unknown"  # whole words only


def test_the_third_list_catches_age_down_and_real_person_prompts() -> None:
    markers = cls.load_markers()
    assert markers.subject_hits("dmrbt, robot, standing in a tidy workshop") == []
    assert markers.subject_hits("dmrbt as a Child, full body") == ["child"]
    assert markers.subject_hits("a celebrity lookalike") == ["celebrity", "lookalike"]
    assert markers.subject_hits("kidney bean") == []


def test_content_markers_file_is_three_lists_and_one_comment() -> None:
    text = (REPO / "data" / "content_markers.yaml").read_text(encoding="utf-8")
    comments = [line for line in text.splitlines() if line.lstrip().startswith("#")]
    assert len(comments) == 1
    data = yaml.safe_load(text)
    assert list(data) == list(cls.MARKER_KEYS)
    for key in cls.MARKER_KEYS:
        terms = data[key]
        assert terms and all(isinstance(t, str) and t == t.lower() for t in terms), key
        assert len(terms) == len(set(terms)), key


def test_a_broken_markers_file_is_refused(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "content_markers.yaml").write_text("sfw_cues: [sfw]\n", encoding="utf-8")
    with pytest.raises(DataError):
        cls.load_markers(tmp_path)


# --- compat rules ------------------------------------------------------------------------------------------------------

def test_shipped_compat_rules_are_valid_unique_and_name_known_families() -> None:
    data = yaml.safe_load((REPO / "data" / "compat.yaml").read_text(encoding="utf-8"))
    assert not schema_errors("compat", data)
    keys = [(r["source_family"], r["target_family"], r["artifact_type"]) for r in data["rules"]]
    assert len(keys) == len(set(keys)) == 154
    families = {p.stem for p in (REPO / "families").glob("*.yaml")}
    assert {k[0] for k in keys} | {k[1] for k in keys} <= families
    rules = compat.load_rules()
    assert all(rules[(f, f, t)].verdict == "native" for (f, g, t) in keys if f == g)


def test_instance_rules_win_and_duplicates_are_reported(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "compat.yaml").write_text(
        "schema: easel/compat/1\nrules:\n"
        "- {source_family: sdxl, target_family: pony, artifact_type: lora, verdict: broken, confidence: 0.9}\n"
        "- {source_family: sdxl, target_family: pony, artifact_type: lora, verdict: degraded, confidence: 0.4}\n",
        encoding="utf-8")
    warnings: list[str] = []
    rules = compat.load_rules(tmp_path, warnings)
    assert rules[("sdxl", "pony", "lora")].verdict == "degraded"
    assert rules[("sdxl", "pony", "lora")].origin == "<home>/data/compat.yaml"
    assert len(warnings) == 1 and "two rules" in warnings[0]
