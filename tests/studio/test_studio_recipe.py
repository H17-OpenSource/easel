"""Recipes on the demo registry: propose, explain, revise, like, approve, and the policy refusals."""

from __future__ import annotations

import json

import pytest
from studio_support import CHECKPOINT, LORA, RECIPE, RUN, run_cli, schema_problems


@pytest.fixture
def home(demo_workspace, frozen_clock):
    return demo_workspace


def _recipe(home, rcp):
    return json.loads((home / "recipes" / f"{rcp}.json").read_text(encoding="utf-8"))


def _third_list_term() -> str:
    """A term of the age-down and real-person list, read at run time (no list text is typed here)."""
    from easel.registry import classify

    return " ".join(classify.load_markers(None).subjects[0])


def test_propose_writes_a_schema_valid_recipe(home, capsys):
    code, out, err = run_cli(["recipe", "propose", "--text", "robot, workshop, reading a book", "--family",
                              "illustrious", "--lora", LORA, "--hires", "none", "--size", "portrait", "--seed", "7"],
                             capsys)
    assert code == 0, err
    assert out.splitlines()[0].startswith("rcp-20300101-002 proposed: illustrious, txt2img, content sfw")
    recipe = _recipe(home, "rcp-20300101-002")
    assert schema_problems("recipe", recipe) == []
    assert recipe["status"] == "proposed" and recipe["checkpoint"] == CHECKPOINT
    assert [r["resource"] for r in recipe["resources"]] == [CHECKPOINT, LORA]
    assert recipe["params"]["hires"] == {"mode": "none"}
    assert (recipe["params"]["width"], recipe["params"]["height"], recipe["params"]["seed"]) == (832, 1216, 7)
    assert recipe["request"]["text"] == "robot, workshop, reading a book"  # the request is kept verbatim
    assert recipe["prompt"].endswith("dmrbt, robot, workshop, reading a book")  # trigger, then the text unchanged
    assert recipe["policy"] == {"prompt_authoring": "sfw-only", "subjects": "fictional-adults-only"}
    assert "next: show the operator this recipe; on their yes: easel recipe approve rcp-20300101-002" in out


def test_explain_names_sources_and_licences(home, capsys):
    code, out, _ = run_cli(["recipe", "propose", "--text", "robot", "--family", "illustrious", "--lora", LORA,
                            "--hires", "none", "--explain"], capsys)
    assert code == 0
    why = out[out.index("why:"):]
    assert "requested by name" in why
    assert "native on illustrious" in why
    assert "licence:" in why and "commercial_use: images" in why
    assert "seed        derived from the request text" in why


def test_hires_mode_is_required(home, capsys):
    code, _out, err = run_cli(["recipe", "propose", "--text", "robot", "--family", "illustrious"], capsys)
    assert code == 2
    assert "hires.mode is required" in err
    assert not (home / "recipes" / "rcp-20300101-002.json").exists()


def test_subjects_policy_refuses_and_writes_nothing(home, capsys):
    term = _third_list_term()
    code, out, err = run_cli(["recipe", "propose", "--text", f"robot and a {term} in a workshop", "--hires", "none"],
                             capsys)
    assert code == 2
    assert "policy.subjects = fictional-adults-only" in err and term in err
    assert "never reword around it" in err
    assert out == ""
    assert sorted(p.name for p in (home / "recipes").iterdir()) == [f"{RECIPE}.json"]
    code, _out, err = run_cli(["recipe", "propose", "--text", "robot", "--name", f"{term} portrait", "--hires",
                               "none"], capsys)
    assert code == 2 and "policy.subjects" in err


def test_approve_refuses_a_recipe_with_a_third_list_term(home, capsys):
    path = home / "recipes" / f"{RECIPE}.json"
    recipe = json.loads(path.read_text(encoding="utf-8"))
    recipe.update(status="proposed", prompt=recipe["prompt"] + ", " + _third_list_term())
    path.write_text(json.dumps(recipe), encoding="utf-8")
    code, _out, err = run_cli(["recipe", "approve", RECIPE], capsys)
    assert code == 2 and "policy.subjects" in err
    assert _recipe(home, RECIPE)["status"] == "proposed"


def test_revise_keeps_the_parent_and_changes_one_thing(home, capsys):
    code, _out, err = run_cli(["recipe", "revise", RECIPE, "--set", "cfg=7", "--explain"], capsys)
    assert code == 0, err
    parent, child = _recipe(home, RECIPE), _recipe(home, "rcp-20300101-002")
    assert child["parent"] == RECIPE and child["request"]["kind"] == "revise"
    assert child["checkpoint"] == parent["checkpoint"]
    assert [(r["resource"], r["weight"]) for r in child["resources"]] == \
        [(r["resource"], r["weight"]) for r in parent["resources"]]
    for key in ("seed", "width", "height", "steps", "sampler", "count"):
        assert child["params"][key] == parent["params"][key], key
    assert child["params"]["cfg"] == 7.0
    assert child["rationale"]["cfg"] == "requested (--set)"
    assert child["rationale"]["steps"] == f"parent {RECIPE}"
    assert child["prompt"] == parent["prompt"] and child["negative"] == parent["negative"]
    assert schema_problems("recipe", child) == []


def test_revise_lora_weight_and_unknown_set_key(home, capsys):
    code, _out, err = run_cli(["recipe", "revise", RECIPE, "--lora-weight", f"{LORA}=0.6"], capsys)
    assert code == 0, err
    lora = [r for r in _recipe(home, "rcp-20300101-002")["resources"] if r["role"] == "lora"]
    assert lora[0]["weight"] == 0.6
    code, _out, err = run_cli(["recipe", "revise", RECIPE, "--set", "colour=red"], capsys)
    assert code == 2 and "--set colour=red" in err


def test_like_takes_the_seed_of_one_output(home, capsys):
    code, _out, err = run_cli(["recipe", "like", RUN, "--output", "4"], capsys)
    assert code == 0, err
    child = _recipe(home, "rcp-20300101-002")
    assert child["params"]["seed"] == 1004
    assert child["request"]["kind"] == "like" and child["request"]["from_run"] == RUN
    assert child["request"]["from_output"] == 4 and child["parent"] == RECIPE


def test_not_sfw_is_user_text_only_and_not_echoed(home, capsys):
    code, out, _ = run_cli(["recipe", "propose", "--text", "robot", "--family", "illustrious", "--rating", "nsfw",
                            "--hires", "none"], capsys)
    assert code == 0
    recipe = _recipe(home, "rcp-20300101-002")
    assert recipe["content_class"] == "nsfw"
    assert recipe["policy"]["prompt_authoring"] == "user-text-only"
    assert "not echoed" in out and recipe["prompt"] not in out


def test_hires_model_picks_the_only_upscaler(home, capsys):
    code, _out, err = run_cli(["recipe", "propose", "--text", "robot", "--family", "illustrious", "--hires", "model"],
                              capsys)
    assert code == 0, err
    recipe = _recipe(home, "rcp-20300101-002")
    hires = recipe["params"]["hires"]
    assert hires["mode"] == "model" and hires["upscaler"] == "99134ef8d78fb1bb"
    assert (hires["scale"], hires["steps"]) == (1.5, 15)  # families/illustrious.yaml inference.hires
    assert 0 < hires["denoise"] <= 1
    assert any(r["role"] == "upscaler" for r in recipe["resources"])


def test_show_list_approve(home, capsys):
    code, out, _ = run_cli(["recipe", "show", RECIPE], capsys)
    assert code == 0 and out.startswith(f"{RECIPE} run: illustrious")
    code, out, _ = run_cli(["recipe", "list"], capsys)
    assert code == 0 and out.splitlines()[0].startswith(RECIPE)
    run_cli(["recipe", "propose", "--text", "robot", "--hires", "none"], capsys)
    code, out, _ = run_cli(["recipe", "approve", "rcp-20300101-002"], capsys)
    assert code == 0 and out.splitlines()[0] == "rcp-20300101-002 approved"
    assert _recipe(home, "rcp-20300101-002")["approved_at"] == "2030-01-01T00:00:00+00:00"
    code, out, _ = run_cli(["recipe", "approve", "rcp-20300101-002"], capsys)
    assert code == 0 and "already approved" in out
    code, _out, err = run_cli(["recipe", "show", "rcp-20300101-099"], capsys)
    assert code == 2 and "no recipe rcp-20300101-099" in err


def test_from_json_request(home, tmp_path, capsys):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"text": "robot", "family": "illustrious", "loras": [f"{LORA}:0.7"],
                                   "hires": "none", "set": {"steps": 20}}), encoding="utf-8")
    code, _out, err = run_cli(["recipe", "propose", "--from-json", str(request)], capsys)
    assert code == 0, err
    recipe = _recipe(home, "rcp-20300101-002")
    assert recipe["params"]["steps"] == 20
    assert [r["weight"] for r in recipe["resources"] if r["role"] == "lora"] == [0.7]


def test_a_requested_lora_picks_a_checkpoint_it_works_on(home, capsys):
    code, _out, err = run_cli(["recipe", "propose", "--text", "robot", "--lora", LORA, "--hires", "none", "--explain"],
                              capsys)
    assert code == 0, err
    recipe = _recipe(home, "rcp-20300101-002")
    assert recipe["checkpoint"] == CHECKPOINT  # native for the LoRA; the SDXL base is not
    assert "the requested LoRAs are native on it" in recipe["rationale"]["checkpoint"]


def test_the_legacy_bridges_say_planned_while_absent(home, monkeypatch, capsys):
    from easel.studio import bridge

    monkeypatch.setattr(bridge, "legacy_module", lambda: None)
    code, out, _ = run_cli(["recipe", "to-legacy", RECIPE], capsys)
    assert (code, out.strip()) == (2, "planned: T4")
    code, out, _ = run_cli(["runs", "import-legacy", "demo-run"], capsys)
    assert (code, out.strip()) == (2, "planned: T4")


def test_explain_flags_commercial_use_none(home, capsys):
    import yaml

    path = home / "registry" / "overrides.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["resources"][CHECKPOINT] = {"set": {"licence.commercial_use": "none"}, "note": "planted by the test",
                                     "at": "2030-01-01T00:00:00+00:00"}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    code, out, err = run_cli(["recipe", "propose", "--text", "robot", "--checkpoint", CHECKPOINT, "--hires", "none",
                              "--explain"], capsys)
    assert code == 0, err
    line = next(x for x in out.splitlines() if x.strip().startswith(f"demo-anime-xl.safetensors ({CHECKPOINT}):"))
    assert "commercial_use: none" in line and "FLAG" in line


def test_a_broken_usage_card_is_read_as_no_card(home, capsys):
    """A card that is not YAML any more (an interrupted hand edit) must not crash propose: it counts as absent."""
    card = home / "standards" / "cards" / f"{LORA}.yaml"
    card.write_bytes(b"schema: easel/usage-card/1\nchosen: {weight: [unclosed\n")
    from easel.studio import common
    from easel.studio import recipe as recipes

    assert recipes.card_chosen(common.open_studio("easel"), LORA) == {}
    code, _out, err = run_cli(["recipe", "propose", "--text", "robot", "--lora", LORA, "--family", "illustrious",
                               "--hires", "none"], capsys)
    assert code == 0, err
