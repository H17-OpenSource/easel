"""Schemas, examples, shipped data and the demo instance, checked against schemas/ (JSON Schema draft 2020-12).

Why: the schemas freeze the file formats this tool writes and reads, and examples/demo/ is the contract the
commands are tested on. A record that drifts from its schema, a schema without an example, a demo hash that no
longer matches the generated media, or a real date in a fixture fails here first.
The file is the same in kiln and easel except PKG; the demo checks at the bottom are per tool.
"""

import hashlib
import importlib.util
import json
import re
import tomllib
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

PKG = "easel"
REPO = Path(__file__).resolve().parents[1]
SCHEMAS = REPO / "schemas"
EXAMPLES = SCHEMAS / "examples"
DEMO = REPO / "examples" / "demo"
DRAFT = "https://json-schema.org/draft/2020-12/schema"
# Schemas another part of the build owns (kiln's LoRA-type files, checked by `kiln types check`): their `$id` and
# record key are checked only where present, and they need no example here.
OTHER_OWNER_PREFIXES = ("type-",)
# Demo files that are not records: prose, card templates and ComfyUI's own API graph format.
NOT_RECORDS = {"README.md", "card.md.tmpl", "card-version.md.tmpl", "graph.json"}
DATE_LIKE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]|$)")
ID_YEAR = re.compile(r"\b(?:run|rcp|job|les)-(\d{4})\d{4}-")
RECORD_SUFFIXES = {".json", ".jsonl", ".yaml", ".yml", ".toml", ".md"}


# --- loading -----------------------------------------------------------------------------------------------------

def schema_files() -> list[Path]:
    return sorted(SCHEMAS.glob("*.schema.json"))


def schema_name(path: Path) -> str:
    return path.name.removesuffix(".schema.json")


def is_other_owner(name: str) -> bool:
    return name.startswith(OTHER_OWNER_PREFIXES)


@cache
def load_schema(name: str) -> dict:
    return json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))


@cache
def validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(load_schema(name))


def record_const(schema: dict) -> str | None:
    return (schema.get("properties", {}).get("schema") or {}).get("const")


def load_records(path: Path) -> list[tuple[str, object]]:
    """Every record in one file as (label, data): a JSONL file gives one per line, a .md file its front matter."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [(f"{path.name}:{n}", json.loads(line)) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    if path.suffix == ".json":
        return [(path.name, json.loads(text))]
    if path.suffix in (".yaml", ".yml"):
        return [(path.name, yaml.safe_load(text))]
    if path.suffix == ".toml":
        return [(path.name, tomllib.loads(text))]
    if path.suffix == ".md":
        parts = text.split("---\n")
        if not text.startswith("---\n") or len(parts) < 3:
            raise AssertionError(f"{path.name}: a record .md starts with YAML front matter between --- lines")
        return [(f"{path.name} (front matter)", yaml.safe_load(parts[1]))]
    raise AssertionError(f"{path.name}: not a record file type")


def name_of_record(data: object, where: str) -> str:
    """The schema name a record declares in its first key `schema: "<pkg>/<name>/<version>"`."""
    assert isinstance(data, dict) and data, f"{where}: a record is a non-empty mapping"
    first = next(iter(data))
    assert first == "schema", f"{where}: the first key must be `schema`, found {first!r}"
    match = re.fullmatch(rf"{PKG}/([a-z][a-z0-9-]*)/([1-9][0-9]*)", str(data["schema"]))
    assert match, f"{where}: schema {data['schema']!r} is not '{PKG}/<name>/<version>'"
    assert (SCHEMAS / f"{match.group(1)}.schema.json").is_file(), f"{where}: no schemas/{match.group(1)}.schema.json"
    return match.group(1)


def check(data: object, where: str, expect: str | None = None) -> None:
    """Validate one record against the schema it names (and, when given, that it names `expect`)."""
    name = name_of_record(data, where)
    if expect is not None:
        assert name == expect, f"{where}: names schema {name!r}, expected {expect!r}"
    errors = sorted(validator(name).iter_errors(data), key=lambda e: list(e.absolute_path))
    assert not errors, f"{where} vs {name}.schema.json:\n" + "\n".join(
        f"  at /{'/'.join(map(str, e.absolute_path))}: {e.message[:300]}" for e in errors[:10])


def errors_of(name: str, data: object) -> list[str]:
    return [e.message for e in validator(name).iter_errors(data)]


@cache
def fixtures_module():
    """tests/fixtures/make_fixtures.py, imported by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("make_fixtures", REPO / "tests" / "fixtures" / "make_fixtures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def demo_files() -> list[Path]:
    if not DEMO.is_dir():
        return []
    return sorted(p for p in DEMO.rglob("*") if p.is_file())


def demo_record_files() -> list[Path]:
    return [p for p in demo_files() if p.name not in NOT_RECORDS]


def walk_strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for key, value in node.items():
            yield str(key)
            yield from walk_strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk_strings(value)


def latest(lines: list[dict], key: str) -> dict[str, dict]:
    """Reduce an event log the way readers do: the last line per key wins."""
    out: dict[str, dict] = {}
    for line in lines:
        out[line[key]] = line
    return out


# --- schemas -------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", schema_files(), ids=schema_name)
def test_schema_is_valid_draft_2020_12(path: Path) -> None:
    schema = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert schema.get("$schema") == DRAFT
    assert schema.get("title") and schema.get("description"), "every schema says what it is"


@pytest.mark.parametrize("path", schema_files(), ids=schema_name)
def test_schema_id_and_record_key_follow_the_convention(path: Path) -> None:
    """$id = urn:<pkg>:<name>:<version>; the record key const = "<pkg>/<name>/<version>", the same version."""
    name, schema = schema_name(path), json.loads(path.read_text(encoding="utf-8"))
    const, sid = record_const(schema), schema.get("$id")
    if is_other_owner(name) and const is None and sid is None:
        pytest.skip(f"{name} is checked by its owner's validator")
    if not is_other_owner(name) or const is not None:
        assert const is not None and re.fullmatch(rf"{PKG}/{re.escape(name)}/[1-9][0-9]*", const), \
            f"properties.schema.const must be '{PKG}/{name}/<version>', found {const!r}"
    if not is_other_owner(name) or sid is not None:
        version = const.rsplit("/", 1)[1] if const else r"[1-9][0-9]*"
        assert sid is not None and re.fullmatch(rf"urn:{PKG}:{re.escape(name)}:{version}", sid), \
            f"$id must be 'urn:{PKG}:{name}:{version}', found {sid!r}"
    if not is_other_owner(name):
        assert schema.get("required", [None])[0] == "schema", "`schema` is the first required key"
        assert next(iter(schema["properties"])) == "schema", "`schema` is the first property"


def test_every_record_schema_has_exactly_one_example() -> None:
    names = {schema_name(p) for p in schema_files() if not is_other_owner(schema_name(p))}
    examples: dict[str, list[str]] = {}
    for path in sorted(EXAMPLES.glob("*.example.*")):
        examples.setdefault(path.name.split(".example.")[0], []).append(path.name)
    missing = sorted(names - set(examples))
    assert not missing, f"schemas without an example in schemas/examples/: {missing}"
    doubled = {k: v for k, v in examples.items() if len(v) > 1}
    assert not doubled, f"more than one example: {doubled}"
    orphans = sorted(set(examples) - {schema_name(p) for p in schema_files()})
    assert not orphans, f"examples without a schema: {orphans}"


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.example.*")), ids=lambda p: p.name)
def test_example_validates(path: Path) -> None:
    assert path.suffix in (".json", ".yaml"), "examples are .json or .yaml"
    (label, data), = load_records(path)
    check(data, f"schemas/examples/{label}", expect=path.name.split(".example.")[0])


def test_example_toml_validates_against_the_config_schema() -> None:
    path = REPO / f"{PKG}.example.toml"
    (label, data), = load_records(path)
    check(data, label, expect="config")


@pytest.mark.parametrize("path", sorted((REPO / "families").glob("*.yaml")), ids=lambda p: p.stem)
def test_shipped_family_validates(path: Path) -> None:
    (label, data), = load_records(path)
    check(data, f"families/{label}", expect="family")
    assert data["id"] == path.stem, "a family's id equals its file stem"


def _shipped_data_files() -> list[Path]:
    found = []
    for folder in ("types", "data", "workflows"):
        root = REPO / folder
        if root.is_dir():
            found += [p for p in sorted(root.rglob("*")) if p.suffix in (".json", ".yaml", ".yml")]
    return found


@pytest.mark.parametrize("path", _shipped_data_files(), ids=lambda p: p.relative_to(REPO).as_posix())
def test_shipped_data_naming_a_schema_validates(path: Path) -> None:
    """Data files that name one of this lane's schemas (for example a type's spec template) must validate."""
    (label, data), = load_records(path)
    declared = data.get("schema") if isinstance(data, dict) else None
    if not isinstance(declared, str) or not declared.startswith(f"{PKG}/"):
        pytest.skip("not a record")
    name = declared.split("/")[1]
    if is_other_owner(name):
        pytest.skip(f"{name} is checked by its owner's validator")
    check(data, path.relative_to(REPO).as_posix())


# --- the schemas refuse what they must refuse ----------------------------------------------------------------------

def _example(name: str) -> dict:
    path = next(EXAMPLES.glob(f"{name}.example.*"))
    return load_records(path)[0][1]


def _set(data: dict, dotted: str, value: object) -> dict:
    node = data
    *parents, last = dotted.split(".")
    for key in parents:
        node = node[key]
    if value is _DELETE:
        del node[last]
    else:
        node[last] = value
    return data


_DELETE = object()
REFUSALS = [
    ("config", "paths.no_such_key", "x", "an unknown key"),
    ("config", "policy.subjects", "anything", "a relaxed subjects policy"),
    ("config", "policy.unknown_rating_is", "sfw", "unknown content treated as sfw"),
    ("config", "scheduler.kind", "broker", "an unknown scheduler kind"),
    ("config", "review.port", "8741", "a port that is not a number"),
    ("family", "licence_summary", _DELETE, "a family without its own licence summary"),
    ("family", "host_ram_gb", _DELETE, "a family without host_ram_gb"),
    ("family", "status", "great", "an unknown status"),
]
if PKG == "kiln":
    REFUSALS += [
        ("config", "train.venv_dir", "runtime/venv-train/Scripts/python.exe", "venv_dir naming the interpreter"),
        ("family", "training", _DELETE, "a root family without its training block"),
        ("image-event", "file", "/" + "abs/demo.png", "an absolute file path"),
        ("image-event", "root", "elsewhere", "an unknown image root"),
        ("image-event", "id", "img-XYZ", "a malformed image id"),
        ("snapshot", "caption_level", "long", "a second caption level"),
        ("run", "base.licence", _DELETE, "a run without the base's licence"),
        ("decision-frontmatter", "status", "decided", "decided without the decision"),
        ("pick-event", "source", "assistant", "a pick from the assistant"),
    ]
else:
    REFUSALS += [
        ("family", "inference", _DELETE, "a root family without its inference block"),
        ("recipe", "params.hires.mode", _DELETE, "a recipe without hires.mode"),
        ("recipe", "policy.subjects", "anything", "a recipe outside the subjects policy"),
        ("resource", "id", None, "a resource with a hash but no id"),
        ("resource", "identity.path", "C:" + "/models/x.safetensors", "a drive path"),
        ("run-review-event", "verdict", "love", "an unknown verdict"),
        ("candidates-entry", "class", "fine", "an unknown vet class"),
    ]


@pytest.mark.parametrize(("name", "dotted", "value", "why"), REFUSALS, ids=[r[3] for r in REFUSALS])
def test_schema_refuses(name: str, dotted: str, value: object, why: str) -> None:
    data = _set(_example(name), dotted, value)
    assert errors_of(name, data), f"{name}.schema.json accepted {why}"


# --- the demo instance -----------------------------------------------------------------------------------------------

needs_demo = pytest.mark.skipif(not DEMO.is_dir(), reason="examples/demo/ not built yet")


@needs_demo
def test_demo_files_are_records_or_listed() -> None:
    odd = [p.relative_to(DEMO).as_posix() for p in demo_files()
           if p.name not in NOT_RECORDS and p.suffix not in RECORD_SUFFIXES]
    assert not odd, f"demo files that are neither records nor in NOT_RECORDS: {odd}"
    binaries = [p.relative_to(DEMO).as_posix() for p in demo_files() if b"\x00" in p.read_bytes()]
    assert not binaries, f"binary files committed in the demo (generate them instead): {binaries}"


@needs_demo
@pytest.mark.parametrize("path", demo_record_files(), ids=lambda p: p.relative_to(DEMO).as_posix())
def test_demo_record_validates(path: Path) -> None:
    for label, data in load_records(path):
        check(data, f"examples/demo/{path.relative_to(DEMO).parent.as_posix()}/{label}")


@needs_demo
def test_demo_and_examples_use_2030_only() -> None:
    """Fixtures carry no real date or id year: every date-like string and every id stamp is in 2030."""
    sources = demo_record_files() + sorted(EXAMPLES.glob("*.example.*"))
    bad = []
    for path in sources:
        for label, data in load_records(path):
            for text in walk_strings(data):
                if DATE_LIKE.match(text) and not text.startswith("2030-"):
                    bad.append(f"{label}: {text}")
                bad += [f"{label}: {text}" for year in ID_YEAR.findall(text) if year != "2030"]
    assert not bad, "dates or ids outside 2030:\n" + "\n".join(bad[:20])


@needs_demo
def test_demo_workspace_places_the_generated_media(demo_workspace: Path) -> None:
    fixtures = fixtures_module()
    files = fixtures.all_files()
    for rel, dest in fixtures.DEMO_MEDIA[PKG].items():
        target = demo_workspace / dest
        assert target.is_file(), f"demo_workspace did not place {dest}"
        assert target.read_bytes() == files[rel], f"{dest} differs from the generated {rel}"


# --- the generated media ---------------------------------------------------------------------------------------------

def test_fixtures_are_deterministic_and_idempotent(tmp_path: Path) -> None:
    fixtures = fixtures_module()
    first = fixtures.generate(tmp_path / "a")
    second = fixtures.generate(tmp_path / "b")
    assert {k: v.read_bytes() for k, v in first.items()} == {k: v.read_bytes() for k, v in second.items()}
    stamps = {k: v.stat().st_mtime_ns for k, v in first.items()}
    fixtures.generate(tmp_path / "a")  # a second run leaves every file alone
    assert stamps == {k: v.stat().st_mtime_ns for k, v in first.items()}
    pngs = [k for k in first if k.startswith("png/")]
    assert len(pngs) == fixtures.PNG_COUNT == 24
    assert len({hashlib.sha256(first[k].read_bytes()).hexdigest() for k in pngs}) == 24, "every PNG differs"


def test_fixture_pngs_open_as_8x8_with_their_text_chunks(tmp_path: Path) -> None:
    from PIL import Image

    fixtures = fixtures_module()
    made = fixtures.generate(tmp_path)
    with_text = 0
    for n in range(1, fixtures.PNG_COUNT + 1):
        with Image.open(made[f"png/{fixtures.png_name(n)}"]) as image:
            image.load()
            assert image.size == (8, 8) and image.mode == "RGB"
            keys = sorted(getattr(image, "text", {}))
            assert keys == sorted(k for k, _ in fixtures.png_text(n))
            for key in keys:
                assert isinstance(json.loads(image.text[key]), dict), f"fixture {n}: {key} is a JSON graph"
            with_text += bool(keys)
    assert with_text == 8


def test_fixture_models_are_safetensors_headers(tmp_path: Path) -> None:
    fixtures = fixtures_module()
    made = fixtures.generate(tmp_path)
    for i in range(len(fixtures.MODELS)):
        data = made[f"models/{fixtures.model_relpath(i)}"].read_bytes()
        size = int.from_bytes(data[:8], "little")
        header = json.loads(data[8:8 + size])
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        assert tensors and len(data) == 8 + size + 2 * len(tensors)


# --- per-tool demo contracts -----------------------------------------------------------------------------------------

def _jsonl(path: Path) -> list[dict]:
    return [data for _, data in load_records(path)]


def _media_sha(dest: str) -> str:
    """sha256 of the generated file the demo expects at `dest` (a path relative to the demo home)."""
    fixtures = fixtures_module()
    by_dest = {d: rel for rel, d in fixtures.DEMO_MEDIA[PKG].items()}
    assert dest in by_dest, f"the demo refers to {dest}, which make_fixtures does not place"
    return hashlib.sha256(fixtures.all_files()[by_dest[dest]]).hexdigest()


if PKG == "kiln":
    PROJECT = DEMO / "projects" / "demo_robot"

    @needs_demo
    def test_kiln_demo_images_match_the_generated_media() -> None:
        config = tomllib.loads((DEMO / "kiln.toml").read_text(encoding="utf-8"))
        roots = {"images": config["paths"]["images_root"], "imports": "projects/demo_robot/images"}
        images = latest(_jsonl(PROJECT / "images.jsonl"), "id")
        for image_id, line in images.items():
            assert image_id == "img-" + line["sha256"][:12], f"{image_id}: id is img-<sha256[:12]>"
            assert line["sha256"] == _media_sha(f"{roots[line['root']]}/{line['file']}"), f"{image_id}: sha256"
        assert len(images) == 19

    @needs_demo
    def test_kiln_demo_snapshots_support_freeze_from() -> None:
        images = latest(_jsonl(PROJECT / "images.jsonl"), "id")
        captions = latest(_jsonl(PROJECT / "captions.jsonl"), "image")
        old = json.loads((PROJECT / "snapshots" / "0.1.0.json").read_text(encoding="utf-8"))
        new = json.loads((PROJECT / "snapshots" / "0.2.0.json").read_text(encoding="utf-8"))
        assert new["from_version"] == old["version"] == "0.1.0" and old["from_version"] is None
        for snap in (old, new):
            members = snap["members"]
            digest = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert snap["members_sha256"] == digest, f"{snap['id']}: members_sha256"
            assert members == sorted(members, key=lambda m: m["image"])
            for split in ("train", "holdout"):
                assert snap["counts"][split] == sum(m["split"] == split for m in members)
            roots: dict[str, set[str]] = {}
            for m in members:
                assert m["image"] in images and images[m["image"]]["sha256"] == m["sha256"]
                assert captions[m["image"]]["status"] == "approved", f"{m['image']} frozen without an approved caption"
                roots.setdefault(images[m["image"]]["lineage"]["root"], set()).add(m["split"])
            assert all(len(s) == 1 for s in roots.values()), f"{snap['id']}: a lineage leaks across splits"
        assert {m["image"] for m in old["members"]} <= {m["image"] for m in new["members"]}
        frozen = {m["image"] for m in new["members"]}
        fresh = [i for i, line in images.items() if line["disposition"] == "ACCEPT_TRAIN" and i not in frozen]
        assert len(new["members"]) == 12 and len(fresh) == 3, "freeze 0.3.0 --from 0.2.0 gives 12 + 3 members"
        assert all(captions[i]["status"] == "approved" for i in fresh)

    @needs_demo
    def test_kiln_demo_project_shape() -> None:
        spec = json.loads((PROJECT / "spec.json").read_text(encoding="utf-8"))
        keys = [item["key"] for item in spec["checklist"]]
        assert len(keys) == 6 == len(set(keys)), "the demo spec has 6 checklist items"
        state = json.loads((PROJECT / "state.json").read_text(encoding="utf-8"))
        assert state["goal"]["from_version"] == "0.2.0"
        images = latest(_jsonl(PROJECT / "images.jsonl"), "id")
        assert sum(line["legacy_untriaged"] for line in images.values()) == 1
        assert {line["root"] for line in images.values()} == {"images", "imports"}
        run = json.loads((PROJECT / "runs" / "run-20300102-001" / "run.json").read_text(encoding="utf-8"))
        assert run["status"] == "ok" and run["snapshot"] == "demo_robot/0.2.0" and "licence" in run["base"]
        for name in ("intake.yaml", "showcase.yaml", "coverage.yaml", "jobs/batch-01.json", "evals/suites/core.yaml",
                     "card.md.tmpl", "card-version.md.tmpl"):
            assert (PROJECT / name).is_file(), name

if PKG == "easel":
    @needs_demo
    def test_easel_demo_resources_match_the_generated_models() -> None:
        resources = {}
        for path in sorted((DEMO / "registry" / "resources").glob("*.yaml")):
            if path.name.endswith(".usage.yaml"):
                continue
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            assert path.stem == data["id"] == data["integrity"]["sha256"][:16]
            assert data["integrity"]["sha256"] == _media_sha(f"models/{data['identity']['path']}")
            resources[data["id"]] = data
        assert len(resources) == 6
        self_trained = [r for r in resources.values() if r["provenance"]["source_class"] == "self-trained"]
        assert len(self_trained) == 1 and self_trained[0]["identity"]["type"] == "lora"

    @needs_demo
    def test_easel_demo_run_recipe_and_compat_agree() -> None:
        resources = {p.stem for p in (DEMO / "registry" / "resources").glob("*.yaml") if ".usage" not in p.name}
        recipe = json.loads((DEMO / "recipes" / "rcp-20300101-001.json").read_text(encoding="utf-8"))
        assert {r["resource"] for r in recipe["resources"]} <= resources and recipe["checkpoint"] in resources
        run_dir = DEMO / "runs" / "run-20300101-001"
        run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert run["recipe"] == recipe["id"] and len(run["outputs"]) == run["count"] == 4
        for out in run["outputs"]:
            assert out["sha256"] == _media_sha(f"runs/{run['id']}/{out['file']}")
        reviews = [r for r in _jsonl(run_dir / "reviews.jsonl") if r["source"] == "operator"]
        current = {r["n"]: r["verdict"] for r in reviews}
        assert current == {1: "keep", 2: "keep", 3: "reject", 4: "keep"}, "kept outputs 1, 2, 4; 3 rejected"
        compat = yaml.safe_load((DEMO / "data" / "compat.yaml").read_text(encoding="utf-8"))
        verdicts = {(r["source_family"], r["target_family"], r["artifact_type"]): r["verdict"] for r in compat["rules"]}
        assert verdicts[("illustrious", "illustrious", "lora")] == "native"
        assert verdicts[("sdxl", "illustrious", "lora")] == "degraded"
