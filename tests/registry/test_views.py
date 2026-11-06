"""Two registry questions checked against a SQL oracle: `registry compatible` and `list --self-trained`.

Each test loads the same records into an in-memory SQLite database, writes the question as a SQL view (an
independent second implementation), and checks that the command's rows are the view's rows, one for one.
Then the CLI is checked on the demo's planted pairs: the self-trained demo_robot LoRA, a native pair (that LoRA
on the Illustrious checkpoint) and a degraded pair (the SDXL ink-style LoRA on the same checkpoint).
"""

from __future__ import annotations

import copy
import sqlite3
from collections import Counter
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli

from easel.registry import compat, views
from easel.registry import resources as res

ANIME_XL = "bbbabeef2c0d57fe"   # checkpoint, illustrious
BASE_XL = "85306657134cbde8"    # checkpoint, sdxl
ROBOT = "cf56db17414bcb55"      # lora, illustrious, self-trained (overrides.yaml), trained_by kiln
INK = "60d12866cf6377b0"        # lora, sdxl
VAE = "b4d12aa2b3121a5e"        # vae, sdxl
UPSCALER = "99134ef8d78fb1bb"   # upscaler, no family

SCHEMA_SQL = """
CREATE TABLE artifact (artifact_id TEXT PRIMARY KEY, filename TEXT NOT NULL, artifact_type TEXT NOT NULL,
                       family_id TEXT, trained_on_family TEXT, source_class TEXT, network_dim INTEGER);
CREATE TABLE compat_rule (source_family TEXT NOT NULL, target_family TEXT NOT NULL, artifact_type TEXT NOT NULL,
                          verdict TEXT NOT NULL, confidence REAL, reason TEXT,
                          PRIMARY KEY (source_family, target_family, artifact_type));
CREATE TABLE training_run (run_id TEXT PRIMARY KEY, output_artifact_id TEXT, dataset_id TEXT, network_dim INTEGER);
-- what can be used with each checkpoint: every non-checkpoint artifact x every checkpoint, the rule found by
-- (trained-on family, else family) -> checkpoint family and the artifact type; no rule = untested, confidence 0
CREATE VIEW oracle_compatible AS
SELECT a.artifact_id, a.filename, a.artifact_type, c.artifact_id AS checkpoint_id,
       c.family_id AS checkpoint_family, COALESCE(r.verdict, 'untested') AS verdict,
       COALESCE(r.confidence, 0) AS confidence, r.reason
  FROM artifact a
  JOIN artifact c ON c.artifact_type = 'checkpoint'
  LEFT JOIN compat_rule r ON r.source_family = COALESCE(a.trained_on_family, a.family_id)
                         AND r.target_family = c.family_id
                         AND r.artifact_type = a.artifact_type
 WHERE a.artifact_type <> 'checkpoint';
-- self-trained artifacts and whether a training record names each one
CREATE VIEW oracle_self_trained AS
SELECT a.artifact_id, a.filename, a.artifact_type, a.family_id,
       CASE WHEN t.run_id IS NULL THEN 'no-record' ELSE 'recorded' END AS training_record,
       t.run_id, t.dataset_id, a.network_dim
  FROM artifact a
  LEFT JOIN training_run t ON t.output_artifact_id = a.artifact_id
 WHERE a.source_class = 'self-trained';
"""


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _registry(home: Path) -> tuple[dict[str, dict], dict[tuple, compat.Rule]]:
    reg = res.Registry(tool="easel", root=home, registry=home / "registry", models_root=home / "models",
                       standards=home / "standards")
    loaded = res.load_records(reg)
    assert not loaded.problems, loaded.problems
    records = res.effective(loaded.records, res.load_overrides(reg))
    return records, compat.load_rules(home)


def _database(records: dict[str, dict], rules: dict[tuple, compat.Rule]) -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA_SQL)
    for rec in records.values():
        db.execute("INSERT INTO artifact VALUES (?,?,?,?,?,?,?)",
                   (rec["id"], rec["identity"]["file_name"], rec["identity"]["type"], rec["identity"]["family"],
                    res.get_field(rec, "lora.trained_on_family"), rec["provenance"]["source_class"],
                    res.get_field(rec, "lora.network_dim")))
        trained = rec["provenance"].get("trained_by") or {}
        if trained.get("run"):
            db.execute("INSERT INTO training_run VALUES (?,?,?,?)",
                       (trained["run"], rec["id"], trained.get("snapshot"), res.get_field(rec, "lora.network_dim")))
    for rule in rules.values():
        db.execute("INSERT INTO compat_rule VALUES (?,?,?,?,?,?)",
                   (rule.source_family, rule.target_family, rule.artifact_type, rule.verdict, rule.confidence,
                    rule.reason))
    return db


def _with_edge_cases(records: dict[str, dict]) -> dict[str, dict]:
    """The demo plus resources that exercise every branch of the join."""
    out = copy.deepcopy(records)

    def clone(src: str, new_id: str, **fields: object) -> None:
        rec = copy.deepcopy(records[src])
        rec["id"] = new_id
        rec["identity"]["file_name"] = f"edge-{new_id[:4]}.safetensors"
        for dotted, value in fields.items():
            res.set_field(rec, dotted.replace("__", "."), value)
        out[new_id] = rec

    clone(INK, "a000000000000001", lora__trained_on_family="pony")             # trained-on beats filed-under
    clone(INK, "a000000000000002", identity__family=None, lora__trained_on_family=None)  # no family at all
    clone(BASE_XL, "a000000000000003", identity__family="pony")              # a third checkpoint family
    clone(BASE_XL, "a000000000000004", identity__family=None)                # a checkpoint of unknown family
    clone(VAE, "a000000000000005", identity__type="embedding")               # another artifact type
    clone(UPSCALER, "a000000000000006", identity__type="lora", identity__family="sd15",
          lifecycle__install_status="missing")                                   # an SD1.5 LoRA, file gone
    return out


def _view_rows(db: sqlite3.Connection) -> Counter:
    query = ("SELECT artifact_id, filename, artifact_type, checkpoint_id, checkpoint_family, verdict, confidence, "
             "reason FROM oracle_compatible")
    return Counter((a, f, t, c, cf, v, round(float(conf), 6), r) for a, f, t, c, cf, v, conf, r in db.execute(query))


def _join_rows(pairs: list[compat.Pair]) -> Counter:
    return Counter((p.resource, p.file_name, p.artifact_type, p.checkpoint, p.checkpoint_family, p.verdict,
                    round(p.confidence, 6), p.reason) for p in pairs)


# --- oracle_compatible ------------------------------------------------------------------------------------------------------

def test_oracle_compatible_join_equals_the_view_on_the_demo(home: Path) -> None:
    records, rules = _registry(home)
    pairs = compat.join(records.values(), rules)
    assert _join_rows(pairs) == _view_rows(_database(records, rules))
    assert len(pairs) == 4 * 2  # four add-ons x two checkpoints


def test_oracle_compatible_join_equals_the_view_on_every_branch(home: Path) -> None:
    records, rules = _registry(home)
    records = _with_edge_cases(records)
    pairs = compat.join(records.values(), rules)
    assert _join_rows(pairs) == _view_rows(_database(records, rules))
    verdicts = {(p.resource, p.checkpoint): p.verdict for p in pairs}
    assert verdicts[("a000000000000001", "a000000000000003")] == "native"      # pony LoRA on a pony checkpoint
    assert verdicts[("a000000000000002", ANIME_XL)] == "untested"              # no family: no rule can match
    assert verdicts[(INK, "a000000000000004")] == "untested"                   # checkpoint of unknown family
    assert verdicts[("a000000000000006", BASE_XL)] == "broken"                 # rules exist for any status


def test_oracle_compatible_planted_pairs_and_the_instance_rule_winning(home: Path) -> None:
    records, rules = _registry(home)
    rows = {p.resource: p for p in compat.for_resource(ANIME_XL, records, rules)}
    assert (rows[ROBOT].verdict, rows[ROBOT].confidence) == ("native", 1.0)
    assert (rows[INK].verdict, rows[INK].confidence) == ("degraded", 0.6)
    assert rows[INK].origin == "<home>/data/compat.yaml"   # the shipped rule says works; the instance's wins
    assert rules[("sdxl", "illustrious", "lora")].verdict == "degraded"
    assert rows[UPSCALER].verdict == "untested"
    order = [p.verdict for p in compat.for_resource(ANIME_XL, records, rules)]
    assert order == sorted(order, key=compat.VERDICT_ORDER.index)


def test_compatible_cli_prints_the_planted_native_and_degraded_pair(home: Path, capsys) -> None:
    code, out, err = run_cli(["registry", "compatible", ANIME_XL], capsys)
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0] == f"checkpoint {ANIME_XL} demo-anime-xl.safetensors (family illustrious): what works with it"
    assert f"{ROBOT} | demo_robot-v1.safetensors | lora | illustrious -> illustrious | native | 1.00 | same family" \
        in lines
    assert any(line.startswith(f"{INK} | demo-style-ink.safetensors | lora | sdxl -> illustrious | degraded | 0.60")
               for line in lines)
    assert not any(line.startswith(UPSCALER) for line in lines)  # untested is not offered
    assert lines[-1] == "not listed: 1 untested, 0 broken (--all lists them)"
    code, out, _err = run_cli(["registry", "compatible", ANIME_XL, "--all"], capsys)
    assert code == 0 and any(line.startswith(f"{UPSCALER} |") and "| untested | 0.00 |" in line
                             for line in out.splitlines())


def test_compatible_cli_from_the_lora_side(home: Path, capsys) -> None:
    code, out, _err = run_cli(["registry", "compatible", INK[:8]], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0].startswith(f"lora {INK} demo-style-ink.safetensors (source family sdxl)")
    assert lines[1].startswith(f"{BASE_XL} | demo-base-xl.safetensors | checkpoint | sdxl -> sdxl | native")
    assert lines[2].startswith(f"{ANIME_XL} | demo-anime-xl.safetensors | checkpoint | sdxl -> illustrious | degraded")


def test_compatible_json_carries_every_row_with_all(home: Path, capsys) -> None:
    import json

    code, out, _err = run_cli(["registry", "compatible", ANIME_XL, "--all", "--json"], capsys)
    assert code == 0
    rows = json.loads(out)
    assert {r["resource"] for r in rows} == {ROBOT, INK, VAE, UPSCALER}
    assert all(r["checkpoint"] == ANIME_XL for r in rows)


# --- oracle_self_trained ----------------------------------------------------------------------------------------------------

def _plant_second_self_trained(home: Path) -> None:
    """Mark the ink LoRA self-trained too, with no training record: the view's no-record branch."""
    path = home / "registry" / "overrides.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["resources"][INK] = {"set": {"provenance.source_class": "self-trained"}, "note": "demo: no record",
                              "at": "2030-01-01T07:00:00+00:00"}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_oracle_self_trained_rows_equal_the_view(home: Path) -> None:
    _plant_second_self_trained(home)
    records, rules = _registry(home)
    db = _database(records, rules)
    view = {row[0]: row for row in db.execute(
        "SELECT artifact_id, filename, artifact_type, family_id, training_record, run_id, dataset_id, network_dim "
        "FROM oracle_self_trained")}
    ours = {r["id"]: views.self_trained_row(r) for r in views.select(records, self_trained=True)}
    assert set(ours) == set(view) == {ROBOT, INK}
    for sha16, row in ours.items():
        assert (row["id"], row["file_name"], row["type"], row["family"], row["training_record"], row["run"],
                row["snapshot"], row["network_dim"]) == view[sha16]
    assert ours[ROBOT]["training_record"] == "recorded" and ours[INK]["training_record"] == "no-record"


def test_list_self_trained_prints_the_planted_row(home: Path, capsys) -> None:
    code, out, err = run_cli(["registry", "list", "--type", "lora", "--self-trained"], capsys)
    assert code == 0, err
    assert out.splitlines() == [
        f"{ROBOT} | demo_robot-v1.safetensors | lora | illustrious | sfw | active | "
        "recorded: kiln demo_robot run-20300102-001"]


def test_list_self_trained_after_a_fresh_scan(home: Path, capsys) -> None:
    from registry_support import empty_registry

    empty_registry(home)
    assert run_cli(["registry", "scan", "--hash"], capsys)[0] == 0
    code, out, _err = run_cli(["registry", "list", "--type", "lora", "--self-trained"], capsys)
    assert code == 0
    assert out.splitlines() == [f"{ROBOT} | demo_robot-v1.safetensors | lora | sdxl | unknown | active | no-record"]
