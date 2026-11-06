"""The questions behind T2b's verbs, checked against a SQL oracle, one row for one row.

Each test loads the same data into an in-memory SQLite database, writes the question as a SQL view (an
independent second implementation), and checks the command's rows against the view's rows:
- `oracle_duplicates` (`easel registry dups`): one table row per file, grouped by sha256;
- `oracle_experience` (`easel standards build`): run resources x current verdicts (the latest per output,
  `oracle_current_verdict`), per resource: verdicts, kept, rejected, mean weight of the kept rows;
- `oracle_undocumented` (`easel registry audit`, check `undocumented`): no origin, no sha256 or no licence, unless
  self-trained;
- `oracle_unattributed` (`audit`, check `unattributed`): LoRAs without a training record. The file registry
  reads it as "no training record and no attributing source": the test shows the view's rows minus exactly the
  downloaded LoRAs whose source link is recorded.
"""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
from pathlib import Path

import pytest
import yaml
from registry_support import header_reader, run_cli

from easel._common import config, hashing
from easel.registry import audit as auditing
from easel.registry import dups as duplicates
from easel.registry import resources as res
from easel.standards import experience

ANIME_XL, BASE_XL, ROBOT, INK, VAE, UPSCALER = ("bbbabeef2c0d57fe", "85306657134cbde8", "cf56db17414bcb55",
                                                "60d12866cf6377b0", "b4d12aa2b3121a5e", "99134ef8d78fb1bb")

ORACLE_SQL = """
CREATE TABLE artifact (artifact_id TEXT PRIMARY KEY, filename TEXT NOT NULL, artifact_type TEXT NOT NULL,
                       bytes INTEGER, sha256 TEXT, origin_url TEXT, license TEXT, source_class TEXT);
CREATE TABLE training_run (run_id TEXT PRIMARY KEY, output_artifact_id TEXT);
CREATE TABLE run_artifact (run_id TEXT NOT NULL, artifact_id TEXT NOT NULL, role TEXT NOT NULL, slot TEXT NOT NULL,
                           weight REAL);
CREATE TABLE run_review (review_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, run_output_id INTEGER,
                         verdict TEXT);
-- the same sha256 under more than one file name
CREATE VIEW oracle_duplicates AS
SELECT sha256, COUNT(*) n, GROUP_CONCAT(filename, ' | ') files
  FROM artifact WHERE sha256 IS NOT NULL GROUP BY sha256 HAVING COUNT(*) > 1;
-- no source link, no sha256 or no licence; a self-trained file has no download page to record
CREATE VIEW oracle_undocumented AS
SELECT artifact_id, filename, artifact_type, bytes FROM artifact
 WHERE (origin_url IS NULL OR sha256 IS NULL OR license IS NULL) AND source_class <> 'self-trained';
-- LoRAs and LyCORIS without a training record
CREATE VIEW oracle_unattributed AS
SELECT a.artifact_id, a.filename FROM artifact a
  LEFT JOIN training_run t ON t.output_artifact_id = a.artifact_id
 WHERE a.artifact_type IN ('lora', 'lycoris') AND t.run_id IS NULL;
-- the current verdict: the latest review per output (or per whole run)
CREATE VIEW oracle_current_verdict AS
SELECT rv.* FROM run_review rv
 WHERE rv.review_id = (SELECT MAX(r2.review_id) FROM run_review r2
                        WHERE r2.run_id = rv.run_id AND r2.run_output_id IS rv.run_output_id);
CREATE VIEW oracle_experience AS
SELECT ra.artifact_id,
       COUNT(DISTINCT rv.review_id) AS reviews,
       SUM(CASE WHEN rv.verdict = 'keep' THEN 1 ELSE 0 END) AS kept,
       SUM(CASE WHEN rv.verdict = 'reject' THEN 1 ELSE 0 END) AS rejected,
       ROUND(AVG(CASE WHEN rv.verdict = 'keep' THEN ra.weight END), 3) AS kept_avg_weight
  FROM run_artifact ra JOIN oracle_current_verdict rv ON rv.run_id = ra.run_id
 GROUP BY ra.artifact_id;
"""


@pytest.fixture
def home(demo_workspace: Path, frozen_clock, monkeypatch: pytest.MonkeyPatch) -> Path:
    header_reader(monkeypatch)
    return demo_workspace


def _open(home: Path) -> tuple[res.Registry, dict[str, dict]]:
    cfg = config.load("easel", home, strict=True)
    reg = res.open_registry(cfg, "easel")
    loaded = res.load_records(reg)
    assert not loaded.problems, loaded.problems
    return reg, res.effective(loaded.records, res.load_overrides(reg))


def _db() -> sqlite3.Connection:
    db = sqlite3.connect(":memory:")
    db.executescript(ORACLE_SQL)
    return db


# --- oracle_duplicates ------------------------------------------------------------------------------------------

def test_oracle_duplicate_files(home: Path, capsys) -> None:
    models = home / "models"
    shutil.copyfile(models / "loras" / "demo-style-ink.safetensors", models / "loras" / "demo-ink-copy.safetensors")
    for name in ("demo-vae-a.safetensors", "demo-vae-b.safetensors"):
        shutil.copyfile(models / "vae" / "demo-xl-vae.safetensors", models / "vae" / name)
    other = bytearray((models / "loras" / "demo-style-ink.safetensors").read_bytes())
    other[-1] ^= 0x55
    (models / "loras" / "demo-ink-other.safetensors").write_bytes(bytes(other))
    toml = home / "easel.toml"
    toml.write_text(toml.read_text(encoding="utf-8").replace('models_root = "models"\n',
                                                             'models_root = "models"\nextra_model_roots = ["more"]\n'),
                    encoding="utf-8")
    (home / "more" / "checkpoints").mkdir(parents=True)
    shutil.copyfile(models / "checkpoints" / "demo-base-xl.safetensors",
                    home / "more" / "checkpoints" / "demo-base-xl.safetensors")

    reg, records = _open(home)
    db = _db()
    for f in duplicates.files_under_roots(reg, lambda line: None):
        db.execute("INSERT INTO artifact (artifact_id, filename, artifact_type, bytes, sha256) VALUES (?,?,?,?,?)",
                   (f.shown, f.shown, "any", f.size, hashing.sha256_file(f.path)))
    view = {sha: (n, sorted(files.split(" | "))) for sha, n, files in db.execute("SELECT * FROM oracle_duplicates")}
    report = duplicates.find(reg, records, full=True)
    ours = {g.key: (len(g.files), sorted(f.shown for f in g.files)) for g in report.groups}
    assert all(g.confirmed for g in report.groups)
    assert ours == view
    assert len(view) == 3 and sorted(n for n, _files in view.values()) == [2, 2, 3]
    code, out, _err = run_cli(["registry", "dups", "--full", "--json"], capsys)
    assert code == 0
    assert {g["sha256"]: sorted(g["files"]) for g in json.loads(out)} == {k: v[1] for k, v in view.items()}


# --- oracle_experience --------------------------------------------------------------------------------------

def _plant_runs(home: Path) -> None:
    runs = home / "runs"
    base = runs / "run-20300101-001"

    def run(run_id: str, resources: list[dict], reviews: list[tuple[int | None, str]]) -> None:
        folder = runs / run_id
        folder.mkdir()
        manifest = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
        manifest["run"], manifest["resources"] = run_id, resources
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        shutil.copyfile(base / "run.json", folder / "run.json")
        lines = [{"schema": "easel/run-review-event/1", "at": f"2030-01-02T00:0{i}:00+00:00", "v": 1,
                  "source": "operator", "run": run_id, "n": n, "verdict": verdict, "checks": {}, "note": None,
                  "via": "cli", "legacy_id": None} for i, (n, verdict) in enumerate(reviews)]
        (folder / "reviews.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")

    def used(role: str, slot: str, sha16: str, weight: float | None) -> dict:
        return {"role": role, "slot": slot, "resource": sha16, "sha256": sha16 * 4, "file_name": "x", "weight": weight}

    run("run-20300101-002", [used("checkpoint", "", ANIME_XL, None), used("lora", "lora.1", INK, 0.6),
                             used("lora", "lora.2", ROBOT, 0.7)],
        [(1, "keep"), (2, "reject"), (1, "maybe"), (None, "keep"), (3, "keep")])
    run("run-20300101-003", [used("checkpoint", "", BASE_XL, None), used("lora", "lora.1", INK, 0.5),
                             used("lora", "lora.2", INK, 0.9)], [(1, "keep"), (2, "keep")])
    run("run-20300101-004", [used("checkpoint", "", BASE_XL, None)], [])


def test_oracle_artifact_experience(home: Path) -> None:
    _plant_runs(home)
    db = _db()
    runs = experience.read_runs(home / "runs")
    for folder in sorted((home / "runs").iterdir()):
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for r in manifest["resources"]:
            db.execute("INSERT INTO run_artifact VALUES (?,?,?,?,?)",
                       (manifest["run"], r["resource"], r["role"], r["slot"], r["weight"]))
        for line in (folder / "reviews.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            db.execute("INSERT INTO run_review (run_id, run_output_id, verdict) VALUES (?,?,?)",
                       (event["run"], event["n"], event["verdict"]))
    view = sorted(db.execute("SELECT * FROM oracle_experience"))
    ours = sorted(e.as_row() for e in experience.reduce(runs).values() if e.reviews)
    assert ours == view
    by_id = {row[0]: row for row in view}
    assert by_id[INK] == (INK, 6, 6, 1, 0.667)  # run 2 (4 current verdicts, 2 kept) + run 3 (INK in two slots x 2 kept)
    assert by_id[ROBOT] == (ROBOT, 8, 5, 2, 0.76)
    assert BASE_XL in by_id and by_id[BASE_XL][1] == 2  # run 4 has no verdict, so it adds no row


def test_experience_feeds_the_cards(home: Path, capsys) -> None:
    _plant_runs(home)
    assert run_cli(["standards", "build"], capsys)[0] == 0
    card = yaml.safe_load((home / "standards" / "cards" / f"{INK}.yaml").read_text(encoding="utf-8"))
    weight = next(v for v in card["values"] if v["source"] == "experience" and v["field"] == "weight")
    assert (weight["value"], weight["n"], weight["evidence"]) == (0.667, 6, "6 kept / 6 reviewed")
    assert card["chosen"]["weight"] == {"value": 0.667, "source": "experience"}  # 6 verdicts beat the author


# --- oracle_undocumented and oracle_unattributed --------------------------------------------------------------------

def _plant_records(home: Path) -> None:
    folder = home / "registry" / "resources"

    def plant(base: str, hex_: str, **dotted: object) -> None:
        rec = copy.deepcopy(yaml.safe_load((folder / f"{base}.yaml").read_text(encoding="utf-8")))
        sha = hex_ * (64 // len(hex_))
        rec["id"], rec["integrity"]["sha256"] = sha[:16], sha
        for key, value in dotted.items():
            res.set_field(rec, key.replace("__", "."), value)
        (folder / f"{sha[:16]}.yaml").write_text(yaml.safe_dump(rec, sort_keys=False), encoding="utf-8")

    plant(INK, "1a", provenance__url=None, provenance__download_url=None, provenance__source_class="unknown")
    plant(INK, "2b", provenance__download_url=None, provenance__url=None)  # downloaded, no link
    plant(INK, "3c", licence__status="unknown")  # documented source, licence unknown
    plant(ROBOT, "4d", provenance__trained_by=None)  # self-trained, no training record
    plant(ROBOT, "5e", identity__type="lycoris")  # self-trained, recorded
    plant(ANIME_XL, "6f", provenance__download_url=None, licence__status="unknown")
    plant(VAE, "70", provenance__source_class="self-trained", provenance__url=None)


def _oracle_db(records: dict[str, dict]) -> sqlite3.Connection:
    db = _db()
    for sha, rec in records.items():
        lic = rec["licence"]
        origin = res.get_field(rec, "provenance.url") or res.get_field(rec, "provenance.download_url")
        db.execute("INSERT INTO artifact VALUES (?,?,?,?,?,?,?,?)",
                   (sha, rec["identity"]["file_name"], rec["identity"]["type"], rec["integrity"]["bytes"],
                    rec["integrity"]["sha256"], origin,
                    None if lic["status"] == "unknown" else (lic.get("name") or lic["status"]),
                    rec["provenance"]["source_class"]))
        trained = res.get_field(rec, "provenance.trained_by")
        if trained:
            db.execute("INSERT INTO training_run VALUES (?,?)", (f"{trained['run']}:{sha}", sha))
    return db


def test_oracle_undocumented(home: Path, capsys) -> None:
    _plant_records(home)
    _reg, records = _open(home)
    db = _oracle_db(records)
    view = {row[0] for row in db.execute("SELECT artifact_id FROM oracle_undocumented")}
    ours = {sha for sha, rec in records.items() if auditing.undocumented(rec)}
    assert ours == view
    assert view == {"1a1a1a1a1a1a1a1a", "2b2b2b2b2b2b2b2b", "3c3c3c3c3c3c3c3c", "6f6f6f6f6f6f6f6f"}
    code, out, _err = run_cli(["registry", "audit", "--only", "undocumented", "--json"], capsys)
    assert code == 0 and {f["id"] for f in json.loads(out)["findings"]} == view


def test_oracle_unattributed_loras(home: Path, capsys) -> None:
    _plant_records(home)
    _reg, records = _open(home)
    db = _oracle_db(records)
    view = {row[0] for row in db.execute("SELECT artifact_id FROM oracle_unattributed")}
    attributed_by_source = {row[0] for row in db.execute(
        "SELECT v.artifact_id FROM oracle_unattributed v JOIN artifact a USING (artifact_id) "
        "WHERE a.source_class = 'downloaded' AND a.origin_url IS NOT NULL")}
    ours = {sha for sha, rec in records.items() if auditing.unattributed(rec)}
    assert ours == view - attributed_by_source
    assert attributed_by_source == {INK, "3c3c3c3c3c3c3c3c"}
    assert ours == {"1a1a1a1a1a1a1a1a", "2b2b2b2b2b2b2b2b", "4d4d4d4d4d4d4d4d"}
    code, out, _err = run_cli(["registry", "audit", "--only", "unattributed", "--json"], capsys)
    assert code == 0 and {f["id"] for f in json.loads(out)["findings"]} == ours
