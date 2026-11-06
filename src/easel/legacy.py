"""Bridges to an adopted workspace: print the host runner's own commands, and read its finished runs back.

An adopted instance (`instance.adopted = true`, `policy.render = "legacy-runner"`) renders through the host's own
runner until cutover. easel never runs those commands; it prints them, and it reads the results read-only.
Every host fact lives in the instance's `[adopt]` table, never in this code:

- `easel recipe to-legacy <rcp>` (`run_to_legacy`) renders the four command templates `adopt.propose_command`,
  `adopt.dryrun_command`, `adopt.submit_command` and `adopt.status_command`, in that order, one line each on
  stdout. A template may use `{recipe_json}` (a request file this command writes to
  `<home>/legacy/requests/<rcp>.json` from the recipe: name, text, negative_extra, family, checkpoint and loras by
  file name, count, seed, size, hires, set, rating), `{recipe_id}` (the host's recipe id: `--host-recipe`, else the
  recipe's `legacy_id`, else the placeholder `<host-recipe-id>` that the propose line's output fills),
  `{count}` and `{job_id}` (`--job-id`, else `<job-id>`, which the submit line's output fills). Anything else in
  angle brackets, such as `<agent>`, is the agent's to fill. A missing template is exit 2 naming the key.
- `easel runs import-legacy <legacy-run-id> [--with-reviews]` (`run_import_legacy`) finds the host run folder
  named by the id under `adopt.output_root` (the folder itself or up to three levels below it, whatever layout the
  host uses) with its `manifest.json`, header-probes and copies each output into `runs/<run-id>/outputs/NNN.png`
  (sha256-checked), and writes `run.json` (with `legacy_id` and `imported_from`), `manifest.json`
  (easel/run-manifest/1) and the host's `graph.json`. Its first stdout line is always
  `run: <run-id> (legacy <legacy-run-id>)`, the id the next commands take. A host id of the shape
  `<letters>-YYYYMMDD-NNN` becomes `run-YYYYMMDD-NNN`, so a run already brought over under that id is found instead
  of duplicated. With `--with-reviews` it opens the database `adopt.run_registry` names read-only (`mode=ro`),
  reads that run's review rows and appends the ones not imported before to `reviews.jsonl`, then prints
  `<n> reviews imported`. `0 reviews imported` means the operator has not reviewed the run on the host's page
  yet: stop and ask.

Both entry points take parsed arguments, so `easel recipe` and `easel runs` register them as their own actions
(`add_to_legacy_args`, `add_import_legacy_args`).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sqlite3
import sys
from collections.abc import Mapping
from pathlib import Path

from easel._common import config, hashing, ids, image_meta, store
from easel.cli import instance_not_found, new_instance_hint
from easel.registry.shipped import DataError, schema_errors, shown

TEMPLATES = (("propose", "adopt.propose_command"), ("dry-run", "adopt.dryrun_command"),
             ("submit", "adopt.submit_command"), ("status", "adopt.status_command"))
PLACEHOLDER = re.compile(r"\{([a-z][a-z_]*)\}")
KNOWN_PLACEHOLDERS = ("recipe_json", "recipe_id", "count", "job_id")
HOST_RECIPE_PLACEHOLDER = "<host-recipe-id>"
JOB_PLACEHOLDER = "<job-id>"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
RECIPE_ID = re.compile(r"^rcp-\d{8}-\d{3}$")
HOST_DATED_ID = re.compile(r"^[A-Za-z]+-(\d{8})-(\d{3,})$")
MANIFEST_DEPTH = 3  # a host run folder sits in adopt.output_root or at most this many folders below it
CHECK_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
MANIFEST_ROLES = ("checkpoint", "lora", "lycoris", "embedding", "vae", "upscaler", "controlnet", "ipadapter", "clip",
                  "text_encoder", "detector")
ROLE_ALIASES = {"base": "checkpoint"}
SLOT = re.compile(r"^([a-z]+(\.\d+)?)?$")
VIA = ("app", "cli", "chat")


class LegacyError(Exception):
    """A refusal or a missing host fact: `.lines` to print, `.code` to exit with (2)."""

    def __init__(self, *lines: str, code: int = 2) -> None:
        super().__init__("\n".join(lines))
        self.lines = list(lines)
        self.code = code


def _err(lines) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _workspace(parser: argparse.ArgumentParser) -> None:
    if not any("--workspace" in a.option_strings for a in parser._actions):
        config.add_workspace_arg(parser)


def _folder(cfg, key: str, default: str) -> Path:
    if cfg.get(key, None) is not None:
        return cfg.path(key)
    if cfg.root is None:
        raise LegacyError(instance_not_found(), new_instance_hint())
    return cfg.root / default


def _need(cfg, key: str, why: str) -> str:
    value = cfg.get(key, None)
    if value is None or not str(value).strip():
        where = cfg.file.name if cfg.file is not None else "the instance's easel.toml"
        raise LegacyError(f"[adopt] {key.split('.', 1)[1]} is not set in {where}: {why}",
                          "the operator fills [adopt] in the instance's easel.toml (docs/runbooks/90-adopt-existing.md)")
    return str(value)


def _guard(handler, args: argparse.Namespace) -> int:
    try:
        return handler(args)
    except config.ConfigError as exc:
        _err(exc.lines)
        return exc.code
    except (LegacyError, DataError) as exc:
        _err(exc.lines)
        return getattr(exc, "code", 2)
    except store.Held as held:
        try:
            from easel._common import journal

            cfg = config.load(_tool(args), getattr(args, "workspace", None), strict=False)
            journal.note(f"stuck: {held.line}", tool=_tool(args), journal_dir=cfg.path("paths.journal"), timeout=2.0)
        except Exception:  # noqa: BLE001  (journalling a held lock must never hide the exit 4)
            pass
        raise


def _tool(args: argparse.Namespace) -> str:
    return str(getattr(args, "tool", None) or "easel")


# --- to-legacy -------------------------------------------------------------------------------------------------------

def add_to_legacy_args(parser: argparse.ArgumentParser) -> None:
    """The arguments of `easel recipe to-legacy` (and `--workspace` unless the parser has it already)."""
    parser.add_argument("rcp", metavar="<rcp>", help="the easel recipe id, for example rcp-20300101-001")
    parser.add_argument("--count", type=int, metavar="N", help="images to render (default: the recipe's count, else 1)")
    parser.add_argument("--host-recipe", metavar="<id>",
                        help="the host's recipe id, once the propose line has printed it (fills {recipe_id})")
    parser.add_argument("--job-id", metavar="<id>",
                        help="the host's job id, once the submit line has printed it (fills {job_id})")
    _workspace(parser)
    parser.set_defaults(func=run_to_legacy)


def render(template: str, values: Mapping[str, str], key: str) -> str:
    """`template` with every `{name}` replaced from `values`; an unknown `{name}` is a refusal naming the key."""
    unknown = sorted({m.group(1) for m in PLACEHOLDER.finditer(template)} - set(values))
    if unknown:
        raise LegacyError(f"[adopt] {key.split('.', 1)[1]} uses {', '.join('{' + u + '}' for u in unknown)}; "
                          f"known placeholders: {', '.join('{' + k + '}' for k in KNOWN_PLACEHOLDERS)}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def quote_path(path: Path) -> str:
    """A path for a command line: forward slashes, double-quoted when it holds a space or a shell character."""
    text = Path(path).as_posix()
    return f'"{text}"' if re.search(r"[\s&|;<>()'\"^%!`$]", text) else text


def _safe(value: str | None, what: str) -> str | None:
    if value is None:
        return None
    if not SAFE_ID.match(value):
        raise LegacyError(f"{what} {value!r} is not an id (letters, digits, . _ : -); it goes into a command line")
    return value


def read_recipe(cfg, rcp: str) -> tuple[Path, dict]:
    if not RECIPE_ID.match(rcp):
        raise LegacyError(f"not a recipe id: {rcp!r} (rcp-YYYYMMDD-NNN; `easel recipe list` shows them)")
    path = _folder(cfg, "paths.recipes", "recipes") / f"{rcp}.json"
    if not path.is_file():
        raise LegacyError(f"no recipe {rcp}: {shown(path, cfg.root)} does not exist")
    try:
        recipe = json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError:  # its text would carry the absolute path; say what is wrong in our own words
        raise LegacyError(f"{shown(path, cfg.root)} cannot be read (held by another program, or no permission)") \
            from None
    except ValueError as exc:
        raise LegacyError(f"{shown(path, cfg.root)} is not readable JSON: {exc}") from None
    if not isinstance(recipe, dict):
        raise LegacyError(f"{shown(path, cfg.root)} is not a recipe record")
    return path, recipe


def _stem(cfg, sha16: object) -> str:
    """The file name stem of a registry resource (what a host matches by name), else the id itself."""
    from easel.registry import resources as res

    try:
        records = res.load_records(res.open_registry(cfg, "easel")).records
    except (config.ConfigError, DataError, OSError):
        return str(sha16)
    record = records.get(str(sha16))
    name = res.get_field(record, "identity.file_name") if record else None
    return Path(str(name)).stem if name else str(sha16)


def request_from_recipe(cfg, recipe: Mapping, count: int) -> dict:
    """The request a host's propose step takes, built from the easel recipe (the keys its flags use)."""
    params = recipe.get("params") or {}
    resources = recipe.get("resources") or []
    loras = [f"{_stem(cfg, r.get('resource'))}:{r['weight']}" if r.get("weight") is not None
             else _stem(cfg, r.get("resource")) for r in resources if r.get("role") in ("lora", "lycoris")]
    embeddings = [_stem(cfg, r.get("resource")) for r in resources if r.get("role") == "embedding"]
    settings = {k: params[k] for k in ("sampler", "scheduler", "steps", "cfg", "clip_skip", "denoise")
                if params.get(k) is not None}
    request = {"name": recipe.get("name") or recipe.get("id"), "text": recipe.get("prompt") or "",
               "family": recipe.get("family"), "checkpoint": _stem(cfg, recipe.get("checkpoint")),
               "loras": loras, "count": count, "seed": params.get("seed"),
               "size": f"{params.get('width')}x{params.get('height')}",
               "hires": ((params.get("hires") or {}).get("mode") or "none") != "none",
               "set": settings, "rating": "sfw" if recipe.get("content_class") == "sfw" else "nsfw", "auto_loras": 0}
    if recipe.get("negative"):
        request["negative_extra"] = recipe["negative"]
    if embeddings:
        request["embeddings"] = embeddings
    return {k: v for k, v in request.items() if v is not None}


def to_legacy_lines(cfg, rcp: str, *, count: int | None = None, host_recipe: str | None = None,
                    job_id: str | None = None, tool: str = "easel") -> tuple[list[str], list[str]]:
    """(the four host lines in order, notes for stderr). Writes the request file the propose line reads."""
    templates = []
    missing = []
    for _label, key in TEMPLATES:
        value = cfg.get(key, None)
        if value is None or not str(value).strip():
            missing.append(key)
        templates.append((key, str(value or "")))
    if missing:
        where = cfg.file.name if cfg.file is not None else "the instance's easel.toml"
        raise LegacyError(*[f"[adopt] {k.split('.', 1)[1]} is not set in {where}" for k in missing],
                          "the operator fills the four command templates in [adopt] of the instance's easel.toml "
                          "(docs/runbooks/90-adopt-existing.md)")
    _path, recipe = read_recipe(cfg, rcp)
    status = recipe.get("status")
    if status not in ("approved", "run"):
        raise LegacyError(f"{rcp} is {status!r}: only an approved recipe is rendered; the operator approves it "
                          f"first (`easel recipe approve {rcp}`)")
    from easel.registry import classify

    hits = classify.load_markers(cfg.root).subject_hits(str(recipe.get("prompt") or ""))
    if hits:
        raise LegacyError(f"{rcp} refused under policy.subjects (fictional adults only): the prompt holds "
                          f"{', '.join(hits)}")
    params = recipe.get("params") or {}
    n = count if count is not None else params.get("count") or params.get("batch") or 1
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise LegacyError(f"--count must be a positive whole number, not {n!r}")
    if cfg.root is None:
        raise LegacyError(instance_not_found(), new_instance_hint())
    request_file = cfg.root / "legacy" / "requests" / f"{rcp}.json"
    store.write_json(request_file, request_from_recipe(cfg, recipe, n))
    legacy_id = recipe.get("legacy_id") if isinstance(recipe.get("legacy_id"), str) else None
    host = _safe(host_recipe, "--host-recipe") or _safe(legacy_id, "the recipe's legacy_id") or HOST_RECIPE_PLACEHOLDER
    values = {"recipe_json": quote_path(request_file), "recipe_id": host, "count": str(n),
              "job_id": _safe(job_id, "--job-id") or JOB_PLACEHOLDER}
    lines = [render(text, values, key) for key, text in templates]
    notes = [f"to-legacy: wrote {shown(request_file, cfg.root)} (the request the propose line reads)",
             "to-legacy: run the lines in order (propose, dry-run, submit), then the status line once; "
             "never route around the host runner"]
    if host == HOST_RECIPE_PLACEHOLDER:
        notes.append(f"to-legacy: {HOST_RECIPE_PLACEHOLDER} is the recipe id the propose line prints; run "
                     f"`easel recipe to-legacy {rcp} --host-recipe <id>` to fill it in")
    elif legacy_id and not host_recipe:
        notes.append(f"to-legacy: {rcp} came from the host as {legacy_id}; the propose line is needed only for a "
                     "new host recipe")
    if values["job_id"] == JOB_PLACEHOLDER:
        notes.append(f"to-legacy: {JOB_PLACEHOLDER} is the job id the submit line prints")
    if any("<agent>" in line for line in lines):
        notes.append("to-legacy: <agent> is your own agent name")
    return lines, notes


def run_to_legacy(args: argparse.Namespace) -> int:
    """`easel recipe to-legacy <rcp>`: print the four host lines; exit 2 naming a missing [adopt] key."""

    def act(a: argparse.Namespace) -> int:
        cfg = config.load(_tool(a), getattr(a, "workspace", None), strict=True)
        lines, notes = to_legacy_lines(cfg, a.rcp, count=getattr(a, "count", None),
                                       host_recipe=getattr(a, "host_recipe", None),
                                       job_id=getattr(a, "job_id", None), tool=_tool(a))
        for line in lines:
            print(line)
        _err(notes)
        return 0

    return _guard(act, args)


# --- import-legacy ---------------------------------------------------------------------------------------------------

def add_import_legacy_args(parser: argparse.ArgumentParser) -> None:
    """The arguments of `easel runs import-legacy` (and `--workspace` unless the parser has it already)."""
    parser.add_argument("legacy_id", metavar="<legacy-run-id>", help="the host's run id, as its submit or status "
                                                                      "line printed it")
    parser.add_argument("--with-reviews", action="store_true",
                        help="also read the host's review rows for this run (read-only) into reviews.jsonl")
    _workspace(parser)
    parser.set_defaults(func=run_import_legacy)


def run_id_for(legacy_id: str, created_at: object = None) -> str:
    """The easel run id of a host run: `<letter>-YYYYMMDD-NNN` -> `run-YYYYMMDD-NNN`, else run-<date>-<slug>."""
    m = HOST_DATED_ID.match(legacy_id)
    if m:
        return f"run-{m.group(1)}-{m.group(2)}"
    day = re.sub(r"[^0-9]", "", str(created_at or "")[:10]) or "00000000"
    slug = re.sub(r"[^a-z0-9_-]+", "-", legacy_id.lower()).strip("-_") or "legacy"
    return f"run-{day}-{slug[:48]}"


def find_manifest(output_root: Path, legacy_id: str) -> Path:
    """`<id>/manifest.json` in `output_root` or up to MANIFEST_DEPTH folders below it: the layout is found, not
    assumed, so any host folder scheme (flat, by date, by project) works without a setting."""
    if not SAFE_ID.match(legacy_id) or ":" in legacy_id:
        raise LegacyError(f"not a host run id: {legacy_id!r}")
    root = Path(output_root)
    found: set[Path] = set()
    for depth in range(MANIFEST_DEPTH + 1):
        pattern = "/".join(["*"] * depth + [legacy_id, "manifest.json"])
        found.update(p for p in root.glob(pattern) if p.is_file())
    hits = sorted(found)
    if not hits:
        raise LegacyError(f"import-legacy: no folder {legacy_id}/ with a manifest.json under <adopt.output_root> "
                          f"(searched {MANIFEST_DEPTH} levels down): a wrong id, or the run has not finished "
                          "writing; check the id the submit or status line printed")
    if len(hits) > 1:
        raise LegacyError(f"import-legacy: {legacy_id} has {len(hits)} manifests under <adopt.output_root>; "
                          "the operator says which run is meant")
    return hits[0]


def _num(value: object, kind: type, minimum: float | None = None, maximum: float | None = None):
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = kind(value)
    except (TypeError, ValueError):
        return None
    if kind is int and isinstance(value, float) and not value.is_integer():
        return None
    if (minimum is not None and number < minimum) or (maximum is not None and number > maximum):
        return None
    return number


def _str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _sha(value: object) -> str | None:
    return value.lower() if isinstance(value, str) and hashing.is_sha256(value) else None


def _host_outputs(manifest: Mapping, folder: Path) -> list[str]:
    """The output file names the manifest lists (relative to its folder), else the images in the folder."""
    names = []
    for item in manifest.get("outputs") or []:
        name = item.get("file") if isinstance(item, Mapping) else item if isinstance(item, str) else None
        if isinstance(name, str) and name:
            names.append(name)
    if names:
        return names
    return sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_meta.IMAGE_SUFFIXES)


def _inside(folder: Path, name: str) -> Path | None:
    path = (folder / name).resolve()
    try:
        path.relative_to(folder.resolve())
    except ValueError:
        return None
    return path


def _copy_output(src: Path, dest: Path) -> dict:
    """Copy one output byte for byte and probe the copy's header; the copy's sha256 must equal the source's."""
    probe = image_meta.probe(src)
    store.write_bytes(dest, src.read_bytes())
    copied = hashing.sha256_file(dest)
    if copied != probe["sha256"]:
        raise LegacyError(f"import-legacy: the copy of {src.name} does not match its source (sha256); nothing more "
                          "is copied: run the import again", code=1)
    return probe


def _manifest_resources(manifest: Mapping) -> tuple[list[dict], list[dict]]:
    """(run-manifest resources, host entries that do not map) from the host manifest's artifacts."""
    out, dropped = [], []
    for art in manifest.get("artifacts") or []:
        if not isinstance(art, Mapping):
            continue
        role = ROLE_ALIASES.get(str(art.get("role")), str(art.get("role")))
        sha = _sha(art.get("sha256"))
        name = Path(str(art.get("filename") or "")).name
        if role not in MANIFEST_ROLES or not name:
            dropped.append({"role": art.get("role"), "slot": art.get("slot")})
            continue
        slot = str(art.get("slot") or "")
        out.append({"role": role, "slot": slot if SLOT.match(slot) else "", "resource": sha[:16] if sha else None,
                    "sha256": sha, "file_name": name, "weight": _num(art.get("weight"), float)})
    return out, dropped


def _recipe_for(cfg, host_recipe: object) -> str | None:
    """The easel recipe whose legacy_id is the host's recipe id, if one was exported or imported."""
    if not isinstance(host_recipe, str) or not host_recipe:
        return None
    folder = _folder(cfg, "paths.recipes", "recipes")
    for path in sorted(folder.glob("rcp-*.json")) if folder.is_dir() else []:
        try:
            rec = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("legacy_id") == host_recipe and RECIPE_ID.match(str(rec.get("id"))):
            return str(rec["id"])
    return None


def build_run(cfg, legacy_id: str, manifest_path: Path, output_root: Path, run_id: str,
              tool: str = "easel") -> tuple[dict, dict, list]:
    """(run.json, manifest.json, [(n, source file or None, dest name)]) for a host run; nothing is written."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict):
        raise LegacyError(f"import-legacy: the manifest of {legacy_id} is not a JSON object")
    folder = manifest_path.parent
    listed = {str(i.get("file")): i for i in manifest.get("outputs") or [] if isinstance(i, Mapping)}
    plan, outputs, host_files = [], [], []
    for n, name in enumerate(_host_outputs(manifest, folder), 1):
        src = _inside(folder, name)
        suffix = Path(name).suffix.lower() or ".png"
        dest = f"outputs/{n:03d}{suffix}"
        entry = listed.get(name, {})
        outputs.append({"n": n, "file": dest, "sha256": _sha(entry.get("sha256")),
                        "bytes": _num(entry.get("bytes"), int, 0), "width": _num(entry.get("width"), int, 1),
                        "height": _num(entry.get("height"), int, 1), "seed": None, "metadata_stripped": False})
        plan.append((n, src if src is not None and src.is_file() else None, dest))
        host_files.append({"n": n, "file": name})
    comfy = manifest.get("comfyui") if isinstance(manifest.get("comfyui"), Mapping) else {}
    seed = _num(manifest.get("seed"), int, 0)
    rel = manifest_path.resolve().relative_to(Path(output_root).resolve()).as_posix()
    run = {"schema": "easel/run/1", "id": run_id, "legacy_id": legacy_id,
           "recipe": _recipe_for(cfg, manifest.get("recipe_id")), "workflow": None, "job": None,
           "created_at": _str(manifest.get("created_at")) or ids.now_iso(tool), "status": "ok",
           "prompt_id": _str(comfy.get("prompt_id")), "seed": seed,
           "count": max(1, len(outputs)), "prompt": _str(manifest.get("prompt")) or "",
           "negative": _str(manifest.get("negative")) or "", "width": _num(manifest.get("width"), int, 64),
           "height": _num(manifest.get("height"), int, 64), "steps": _num(manifest.get("steps"), int, 1),
           "cfg": _num(manifest.get("cfg"), float, 0), "sampler": _str(manifest.get("sampler")),
           "scheduler": _str(manifest.get("scheduler")), "denoise": _num(manifest.get("denoise"), float, 0, 1),
           "started_at": None, "finished_at": None, "duration_s": _num(manifest.get("duration_s"), float, 0),
           "gpu_peak_mb": None, "graph": "graph.json", "manifest": "manifest.json", "outputs": outputs,
           "imported_from": {"kind": "legacy", "manifest": rel}, "error": None, "notes": None,
           "legacy": {"recipe_id": manifest.get("recipe_id"), "task": manifest.get("task"),
                      "batch": manifest.get("batch"), "output_files": host_files}}
    resources, dropped = _manifest_resources(manifest)
    if dropped:
        run["legacy"]["unmapped_artifacts"] = dropped
    easel_manifest = {"schema": "easel/run-manifest/1", "run": run_id, "written_at": ids.now_iso(tool),
                      "graph_sha256": None, "comfyui": {"version": _str(comfy.get("version")), "node_packs": []},
                      "resources": resources, "seeds": [seed] if seed is not None else []}
    return run, easel_manifest, plan


def _validate(name: str, record: Mapping, what: str) -> None:
    problems = schema_errors(name, record)
    if problems:
        raise LegacyError(f"import-legacy: {what} would not be a valid {record.get('schema')} record; nothing "
                          "written:", *[f"  {p}" for p in problems])


def import_run(cfg, legacy_id: str, *, tool: str = "easel") -> tuple[str, Path, list[str], list[str]]:
    """Import (or find) the host run; (run id, run folder, stdout lines, warnings)."""
    output_root = Path(_need(cfg, "adopt.output_root", "import-legacy reads the host's run folders under it"))
    if not output_root.is_absolute() and cfg.root is not None:
        output_root = cfg.root / output_root
    manifest_path = find_manifest(output_root, legacy_id)
    try:
        created = json.loads(manifest_path.read_text(encoding="utf-8-sig")).get("created_at")
    except (OSError, ValueError, AttributeError):
        raise LegacyError(f"import-legacy: the manifest of {legacy_id} is not readable JSON") from None
    run_id = run_id_for(legacy_id, created)
    runs = _folder(cfg, "paths.runs", "runs")
    run_dir = runs / run_id
    warnings: list[str] = []
    if run_dir.exists():
        return run_id, run_dir, _refresh(cfg, legacy_id, run_id, run_dir, manifest_path, output_root, warnings), \
            warnings
    run, easel_manifest, plan = build_run(cfg, legacy_id, manifest_path, output_root, run_id, tool)
    _validate("run", run, "run.json")
    _validate("run-manifest", easel_manifest, "manifest.json")
    runs.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()  # exclusive: the claim on the id
    except FileExistsError:
        raise LegacyError(f"import-legacy: {shown(run_dir, cfg.root)} appeared while importing; run it again") \
            from None
    copied = 0
    for n, src, dest in plan:
        out = run["outputs"][n - 1]
        if src is None:
            warnings.append(f"import-legacy: output {n} of {legacy_id} is missing in the host folder; recorded "
                            "without a copy")
            continue
        probe = _copy_output(src, run_dir / dest)
        if out["sha256"] and out["sha256"] != probe["sha256"]:
            warnings.append(f"import-legacy: output {n}: the bytes differ from the sha256 the host manifest lists; "
                            "the copy's own sha256 is recorded")
        out.update(sha256=probe["sha256"], bytes=probe["bytes"], width=probe["width"] or out["width"],
                   height=probe["height"] or out["height"])
        if not probe.get("readable"):
            warnings.append(f"import-legacy: output {n} is not a readable image header ({probe.get('error')})")
        copied += 1
    graph_src = manifest_path.parent / "graph.json"
    if graph_src.is_file():
        store.write_bytes(run_dir / "graph.json", graph_src.read_bytes())
        easel_manifest["graph_sha256"] = hashing.sha256_file(run_dir / "graph.json")
    else:
        warnings.append(f"import-legacy: {legacy_id} has no graph.json beside its manifest")
    store.write_json(run_dir / "manifest.json", easel_manifest)
    store.write_json(run_dir / "run.json", run)  # last: a run folder with run.json is a complete import
    rel = shown(run_dir, cfg.root)
    return run_id, run_dir, [f"imported {legacy_id} as {run_id}: {copied} of {len(plan)} outputs copied to "
                             f"{rel}/outputs/"], warnings


def _refresh(cfg, legacy_id: str, run_id: str, run_dir: Path, manifest_path: Path, output_root: Path,
             warnings: list[str]) -> list[str]:
    """A run folder that exists: the same host run (fill any output copy that is missing) or a collision."""
    run_json = run_dir / "run.json"
    try:
        run = json.loads(run_json.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        raise LegacyError(f"import-legacy: {shown(run_dir, cfg.root)} exists without a readable run.json (an "
                          "interrupted import?); the operator moves it aside, then import again") from None
    if str(run.get("legacy_id")) != legacy_id:
        raise LegacyError(f"import-legacy: {run_id} exists and holds another run (legacy_id "
                          f"{run.get('legacy_id')!r}); nothing was changed")
    _new, _manifest, plan = build_run(cfg, legacy_id, manifest_path, output_root, run_id)
    sources = {n: src for n, src, _dest in plan}
    filled = 0
    for out in run.get("outputs") or []:
        dest = run_dir / str(out.get("file"))
        src = sources.get(out.get("n"))
        if dest.exists() or src is None:
            continue
        if out.get("sha256") and hashing.sha256_file(src) != out["sha256"]:
            warnings.append(f"import-legacy: output {out.get('n')}: the host file no longer matches run.json; not "
                            "copied")
            continue
        _copy_output(src, dest)
        filled += 1
    line = f"{run_id} already holds {legacy_id}; nothing imported twice"
    if filled:
        line += f"; {filled} missing output cop{'y' if filled == 1 else 'ies'} added"
    return [line]


# --- reviews ---------------------------------------------------------------------------------------------------------

def _ro_connect(db: Path) -> sqlite3.Connection:
    """Open a database read-only (`mode=ro`, so the write-ahead log is read too); never a write."""
    return sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)


def host_review_rows(db: Path, legacy_id: str) -> tuple[list[dict], list[dict]]:
    """(run_review rows, run_output rows) of one host run, oldest first."""
    if not Path(db).is_file():
        raise LegacyError(f"import-legacy: adopt.run_registry names no file ({Path(db).name})")
    try:
        conn = _ro_connect(db)
    except sqlite3.Error as exc:
        raise LegacyError(f"import-legacy: cannot open adopt.run_registry read-only: {exc}") from None
    try:
        conn.row_factory = sqlite3.Row
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "run_review" not in tables:
            raise LegacyError("import-legacy: the database adopt.run_registry names has no run_review table; it "
                              "must name the host's catalog (the database that holds run and run_review)")
        reviews = [dict(r) for r in conn.execute("SELECT * FROM run_review WHERE run_id = ? ORDER BY review_id",
                                                 (legacy_id,))]
        outputs = [dict(r) for r in conn.execute("SELECT * FROM run_output WHERE run_id = ? ORDER BY run_output_id",
                                                 (legacy_id,))] if "run_output" in tables else []
    except sqlite3.Error as exc:
        raise LegacyError(f"import-legacy: reading the host's reviews failed: {exc}") from None
    finally:
        conn.close()
    return reviews, outputs


def _checks(raw: object, review_id: object, warnings: list[str]) -> dict | None:
    if raw in (None, ""):
        return None
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        warnings.append(f"import-legacy: review {review_id}: checks are not JSON; left out")
        return None
    if not isinstance(data, Mapping):
        return None
    out = {}
    for key, value in data.items():
        mapped = {True: "pass", False: "fail"}.get(value, value) if isinstance(value, bool) else value
        if CHECK_KEY.match(str(key)) and mapped in ("pass", "fail", "na"):
            out[str(key)] = mapped
        else:
            warnings.append(f"import-legacy: review {review_id}: check {key!r} = {value!r} left out (not pass, fail "
                            "or na under a lower-case key)")
    return out


def review_lines(run: Mapping, reviews: list[dict], outputs: list[dict], *, tool: str,
                 warnings: list[str]) -> list[dict]:
    """The reviews.jsonl lines (easel/run-review-event/1) of the host rows, one per row that maps."""
    by_sha = {o.get("sha256"): o.get("n") for o in run.get("outputs") or [] if o.get("sha256")}
    n_of = {}
    for position, row in enumerate(outputs, 1):
        n_of[row.get("run_output_id")] = by_sha.get(_sha(row.get("sha256"))) or position
    lines = []
    for row in reviews:
        out_id = row.get("run_output_id")
        n = None if out_id is None else n_of.get(out_id)
        if out_id is not None and n is None:
            warnings.append(f"import-legacy: review {row.get('review_id')} names an output the host run does not "
                            "list; left out")
            continue
        verdict = row.get("verdict") if row.get("verdict") in ("keep", "maybe", "reject") else None
        line = {"schema": "easel/run-review-event/1", "at": _str(row.get("created_at")) or ids.now_iso(tool), "v": 1,
                "source": "operator", "by": "easel runs import-legacy", "run": run["id"], "n": n, "verdict": verdict}
        checks = _checks(row.get("checks"), row.get("review_id"), warnings)
        if checks is not None:
            line["checks"] = checks
        line.update({"note": _str(row.get("note")), "via": row.get("source") if row.get("source") in VIA else "import",
                     "legacy_id": row.get("review_id")})
        problems = schema_errors("run-review-event", line)
        if problems:
            warnings.append(f"import-legacy: review {row.get('review_id')} left out: {problems[0]}")
            continue
        lines.append(line)
    return lines


def import_reviews(cfg, legacy_id: str, run_dir: Path, *, tool: str = "easel") -> tuple[int, list[str]]:
    """Append the host's review rows not imported before; (how many, warnings)."""
    db = Path(_need(cfg, "adopt.run_registry", "--with-reviews reads the host's review rows from it"))
    if not db.is_absolute() and cfg.root is not None:
        db = cfg.root / db
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8-sig"))
    reviews, outputs = host_review_rows(db, legacy_id)
    warnings: list[str] = []
    lines = review_lines(run, reviews, outputs, tool=tool, warnings=warnings)
    path = run_dir / "reviews.jsonl"
    with store.lock(run_dir, tool):
        seen = {str(r.get("legacy_id")) for r in store.read_jsonl(path).records if r.get("legacy_id") is not None}
        new = [line for line in lines if str(line["legacy_id"]) not in seen]
        if new:
            data = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in new).encode("utf-8")
            store.append_bytes_locked(path, data)
    return len(new), warnings


def run_import_legacy(args: argparse.Namespace) -> int:
    """`easel runs import-legacy <id> [--with-reviews]`: import a host run, then (optionally) its reviews."""

    def act(a: argparse.Namespace) -> int:
        tool = _tool(a)
        cfg = config.load(tool, getattr(a, "workspace", None), strict=True)
        run_id, run_dir, lines, warnings = import_run(cfg, a.legacy_id, tool=tool)
        print(f"run: {run_id} (legacy {a.legacy_id})")  # always first: the easel run id later commands take
        for line in lines:
            print(line)
        _err(warnings)
        if not getattr(a, "with_reviews", False):
            print(f"next: easel runs import-legacy {shlex.quote(a.legacy_id)} --with-reviews (after the operator "
                  "reviewed it on the host's page)")
            return 0
        count, warnings = import_reviews(cfg, a.legacy_id, run_dir, tool=tool)
        _err(warnings)
        print(f"{count} reviews imported")
        if count == 0:
            _err([f"stop: no new review rows for {a.legacy_id}; the operator has not reviewed it on the host's page "
                  "(or every row is imported already). Ask, and import again when they say done."])
        else:
            print("next: easel runs reviews --current")
        return 0

    return _guard(act, args)
