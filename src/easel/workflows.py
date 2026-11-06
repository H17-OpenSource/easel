"""Workflow templates: a ComfyUI API graph with named slots, its facts and its node packs, checked and approved.

A template is a folder `workflows/<id>/` with `graph.json` (the API format `POST /prompt` takes), `slots.json`
(easel/workflow-slots/1: which node input each named value fills) and `meta.yaml` (easel/workflow-meta/1: task,
family, status, and every custom node pack the graph needs with its licence), plus a README for shipped ones.
`comfy_graph.fill_template` fills it for a run.

Where templates live: the repository ships three (`txt2img`, `txt2img-lora-hires`, `detailer`); an instance keeps
its own under `<home>/workflows/<id>/` (`paths.workflows`). An instance template with a shipped id shadows the
shipped one (that is how approving a shipped template works: the approved copy lives in the instance).

The verbs:
- `check [<id>...]`: offline. Schema validity, the files the meta names, slots pointing at real nodes and
  inputs, links between nodes, no model file name left in the graph without a slot, and the node packs: a node
  class that `runtime/comfyui.lock.json` says comes from a pack (its `needed_by` nodes) needs that pack in the
  meta with the lock's licence and commit, and the packages the pack installs. Exit 1 on any problem; CI runs it
  on the shipped templates.
- `validate <id>`: `check`, then the server's `/object_info` for the graph's node classes (every class and input
  name must exist there). Records `validated_at` on an instance template; a shipped one is only reported.
- `register <graph.json> --slots <slots.json> --task <task>`: a new instance template, status draft.
- `approve <id> --run <run-id> --note "<the operator's words>"`: only after a run with this template finished
  ok (`run.json` names it in `workflow`); writes status approved (into the instance copy).
- `list`: one line per template.
A template whose node packs are missing on the server is unavailable: `validate` says which class is missing; it
is never silently rendered without them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from easel._common import comfy_graph, ids, store
from easel.cli import instance_not_found, new_instance_hint
from easel.registry.shipped import REPO, reason, schema_errors, shown

SHIPPED = REPO / "workflows"
LOCK = REPO / "runtime" / "comfyui.lock.json"
META_SCHEMA = "easel/workflow-meta/1"
SLOTS_SCHEMA = "easel/workflow-slots/1"
TASKS = ("txt2img", "img2img", "inpaint", "upscale", "detail", "i2v", "t2v")
DOMAINS = ("image", "video", "upscale")
WORKFLOW_ID = re.compile(r"^[a-z][a-z0-9_-]*$")


class WorkflowError(Exception):
    """A template that cannot be registered, validated or approved: `.lines` to print, exit 2."""

    def __init__(self, *lines: str) -> None:
        super().__init__("\n".join(lines))
        self.lines = list(lines)


@dataclass
class Template:
    """One template folder as read: what it holds, where it came from, what could not be read."""

    id: str
    folder: Path
    origin: str  # shipped | instance
    meta: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)
    graph: dict = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return str(self.meta.get("status") or "?")


@dataclass
class Report:
    """The result of checking one template: problems (exit 1) and warnings (printed, not counted)."""

    template: Template
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --- reading ---------------------------------------------------------------------------------------------------------

def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_yaml(path: Path) -> object:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def load(folder: Path, origin: str) -> Template:
    """Read a template folder; anything unreadable is listed, never raised."""
    folder = Path(folder)
    t = Template(id=folder.name, folder=folder, origin=origin)
    meta_file = folder / "meta.yaml"
    try:
        meta = _read_yaml(meta_file)
        t.meta = meta if isinstance(meta, dict) else {}
        if not isinstance(meta, dict):
            t.unreadable.append("meta.yaml is not a mapping")
    except FileNotFoundError:
        t.unreadable.append("meta.yaml is missing")
    except Exception as exc:  # noqa: BLE001  (any parse error is a problem line, not a crash)
        t.unreadable.append(f"meta.yaml does not parse: {reason(exc)}")
    files = t.meta.get("files") if isinstance(t.meta.get("files"), dict) else {}
    for attr, default in (("graph", "graph.json"), ("slots", "slots.json")):
        rel = str(files.get(attr) or default)
        try:
            data = _read_json(folder / rel)
        except FileNotFoundError:
            t.unreadable.append(f"{rel} is missing")
            continue
        except (OSError, ValueError) as exc:
            t.unreadable.append(f"{rel} is not JSON: {reason(exc)}")
            continue
        if not isinstance(data, dict):
            t.unreadable.append(f"{rel} is not a JSON object")
            continue
        setattr(t, attr, data)
    return t


def instance_dir(cfg) -> Path | None:
    """`paths.workflows` of the instance (default `workflows`), or None without an instance."""
    if cfg is None or cfg.root is None:
        return None
    if cfg.get("paths.workflows", None) is not None:
        return cfg.path("paths.workflows")
    return cfg.root / "workflows"


def _folders(base: Path | None) -> list[Path]:
    if base is None or not base.is_dir():
        return []
    return [p for p in sorted(base.iterdir()) if p.is_dir() and not p.name.startswith((".", "_"))]


def discover(cfg=None, *, shipped: Path = SHIPPED) -> dict[str, Template]:
    """Every template by id: the shipped ones, then the instance's (an instance id shadows a shipped one)."""
    found = {p.name: load(p, "shipped") for p in _folders(shipped)}
    found.update({p.name: load(p, "instance") for p in _folders(instance_dir(cfg))})
    return found


def get(cfg, wid: str, *, shipped: Path = SHIPPED) -> Template:
    templates = discover(cfg, shipped=shipped)
    if wid not in templates:
        raise WorkflowError(f"no workflow template {wid!r}; `easel workflows list` shows them "
                            f"({', '.join(sorted(templates)) or 'none'})")
    return templates[wid]


def load_lock(path: Path = LOCK) -> dict[str, dict]:
    """The node packs of runtime/comfyui.lock.json by name (empty when the lock is absent or unreadable)."""
    try:
        data = _read_json(path)
    except (OSError, ValueError):
        return {}
    packs = data.get("node_packs") if isinstance(data, dict) else None
    return {str(p["name"]): p for p in packs or [] if isinstance(p, dict) and p.get("name")}


def pack_of_class(lock: Mapping[str, Mapping]) -> dict[str, str]:
    """Node class -> the pack that provides it, from the lock's `needed_by` entries."""
    out: dict[str, str] = {}
    for name, pack in lock.items():
        for need in pack.get("needed_by") or []:
            for cls in (need or {}).get("nodes") or []:
                out[str(cls)] = name
    return out


# --- checking --------------------------------------------------------------------------------------------------------

def _is_link(value: object) -> bool:
    return (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            and isinstance(value[1], int) and not isinstance(value[1], bool))


def graph_problems(graph: Mapping) -> list[str]:
    """API-format shape and links: every node has class_type and inputs; every link names an existing node."""
    if "nodes" in graph and isinstance(graph.get("nodes"), list):
        return ["graph.json is in the editor's UI format; export it with Export (API) in ComfyUI"]
    if not graph:
        return ["graph.json has no nodes"]
    problems = []
    for node_id, node in graph.items():
        if not isinstance(node, dict) or not isinstance(node.get("class_type"), str) \
                or not isinstance(node.get("inputs"), dict):
            problems.append(f"node {node_id}: not an API-format node (class_type and inputs)")
            continue
        for name, value in node["inputs"].items():
            if _is_link(value) and value[0] not in graph:
                problems.append(f"node {node_id} input {name}: links to node {value[0]}, which the graph does not have")
    return problems


def slot_problems(t: Template) -> tuple[list[str], list[str]]:
    """(problems, warnings) of the slot map against the graph."""
    problems, warnings = [], []
    slots = [s for s in t.slots.get("slots") or [] if isinstance(s, dict)]
    names = [s.get("name") for s in slots]
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"slot {name!r} is defined twice")
    covered = set()
    for s in slots:
        node, inp = str(s.get("node")), str(s.get("input"))
        if node not in t.graph:
            problems.append(f"slot {s.get('name')!r} points at node {node}, which the graph does not have")
            continue
        covered.add((node, inp))
        inputs = (t.graph[node] or {}).get("inputs") or {}
        if inp not in inputs:
            warnings.append(f"slot {s.get('name')!r}: node {node} has no input {inp!r} in the graph yet (it is added "
                            "when filled)")
        elif _is_link(inputs[inp]):
            problems.append(f"slot {s.get('name')!r} would overwrite the link of node {node} input {inp}")
    for node_id, node in t.graph.items():
        if not isinstance(node, dict):
            continue
        for inp in comfy_graph.NAME_INPUTS.get(str(node.get("class_type")), ()):
            if inp in (node.get("inputs") or {}) and (str(node_id), inp) not in covered:
                problems.append(f"node {node_id} ({node.get('class_type')}): the file name in {inp} has no slot; "
                                "a template never hard-codes a model file")
    return problems, warnings


def pack_problems(t: Template, lock: Mapping[str, Mapping]) -> tuple[list[str], list[str]]:
    """(problems, warnings): every pack class is covered by a named pack, with the lock's licence and commit."""
    problems, warnings = [], []
    listed = {str(p.get("name")): p for p in t.meta.get("node_packs") or [] if isinstance(p, dict)}
    owners = pack_of_class(lock)
    for cls in comfy_graph.node_classes(t.graph) if t.graph else []:
        pack = owners.get(cls)
        if pack and pack not in listed:
            problems.append(f"node class {cls} comes from {pack} ({lock[pack].get('licence')}); meta.yaml must "
                            "list it under node_packs")
    for name, entry in listed.items():
        licence = str(entry.get("licence") or "")
        pinned = lock.get(name)
        if pinned is None:
            warnings.append(f"node pack {name} is not in runtime/comfyui.lock.json: its licence ({licence or 'none'}) "
                            "and commit are not pinned")
            continue
        if licence != str(pinned.get("licence")):
            problems.append(f"node pack {name}: licence {licence!r} differs from the lock's {pinned.get('licence')!r}")
        if entry.get("commit") and entry.get("commit") != pinned.get("commit"):
            problems.append(f"node pack {name}: commit {entry['commit']} differs from the lock's {pinned.get('commit')}")
        installs = {str(i.get("name")): str(i.get("licence")) for i in entry.get("installs") or [] if isinstance(i, dict)}
        for req in pinned.get("requires") or []:
            if req.get("name") not in installs:
                problems.append(f"node pack {name} installs {req.get('name')} ({req.get('licence')}); meta.yaml must "
                                "name it under installs")
            elif installs[req["name"]] != str(req.get("licence")):
                problems.append(f"node pack {name}: {req['name']} licence {installs[req['name']]!r} differs from the "
                                f"lock's {req.get('licence')!r}")
        if licence == "unverified":
            warnings.append(f"node pack {name}: licence unverified; read its LICENSE before relying on it")
    return problems, warnings


def check(t: Template, lock: Mapping[str, Mapping] | None = None) -> Report:
    """Every offline check of one template."""
    lock = load_lock() if lock is None else lock
    report = Report(t, problems=list(t.unreadable))
    if t.meta:
        report.problems += [f"meta.yaml {p}" for p in schema_errors("workflow-meta", t.meta)]
        if t.meta.get("id") != t.id:
            report.problems.append(f"meta.yaml id {t.meta.get('id')!r} differs from the folder name {t.id!r}")
        if t.meta.get("status") == "approved" and not t.meta.get("approved_at"):
            report.problems.append("status approved without approved_at: approve it with `easel workflows approve`")
    if t.slots:
        report.problems += [f"slots.json {p}" for p in schema_errors("workflow-slots", t.slots)]
        if t.slots.get("workflow") != t.id:
            report.problems.append(f"slots.json workflow {t.slots.get('workflow')!r} differs from the folder name")
    if t.graph:
        report.problems += graph_problems(t.graph)
    if t.graph and t.slots:
        more, warn = slot_problems(t)
        report.problems += more
        report.warnings += warn
    if t.meta:
        more, warn = pack_problems(t, lock)
        report.problems += more
        report.warnings += warn
    if t.origin == "shipped" and not (t.folder / "README.md").is_file():
        report.problems.append("README.md is missing (every shipped template says what it does and needs)")
    return report


def summary(t: Template) -> str:
    """`<id> | <status> | <task> | <family> | <origin> | <packs>` for `list` and `check`."""
    packs = [f"{p.get('name')} ({p.get('licence')})" for p in t.meta.get("node_packs") or [] if isinstance(p, dict)]
    return (f"{t.id} | {t.status} | {t.meta.get('task') or '?'} | {t.meta.get('family') or '-'} | {t.origin} | "
            f"{', '.join(packs) or 'core nodes only'}")


# --- writing ---------------------------------------------------------------------------------------------------------

def _changelog(cfg, *, tool: str, source: str, by: str, action: str, wid: str, changes: Mapping,
               note: str | None = None) -> None:
    from easel.registry import resources as res

    reg = res.open_registry(cfg, tool)
    res.append_changelog(reg, [res.changelog_line(tool=tool, source=source, by=by, action=action, kind="workflow",
                                                  target=wid, changes=changes, note=note)])


def _write(folder: Path, meta: Mapping, slots: Mapping | None = None, graph: Mapping | None = None,
           head: str = "") -> None:
    if graph is not None:
        store.write_json(folder / str(meta["files"]["graph"]), graph)
    if slots is not None:
        store.write_json(folder / str(meta["files"]["slots"]), slots)
    store.write_text(folder / "meta.yaml", head + store.dump_yaml(dict(meta)))


def register(cfg, graph_file: Path, slots_file: Path, *, task: str, wid: str | None = None, name: str | None = None,
             domain: str = "image", family: str | None = None, packs: Iterable[tuple[str, str]] = (),
             notes: str | None = None, tool: str = "easel") -> Report:
    """Copy a graph and its slot map into a new instance template (status draft), checked first."""
    base = instance_dir(cfg)
    if base is None:
        raise WorkflowError(instance_not_found() if cfg is not None else "no instance",
                            new_instance_hint())
    wid = wid or re.sub(r"[^a-z0-9_-]+", "-", Path(graph_file).stem.lower()).strip("-")
    if not WORKFLOW_ID.match(wid or ""):
        raise WorkflowError(f"not a workflow id (lower-case letters, digits, - and _, starting with a letter): {wid!r}")
    if wid in {p.name for p in _folders(SHIPPED)}:
        raise WorkflowError(f"{wid!r} is a shipped template's id; pick another with --id")
    try:
        graph = _read_json(Path(graph_file))
        slots = _read_json(Path(slots_file))
    except (OSError, ValueError) as exc:
        raise WorkflowError(f"cannot read the template: {reason(exc)}") from None
    if not isinstance(graph, dict) or not isinstance(slots, dict):
        raise WorkflowError("graph.json and slots.json are JSON objects")
    if isinstance(slots.get("slots"), list):
        slots = {**slots, "schema": SLOTS_SCHEMA, "workflow": wid}
    lock = load_lock()
    owners = pack_of_class(lock)
    node_packs = {}
    for cls in comfy_graph.node_classes(graph) if not graph_problems(graph) else []:
        pack = owners.get(cls)
        if pack and pack not in node_packs:
            pinned = lock[pack]
            node_packs[pack] = {"name": pack, "licence": str(pinned.get("licence")), "commit": pinned.get("commit"),
                                "url": pinned.get("repo"), "required": True,
                                "installs": [{"name": r["name"], "licence": r["licence"]}
                                             for r in pinned.get("requires") or []], "note": None}
    for pack, licence in packs:
        node_packs.setdefault(pack, {"name": pack, "licence": licence, "commit": None, "url": None, "required": True,
                                     "installs": [], "note": None})
    meta = {"schema": META_SCHEMA, "id": wid, "name": name or wid, "legacy_id": None, "domain": domain, "task": task,
            "status": "draft", "family": family, "comfyui_min_version": cfg.get("comfyui.min_version", None) or None,
            "node_packs": list(node_packs.values()), "files": {"graph": "graph.json", "slots": "slots.json"},
            "validated_at": None, "validator_note": None, "approved_at": None, "notes": notes}
    folder = base / wid
    report = check(Template(wid, folder, "instance", meta=meta, slots=slots, graph=graph), lock)
    if report.problems:
        return report
    base.mkdir(parents=True, exist_ok=True)
    try:
        folder.mkdir()  # exclusive: an existing template is never replaced
    except FileExistsError:
        raise WorkflowError(f"{shown(folder, cfg.root)} exists; a template is never replaced (pick another --id)") \
            from None
    _write(folder, meta, slots, graph)
    _changelog(cfg, tool=tool, source="operator", by="easel workflows register", action="add", wid=wid,
               changes={"status": {"from": None, "to": "draft"}})
    report.template = load(folder, "instance")
    return report


def record_validation(cfg, t: Template, note: str, *, tool: str = "easel") -> bool:
    """Write validated_at and the validator's note into an instance template's meta; False for a shipped one."""
    if t.origin != "instance":
        return False
    with store.lock(t.folder, tool):
        meta = dict(load(t.folder, "instance").meta)
        before = meta.get("validated_at")
        meta["validated_at"] = ids.now_iso(tool)
        meta["validator_note"] = note
        _write(t.folder, meta)
    _changelog(cfg, tool=tool, source=f"{tool}:workflows validate", by="easel workflows validate", action="update",
               wid=t.id, changes={"validated_at": {"from": before, "to": meta["validated_at"]}}, note=note)
    return True


def approve(cfg, wid: str, run_id: str, note: str, *, source: str = "operator", tool: str = "easel") -> Template:
    """Mark a template approved after a run with it finished ok; a shipped template is copied into the instance."""
    t = get(cfg, wid)
    base = instance_dir(cfg)
    if base is None:
        raise WorkflowError(instance_not_found(), new_instance_hint())
    runs =cfg.path("paths.runs") if cfg.get("paths.runs", None) is not None else cfg.root / "runs"
    run_json = runs / run_id / "run.json"
    if not run_json.is_file():
        raise WorkflowError(f"no run {run_id}: {shown(run_json, cfg.root)} does not exist")
    run = _read_json(run_json)
    if run.get("status") != "ok":
        raise WorkflowError(f"{run_id} is {run.get('status')!r}, not ok: a template is approved only after it "
                            "produced output end to end")
    if run.get("workflow") != wid:
        raise WorkflowError(f"{run_id} did not use the template {wid!r} (its run.json says workflow "
                            f"{run.get('workflow')!r})")
    report = check(t)
    if report.problems:
        raise WorkflowError(f"{wid} fails `easel workflows check`; fix it first:", *[f"  {p}" for p in report.problems])
    clean = " ".join(str(note).split())
    if not clean:
        raise WorkflowError("--note is empty: quote the operator's words")
    target = base / wid
    meta = dict(t.meta)
    before = meta.get("status")
    meta.update(status="approved", approved_at=ids.now_iso(tool), validator_note=f"{run_id}: {clean}")
    if t.origin == "shipped":
        if target.exists():
            raise WorkflowError(f"{shown(target, cfg.root)} exists but is not a template; move it aside first")
        target.mkdir(parents=True)
        _write(target, meta, t.slots, t.graph)
        readme = t.folder / "README.md"
        if readme.is_file():
            store.write_text(target / "README.md", readme.read_text(encoding="utf-8"))
    else:
        with store.lock(target, tool):
            _write(target, meta)
    _changelog(cfg, tool=tool, source=source, by="easel workflows approve", action="set", wid=wid,
               changes={"status": {"from": before, "to": "approved"}}, note=clean)
    return load(target, "instance")
