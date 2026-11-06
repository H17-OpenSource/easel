"""`easel doctor`: one line per check, then `next:`. Exit 0 when no line is a FAIL, 1 otherwise.

Line shapes: `<check>: <what was found> ... ok` and `... warn` (a warn is reported to the operator, it does not
block), and `FAIL: <check>: <what is wrong>` (a FAIL comes first: fix it before any stage). A `fix:` line under a
warn or FAIL says what to do. Paths are shown as `<models_root>` or relative to the instance, never absolute.

The checks: the install; the instance file (found, valid, every `${VAR}` set); native or adopted; the models
folder; the registry; the scheduler (free RAM measurable, the disk floor); `jobs.allowed`; the policy; on an
adopted instance the four `[adopt]` command templates; the ComfyUI server.

The server on an adopted workspace (`instance.adopted = true` or `policy.render = "legacy-runner"`): the host
runner starts it per job, so a stopped server is expected. Then doctor prints exactly
    comfyui: not running at <url> (the host runner starts it per job) ... ok
    comfyui version: unknown until a job has run (lock says <min_version>) ... warn
and never a FAIL for it; `min_version` and drift from runtime/comfyui.lock.json are checked only when the server
answers. On a native instance a silent server is a FAIL unless easel starts it per job
(`comfyui.manage_lifecycle = true`). The tool's default port is not necessarily the server's: the fix line says
to set `comfyui.url`.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

from easel._common import config, scheduler, sysinfo
from easel._common.comfy_client import ComfyError

from .common import REPO, TOOL, Studio, is_adopted, legacy_runner
from .lifecycle import PORT_LESSON, client

ADOPT_TEMPLATES = ("propose_command", "dryrun_command", "submit_command", "status_command")
DEFAULT_DISK_FLOOR_GB = 20.0  # the shipped easel.example.toml value; an instance sets its own


@dataclass
class Check:
    state: str  # ok | warn | FAIL
    text: str
    fix: str | None = None

    def lines(self) -> list[str]:
        head = f"FAIL: {self.text}" if self.state == "FAIL" else f"{self.text} ... {self.state}"
        return [head] + ([f"  fix: {self.fix}"] if self.fix else [])


def _version() -> str:
    from easel import cli

    return cli.version()


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in str(text).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def lock_version() -> str | None:
    """The ComfyUI version runtime/comfyui.lock.json was tested with."""
    try:
        data = json.loads((REPO / "runtime" / "comfyui.lock.json").read_text(encoding="utf-8"))
        return (data.get("comfyui") or {}).get("version")
    except (OSError, ValueError):
        return None


def comfy_checks(studio: Studio) -> list[Check]:
    cfg = studio.cfg
    min_version = str(cfg.get("comfyui.min_version", "") or "")
    tested = lock_version()
    try:
        the_client = client(studio)
    except ComfyError as exc:
        return [Check("FAIL", f"comfyui: {exc.lines[0]}", "easel config set comfyui.url <url>")]
    stats = the_client.ping()
    if stats is None:
        if studio.adopted:
            return [Check("ok", f"comfyui: not running at {the_client.url} (the host runner starts it per job)"),
                    Check("warn", f"comfyui version: unknown until a job has run (lock says {min_version or tested})")]
        if bool(cfg.get("comfyui.manage_lifecycle", False)):
            return [Check("ok", f"comfyui: not running at {the_client.url} (easel starts it per job: "
                                "comfyui.manage_lifecycle = true)"),
                    Check("warn", f"comfyui version: unknown until a job has run (lock says {min_version or tested})")]
        return [Check("FAIL", f"comfyui: nothing answers at {the_client.url}",
                      "report to the operator (the server is theirs to start); " + PORT_LESSON)]
    system = stats.get("system") or {}
    version = str(system.get("comfyui_version") or "")
    out = [Check("ok", f"comfyui: up at {the_client.url} (version {version or 'not reported'})")]
    if version and min_version and _version_tuple(version) < _version_tuple(min_version):
        out.append(Check("FAIL", f"comfyui version: {version} is older than comfyui.min_version {min_version}",
                         "report to the operator: the server needs the version in runtime/comfyui.lock.json"))
    elif version and tested and version != tested:
        out.append(Check("warn", f"comfyui version: {version} differs from the tested {tested} "
                                 "(runtime/comfyui.lock.json)"))
    elif version:
        out.append(Check("ok", f"comfyui version: {version} (lock says {tested or min_version or '-'})"))
    argv = system.get("argv")
    if isinstance(argv, list) and argv and "--output-directory" not in argv:
        out.append(Check("warn", "comfyui: the server writes its outputs into its own install folder (started "
                                 "without --output-directory)",
                         "tell the operator; do not render against a server you did not set up"))
    return out


def run_checks(tool: str = TOOL, workspace: str | Path | None = None) -> tuple[list[Check], str]:
    """(checks, the next command)."""
    checks = [Check("ok", f"{tool}: {_version()} (python {platform.python_version()})")]
    try:
        cfg = config.load(tool, workspace, strict=False)
    except config.ConfigError as exc:
        checks += [Check("FAIL", f"config: {line}") for line in exc.lines]
        return checks, f"fix the instance file, then `{tool} doctor`"
    if cfg.file is None or cfg.root is None:
        from easel.cli import instance_not_found, new_instance_hint

        checks.append(Check("FAIL", f"config: {instance_not_found(tool)}"))
        return checks, new_instance_hint(tool)
    studio = Studio(cfg=cfg, tool=tool, root=cfg.root)
    checks.append(Check("ok", f"config: {cfg.file.name} (found by {cfg.found_by})"))
    for line in cfg.unset_lines():
        key = line.rsplit("(", 1)[-1].rstrip(")")
        checks.append(Check("FAIL", line.removeprefix("FAIL: "), f"set the variable, or {tool} config set {key} <value>"))
    unset_keys = {key for _var, key in cfg.unset}
    adopted = is_adopted(cfg)
    if adopted:
        how = "policy.render = legacy-runner: the host runner renders" if legacy_runner(cfg) else \
            "instance.adopted = true"
        checks.append(Check("ok", f"instance: adopted ({how})"))
    else:
        checks.append(Check("ok", "instance: native"))

    if "paths.models_root" not in unset_keys:
        try:
            models = cfg.path("paths.models_root")
            if models.is_dir():
                checks.append(Check("ok", "models: <models_root> exists"))
            else:
                checks.append(Check("FAIL", "models: paths.models_root does not exist (<models_root>)",
                                    f"{tool} config set paths.models_root <models_root>"))
        except config.ConfigError as exc:
            checks.append(Check("FAIL", f"models: {exc.lines[0]}"))

    try:
        from easel.registry import resources as res

        reg = res.open_registry(cfg, tool)
        loaded = res.load_records(reg)
        n = len(loaded.records)
        if loaded.problems:
            checks.append(Check("warn", f"registry: {n} resources, {len(loaded.problems)} record(s) not readable",
                                f"{tool} registry list (it names them)"))
        elif n == 0:
            checks.append(Check("warn", "registry: no resources yet", f"{tool} registry scan --hash"))
        else:
            checks.append(Check("ok", f"registry: {n} resources"))
    except Exception as exc:  # noqa: BLE001  (doctor reports every problem as a line; it never stops early)
        from easel.registry.shipped import reason

        checks.append(Check("warn", f"registry: not readable ({reason(exc)[:200]})"))

    kind = str(cfg.get("scheduler.kind", "none"))
    try:
        scheduler.from_config(cfg)
        if kind == "none":
            try:
                free = sysinfo.free_ram_gb()
                checks.append(Check("ok", f"scheduler: none (this machine alone); free RAM {free:.1f} GB"))
            except sysinfo.SysinfoUnavailable:
                checks.append(Check("warn", "scheduler: " + scheduler.CANNOT_MEASURE,
                                    "the operator decides; never set the key yourself"))
        else:
            checks.append(Check("ok", f"scheduler: {kind} (an external scheduler decides; fail_open = "
                                      f"{str(bool(cfg.get('scheduler.fail_open', False))).lower()})"))
    except scheduler.SchedulerError as exc:
        checks.append(Check("FAIL", f"scheduler: {exc.lines[0]}"))
    try:
        floor = float(cfg.get("limits.disk_floor_gb", DEFAULT_DISK_FLOOR_GB) or 0)
        free_disk = sysinfo.disk_free_gb(cfg.root)
        if free_disk < floor:
            checks.append(Check("warn", f"disk: free {free_disk:.0f} GB < {floor:g} GB floor (limits.disk_floor_gb); "
                                        "renders and downloads wait",
                                "report to the operator; never delete files or lower the floor yourself"))
        else:
            checks.append(Check("ok", f"disk: free {free_disk:.0f} GB >= {floor:g} GB floor"))
    except (sysinfo.SysinfoUnavailable, OSError, ValueError, TypeError):
        checks.append(Check("warn", "disk: free space cannot be measured"))

    allowed = [e for e in cfg.get("jobs.allowed", []) or [] if isinstance(e, dict)]
    if legacy_runner(cfg):
        checks.append(Check("ok", "jobs: GPU verbs refuse here; renders go through the host runner (runbook 90)"))
    elif any(e.get("tool") == tool and e.get("verb") == "run" for e in allowed):
        checks.append(Check("ok", f"jobs: {tool} run allowed (jobs.allowed)"))
    else:
        checks.append(Check("warn", f"jobs: jobs.allowed lacks {tool} run: {tool} jobs submit refuses",
                            "the operator decides what the scheduler may start"))

    checks.append(Check("ok", f"policy: subjects {cfg.get('policy.subjects', 'fictional-adults-only')}, prompt "
                              f"authoring {cfg.get('policy.prompt_authoring', 'sfw-only')}, reviewer "
                              f"{cfg.get('policy.reviewer', 'human-only')}, render {cfg.get('policy.render', 'native')}"))
    if adopted:
        adopt = cfg.get("adopt", {}) or {}
        missing = [k for k in ADOPT_TEMPLATES if not str(adopt.get(k) or "").strip()]
        if missing:
            checks.append(Check("warn", f"adopt: [adopt] lacks {', '.join(missing)}: {tool} recipe to-legacy cannot "
                                        "print those lines", "the operator fills [adopt] in the instance's file"))
        else:
            checks.append(Check("ok", "adopt: the four host command templates are set"))
    checks += comfy_checks(studio)

    first = next((c for c in checks if c.state == "FAIL"), None)
    if first is not None:
        return checks, first.fix or f"fix the line above, then `{tool} doctor`"
    return checks, f"{tool} next"


def report(tool: str = TOOL, workspace: str | Path | None = None) -> tuple[int, list[str]]:
    """(exit code, every line incl. `next:`)."""
    checks, nxt = run_checks(tool, workspace)
    lines = [line for check in checks for line in check.lines()]
    lines.append(f"next: {nxt}")
    return (1 if any(c.state == "FAIL" for c in checks) else 0), lines
