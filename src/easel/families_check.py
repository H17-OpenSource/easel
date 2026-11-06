"""`easel families check`: the rules every model family file in `families/` must follow.

Why: a family file tells the tool how a base model family behaves (architecture, the tool's own block of
settings), how much free host RAM its jobs wait for, and under which licence its bases come. Families inherit
settings from a parent (illustrious from sdxl, for example), but a licence never travels that way: two bases of
one architecture rarely share terms, so every family must carry its own licence summary.

Rules (each failure is one problem line; exit 1):
- the file parses as a YAML mapping; `schema` is `<tool>/family/1`; `id` equals the file name; `name` is set;
- `status` is one of `STATUSES`;
- `inherits` is null or the id of another family file, and the chain has no cycle;
- `licence_summary` is the family's own mapping, never only inherited, with `status` (verified, unverified or
  unknown), `name` (a string or null), `text` and a `url` key; the url is an http(s) address or null, and null
  is a warning (no public model card recorded yet);
- `host_ram_gb`, after inheritance, holds `train`, `eval` (a number of GB or null), `eval_includes_comfyui_server`
  (true, false or null; not null when `eval` is a number) and `basis`;
- the tool's own block, after inheritance, is a mapping; a training block that says `supported: true` needs a
  `host_ram_gb.train` number.
Unknown keys, at the top level and inside `licence_summary` and `host_ram_gb`, are warnings. `--strict` turns
every warning into a problem. Inheritance is a deep merge in which the child's keys win.

This file has a twin in the other tool (same rules, other constants); keep the two in step.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# --- the constants that differ between the two tools' copies -------------------------------------------------
PKG = "easel"
OWN_BLOCK = "inference"  # this tool's settings block in a family file
OTHER_BLOCK = "training"  # the other tool's block; it lives only in the other tool's copy of the file
# --------------------------------------------------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]  # src/<pkg>/families_check.py -> the repository root
SCHEMA = f"{PKG}/family/1"
STATUSES = ("tested", "experimental", "unverified", "listed")
TOP_KEYS = ("schema", "id", "name", "status", "inherits", "architecture", "description", "licence_summary", "host_ram_gb", OWN_BLOCK)
LICENCE_KEYS = ("status", "name", "text", "url")
LICENCE_REQUIRED = LICENCE_KEYS
LICENCE_STATUSES = ("verified", "unverified", "unknown")
RAM_KEYS = ("train", "eval", "eval_includes_comfyui_server", "basis")


@dataclass
class Report:
    """What one check found. Problems fail the check; warnings fail it only under --strict."""

    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)


def deep_merge(parent: dict, child: dict) -> dict:
    """A new mapping: `parent` overlaid by `child`; nested mappings merge, every other value is replaced."""
    merged = dict(parent)
    for key, value in child.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _first_line(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:157] + "..." if len(text) > 160 else text


def load_families(families_dir: Path, report: Report) -> dict[str, dict]:
    """Every `*.yaml` under `families_dir` that parses as a mapping, keyed by file stem."""
    families: dict[str, dict] = {}
    for path in sorted(families_dir.glob("*.yaml")):
        rel = f"{families_dir.name}/{path.name}"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            report.problems.append(f"{rel}: cannot be read: {_first_line(exc)}")
            continue
        except yaml.YAMLError as exc:
            report.problems.append(f"{rel}: does not parse as YAML: {_first_line(exc)}")
            continue
        if not isinstance(data, dict):
            report.problems.append(f"{rel}: top level must be a mapping")
            continue
        families[path.stem] = data
    return families


def resolve(fid: str, families: dict[str, dict], cache: dict[str, dict | None], trail: tuple[str, ...] = ()) -> dict | None:
    """The family after inheritance, or None when its chain is broken (a missing parent or a cycle)."""
    if fid in cache:
        return cache[fid]
    if fid in trail or fid not in families:
        return None
    raw = families[fid]
    parent_id = raw.get("inherits")
    if parent_id is None:
        resolved: dict | None = dict(raw)
    else:
        parent = resolve(str(parent_id), families, cache, (*trail, fid)) if isinstance(parent_id, str) else None
        resolved = deep_merge(parent, raw) if parent is not None else None
    cache[fid] = resolved
    return resolved


def _chain_problem(fid: str, families: dict[str, dict]) -> str | None:
    """Why the inheritance chain of `fid` is broken, or None."""
    seen = [fid]
    current = families[fid].get("inherits")
    while current is not None:
        if not isinstance(current, str):
            return f"inherits must be a family id or null, not {current!r}"
        if current in seen:
            return f"inheritance cycle: {' -> '.join([*seen, current])}"
        if current not in families:
            return f"inherits {current!r}, but there is no {current}.yaml" if len(seen) == 1 else f"its ancestor {seen[-1]!r} inherits {current!r}, which does not exist"
        seen.append(current)
        current = families[current].get("inherits")
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_licence(raw: dict, rel: str, report: Report) -> None:
    licence = raw.get("licence_summary")
    if licence is None:
        if raw.get("inherits") is not None:
            report.problems.append(f"{rel}: no own licence_summary; a licence is never inherited from {raw.get('inherits')!r}, so state this family's own")
        else:
            report.problems.append(f"{rel}: licence_summary is missing")
        return
    if not isinstance(licence, dict):
        report.problems.append(f"{rel}: licence_summary must be a mapping")
        return
    for key in LICENCE_REQUIRED:
        if key not in licence:
            report.problems.append(f"{rel}: licence_summary.{key} is missing")
    for key in sorted(set(licence) - set(LICENCE_KEYS)):
        report.warnings.append(f"warn: {rel}: unknown key licence_summary.{key}")
    if "status" in licence and licence["status"] not in LICENCE_STATUSES:
        report.problems.append(f"{rel}: licence_summary.status must be one of {', '.join(LICENCE_STATUSES)}")
    if "text" in licence and not (isinstance(licence["text"], str) and licence["text"].strip()):
        report.problems.append(f"{rel}: licence_summary.text must be a non-empty string")
    if "name" in licence and licence["name"] is not None and not isinstance(licence["name"], str):
        report.problems.append(f"{rel}: licence_summary.name must be a string or null")
    url = licence.get("url")
    if "url" in licence:
        if url is None:
            report.warnings.append(f"warn: {rel}: licence_summary.url is null: no public model card recorded yet")
        elif not (isinstance(url, str) and url.startswith(("https://", "http://")) and len(url) > len("https://")):
            report.problems.append(f"{rel}: licence_summary.url must be an http(s) address or null")


def _check_ram(resolved: dict, rel: str, report: Report) -> None:
    ram = resolved.get("host_ram_gb")
    if ram is None:
        report.problems.append(f"{rel}: host_ram_gb is missing, also after inheritance")
        return
    if not isinstance(ram, dict):
        report.problems.append(f"{rel}: host_ram_gb must be a mapping of {', '.join(RAM_KEYS)}")
        return
    for key in RAM_KEYS:
        if key not in ram:
            report.problems.append(f"{rel}: host_ram_gb.{key} is missing")
    for key in sorted(set(ram) - set(RAM_KEYS)):
        report.warnings.append(f"warn: {rel}: unknown key host_ram_gb.{key}")
    for key in ("train", "eval"):
        value = ram.get(key)
        if value is not None and not (_is_number(value) and value > 0):
            report.problems.append(f"{rel}: host_ram_gb.{key} must be a positive number of GB or null")
    includes = ram.get("eval_includes_comfyui_server")
    if includes is not None and not isinstance(includes, bool):
        report.problems.append(f"{rel}: host_ram_gb.eval_includes_comfyui_server must be true, false or null")
    if _is_number(ram.get("eval")) and includes is None:
        report.problems.append(f"{rel}: host_ram_gb.eval is a number, so eval_includes_comfyui_server must say true or false")
    if "basis" in ram and not (isinstance(ram["basis"], str) and ram["basis"].strip()):
        report.problems.append(f"{rel}: host_ram_gb.basis must say where the numbers come from")


def _check_own_block(resolved: dict, rel: str, report: Report) -> None:
    block = resolved.get(OWN_BLOCK)
    if block is None:
        return
    if not isinstance(block, dict):
        report.problems.append(f"{rel}: {OWN_BLOCK} must be a mapping")
        return
    if OWN_BLOCK == "training" and block.get("supported") is True:
        ram = resolved.get("host_ram_gb")
        if not (isinstance(ram, dict) and _is_number(ram.get("train"))):
            report.problems.append(f"{rel}: training.supported is true, so host_ram_gb.train must be a number")


def check_family(fid: str, families: dict[str, dict], resolved: dict | None, rel: str, report: Report) -> None:
    """Check one family file as written and, through `resolved`, after inheritance; findings go into `report`."""
    raw = families[fid]
    if raw.get("schema") != SCHEMA:
        report.problems.append(f"{rel}: schema must be {SCHEMA!r}")
    if raw.get("id") != fid:
        report.problems.append(f"{rel}: id must equal the file name {fid!r}")
    if not (isinstance(raw.get("name"), str) and raw["name"].strip()):
        report.problems.append(f"{rel}: name is missing")
    if raw.get("status") not in STATUSES:
        report.problems.append(f"{rel}: status must be one of {', '.join(STATUSES)}")
    for key in sorted(set(raw) - set(TOP_KEYS)):
        where = f"; it belongs in the other tool's copy of {fid}.yaml" if key == OTHER_BLOCK else ""
        report.warnings.append(f"warn: {rel}: unknown key {key}{where}")
    _check_licence(raw, rel, report)
    chain = _chain_problem(fid, families)
    if chain is not None:
        report.problems.append(f"{rel}: {chain}")
        return
    if resolved is None:  # unreachable while the chain is sound; kept as a guard
        report.problems.append(f"{rel}: inheritance could not be resolved")
        return
    _check_ram(resolved, rel, report)
    _check_own_block(resolved, rel, report)


def check_families(families_dir: Path) -> Report:
    """Check every family file under `families_dir`."""
    report = Report()
    families = load_families(families_dir, report)
    cache: dict[str, dict | None] = {}
    for fid in sorted(families):
        rel = f"{families_dir.name}/{fid}.yaml"
        check_family(fid, families, resolve(fid, families, cache), rel, report)
        report.checked.append(fid)
    return report


def print_report(report: Report, strict: bool, out: Callable[[str], None] = print) -> int:
    """Print the findings and the summary line; return the exit code (0 ok, 1 problems)."""
    problems = list(report.problems)
    if strict:
        problems += [w.removeprefix("warn: ") for w in report.warnings]
    _emit(problems, out)
    if not strict:
        _emit(report.warnings, out)
    count = len(report.checked)
    if problems:
        out(f"FAIL: {len(problems)} problem(s) in {count} family file(s)")
        return 1
    tail = f", {len(report.warnings)} warning(s)" if report.warnings else ""
    out(f"ok: {count} family file(s) checked{tail}")
    return 0


def _emit(lines: Iterable[str], out: Callable[[str], None]) -> None:
    for line in lines:
        out(line)


def run(*, families_dir: Path | None = None, strict: bool = False, out: Callable[[str], None] = print) -> int:
    """The `families check` command: check the shipped families (or `families_dir`) and return the exit code."""
    families_dir = families_dir or REPO / "families"
    if not families_dir.is_dir():
        out("no families folder found: pass --dir <folder holding the family files>")
        return 2
    report = check_families(families_dir)
    if not report.checked and not report.problems:
        out(f"no family files in {families_dir.name}/")
        return 1
    return print_report(report, strict, out)
