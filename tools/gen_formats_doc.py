"""Generate the docs that must agree with the code: docs/STUCK.md, docs/reference/formats.md and
.github/ISSUE_TEMPLATE/pitfall.yml.

Why: the STUCK template has one owner, `src/<pkg>/_common/journal.py`. The STUCK page and the issue form for
pitfalls are rendered from that one text, so they cannot drift apart. The record-formats page is rendered
from `schemas/*.schema.json` for the same reason.

usage: python tools/gen_formats_doc.py [--repo <dir>] [--check]

- Without `--check`: writes the three files and prints one `wrote <path>` line per file.
- `--check` (CI): writes nothing. Exit 1 when a file is missing, differs from what would be generated, or
  the STUCK renderings on disk disagree with the template; exit 0 prints `in step: ...`.
- Exit 2 when an input does not exist yet (the journal module or `schemas/`); whatever can be generated
  is still written.

The template is read by running a command with capture_output=True and the clock frozen
(KILN_NOW and EASEL_NOW = 2030-01-01T00:00:00+00:00); only stdout is kept, because the frozen-clock warning
goes to stderr. The first command is `python -m <pkg> journal stuck --template`; when that is not
available, a one-line Python command prints `<pkg>._common.journal.STUCK_TEMPLATE` (or the result of
`stuck_template()`). The same file serves both repos; the package name comes from pyproject.toml.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import yaml

FROZEN_NOW = "2030-01-01T00:00:00+00:00"
STUCK_PATH = Path("docs/STUCK.md")
FORMATS_PATH = Path("docs/reference/formats.md")
PITFALL_PATH = Path(".github/ISSUE_TEMPLATE/pitfall.yml")
MAX_DEPTH = 3  # nested fields shown in formats.md, counted from the record's top level
KEY_RE = re.compile(r"^[a-z][a-z0-9 ()_-]*$")
READ_ATTR = """\
import importlib, sys
m = importlib.import_module(sys.argv[1] + "._common.journal")
t = getattr(m, "STUCK_TEMPLATE", None)
if not isinstance(t, str):
    f = getattr(m, "stuck_template", None)
    t = f() if callable(f) else None
if not isinstance(t, str):
    sys.exit(3)
sys.stdout.write(t)
"""


class InputMissing(Exception):
    """An input the generator needs does not exist yet; the message says which and who builds it."""


@dataclass(frozen=True)
class Field:
    """One `key: hint` line of the STUCK template."""

    key: str
    hint: str


@dataclass(frozen=True)
class Template:
    """The STUCK template: its header line, the placeholders in the header, and its fields in order."""

    text: str
    header: str
    placeholders: tuple[str, ...]
    fields: tuple[Field, ...]


# --- inputs -----------------------------------------------------------------------------------------------------

def package_name(repo: Path) -> str:
    """The distribution name in the repo's pyproject.toml (`kiln` or `easel`)."""
    data = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["name"])


def frozen_env(repo: Path, home: Path) -> dict[str, str]:
    """The caller's environment without any KILN_*/EASEL_* value, plus the frozen clock and the repo's src/.

    Both tools' home variables point at an empty scratch folder, so no command run here can find, read or
    write a real instance (or its journal) by searching upward from the repo or in the user's config folder.
    """
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("KILN_", "EASEL_"))}
    env["KILN_NOW"] = FROZEN_NOW
    env["EASEL_NOW"] = FROZEN_NOW
    env["KILN_HOME"] = str(home)
    env["EASEL_HOME"] = str(home)
    env["PYTHONIOENCODING"] = "utf-8"
    src = str(repo / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_stdout(repo: Path, args: list[str]) -> str | None:
    """Run `python <args>` in the repo with the frozen clock; stdout on exit 0, else None (stderr is dropped)."""
    with tempfile.TemporaryDirectory(prefix="gen-formats-") as scratch:
        proc = subprocess.run(
            [sys.executable, *args], cwd=repo, env=frozen_env(repo, Path(scratch)), capture_output=True,
            text=True, encoding="utf-8", timeout=120, check=False,
        )
    return proc.stdout if proc.returncode == 0 and proc.stdout.strip() else None


def read_template(repo: Path, pkg: str) -> Template:
    """Get the STUCK template from the CLI, else from the journal module; parse and check its shape."""
    text = run_stdout(repo, ["-m", pkg, "journal", "stuck", "--template"])
    if text is None:
        text = run_stdout(repo, ["-c", READ_ATTR, pkg])
    if text is None:
        raise InputMissing(
            f"the STUCK template is not available: `{pkg} journal stuck --template` failed and "
            f"{pkg}._common.journal has no STUCK_TEMPLATE (T1a builds it; run this after M1)"
        )
    return parse_template(text)


def parse_template(text: str) -> Template:
    """Split the template into header, placeholders and `key: hint` fields; refuse a shape it cannot render."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    if not lines or not lines[0].startswith("## STUCK"):
        raise ValueError("the STUCK template must start with a line `## STUCK ...`")
    if any("clock frozen" in line.lower() for line in lines):
        raise ValueError("the frozen-clock warning reached stdout; it must go to stderr")
    fields: list[list[str]] = []
    for line in lines[1:]:
        key, sep, hint = line.partition(":")
        if sep and KEY_RE.match(key):
            fields.append([key, hint.strip()])
        elif fields:
            fields[-1][1] = (fields[-1][1] + " " + line.strip()).strip()
        elif line.strip():
            raise ValueError(f"the STUCK template has a line before its first field: {line!r}")
    if not fields:
        raise ValueError("the STUCK template has no `key: hint` fields")
    header = lines[0]
    return Template(
        text="\n".join(lines),
        header=header,
        placeholders=tuple(re.findall(r"<([^<>]+)>", header)),
        fields=tuple(Field(k, h) for k, h in fields),
    )


def load_schemas(repo: Path) -> list[tuple[str, Path, dict]]:
    """Every `schemas/*.schema.json` as (name, path, parsed schema), sorted by name."""
    folder = repo / "schemas"
    files = sorted(folder.glob("*.schema.json")) if folder.is_dir() else []
    if not files:
        raise InputMissing("schemas/*.schema.json not found (T6 writes them)")
    out = []
    for path in files:
        try:
            out.append((path.name[: -len(".schema.json")], path, json.loads(path.read_text(encoding="utf-8"))))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.relative_to(repo).as_posix()} is not valid JSON: {exc}") from exc
    return out


# --- docs/STUCK.md ----------------------------------------------------------------------------------------------

STUCK_MD = """\
# STUCK: when a step fails

<!-- generated by tools/gen_formats_doc.py from the STUCK template in src/{{pkg}}/_common/journal.py; do not edit by hand -->

Read this when a command fails, a gate refuses, or what you see does not match the runbook. Do the steps in
order, once each.

## Steps

1. **Stop.** Never run the same failing command a third time, and never route around a gate: no other flag,
   no hand edit, no other tool that does the same thing. Reading a job's state once is not a retry.
2. **Check the install:** `{{pkg}} doctor`. Keep its `FAIL` and `warn` lines.
3. **Write the note:** `{{pkg}} journal stuck "<one line: what failed>" --stage <stage> --command "<exact>"
   --exit-code <n>`, plus `--expected`, `--tried` (once per attempt), `--hypothesis` and `--need`. It adds a
   block in the shape below to today's journal and prints it for your hand-off; a field you leave out keeps
   its placeholder, so the gap shows. Give the output with `--output-file <file>` or `--output -` (stdin)
   from what you already have: never run the failing command again just to capture it.
4. **Look the symptom up once:** the failure table (section 6) of the runbook you were following, then
   `docs/troubleshooting/`. If a row matches, do its action once and record the result with
   `{{pkg}} journal note "<line>"`.
5. **Hand off and stop:** one message with what you did, the last 20 lines of output, what the runbook said
   to expect, and what you need from whom (the `need:` line). Then wait for the answer.
6. **Two rules disagree:** do neither; put both, with their files and lines, in the hand-off.
7. **Fixed?** Add a row (symptom, cause, action) to the runbook's failure table, dated from `date`, and a
   journal line, in the same session.

## The STUCK note

```
{{template}}
```

- Every `<...>` is filled from the clock, the command and its output, never from memory.
- `need:` names one of three: the operator's yes, an image judgement (only a human gives one), or a
  maintainer (the tool or the docs are wrong).
- The issue form for pitfalls (`.github/ISSUE_TEMPLATE/pitfall.yml`) has the same fields. A fix that
  belongs in the tool or the docs goes there, without private values.

## Exit codes

| code | meaning | what to do |
|---|---|---|
| 0 | ok | continue |
| 1 | a check found problems | read the lines it printed; fix the cause, never the check |
| 2 | usage error, refusal, or a planned verb | read `--help` or the refusal line; a refusal is a gate, so do not route around it |
| 3 | deferred by a guard: not enough free RAM or disk, or the scheduler said wait | wait as the line says; never start the work another way |
| 4 | a lock or a file is held by another process | see Locks below |

## Locks

Every store has a `.lock` file. A writer takes an OS lock on an open handle of it, so the lock ends when the
holding process ends, even when that process is killed; the pid, host, time and tool written inside are a
hint only. A writer waits up to 30 s for the lock, then exits 4. Windows also refuses to replace a file that
another process holds open (a backup, a review page, an antivirus scan); after a short retry loop that
exits 4 too, naming the file.

To confirm that a holder is gone:
1. Read the pid in the `.lock` file; do not edit the file.
2. Check whether that process runs: `tasklist /FI "PID eq <pid>"` on Windows, `ps -p <pid>` elsewhere.
3. Not running: the lock is already free, so run the command once more. Running: it is a real holder; wait a
   minute, try once more, then write a STUCK note with `need: operator yes`.

Never delete a `.lock` file: deleting it does not release an OS lock, and nothing is deleted anyway.
"""


def render_stuck(pkg: str, template: Template) -> str:
    """docs/STUCK.md for this repo, with the template in one fenced block."""
    if "```" in template.text:
        raise ValueError("the STUCK template must not contain a code fence")
    return STUCK_MD.replace("{{pkg}}", pkg).replace("{{template}}", template.text)


# --- .github/ISSUE_TEMPLATE/pitfall.yml -------------------------------------------------------------------------

def _q(text: str) -> str:
    """A YAML double-quoted scalar (a JSON string is one)."""
    return json.dumps(text, ensure_ascii=False)


def _form_ids(template: Template) -> list[tuple[str, str, str, str]]:
    """(id, kind, label, description) per form element, header placeholders first, ids unique.

    An id is the label up to its first ` (`, like the journal's own field keys: `output (last 20 lines)` ->
    `output`, `exit code` -> `exit_code`.
    """
    rows = [(p, "input", p, f"from the note's first line: {template.header}") for p in template.placeholders]
    rows += [(f.key, "textarea", f.key, f.hint) for f in template.fields]
    out, seen = [], set()
    for key, kind, label, desc in rows:
        base = re.sub(r"[^a-z0-9]+", "_", key.split(" (", 1)[0].lower()).strip("_") or "field"
        ident, n = base, 2
        while ident in seen:
            ident, n = f"{base}_{n}", n + 1
        seen.add(ident)
        out.append((ident, kind, label, desc))
    return out


def render_pitfall(pkg: str, template: Template) -> str:
    """A GitHub issue form whose fields are the STUCK template's placeholders and fields, in order."""
    intro = (
        f"Fill this from the STUCK note in your journal (`{pkg} journal report` collects the day's notes). "
        "Leave out private values: no absolute paths (write `<home>`, `<models_root>`), no names, no ids of "
        "your own records, no tokens, and no description of any image."
    )
    lines = [
        f"# generated by tools/gen_formats_doc.py from the STUCK template in src/{pkg}/_common/journal.py; "
        "do not edit by hand",
        f"name: {_q('Pitfall (a STUCK note)')}",
        f"description: {_q('A step failed, or a runbook did not match what happened.')}",
        f"title: {_q('[pitfall] <stage>: <one line>')}",
        f"labels: [{_q('pitfall')}]",
        "body:",
        "  - type: markdown",
        "    attributes:",
        f"      value: {_q(intro)}",
    ]
    for ident, kind, label, desc in _form_ids(template):
        lines += [
            f"  - type: {kind}",
            f"    id: {ident}",
            "    attributes:",
            f"      label: {_q(label)}",
            f"      description: {_q(desc)}",
        ]
        if kind == "textarea" and ident.startswith("output"):
            lines.append("      render: text")
        lines += ["    validations:", "      required: true"]
    return "\n".join(lines) + "\n"


# --- docs/reference/formats.md ----------------------------------------------------------------------------------

def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def _value(v: object) -> str:
    return f"`{v}`" if isinstance(v, str) else f"`{json.dumps(v)}`"


def _resolve(node: object, root: dict, depth: int = 0) -> tuple[dict, str | None]:
    """Follow a local `$ref` (siblings win); returns the node and the ref it came through, if any."""
    if not isinstance(node, dict):
        return {}, None
    ref = node.get("$ref")
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return _merge_all_of(node, root, depth), None
    target: object = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        target = target.get(part, {}) if isinstance(target, dict) else {}
    merged = {**(target if isinstance(target, dict) else {}), **{k: v for k, v in node.items() if k != "$ref"}}
    return _merge_all_of(merged, root, depth), ref


def _merge_all_of(node: dict, root: dict, depth: int) -> dict:
    """Fold the `properties` and `required` of `allOf` parts into the node, so composed records list them."""
    parts = node.get("allOf")
    if not isinstance(parts, list) or depth > 8:  # the depth stops a cycle of references
        return node
    props = dict(node.get("properties") or {})
    required = list(node.get("required") or [])
    for part in parts:
        resolved, _ = _resolve(part, root, depth + 1)
        props.update(resolved.get("properties") or {})
        required += [r for r in resolved.get("required") or [] if r not in required]
    return {**node, "properties": props, "required": required}


def _type(node: dict, root: dict) -> str:
    """A short human type: a const, an enum, `array of x`, alternatives joined by `or`, a format."""
    if "const" in node:
        return _value(node["const"])
    if "enum" in node:
        return "one of " + ", ".join(_value(v) for v in node["enum"])
    ref = node.get("$ref")
    if isinstance(ref, str) and not ref.startswith("#"):
        return f"see `{ref}`"
    kind = node.get("type")
    if kind is None:
        options = node.get("anyOf") or node.get("oneOf")
        if isinstance(options, list) and options:
            return " or ".join(_type(_resolve(o, root)[0], root) for o in options)
        return "object" if node.get("properties") or isinstance(node.get("additionalProperties"), dict) else "any"
    kinds = kind if isinstance(kind, list) else [kind]
    names = []
    for k in kinds:
        if k == "array":
            items = node.get("items")
            names.append("array of " + (_type(_resolve(items, root)[0], root) if isinstance(items, dict) else "any"))
        else:
            names.append(str(k))
    out = " or ".join(names)
    return f"{out} ({node['format']})" if node.get("format") else out


def _meaning(node: dict) -> str:
    text = str(node.get("description") or node.get("title") or "")
    if "default" in node:
        text = (text + f" Default: {_value(node['default'])}.").strip()
    return _cell(text)


def _fields(node: dict, root: dict, prefix: str, depth: int, refs: frozenset[str]) -> Iterator[tuple[str, ...]]:
    """(field, type, required, meaning) rows, depth first, in the schema's own order."""
    required = set(node.get("required") or [])
    for key, raw in (node.get("properties") or {}).items():
        sub, ref = _resolve(raw, root)
        name = f"{prefix}{key}"
        yield (f"`{name}`", _cell(_type(sub, root)), "yes" if key in required else "no", _meaning(sub))
        if depth >= MAX_DEPTH or (ref and ref in refs):
            continue
        seen = refs | {ref} if ref else refs
        if sub.get("properties") or isinstance(sub.get("additionalProperties"), dict):
            yield from _fields(sub, root, name + ".", depth + 1, seen)
        items, item_ref = _resolve(sub.get("items"), root)
        if items.get("properties") and not (item_ref and item_ref in seen):
            yield from _fields(items, root, name + "[].", depth + 1, seen | {item_ref} if item_ref else seen)
    extra, extra_ref = _resolve(node.get("additionalProperties"), root)
    if extra and depth < MAX_DEPTH and not (extra_ref and extra_ref in refs):
        name = f"{prefix}<key>"
        yield (f"`{name}`", _cell(_type(extra, root)), "no", _meaning(extra))
        if extra.get("properties") or isinstance(extra.get("additionalProperties"), dict):
            yield from _fields(extra, root, name + ".", depth + 1, refs | {extra_ref} if extra_ref else refs)


def _anchor(name: str) -> str:
    return re.sub(r"[^a-z0-9 _-]", "", name.lower()).replace(" ", "-")


def _schema_id(schema: dict) -> str:
    """The record's `schema` value in backticks (its const, else the `$id`), or `-` when it declares none."""
    const = ((schema.get("properties") or {}).get("schema") or {}).get("const")
    value = str(const or schema.get("$id") or "")
    return f"`{value}`" if value else "-"


def render_formats(pkg: str, repo: Path, schemas: list[tuple[str, Path, dict]]) -> str:
    """docs/reference/formats.md: an index table, then one section per record type with its fields."""
    out = [
        f"# Record formats ({pkg})",
        "",
        "<!-- generated by tools/gen_formats_doc.py from schemas/*.schema.json; do not edit by hand -->",
        "",
        f"Every record {pkg} writes is a YAML or JSON file, or one line of a JSONL log, described by a JSON Schema",
        "(draft 2020-12) in `schemas/`. A record's first key is `schema: \"<repo>/<name>/<version>\"`; a new",
        "version always comes with a reader for the old one. Worked examples are in `schemas/examples/`. Read",
        "records through the commands in [REGISTRY.md](../REGISTRY.md); never edit one by hand.",
        "",
        "| record | schema | file |",
        "|---|---|---|",
    ]
    for name, path, schema in schemas:
        rel = path.relative_to(repo).as_posix()
        out.append(f"| [{name}](#{_anchor(name)}) | {_schema_id(schema)} | [{rel}](../../{rel}) |")
    for name, path, schema in schemas:
        rel = path.relative_to(repo).as_posix()
        out += ["", f"## {name}", ""]
        title, desc = schema.get("title"), schema.get("description")
        if title or desc:
            out += [": ".join(_cell(t) for t in (title, desc) if t), ""]
        out.append(f"- schema: {_schema_id(schema)}")
        out.append(f"- file: [{rel}](../../{rel})")
        for ext in ("yaml", "yml", "json", "jsonl"):
            example = repo / "schemas" / "examples" / f"{name}.example.{ext}"
            if example.is_file():
                erel = example.relative_to(repo).as_posix()
                out.append(f"- example: [{erel}](../../{erel})")
                break
        rows = list(_fields(schema, schema, "", 1, frozenset()))
        if rows:
            out += ["", "| field | type | required | meaning |", "|---|---|---|---|"]
            out += [f"| {' | '.join(row)} |" for row in rows]
    return "\n".join(out) + "\n"


# --- agreement and the main flow -------------------------------------------------------------------------------

def agreement_problems(repo: Path, template: Template) -> list[str]:
    """What on disk disagrees with the template: the block in STUCK.md, the fields of pitfall.yml."""
    problems = []
    stuck = repo / STUCK_PATH
    if stuck.is_file() and f"```\n{template.text}\n```" not in stuck.read_text(encoding="utf-8"):
        problems.append(f"disagrees: {STUCK_PATH.as_posix()} does not carry the STUCK template verbatim")
    form = repo / PITFALL_PATH
    if form.is_file():
        try:
            data = yaml.safe_load(form.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            return [*problems, f"disagrees: {PITFALL_PATH.as_posix()} is not valid YAML: {exc}"]
        got = [
            (el.get("id"), (el.get("attributes") or {}).get("label"), (el.get("attributes") or {}).get("description"))
            for el in data.get("body") or [] if isinstance(el, dict) and el.get("type") in ("input", "textarea")
        ]
        want = [(i, label, desc) for i, _, label, desc in _form_ids(template)]
        if got != want:
            problems.append(f"disagrees: the fields of {PITFALL_PATH.as_posix()} are not the STUCK template's")
    return problems


def generate(repo: Path) -> tuple[dict[Path, str], list[str], Template | None]:
    """Render what the inputs allow: ({relative path: text}, [missing-input lines], the template or None)."""
    pkg = package_name(repo)
    docs: dict[Path, str] = {}
    missing: list[str] = []
    template = None
    try:
        template = read_template(repo, pkg)
        docs[STUCK_PATH] = render_stuck(pkg, template)
        docs[PITFALL_PATH] = render_pitfall(pkg, template)
    except InputMissing as exc:
        missing.append(f"skipped {STUCK_PATH.as_posix()} and {PITFALL_PATH.as_posix()}: {exc}")
    try:
        docs[FORMATS_PATH] = render_formats(pkg, repo, load_schemas(repo))
    except InputMissing as exc:
        missing.append(f"skipped {FORMATS_PATH.as_posix()}: {exc}")
    return docs, missing, template


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render STUCK.md, formats.md and pitfall.yml from their sources.")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1], help="repo root")
    parser.add_argument("--check", action="store_true", help="compare with the files on disk; write nothing")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        docs, missing, template = generate(repo)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    for line in missing:
        print(line)
    if args.check:
        problems = []
        for rel, text in docs.items():
            path = repo / rel
            if not path.is_file():
                problems.append(f"missing: {rel.as_posix()} (run python tools/gen_formats_doc.py)")
            elif path.read_text(encoding="utf-8") != text:
                problems.append(f"stale: {rel.as_posix()} (run python tools/gen_formats_doc.py)")
        if template is not None:
            problems += agreement_problems(repo, template)
        for line in problems:
            print(line)
        if missing:
            return 2
        if problems:
            return 1
        print("in step: " + ", ".join(rel.as_posix() for rel in docs))
        return 0
    for rel, text in docs.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        print(f"wrote {rel.as_posix()}")
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
