"""`qa docs|agents|tests` on small synthetic repositories, plus the structure checks on the shipped docs.

The same file serves both repositories: the package name comes from pyproject.toml. Needles that the privacy
scan would flag (absolute paths, the frozen-clock warning, old dates) are built at run time from fragments.
"""

import argparse
import importlib
import json
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PKG = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["name"]
OTHER = "easel" if PKG == "kiln" else "kiln"
qa = importlib.import_module(f"{PKG}.qa")
cli = importlib.import_module(f"{PKG}.cli")

SECTIONS = [
    "## 1. Purpose and who decides",
    "## 2. Entry conditions",
    "## 3. Commands",
    "## 4. Expected output",
    "## 5. Gate and operator question",
    "## 6. Symptom, cause, action",
    "## 7. Record",
]
FENCE = "`" * 3


def runbook(stem: str, commands: list[str] | None = None, output: list[str] | None = None,
            sections: list[str] | None = None, extra: str = "") -> str:
    """A runbook with the seven sections, an entry table for `stem`, a command block and an output block."""
    commands = commands if commands is not None else [f"{PKG} families check --strict", f"{PKG} next"]
    output = output if output is not None else ["checked 15 family files at 2030-01-01T00:00:00+00:00"]
    body = {
        SECTIONS[0]: "The demo stage. The operator decides.",
        SECTIONS[1]: f"| command | literal line |\n|---|---|\n| `{PKG} next --stage` | `{stem}` |",
        SECTIONS[2]: "\n".join([FENCE, *commands, FENCE]),
        SECTIONS[3]: "\n".join([FENCE, *output, FENCE]),
        SECTIONS[4]: "Done when the check exits 0.",
        SECTIONS[5]: "| symptom | cause | action |\n|---|---|---|\n| exit 1 | a family file | fix it |",
        SECTIONS[6]: "\n".join([FENCE, f'{PKG} journal note "{stem}: done"', FENCE]),
    }
    parts = [f"# {stem}: a demo stage", ""]
    for heading in sections if sections is not None else SECTIONS:
        parts += [heading, "", body.get(heading, "text"), ""]
    return "\n".join(parts) + extra


def make_repo(tmp_path: Path, files: dict[str, str], toml: dict[str, str] | None = None) -> Path:
    """A synthetic checkout: this repository's qa.toml (with each `toml` key replaced by its value) plus the
    given files."""
    root = tmp_path / "repo"
    root.mkdir(parents=True)
    rules = (REPO / "qa.toml").read_text(encoding="utf-8")
    for old, new in (toml or {}).items():
        assert old in rules, old
        rules = rules.replace(old, new)
    (root / "qa.toml").write_text(rules, encoding="utf-8")
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text) if text.startswith("\n") else text, encoding="utf-8")
    return root


def docs(root: Path, **kwargs) -> "qa.Report":
    return qa.check_docs(root, qa.load_rules(root), **kwargs)


def found(report, check: str | None = None) -> list[str]:
    return [str(f) for f in report.findings if check is None or f.check == check]


# --- qa docs ----------------------------------------------------------------------------------------------------

def test_a_clean_runbook_has_no_finding(tmp_path):
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": runbook("05-demo")})
    report = docs(root)
    assert found(report) == []
    assert report.lines()[-1] == "docs: 0 finding(s) in 1 file(s)"


def test_sections_missing_extra_and_out_of_order(tmp_path):
    shuffled = [SECTIONS[1], SECTIONS[0], *SECTIONS[2:4], "## 5. Something else", *SECTIONS[5:]]
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": runbook("05-demo", sections=shuffled)})
    lines = found(docs(root), "sections")
    assert any("missing the section heading '## 5. Gate and operator question'" in line for line in lines)
    assert any("an extra section heading '## 5. Something else'" in line for line in lines)
    assert any("out of order" in line for line in lines)


def test_the_entry_literal_must_be_the_runbooks_own_id(tmp_path):
    root = make_repo(tmp_path, {"docs/runbooks/06-demo.md": runbook("05-demo")})
    lines = found(docs(root), "entry")
    assert len(lines) == 1 and "this runbook is '06-demo'" in lines[0]


def test_an_entry_condition_without_code_cells_or_with_jq_is_a_finding(tmp_path):
    text = runbook("05-demo").replace(f"| `{PKG} next --stage` | `05-demo` |",
                                      f"| `{PKG} next --stage` | `05-demo` |\n| {PKG} next --json \\| jq .stage | 05 |")
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": text})
    lines = found(docs(root), "entry")
    assert any("two code cells" in line for line in lines)
    assert any("jq" in line for line in lines)


def test_the_pending_marker_is_a_finding_unless_allowed(tmp_path):
    marker = "<!-- pending: fixture output -->"
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": runbook("05-demo", output=[]).replace(
        f"{SECTIONS[3]}\n\n{FENCE}\n{FENCE}", f"{SECTIONS[3]}\n\n{marker}")})
    assert len(found(docs(root), "pending")) == 1
    allowed = docs(root, allow_pending=True)
    assert found(allowed) == []
    assert allowed.lines()[-1].endswith("1 pending marker(s) tolerated (--allow-pending)")


def test_commands_are_checked_against_the_clis_own_help(tmp_path):
    commands = [
        f"{PKG} families check --strict",
        f"{PKG} families check --no-such-flag",
        f"{PKG} families no-such-action",
        f"{PKG} no-such-verb",
        f"{PKG} journal note|stuck|report",
        f'{PKG} journal stuck "<line>" --stage <stage> --command "<exact>" --exit-code <n>',
        f"{PKG} journal <action> --help",
        f"python -m {PKG} qa docs --allow-pending",
        f"{PKG} --version",
    ]
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": runbook("05-demo", commands=commands)})
    lines = found(docs(root), "commands")
    assert len(lines) == 3, lines
    assert any(f"`{PKG} families check --no-such-flag` -- no such flag" in line for line in lines)
    assert any(f"`{PKG} families no-such-action` -- no such subcommand (has: check)" in line for line in lines)
    assert any(f"`{PKG} no-such-verb` -- no such verb" in line for line in lines)


def test_inline_spans_and_escaped_table_pipes_are_commands_too(tmp_path):
    text = f"""
    # Notes
    Run `{PKG} families check --nope`, then `{PKG} journal note\\|stuck`.
    | ask | first command |
    |---|---|
    | check | `{PKG} journal note\\|nosuch` |
    """
    lines = found(docs(make_repo(tmp_path, {"docs/notes.md": text})), "commands")
    assert len(lines) == 2, lines
    assert any("--nope" in line for line in lines) and any("journal nosuch" in line for line in lines)


def test_a_planned_verb_is_counted_not_checked(tmp_path):
    planned = [c.verb for c in cli.COMMANDS if not cli.module_exists(c.module)]
    if not planned:
        pytest.skip("every verb is built")
    root = make_repo(tmp_path, {"docs/notes.md": f"Later: `{PKG} {planned[0]} anything --whatever`.\n"})
    report = docs(root)
    assert found(report) == []
    assert "1 name a planned verb" in report.notes["commands"]


def test_a_subcommand_marked_planned_is_counted_not_checked():
    """A built verb may register an unbuilt subcommand with set_defaults(planned=...): counted, flags unchecked."""
    parser = argparse.ArgumentParser(prog="demo")
    actions = parser.add_subparsers(dest="action")
    built = actions.add_parser("list")
    built.add_argument("--type")
    actions.add_parser("audit", add_help=False).set_defaults(planned="T2b")
    tree = qa._structure(parser)
    assert tree["subcommands"]["audit"]["planned"] == "T2b" and tree["subcommands"]["list"]["planned"] is None
    assert qa._walk("demo", tree, ["audit", "--whatever"]) == ([], {f"{PKG} demo audit": "T2b"}, False)
    assert qa._walk("demo", tree, ["list", "--type", "lora"]) == ([], {}, True)
    assert qa._walk("demo", tree, ["list|audit"]) == ([], {f"{PKG} demo audit": "T2b"}, True)
    problems, planned, _built = qa._walk("demo", tree, ["dups"])
    assert not planned and problems == [f"`{PKG} demo dups` -- no such subcommand (has: audit, list)"]


# --- planned markers, required options, the sister tool (a synthetic CLI, so nothing depends on what is built) ---

def synthetic_cli() -> argparse.ArgumentParser:
    """`<pkg> demo list|make|audit`: make needs --name, list takes one of two required filters, audit is planned."""
    parser = argparse.ArgumentParser(prog="demo")
    actions = parser.add_subparsers(dest="action")
    listing = actions.add_parser("list")
    one_of = listing.add_mutually_exclusive_group(required=True)
    one_of.add_argument("--all", action="store_true")
    one_of.add_argument("--type")
    make = actions.add_parser("make")
    make.add_argument("--name", required=True)
    make.add_argument("--note")
    actions.add_parser("audit", add_help=False).set_defaults(planned="T2b")
    return parser


class FakeHelps:
    """A HelpCache stand-in over the synthetic CLI: verb `demo` is built, verb `later` is planned (session-2)."""

    def __init__(self, pkg: str = PKG) -> None:
        self.pkg, self.root = pkg, REPO
        self.tree = qa._structure(synthetic_cli())

    def top(self) -> dict:
        return {"options": ["--version"], "verbs": {"demo": {"planned": None}, "later": {"planned": "session-2"}}}

    def verb(self, name: str) -> dict:
        state = self.top()["verbs"].get(name)
        if state is None:
            return {"unknown": True}
        return {"planned": state["planned"]} if state["planned"] else {"tree": self.tree}


def fake_docs(tmp_path: Path, text: str, *, sister: FakeHelps | None = None) -> "qa.Report":
    """_check_commands over one synthetic doc with the FakeHelps CLI as this tool (and optionally as the sister)."""
    path = tmp_path / "notes.md"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    report = qa.Report("docs", qa.DOC_CHECKS)
    counts = {"checked": 0, "planned": 0, "skipped": 0, "pending": 0, "sister": 0, "sister_unchecked": 0}
    tools = [(PKG, FakeHelps()), (OTHER, sister)]
    qa._check_commands(report, qa.parse_doc(path, "notes.md", []), tools, counts)
    report.notes["counts"] = json.dumps(counts, sort_keys=True)
    return report


def test_split_comment_reads_a_shell_comment_only():
    assert qa.split_comment(f"{PKG} next   # planned: T3") == (f"{PKG} next", "planned: T3")
    assert qa.split_comment(f'{PKG} journal note "fix #3"') == (f'{PKG} journal note "fix #3"', None)
    assert qa.split_comment("read docs/a.md#part") == ("read docs/a.md#part", None)
    assert qa.split_comment(f"{PKG} config set paths.models_root <the operator's folder> # mine") == (
        f"{PKG} config set paths.models_root <the operator's folder>", "mine")


def test_a_fenced_line_naming_a_planned_command_ends_with_the_marker(tmp_path):
    text = f"""
    # Notes
    Inline mentions are free: `{PKG} later go` and `{PKG} demo audit`.
    {FENCE}
    {PKG} later go
    {PKG} later go  # planned: session-2
    {PKG} demo audit --anything
    {PKG} demo audit  # planned: T2b
    {PKG} demo audit  # planned: T9
    {PKG} demo list|audit --all
    {PKG} demo list --all  # planned: T2b
    {PKG} demo list --all  # lists everything
    {PKG} demo nosuch  # planned: T2b
    {FENCE}
    """
    report = fake_docs(tmp_path, text)
    lines = found(report, "planned")
    assert len(lines) == 5, lines
    assert any(":5: names `" + PKG + " later` (planned: session-2), which is not built; end the line with "
               "`# planned: session-2`" in line for line in lines)
    assert any(":7: names `" + PKG + " demo audit` (planned: T2b)" in line for line in lines)
    assert any(":9: says `# planned: T9`, but the CLI prints planned: T2b" in line for line in lines)
    assert any(":10: names `" + PKG + " demo audit`" in line for line in lines)
    assert any(":11: says `# planned: T2b`, but every command on it is built now" in line for line in lines)
    # an unknown subcommand is the commands check's finding, not a stale marker
    assert found(report, "commands") == [f"commands: notes.md:13: `{PKG} demo nosuch` -- no such subcommand "
                                         "(has: audit, list, make)"]


def test_a_fenced_line_carries_every_required_option(tmp_path):
    text = f"""
    # Notes
    Inline is free: `{PKG} demo make`.
    {FENCE}
    {PKG} demo make
    {PKG} demo make --name robot
    {PKG} demo make --help
    {PKG} demo list
    {PKG} demo list --type lora
    {PKG} demo make --note "first" \\
        --name robot
    {FENCE}
    """
    lines = found(fake_docs(tmp_path, text), "commands")
    assert lines == [
        f"commands: notes.md:5: `{PKG} demo make` needs --name: a command line carries every option the verb "
        f"requires (see `{PKG} demo make --help`)",
        f"commands: notes.md:8: `{PKG} demo list` needs --all or --type: a command line carries every option the "
        f"verb requires (see `{PKG} demo list --help`)",
    ]


def test_the_sister_tools_commands_are_checked_against_its_own_cli(tmp_path):
    text = f"""
    # Notes
    `{OTHER} demo make --nope`
    {FENCE}
    {OTHER} later go
    {FENCE}
    """
    report = fake_docs(tmp_path, text, sister=FakeHelps(OTHER))
    assert any(f"`{OTHER} demo make --nope` -- no such flag" in line for line in found(report, "commands"))
    assert any(f"names `{OTHER} later` (planned: session-2)" in line for line in found(report, "planned"))
    assert json.loads(report.notes["counts"])["sister"] == 2
    unchecked = fake_docs(tmp_path, text, sister=None)
    assert found(unchecked) == [] and json.loads(unchecked.notes["counts"])["sister_unchecked"] == 2


def test_a_sister_that_is_not_importable_is_counted_not_checked(tmp_path):
    root = make_repo(tmp_path, {"docs/notes.md": "See `nosuchtool no-such-verb --no-such-flag`.\n"},
                     toml={f'sister = "{OTHER}"': 'sister = "nosuchtool"'})
    report = docs(root)
    assert found(report) == []
    assert "1 nosuchtool command(s) not checked (nosuchtool is not importable)" in report.notes["commands"]


def test_the_sister_cli_is_used_when_it_is_installed(tmp_path):
    pytest.importorskip(f"{OTHER}.cli")
    root = make_repo(tmp_path, {"docs/notes.md": f"See `{OTHER} no-such-verb`.\n"})
    report = docs(root)
    assert found(report, "commands") == [f"commands: docs/notes.md:1: `{OTHER} no-such-verb` -- no such verb "
                                         f"(see `{OTHER} --help`)"]
    assert f"checked against {OTHER}'s CLI" in report.notes["commands"]


def test_table_actions_never_run_gpu_work_where_qa_toml_asks(tmp_path):
    rule = ("[[docs.gpu_actions]]\nregex = '\\b(?:start|restart|stop|suspend)\\b[^|]*?\\b(?:server|queue|process)\\b'"
            "\nwhy = \"report to the operator\"\n\n")
    table = ("# Render\n\n| symptom | cause | action |\n|---|---|---|\n"
             "| the server is gone | it stopped the server | restart the server, then retry |\n"
             "| a job waits | no room | report the line to the operator; need: operator |\n")
    scope = '[docs]\ngpu_actions_in = ["docs/troubleshooting/*.md"]\n'
    root = make_repo(tmp_path, {"docs/troubleshooting/render.md": table, "docs/notes.md": table},
                     toml={"[docs]\n": scope, "[tests]\n": rule + "[tests]\n"})
    lines = found(docs(root), "gpu")
    assert lines == ["gpu: docs/troubleshooting/render.md:5: a table action that runs GPU work by hand "
                     "('restart the server'); report to the operator"]


def test_the_planned_rule_sentence_where_qa_toml_asks_for_it(tmp_path):
    rule = "A step marked planned is not built: report it once and stop at that stage."
    keys = f'planned_rule = "{rule}"\nplanned_rule_in = ["AGENTS.md", "docs/STUCK.md", "docs/runbooks/*.md"]\n'
    wrapped = rule.replace(" report", "\nreport")
    root = make_repo(tmp_path, {"AGENTS.md": f"# Agents\n{wrapped}\n", "docs/runbooks/README.md": "# Map\n"},
                     toml={"[docs]\n": "[docs]\n" + keys})
    lines = found(docs(root), "planned")
    assert lines == ["planned: docs/STUCK.md: missing; it must carry the planned-step rule",
                     f"planned: docs/runbooks/README.md: does not carry the planned-step rule: {rule!r}"]


def test_script_flags_are_checked_against_the_scripts_help(tmp_path):
    script = textwrap.dedent('''\
        import argparse

        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--check", action="store_true")
            parser.parse_args()
        ''')
    root = make_repo(tmp_path, {
        "tools/demo_tool.py": script,
        "docs/notes.md": "Run `python tools/demo_tool.py --check` (CI) or `python tools/demo_tool.py --nope`.\n",
    })
    lines = found(docs(root), "commands")
    assert len(lines) == 1 and "`python tools/demo_tool.py --nope` -- no such flag" in lines[0]


def test_paths_and_links_must_exist(tmp_path):
    text = """
    # Notes
    Read `docs/real.md`, [the stage map](runbooks/README.md) and [the top](../README.md#start).
    Not here: `docs/gone.md`, [a link](missing.md), [outside](../../elsewhere.md).
    Free: `docs/troubleshooting/<stage>.md`, `tests/fixtures/_gen/`, [web](https://example.org/x), [anchor](#notes).
    Also free: the sister repository's file, SISTER's `docs/only-there.md`.
    """.replace("SISTER", OTHER)
    root = make_repo(tmp_path, {
        "docs/notes.md": text,
        "docs/real.md": "real\n",
        "docs/runbooks/README.md": "map\n",
        "README.md": "top\n",
    })
    lines = found(docs(root), "paths")
    assert len(lines) == 3, lines
    assert any("names docs/gone.md" in line for line in lines)
    assert any("links to missing.md, which does not exist" in line for line in lines)
    assert any("outside the repository" in line for line in lines)


def test_a_direct_gpu_command_on_a_fenced_command_line(tmp_path):
    commands = ["easel run <rcp>", "easel run <rcp> --dry-run", "easel jobs submit <rcp> --count <n>",
                "kiln train launch <run-id>", "accelerate launch train.py"]
    text = runbook("05-demo", commands=[f"{PKG} next", *commands], output=["easel comfy start"])
    text += "\nProse may name `easel run <rcp>` and `easel comfy start` on purpose.\n"
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": text})
    lines = found(docs(root), "gpu")
    assert len(lines) == 3, lines
    assert all("a direct GPU command" in line for line in lines)


def test_no_absolute_path_anywhere(tmp_path):
    drive = "C" + ":" + "\\" + "work\\demo"
    posix = "/" + "home" + "/demo/models"
    root = make_repo(tmp_path, {"docs/notes.md": f"Put it in {drive}.\n\n{FENCE}\nls {posix}\n{FENCE}\n"})
    assert len(found(docs(root), "abspath")) == 2


def test_output_dates_and_the_clock_warning_in_the_output_scope_only(tmp_path):
    old = f"{2030 - 4}-11-05"
    warning = "clock " + "frozen (" + PKG.upper() + "_NOW set)"
    output = [f"at {old}T10:00:00+00:00", warning, "run-20300101-001 at 2030-01-01T00:00:00+00:00",
              f"recipe rcp-{2030 - 4}1105-001"]
    root = make_repo(tmp_path, {
        "docs/runbooks/05-demo.md": runbook("05-demo", output=output),
        "docs/troubleshooting/render.md": f"# Render\n\n{FENCE}\n{output[0]}\n{warning}\n{FENCE}\n",
    })
    lines = found(docs(root), "dates")
    assert len(lines) == 3, lines
    assert all("docs/runbooks/05-demo.md" in line for line in lines)
    assert any("frozen-clock warning" in line for line in lines)


def test_adopt_runbooks_use_placeholders_on_command_lines(tmp_path):
    commands = [f'{PKG} journal note "froze 0.3.0 from <old-version>"', f'{PKG} journal note "<version> done"',
                f'{PKG} journal note "ran host_tool.py"']
    text = runbook("90-adopt-existing", commands=commands)
    text += "\nThe literal `0.3.0` in prose is not a command line.\n"
    root = make_repo(tmp_path, {"docs/runbooks/90-adopt-existing.md": text})
    lines = found(docs(root), "adopt")
    assert len(lines) == 2, lines
    assert any("a concrete version (0.3.0)" in line for line in lines)
    assert any("a host script name" in line for line in lines)


def test_an_unclosed_fence_is_a_finding(tmp_path):
    root = make_repo(tmp_path, {"docs/notes.md": f"# Notes\n\n{FENCE}\n{PKG} next\n"})
    assert len(found(docs(root), "fences")) == 1


def test_the_help_cache_is_written_and_reused(tmp_path):
    root = make_repo(tmp_path, {"docs/runbooks/05-demo.md": runbook("05-demo")})
    cache = tmp_path / "cache" / "help.json"
    first = docs(root, cache_file=cache)
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["fingerprint"] and "families" in data["verbs"]
    second = docs(root, cache_file=cache)
    assert found(first) == found(second) == []


def test_the_cli_exit_codes(tmp_path, capsys):
    clean = make_repo(tmp_path / "a", {"docs/runbooks/05-demo.md": runbook("05-demo")})
    assert cli.main(["qa", "docs", "--root", str(clean), "--no-help-cache"]) == 0
    capsys.readouterr()
    dirty = make_repo(tmp_path / "b", {"docs/notes.md": "`docs/gone.md`\n"})
    assert cli.main(["qa", "docs", "--root", str(dirty), "--no-help-cache"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "paths: docs/notes.md:1: names docs/gone.md, which does not exist"
    assert out[-1] == "docs: 1 finding(s) in 1 file(s)"
    (dirty / "qa.toml").write_text('[docs]\ninclude = ["*.md"]\nno_such_key = 1\n', encoding="utf-8")
    assert cli.main(["qa", "docs", "--root", str(dirty)]) == 2
    (dirty / "qa.toml").unlink()
    assert cli.main(["qa", "tests", "--root", str(dirty)]) == 2
    assert "qa.toml is missing" in capsys.readouterr().err


def test_the_shipped_docs_pass_the_structure_checks():
    """Sections, entry tables, fences, GPU lines, absolute paths, output dates and adopt placeholders on the real
    docs. Commands, paths and pending markers depend on verbs and files still being built; CI runs `qa docs`."""
    report = docs(REPO, allow_pending=True)
    structural = {"fences", "sections", "entry", "gpu", "abspath", "dates", "adopt"}
    assert [str(f) for f in report.findings if f.check in structural] == []


# --- qa agents --------------------------------------------------------------------------------------------------

AGENT_PARTS = ["## 1. What this repo is", "## 2. Read order (cold start)", "## 3. Routing", "## 4. Never",
               "## 5. When stuck", "## 6. Record", "## 7. Hosts"]
NEVER = ("- Never describe an image; never a minor or a real person; GPU work is submitted; never without "
         "--dry-run; never open a window; never publish; run `date` and `--help`; never a third time; two rules "
         "that disagree: neither; never edit by hand; never delete.")


def agents_md(parts: list[str] = AGENT_PARTS, never: str = NEVER, filler: int = 0) -> str:
    lines = ["# AGENTS.md", ""]
    for part in parts:
        lines += [part, never if part == "## 4. Never" else "text", ""]
    return "\n".join(lines + ["filler"] * filler) + "\n"


def agent_repo(tmp_path: Path, agents: str, host: str = "# Host\nRead AGENTS.md first.\nThen the runbook.\n") -> Path:
    return make_repo(tmp_path, {"AGENTS.md": agents, "CLAUDE.md": host, ".github/copilot-instructions.md": host})


def test_agents_clean(tmp_path):
    root = agent_repo(tmp_path, agents_md())
    report = qa.check_agents(root, qa.load_rules(root))
    assert found(report) == []


def test_agents_too_long_missing_part_and_never_rule(tmp_path):
    text = agents_md(parts=AGENT_PARTS[:5] + AGENT_PARTS[6:], never=NEVER.replace("minor", "child"), filler=60)
    root = agent_repo(tmp_path, text)
    report = qa.check_agents(root, qa.load_rules(root))
    assert len(found(report, "length")) == 1
    assert any("'## 6. Record'" in line for line in found(report, "parts"))
    assert found(report, "never") == ["never: AGENTS.md: the part '## 4. Never' does not mention 'minor'"]


def test_agents_host_files_are_short_and_point_at_agents_md(tmp_path):
    root = agent_repo(tmp_path, agents_md(), host="# Host\none\ntwo\nthree\n")
    lines = found(qa.check_agents(root, qa.load_rules(root)), "hosts")
    assert len(lines) == 4  # two files, each too long and not pointing at AGENTS.md


def test_agents_lines_that_name_a_planned_command_say_so(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, "HelpCache", lambda root, cache_file=None, pkg=PKG: FakeHelps(pkg))
    monkeypatch.setattr(qa, "sister_cache", lambda sister, cache_file, root: FakeHelps(sister))
    never = "\n".join([NEVER, f"- Unmarked: `{PKG} later go`, then `{OTHER} demo audit`.",
                       f"- Marked: `{PKG} later go` (planned: session-2) and `{OTHER} later` # planned: session-2",
                       f"- Wrong task: `{PKG} demo audit` (planned: T9).",
                       f"- Built: `{PKG} demo make --name x`; prose may say a feature is planned: later."])
    root = agent_repo(tmp_path, agents_md(never=never))
    lines = found(qa.check_agents(root, qa.load_rules(root)), "planned")
    assert len(lines) == 2, lines
    assert any("names `" in line and "without `planned: " in line and f"`{PKG} later` (planned: session-2)" in line
               and f"`{OTHER} demo audit` (planned: T2b)" in line for line in lines)
    assert any("says planned: T9, but the CLI prints planned: T2b" in line for line in lines)


def test_the_shipped_agents_file_passes():
    assert found(qa.check_agents(REPO, qa.load_rules(REPO))) == []


# --- qa tests ---------------------------------------------------------------------------------------------------

def test_tests_every_file_has_a_test_and_names_are_unique(tmp_path):
    root = make_repo(tmp_path, {
        "tests/test_ok.py": "def test_one():\n    pass\n",
        "tests/a/test_twice.py": "class TestThing:\n    def test_it(self):\n        pass\n",
        "tests/b/test_twice.py": "import sys\nif sys:\n    def test_inside_if():\n        pass\n",
        "tests/test_empty.py": "def helper():\n    pass\n",
        "tests/test_broken.py": "def test_(:\n",
        "tests/a/__init__.py": "",
    })
    report = qa.check_tests(root, qa.load_rules(root))
    assert [f.file for f in report.findings if f.check == "has-test"] == ["tests/test_broken.py", "tests/test_empty.py"]
    assert sorted(f.file for f in report.findings if f.check == "names") == ["tests/a/test_twice.py",
                                                                              "tests/b/test_twice.py"]
    assert [f.file for f in report.findings if f.check == "layout"] == ["tests/a/__init__.py"]


def test_the_shipped_tests_pass():
    assert found(qa.check_tests(REPO, qa.load_rules(REPO))) == []
