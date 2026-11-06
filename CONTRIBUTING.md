# Contributing to easel

easel is an MIT-licensed tool for rendering images with ComfyUI from plain-file registries, written so that a
person or an AI agent can run it from the docs alone. Contributions are welcome from both. This guide says how
work is picked up, how a change is proposed and what it must pass before it is merged. Read it to the end once;
agents doing contributor work follow it exactly like people do.

## Start here

Do these five things, in this order, before you change a file:

1. Read [AGENTS.md](AGENTS.md), then [docs/runbooks/README.md](docs/runbooks/README.md), then this guide. The
   "Never" list in AGENTS.md binds every text an agent will read, so it binds what you write as well.
2. Install from a clone: `python -m venv .venv` and `.venv/Scripts/pip install -e ".[dev]"`
   (`.venv/bin/pip` on Linux and macOS). The first install needs network access; see the contributor notes.
3. Run the offline suite on the untouched checkout:
   `python -m pytest -q -m "not gpu and not machine and not comfy_live and not net"`. If it fails before you
   changed anything, open an issue with the output instead of working around it.
4. Install the pre-push gate (below, "Your private word list and the pre-push gate").
5. Find the issue your change belongs to, or open one with the template that fits.

## Contribution terms

- **Inbound = outbound.** Everything you contribute is licensed under this repository's MIT licence
  ([LICENSE](LICENSE)). There is no contributor licence agreement.
- **Only what you may give.** Contribute your own work, or third-party material whose licence allows it and is
  named in the pull request. Never paste code, prompts or text from a source whose licence you have not read.
- **Sign-off is optional.** `git commit -s` adds a `Signed-off-by:` line (the Developer Certificate of Origin);
  it is welcome and not required.
- **Nothing private, ever.** No personal names, account names, e-mail addresses, paths from your machine, host
  names or ports, model files, images or dataset files. The scanner and the pre-push gate look for these
  shapes, and a maintainer refuses a pull request that carries one, however small.
- **Who decides.** The maintainers (`.github/CODEOWNERS`) review and merge. In a running instance the operator,
  the person who owns it, decides every image verdict, every release and every licence question; no
  contribution may hand such a decision to a tool or an agent.
- **Conduct and security.** [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) (Contributor Covenant 2.1) applies
  everywhere the project lives. Report a vulnerability privately as [SECURITY.md](SECURITY.md) describes, never
  in a public issue. Neither has an e-mail contact: GitHub's private reporting and the issue tracker are the
  only channels.

## How work is tracked

The issue tracker is the agenda; there is no other list. Each issue states its acceptance criteria before the
work starts, written so that a command or a test can check them, and the pull request is judged against
exactly those criteria.

| label | used for |
|---|---|
| `bug` | a command that does something other than its help or runbook says |
| `pitfall` | a STUCK note from a session: a step failed or a runbook did not match (template generated from the one STUCK template) |
| `runbook-gap` | a runbook that is missing, unclear or wrong about a literal line |
| `model-family` | a new or corrected `families/<id>.yaml` |
| `workflow-template` | a new or changed `workflows/<id>/` template |
| `resource-knowledge` | how a public model behaves in practice: strengths, stacking, negatives |
| `good-first-issue` | small, well-specified, with the acceptance criteria already written |

- **Taking an issue.** Comment on it to take it; a maintainer then assigns you. An issue with an assignee is
  taken, even when it looks quiet: ask on it instead of starting in parallel. When you stop, say so on the issue
  so it can be assigned again.
- **Announce shared files first.** Before you change `src/easel/_common/` (shared byte for byte with kiln),
  `families/*.yaml` (shared keys), `schemas/`, `tests/conftest.py`, `pyproject.toml` or
  `.github/workflows/ci.yml`, say on the issue which of them you will touch and why. Others may be changing
  them at the same time.

## How to propose a change

1. Fork the repository and branch from `main`. One idea per branch and per pull request.
2. A complete pull request carries, together:
   - the code and its tests, with a sabotage check for every new guard (see "Tests");
   - the doc section that describes the behaviour (a runbook, `docs/REGISTRY.md`, `docs/reference/`);
   - a row in the runbook's failure table or in `docs/lessons/` when you learned something the hard way;
   - an ADR in `docs/decisions/` (`ADR-NNNN-<topic>.md`, one decision per file) when you chose between options
     that a later contributor will ask about;
   - one line in [CHANGELOG.md](CHANGELOG.md) under "Unreleased".
3. **A format change is a version bump.** A record format changes only as a new schema version
   (`easel/<name>/<n+1>`) plus a reader for the old version and a test of both. A published schema is never
   edited in place.
4. **Shared code starts in easel.** `_common` is changed in this repository only, then copied with
   `python tools/sync_common.py --to ../kiln`; open the twin pull request in kiln, and `--check` must print
   `identical` in both checkouts ([ADR-0001](docs/decisions/ADR-0001-vendored-common.md)).
5. Fill in the pull request template. A maintainer reviews it; it merges once CI is green and the review
   comments are answered.

## What a change has to meet

### The gates

Run them from the checkout before you open the pull request; CI runs the same set on Linux and Windows, plus a
gitleaks scan of the history ([ADR-0002](docs/decisions/ADR-0002-secret-scanning.md)).

```
python -m ruff check .
python -m pytest -q -m "not gpu and not machine and not comfy_live and not net"
python tools/privacy_check.py --public .
python tools/gen_formats_doc.py --check
python -m easel qa docs
python -m easel qa agents
python -m easel qa tests
python tools/sync_common.py --check
python -m easel families check
python -m easel workflows check
python -m piplicenses --from=mixed --ignore-packages easel kiln pip setuptools --allow-only "<the list in ci.yml>"
```

- **The licence gate** needs `pip-licenses>=5.0` (part of `[dev]`; older versions cannot read the
  `license = "MIT"` expression and report `UNKNOWN`). Copy the `--allow-only` value from the "Licence gate" step
  of `.github/workflows/ci.yml`. A new dependency whose licence string is not on that list fails until you add
  the exact string there, in the same pull request, with one sentence saying you read the licence. GPL and AGPL
  packages never go on the list: such programs stay outside the package and are driven over HTTP or as a
  subprocess (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). `tests/test_licence_gate.py` proves the gate
  refuses a GPL-classified package; it was run on 2026-11-06 and refused it with exit 1.
- **Notices** are regenerated with `python tools/gen_notices.py` whenever a dependency changes.
- **Planned steps.** A step marked planned is not built: report it once and stop at that stage. For a
  contributor that means: a verb that is not built has a `COMMANDS` row in `src/easel/cli.py` with its task and
  prints `planned: <task>`; a subcommand of a built verb that the docs name but that is not built yet is
  registered as a placeholder with `set_defaults(planned="<task>")`; a fenced doc line that names either ends
  with `# planned: <task>`. The change that builds the step drops the marker and the stub's test assertion.

### Tests

- Every new behaviour has a test. Every new guard (a refusal, a gate, a lock, a validator, a privacy pattern)
  also gets a sabotage check: break the guard on purpose, run the test and watch it fail, then restore the
  guard. List each `mutation -> failing test` in the pull request.
- Tests run offline. Anything that needs a GPU, a configured machine, a live ComfyUI or the network carries the
  marker `gpu`, `machine`, `comfy_live` or `net`; CI deselects those.
- Fixture images are generated at test time into the ignored `tests/fixtures/_gen/`; nothing binary is ever
  committed. Docstrings and fixtures use the demo subject (`demo_robot`) and 2030 dates.
- `tests/` has no `__init__.py`, so every test file name must be unique across all its subfolders.

### Standing rules

- Never loosen a gate, a pattern, an allowlist or a test to make something pass. Fix the cause, or open an issue
  that says why the rule is wrong.
- Only a human judges images. No code path turns an assistant's reading of an image into a verdict.
- Subjects are fictional adults only: no recipe, template, example or test prompt that makes a subject read as a
  minor or depicts a real person.
- GPU work is submitted through the scheduler interface; no command starts a render or a training run that the
  scheduler did not grant.
- No absolute path, host-specific port or e-mail address in a shipped file. The one address allowed repo-wide is
  ComfyUI's public default, `127.0.0.1:8188`.
- Records are written through the store (atomic writes, OS locks), never by hand. Generated files
  (`docs/STUCK.md`, `docs/reference/formats.md`, `.github/ISSUE_TEMPLATE/pitfall.yml`, the tables in
  THIRD_PARTY_NOTICES.md, every `INDEX.md`) are regenerated, never edited.

### Where to contribute, and the validator for each

| you add | where | checked by |
|---|---|---|
| a model family | `families/<id>.yaml`, with its own `licence_summary` (and source URL) and `host_ram_gb` | `easel families check --strict` (a missing card URL is a warning without `--strict`); the keys shared with kiln must agree (`sync_common.py --check`) |
| a workflow template | `workflows/<id>/` with `graph.json`, `slots.json`, `meta.yaml`, `README.md`; `meta.yaml` names every node pack and its licence | `easel workflows check` ([runbook 05](docs/runbooks/05-workflows.md)) |
| resource knowledge | a dated row in `docs/WHEN-TO-APPLY.md` or a lessons entry: what a public model does, with the evidence | review against the cited evidence |
| a scheduler adapter | an implementation of the scheduler interface in `_common` (through easel, see above) | its tests, including the fail-closed path |
| a runbook fix | the runbook's failure table, dated | `easel qa docs` |

### Doc rules

- A runbook has seven sections, in this order and with these exact headings, which `easel qa docs` checks:
  `## 1. Purpose and who decides`, `## 2. Entry conditions`, `## 3. Commands`, `## 4. Expected output`,
  `## 5. Gate and operator question`, `## 6. Symptom, cause, action`, `## 7. Record`.
- Every command in a doc must run as written; `qa docs` checks commands against `--help`, and a fenced command
  line must carry every option the verb requires. Write variable parts as `<id>`, `<n>`, `<version>`, and paths
  as `<home>`, `<models_root>` or `~/easel-home`, never as a real path. Commands of kiln in these docs are
  checked against kiln's CLI when it is installed beside easel.
- Every entry condition is a command plus a literal line that a built command prints (`easel next --stage`, a
  doctor line); never guess the output of an unbuilt verb. A runbook that a request can reach has a request row.
- Describe easel in its own terms: no history of other tools, no export mapping notes, no description of the
  machine it was developed on. An external scheduler is "the external scheduler configured in `[scheduler]`";
  host facts of an adopted workspace live in the instance's `[adopt]` table only.
- An agent-facing failure row never tells an agent to start, stop, restart or pause a server or a GPU process;
  its action is "report to the operator", or `easel jobs submit` (the host runner's line on an adopted
  workspace).
- Expected-output blocks come from the demo under the frozen clock (`EASEL_NOW=2030-01-01T00:00:00+00:00`) with
  `--redact-paths`, stdout only. Their dates are 2030 or later; the clock warning never appears in them.
- A dated fact says its date. Troubleshooting pages, lessons and the changelog carry real dates; runbook output
  blocks never do.
- The maintainers' private sweep: before a release, a maintainer scans the whole tree and its history with their
  own private word list. Your gate does the same for every push you make.

### The rules file of `qa`

`qa.toml` at the repository root holds what `easel qa docs|agents|tests` checks in this repository (the logic is
`src/easel/qa.py`, the same file in both repositories). Change a rule in the same commit as the docs or code it
describes, and never loosen one to make a check pass.

| key | what it sets |
|---|---|
| `[docs] tool`, `sister` | this tool's name (its commands are checked against its `--help`) and the sister tool's (a path right after "kiln's" names a file of the sister repository) |
| `include`, `exclude` | the Markdown files `qa docs` reads |
| `runbooks`, `adopt_runbooks` | which files are runbooks (seven sections, entry table) and which are adopt runbooks (placeholders only on command lines) |
| `sections`, `entry_section`, `entry_header`, `entry_tools` | the seven headings, the entry-condition section, its table header, and the tools an entry command may start with |
| `output_sections`, `output_scope`, `min_year`, `clock_warning` | which fenced blocks are output, where dates must be from `min_year` on, and the warning text that must never appear there |
| `pending_marker` | the marker of an output block not filled yet (only `--allow-pending` tolerates it) |
| `path_roots`, `path_ignore` | the folders whose paths must exist when a doc names them, and the paths that are exempt |
| `planned_rule`, `planned_rule_in` | the planned-step sentence and the files that must carry it word for word |
| `[[docs.gpu]]` | direct GPU commands a fenced command line must never show (`command` or `regex`, `unless`, `why`) |
| `[[docs.adopt_concrete]]` | concrete values (versions, record ids, host scripts, host and port) refused on adopt command lines |
| `[agents]` | the AGENTS.md line limit, its parts, the words its "Never" part must carry, and the host files |
| `[tests]` | the test folders and the file pattern of `qa tests` |

### Your private word list and the pre-push gate

The shipped patterns catch shapes (paths, e-mail addresses, id formats); only you know your own names, folders
and ports. Keep them in a private list and let the gate check every push:

1. Copy `data/privacy/private-words.example.txt` outside every repository and outside any folder that is synced
   or shared, and fill in one value per line under each heading.
2. Run `python tools/install-hooks.py --words <your list> --identity "<your name> <your public e-mail>"`.
   Use the address your public commits should carry, for example your GitHub noreply address; the gate refuses
   commits authored under any other identity, so a commit made with a private address cannot slip out.
3. The hook runs `tools/pre-push-gate` on every push: added lines, file names, commit messages and identities
   against both lists, plus gitleaks when it is installed. It prints counts, never the matched text, and it
   refuses the push when your list cannot be read. Rewrite the offending commit, then push again.
4. Commit in UTC: a commit records the committer's local UTC offset, which tells where they live and when they
   work, and the gate refuses an offset other than `+0000`. Set `TZ=UTC` (or `GIT_AUTHOR_DATE` and
   `GIT_COMMITTER_DATE` ending in `+0000`) for your commits in this repository.

### Hand-offs

A session that stops before its pull request is finished ends with one message in the shape of
[docs/HANDOFF-TEMPLATE.md](docs/HANDOFF-TEMPLATE.md), posted on the issue: what was done, what is open, what the
next person runs first.

### Contributor notes: pitfalls already paid for

The failure table for contributors is
[docs/troubleshooting/contributing.md](docs/troubleshooting/contributing.md): tests and fixtures, privacy
needles, editing files safely, schemas and records, planned verbs, network clients, docs, the licence gate and
the hook. Read it once before your first change; add a row there when you pay for a new one. The three that
cost the most:

- The privacy scan reads `tests/` too, with no allowlist there. A test that needs a value which looks private
  (a path, an e-mail address, a placeholder) builds it at run time from fragments, so the literal never sits in
  the file.
- Change lines that hold backslash escapes with an editor, never through a shell heredoc or inline script,
  which can silently collapse or expand them.
- When you build a verb, register every subcommand the docs name as a planned placeholder in the same change,
  so an agent gets `planned: <task>` instead of an argparse error.

## Questions

Ask on the issue you are working on, or open a new one with the closest template. Security reports go through
the private route in [SECURITY.md](SECURITY.md).
