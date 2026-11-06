# Runbooks: the stage table (easel)

One runbook per stage of the image workflow. `easel next` names the stage you are in and the runbook to read;
read only that runbook. Rules for agents: [AGENTS.md](../../AGENTS.md). When a step fails:
[STUCK.md](../STUCK.md).

## How to use the table

1. Run `easel next --stage`. It prints one line, the stage id, for example `03-render`.
2. Find the row whose literal line is exactly what it printed, and read only that runbook. A task that the
   routing table of [AGENTS.md](../../AGENTS.md) sends to a runbook (a render, a template, a search, a lesson)
   uses that runbook's request row in section 2 instead of `next --stage`; `next` never prints `05-workflows`,
   `06-search` or `07-lessons`. If `next --stage` prints `01-setup` or `02-resources`, do that stage first.
3. Check that runbook's section 2 (entry conditions) before its first command. Every entry condition there,
   as here, is a command plus the literal line it must print.
4. **A step marked planned is not built: report it once and stop at that stage.** A planned runbook has no
   link below; a planned verb is marked `[planned: <task>]` by `easel --help` and prints
   `planned: <task> (<help>)` with exit 2; a command line in a doc that names one ends with
   `# planned: <task>`. Never work around a planned step.

On an adopted instance (`instance.adopted = true` or `policy.render = "legacy-runner"`), rendering and review
go through the host workspace's own runner and review page, so `easel next --stage` prints
`90-adopt-existing` wherever a native instance would print `03-render` or `04-review-and-learn`.

## Stage table

| stage | runbook | entry condition: command | literal line | `next` routes here when | first command | state |
|---|---|---|---|---|---|---|
| 01 | [01-setup](01-setup.md) | `easel next --stage` | `01-setup` | no instance is found, a `${VAR}` is unset, the models folder is missing, or (native) no ComfyUI server answers | `easel doctor` | built |
| 02 | [02-resources](02-resources.md) | `easel next --stage` | `02-resources` | the registry has no resources yet | `easel registry scan` | built |
| 03 | [03-render](03-render.md) | `easel next --stage` | `03-render` | a job is open, an approved recipe has no run yet, or nothing is in flight; a render request uses the request row | `easel registry list --type checkpoint --family <family>` | built |
| 04 | [04-review-and-learn](04-review-and-learn.md) | `easel next --stage` | `04-review-and-learn` | a finished run has outputs without a current verdict | `easel runs list` | built; the review page is planned: session-2 |
| 05 | [05-workflows](05-workflows.md) | `easel doctor` | `config: easel.toml (found by <how>) ... ok` | never: a request routes here (a template to check, register or approve) | `easel workflows check` | built |
| 06 | [06-search](06-search.md) | `easel doctor` | `models: <models_root> exists ... ok` | never: a request routes here (a model the registry does not have) | `easel search civitai --query "<words>" --type <type>` | built |
| 07 | [07-lessons](07-lessons.md) | `easel doctor` | `config: easel.toml (found by <how>) ... ok` | never: a request routes here (a lesson the operator stated) | `easel lessons list` | built |
| 08 | 08-references | none yet | none yet | not routed | none yet | planned: session-2 |
| 90 | [90-adopt-existing](90-adopt-existing.md) | `easel next --stage` | `90-adopt-existing` | the instance is adopted and a render or a review is the next step | `easel next` | built |

Every stage also needs a working install: `easel --version` prints `easel <version>` (any version string
matches, for example `easel 0.1.0.dev0`), and `easel doctor` prints no line starting with `FAIL`. A `FAIL` line
comes first; fix it before the stage. On an adopted instance doctor prints
`instance: adopted (policy.render = legacy-runner: the host runner renders) ... ok` whether the host's server
is up or down; with the server stopped it also prints
`comfyui: not running at <url> (the host runner starts it per job) ... ok`, which is not a failure.

## The shape of a runbook

Each runbook has the same seven sections, in this order (`easel qa docs` checks them):

1. **Purpose and who decides.** What the stage produces; what the operator decides and what you do.
2. **Entry conditions.** Each one a command plus the literal line it prints (no `jq`, no parsing). A runbook
   that a request can reach has a request row as well.
3. **Commands.** In order, `--dry-run` or read-only checks first.
4. **Expected output.** What the commands print on the shipped demo instance (`examples/demo/`) under the
   frozen clock (2030 dates) with `--redact-paths`. Variable parts are written `<n>`, `<id>`, `<sha16>`. A
   block not filled yet carries the pending marker (`pending_marker` in `qa.toml`), which `easel qa docs`
   reports until the block is generated.
5. **Gate and operator question.** When the stage is done, and the one question to ask, as a template.
6. **Symptom, cause, action.** A table of known failures; anything else goes to [STUCK.md](../STUCK.md).
7. **Record.** The journal line to write, and where a new pitfall goes.

**Match the shape, not the numbers.** The expected output comes from the demo, so your counts, ids, hashes
and dates differ. Compare the lines that appear, their order and their words. A line missing, a different
line in its place, or an exit code other than the one the runbook names is a failure; another count is not.

## Where else to look

- What a record holds, field by field: [reference/formats.md](../reference/formats.md).
- Which question is answered by which command: [REGISTRY.md](../REGISTRY.md).
- When to load or apply a resource, and under which licence: [WHEN-TO-APPLY.md](../WHEN-TO-APPLY.md).
- What easel may and may not do on Civitai: [reference/civitai-rules.md](../reference/civitai-rules.md).
- How the journal works and how to hand off: [JOURNAL.md](../JOURNAL.md),
  [HANDOFF-TEMPLATE.md](../HANDOFF-TEMPLATE.md).
- Known failures by topic: `docs/troubleshooting/`.
