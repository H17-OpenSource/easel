# 90-adopt-existing: render and review through an existing (adopted) workspace

An adopted workspace keeps its own render runner, review page and catalog until cutover. easel writes the
recipe, prints the host's own commands from the instance config and reads the host's results back; it never
starts a server or a render here. Every host fact (paths, the four command templates, the run registry) lives
in the instance's `easel.toml` (`[adopt]`, `[env_aliases]`), never in this repo. Stage map:
[README.md](README.md).

## 1. Purpose and who decides

Get images rendered by the host runner, reviewed by the operator on the host's own review page, and brought
into easel's run records with their verdicts, so kiln can import the kept images (kiln runbook 90).

- **State lives** in the instance folder (`<home>`): `recipes/`, `legacy/requests/`, `runs/<run-id>/` with
  `reviews.jsonl`. The host catalog is only read.
- **The operator decides** the request, the recipe's approval, every verdict (on the host's own review page)
  and when a render is done.
- **You:** propose the recipe, print and run the host lines in order, read the job state once per check-in,
  and import.

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel next --stage` | `90-adopt-existing` |
| `easel doctor` | `instance: adopted (policy.render = legacy-runner: the host runner renders) ... ok` |

On an adopted instance `next --stage` prints `90-adopt-existing` wherever a native instance would print
`03-render` or `04-review-and-learn`. The doctor line holds whether the host's server is up or down; with it
down doctor also prints `comfyui: not running at <url> (the host runner starts it per job) ... ok`, which is
not a failure. A step marked planned is not built: report it once and stop at that stage.

## 3. Commands

**The recipe.** `recipe propose`, `revise`, `show` and `approve` work on an adopted instance as on a native
one, and so does `easel run <rcp> --dry-run`; only the render itself goes through the host.

```
easel next
easel recipe propose --text "<the scene>" --family <family> --hires <none|latent|model> --explain
easel recipe approve <rcp>
easel recipe to-legacy <rcp>
easel recipe to-legacy <rcp> --host-recipe <host-recipe-id>
easel recipe to-legacy <rcp> --host-recipe <host-recipe-id> --job-id <job-id>
```

- Ask the operator before `approve` (the question of [03-render.md](03-render.md), section 5).
- **Rendering:** `to-legacy <rcp>` writes the host's request file `<home>/legacy/requests/<rcp>.json` and
  prints four lines from the `[adopt]` templates: the propose line, the dry-run line, the submit line and the
  status line. Run the first three in order. Until they are known, the host's recipe id prints as
  `<host-recipe-id>` (the propose line's output names it) and the job id as `<job-id>` (the submit line's
  output names it); run `to-legacy` again with `--host-recipe` and then `--job-id` to get the lines filled
  in. `<agent>` in a line is your own agent name.
- The submit line hands the render to the host runner, which waits for room, starts the server per job and
  stops it after. It is the only way a render starts here.
- **Status:** run the status line once. If it does not say `done`, report the state and stop; continue when
  the operator says the render is done. Reading the state once is not a retry.

**The review.** The operator reviews on the host's own review page. Then:

```
easel runs import-legacy <legacy-run-id> --with-reviews
easel runs reviews <run-id> --current
easel next
```

- `import-legacy` finds the host run folder under `adopt.output_root` (from `[adopt]`), copies each output
  into `runs/<run-id>/outputs/` (sha256-checked) and writes `run.json` with `legacy_id`. It prints
  `imported <legacy-run-id> as <run-id>: ...`, which names the `<run-id>` for the next lines. With
  `--with-reviews` it then reads that run's review rows from the database `adopt.run_registry` names
  (read-only) into `reviews.jsonl` and prints `<n> reviews imported`. Importing the same run again adds only
  what is missing.
- easel's own review page (planned: session-2) is not used on an adopted instance.
- `easel run` without `--dry-run`, `easel jobs submit` and `easel comfy start` refuse here (exit 2, pointing
  at this runbook).

## 4. Expected output

The demo made adopted for this page: `policy.render = "legacy-runner"`, `instance.adopted = true` and four
synthetic `[adopt]` templates: a `host-tool propose` line that reads `{recipe_json}`, a `host-tool run` dry-run
line, a `host-runner submit` line and a `host-runner status {job_id}` line. Your host's lines come from your
own `[adopt]` table. `easel next`:

```
stage: 90-adopt-existing
read: docs/runbooks/90-adopt-existing.md
next: easel recipe propose --text "<the scene>" --family <family> --hires <none|latent|model> --explain
```

`easel doctor` (server stopped):

```
easel: 0.1.0.dev0 (python <python version>) ... ok
config: easel.toml (found by --workspace) ... ok
instance: adopted (policy.render = legacy-runner: the host runner renders) ... ok
models: <models_root> exists ... ok
registry: 6 resources ... ok
scheduler: none (this machine alone); free RAM <x> GB ... ok
disk: free <x> GB >= <y> GB floor ... ok
jobs: GPU verbs refuse here; renders go through the host runner (runbook 90) ... ok
policy: subjects fictional-adults-only, prompt authoring sfw-only, reviewer human-only, render legacy-runner ... ok
adopt: the four host command templates are set ... ok
comfyui: not running at http://127.0.0.1:8188 (the host runner starts it per job) ... ok
comfyui version: unknown until a job has run (lock says 0.43.0) ... warn
next: easel next
```

`easel recipe approve <rcp>` after a propose (the demo's new recipe is rcp-20300101-002):

```
rcp-20300101-002 approved
next: easel recipe to-legacy rcp-20300101-002
```

`easel recipe to-legacy <rcp>` (stdout first; the `to-legacy:` notes go to stderr):

```
host-tool propose --from-json <home>/legacy/requests/rcp-20300101-002.json
host-tool run <host-recipe-id> --count 1 --dry-run
host-runner submit --by <agent> --title <host-recipe-id> --cmd "host-tool run <host-recipe-id> --count 1"
host-runner status <job-id>
```

```
to-legacy: wrote legacy/requests/rcp-20300101-002.json (the request the propose line reads)
to-legacy: run the lines in order (propose, dry-run, submit), then the status line once; never route around the host runner
to-legacy: <host-recipe-id> is the recipe id the propose line prints; run `easel recipe to-legacy rcp-20300101-002 --host-recipe <id>` to fill it in
to-legacy: <job-id> is the job id the submit line prints
to-legacy: <agent> is your own agent name
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. The host status line says `done` (or the operator said the render is done).
2. `import-legacy --with-reviews` printed `<n> reviews imported` with `<n>` above 0.
3. `easel runs reviews <run-id> --current` shows the operator's verdicts.

After the submit, once, then wait:

> The render for `<rcp>` is submitted to the host runner as `<job id>`; it runs when the machine has room.
> When it is done, please review the images on the host's own review page and tell me "done".

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| the submit line prints `REFUSED` | the command is not on the host runner's allowed list | stop; report the line; never change the allowed list or route around it |
| the host refuses a direct render command | GPU work goes only through the host runner's submit line | use the submit line; never route around the refusal |
| the status line says `failed` | a host command exited non-zero | paste the tail it shows, then [docs/STUCK.md](../STUCK.md) |
| the status line says `cancelled` | someone cancelled the job | ask the operator whether to submit again |
| the status line says the scheduler could not be reached | the host runner cannot get its grant | report it; do not resubmit |
| `to-legacy: <host-recipe-id> is the recipe id the propose line prints; ...` | the host's recipe id is not known yet | run the propose line, then `easel recipe to-legacy <rcp> --host-recipe <id>` |
| `<rcp> is 'proposed': only an approved recipe is rendered; ...` | the operator has not approved the recipe | ask; on their yes, `easel recipe approve <rcp>` |
| `[adopt] <key> is not set in <file>: ...` (exit 2) | the instance config lacks a command template | the operator fills `[adopt]` in the instance's `easel.toml` |
| `adopted workspace: run the lines easel recipe to-legacy <rcp> prints (runbook 90)` | a render verb was used on the adopted instance | use the printed host lines instead |
| `import-legacy: no manifest for <legacy-run-id> under <adopt.output_root>/...` | wrong id, or the run has not finished writing | check the id the submit or status line printed; ask |
| `import-legacy: the database adopt.run_registry names has no run_review table; ...` | `adopt.run_registry` names the wrong database | report it; the operator points `adopt.run_registry` at the host catalog (the database with the run and review tables) |
| `import-legacy: output <n> ... the bytes differ from the sha256 the host manifest lists; ...` | a host output changed after the run | report it; never copy files by hand |
| `0 reviews imported` | the operator has not reviewed the run yet | stop; ask the operator; import again when they say done |
| `<run-id> already holds <legacy-run-id>; nothing imported twice` | the run was imported before | expected; continue with `easel runs reviews <run-id> --current` |

Not in the table: follow [docs/STUCK.md](../STUCK.md).

## 7. Record

```
easel journal note "90-adopt-existing: <rcp> submitted as <job id>, <state>; imported <legacy-run-id> as <run-id> with <n> reviews"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
