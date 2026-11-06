# 03-render: from a request to an approved recipe to a submitted render

Native instance only. On an adopted workspace the render verbs refuse; read
[90-adopt-existing.md](90-adopt-existing.md). Stage map: [README.md](README.md).

## 1. Purpose and who decides

Turn a request into a recipe (everything needed to make the image again: checkpoint, LoRAs and weights,
prompt, negative, sampler, steps, CFG, clip skip, size, seed, hires mode, and why each was chosen), get it
approved, check its graph, and submit the render as a GPU job.

- **The operator decides** the request, the approval of the recipe, and anything beyond the policy.
- **You:** propose, explain, check with `--dry-run`, submit, and report the job state. Under
  `policy.prompt_authoring = "sfw-only"` you may write the scene for an sfw request; otherwise the prompt
  is the operator's words, verbatim. Never a prompt that makes a subject read as a minor or depicts a real
  person (`policy.subjects`); `recipe propose` refuses such tokens.
- A render without `--dry-run` is never typed by you: it is submitted (`easel jobs submit`).

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel next --stage` | `03-render` |
| `easel doctor` | `instance: native ... ok` |

Sent here by `easel next`: both rows hold. Sent here by a render request (the routing table of AGENTS.md):
the second row must hold, and `next --stage` may print another stage; if it prints `01-setup` or
`02-resources`, do that stage first. `instance: adopted ...` instead: read
[90-adopt-existing.md](90-adopt-existing.md). A step marked planned is not built: report it once and stop at
that stage.

## 3. Commands

Explain first, the graph check (`--dry-run`) before any submit.

```
easel registry list --type checkpoint --family <family>
easel recipe propose --text "<the scene>" --family <family> --hires <none|latent|model> --explain
easel recipe show <rcp> --explain
easel recipe approve <rcp>
easel run <rcp> --dry-run
easel jobs submit <rcp> --count <n>
easel jobs status <job-id>
easel runs show <run-id>
```

- `propose --explain` names every choice and its source (manual, experience, author, references, prior)
  and the licence terms of each resource, such as `commercial_use: none`. `--hires` is required: `none`,
  `latent` or `model` ([WHEN-TO-APPLY.md](../WHEN-TO-APPLY.md), section Hires). A subject LoRA is suggested,
  never added: the operator names it, then `--lora <sha16|file>`.
- A change the operator asks for is a revision: `easel recipe revise <rcp> --set <key>=<value>` keeps the
  parent's checkpoint, LoRAs, seed and hires mode and changes only what you name, so the next render compares
  one change. It writes a new recipe id; show it to the operator again before `approve`.
- `recipe approve` only after the operator's yes for that recipe.
- `run <rcp> --dry-run` compiles the ComfyUI graph and prints it; it renders nothing and needs no GPU.
- `jobs submit` checks the recipe, writes `jobs/<job-id>.json` and returns at once. It starts the job runner
  hidden when none runs; the runner waits until the machine has room (`families/<family>.yaml` `host_ram_gb`,
  the disk floor, or the external scheduler of `[scheduler]`), renders, and exits when the queue is empty.
  You never type the runner. `jobs status` once per check-in shows the state; `easel jobs cancel <job-id>`
  cancels a job that has not started.
- The run is recorded when it starts: `runs/<run-id>/{run.json, graph.json, manifest.json, outputs/}`.

## 4. Expected output

On the demo, with the scene "demo robot reading a manual at a workbench":
`easel recipe propose --text "<the scene>" --family illustrious --hires none --explain`



```
rcp-20300101-002 proposed: illustrious, txt2img, content sfw, prompt authoring sfw-only
  checkpoint  demo-anime-xl.safetensors (bbbabeef2c0d57fe)
  sampler     euler_ancestral / normal, 28 steps, CFG 5.5, clip skip 2
  size        1024x1024, count 1, seed <seed>, hires none
  prompt      masterpiece, best quality, amazing quality, demo robot reading a manual at a workbench
  negative    worst quality, low quality, lowres, bad anatomy, bad hands, extra digits, signature, watermark, jpeg artifacts
why:
  checkpoint  best match of 1: score 3 (matches demo)
  sampler     prior: families/illustrious.yaml
  ...
  hires       none: requested
  suggested LoRAs, not added (name one with --lora): demo_robot-v1.safetensors (cf56db17414bcb55) (native; matches demo, robot)
licence:
  demo-anime-xl.safetensors (bbbabeef2c0d57fe): unverified - Demo Open Licence; commercial_use: images; credit required yes; derivatives yes
  WARNING     illustrious reads comma-separated tags best; the request text is prose
next: show the operator this recipe; on their yes: easel recipe approve rcp-20300101-002
```

`easel recipe approve rcp-20300101-002`:

```
rcp-20300101-002 approved
next: easel run rcp-20300101-002 --dry-run, then easel jobs submit rcp-20300101-002 --count 1
```

`easel run rcp-20300101-002 --dry-run` prints the graph as JSON, then one summary line:

```
{
  "1": {
    "class_type": "CheckpointLoaderSimple",
    ...
}
(dry run: 8 nodes, 1 image(s), seed <seed>, hires none; nothing queued, no server asked)
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. The recipe is approved by the operator, and `easel run <rcp> --dry-run` exits 0.
2. The job reached `done`, and `easel runs show <run-id>` lists the outputs.
3. Then the operator reviews them ([04-review-and-learn.md](04-review-and-learn.md)).

Ask before `approve`, once, then wait:

> Recipe `<rcp>`: `<checkpoint>` + `<loras and weights>`, `<size>`, `<steps>` steps, CFG `<cfg>`, seed
> `<seed>`, hires `<mode>`. Licence notes: `<notes or "none">`. Reply "approve" to render `<n>` images, or
> tell me what to change.

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| `adopted workspace: run the lines easel recipe to-legacy <rcp> prints (runbook 90)` (exit 2) | `policy.render = legacy-runner` | follow [90-adopt-existing.md](90-adopt-existing.md) |
| `propose` exits 2 and names the subjects policy | an age-down or real-person token in the request | never reword around it; tell the operator the request cannot be rendered |
| `hires.mode is required: pass --hires none\|latent\|model (...); nothing was written` | every recipe states its hires mode | ask the operator which mode, then propose again with `--hires` |
| `<rcp> is proposed: it renders once the operator approved it` | the recipe is not approved yet | ask the operator; on their yes, `easel recipe approve <rcp>` |
| `cannot run:` then `<file> is not on disk under <models_root>` | a model file moved or is gone | `easel registry scan --hash`, then report; or revise the recipe as the operator says |
| `REFUSED: easel run is not in jobs.allowed of this instance (...)` | the instance allows no render jobs | the operator decides; never edit the list yourself |
| `waiting: free <x> GB < <y> GB needed by family <family>` | the machine lacks free RAM for this family (`families/<family>.yaml` `host_ram_gb`) | expected; let it wait; tell the operator if it waits long; never free memory yourself |
| `family <f>: host_ram_gb.eval is not measured; waiting for limits.host_ram_floor_gb = <n> GB instead` | the family file has no measured RAM for this step | expected; the job waits for the floor; a maintainer can measure and fill the family file |
| `waiting: free disk <x> GB < <y> GB floor (limits.disk_floor_gb)` | less free disk than the instance's floor | tell the operator; freeing space is their call |
| `cannot measure free RAM; set limits.skip_ram_check = true to skip` | the RAM reading failed; nothing starts | the operator decides; never set the key yourself |
| `cannot measure free disk space; nothing starts (...)` | the disk reading failed | report it to the operator |
| `scheduler unreachable: <verb> ...` then `failing closed: nothing starts (scheduler.fail_open = false)` (exit 3) | the external scheduler of `[scheduler]` did not answer | report it; never set `scheduler.fail_open` yourself; do not resubmit |
| `<job-id> is running: a render is not stopped halfway` | `jobs cancel` on a running job | expected; wait for it to finish, or tell the operator |
| the job ends `failed` | a command exited non-zero | read `easel jobs status <job-id>` once, paste the tail of `jobs/<job-id>.log`, then [docs/STUCK.md](../STUCK.md) |
| the job log says the GPU ran out of memory | the graph did not fit the GPU | report it; with the operator's yes, revise to a smaller size, `--count 1` or no hires pass |
| ComfyUI rejects the sampler name | an A1111 name (`euler_a`) instead of ComfyUI's (`euler_ancestral`) | revise the sampler to ComfyUI's spelling |
| model names not found by the server | the server was started before the models were added | report to the operator; restarting the server is their call or the scheduler's |
| a rerun looks different | a setting changed, or a node pack drifted | `easel runs compare <old-run> <new-run>`: settings first, then pixels |

More render symptoms: [docs/troubleshooting/render.md](../troubleshooting/render.md).
Not in the table: follow [docs/STUCK.md](../STUCK.md).

## 7. Record

```
easel journal note "03-render: <rcp> approved by the operator, job <job-id> <state>, run <run-id>"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
