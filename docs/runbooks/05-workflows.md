# 05-workflows: check, register and approve workflow templates

Stage map: [README.md](README.md). `easel next` does not route here; a request does (the routing table of
[AGENTS.md](../../AGENTS.md)).

## 1. Purpose and who decides

A workflow template is a ComfyUI graph in API format with named slots, kept in a folder `workflows/<id>/`:
`graph.json` (the graph), `slots.json` (which node input each named value fills), `meta.yaml` (task, family,
status and every custom node pack the graph needs, with its licence) and, for shipped ones, a README. The
repository ships three (`txt2img`, `txt2img-lora-hires`, `detailer`); an instance keeps its own under
`<home>/workflows/<id>/`, and an instance template with a shipped id shadows the shipped one.

- **The operator decides** which graph becomes a template, whether a node pack may be installed (its licence
  binds whoever runs it), and whether a template is approved after a run with it.
- **You:** check, list, validate, register the graph the operator gave you, and report every problem line.
  You never install a node pack and never start a server to validate.

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel doctor` | `config: easel.toml (found by <how>) ... ok` |
| `easel workflows list` | `<n> templates` |

`validate` also needs a ComfyUI server that answers (`easel comfy status` prints `comfyui: up at <url> ...`);
without one it stops after the offline checks with exit 3. A step marked planned is not built: report it once and
stop at that stage.

## 3. Commands

Offline checks first; `validate` asks the server; `register` and `approve` write into the instance.

```
easel workflows check
easel workflows list
easel workflows check <id>
easel workflows validate <id>
easel workflows register <graph.json> --slots <slots.json> --task <task> --family <family> --id <id> --notes "<what it is for>"
easel workflows approve <id> --run <run-id> --note "<the operator's words>"
```

- `check` is offline: schema validity, the files the meta names, slots pointing at real nodes and inputs,
  links between nodes, no model file name left in the graph without a slot, and the node packs: a node class
  that `runtime/comfyui.lock.json` assigns to a pack needs that pack in the meta, with the lock's licence and
  commit and the packages it installs. Exit 1 on any problem; CI runs it on the shipped templates.
- `validate <id>` runs `check`, then asks the server's `/object_info` whether every node class and input of the
  graph exists there. On an instance template it records `validated_at`; a shipped one is only reported.
- `register` makes a new instance template with status `draft` from a graph and a slot map the operator gave
  you. Name each custom node pack with `--node-pack <name>=<licence>` (an SPDX id, or `unverified`).
- `approve` writes status `approved` (into the instance copy) only after a run that used the template ended
  ok. A recipe compiles its graph from the family (`inference.graph_compiler`) and does not select a template
  yet, so a run names a template only when it was made from one; until then `approve` refuses (section 6).
- A template whose node packs are missing on the server is unavailable: `validate` names the missing class,
  and the template is never rendered without it. The detailer needs ComfyUI-Impact-Pack (GPL-3.0) and
  ComfyUI-Impact-Subpack (AGPL-3.0, which installs ultralytics) inside the operator's ComfyUI.

## 4. Expected output

`easel workflows check` on the demo:

```
ok detailer | draft | detail | sdxl | shipped | ComfyUI-Impact-Pack (GPL-3.0), ComfyUI-Impact-Subpack (AGPL-3.0) (9 nodes, 19 slots)
ok txt2img | draft | txt2img | sdxl | shipped | core nodes only (7 nodes, 12 slots)
ok txt2img-lora-hires | draft | txt2img | sdxl | shipped | core nodes only (10 nodes, 22 slots)
workflows: 3 checked, 0 with problems
```

Registering a copy of the shipped `txt2img` graph and slots as `demo-portrait`:
`easel workflows register <graph.json> --slots <slots.json> --task txt2img --family sdxl --id demo-portrait`


```
ok demo-portrait | draft | txt2img | sdxl | instance | core nodes only (7 nodes, 12 slots)
registered demo-portrait as draft in workflows/demo-portrait
next: easel workflows validate demo-portrait
```

`easel workflows list` after it:

```
demo-portrait | draft | txt2img | sdxl | instance | core nodes only
detailer | draft | detail | sdxl | shipped | ComfyUI-Impact-Pack (GPL-3.0), ComfyUI-Impact-Subpack (AGPL-3.0)
txt2img | draft | txt2img | sdxl | shipped | core nodes only
txt2img-lora-hires | draft | txt2img | sdxl | shipped | core nodes only
4 templates
```

`easel workflows validate txt2img` on the demo (no server; exit 3) prints the check line on stdout and the
reason on stderr:

```
ok txt2img | draft | txt2img | sdxl | shipped | core nodes only (7 nodes, 12 slots)
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. `easel workflows check` prints `workflows: <n> checked, 0 with problems` and exits 0.
2. The template the operator asked about is registered, and `easel workflows validate <id>` printed
   `<id>: valid on the server (...)` while the server answered.
3. Approval: the operator said yes after an ok run with the template, in their own words.

Ask once, then wait:

> Template `<id>` (`<task>`, `<family>`) passes the offline check. Node packs: `<packs and licences, or "core
> nodes only">`. Server check: `<valid | not run: no server | the missing classes>`. Shall I keep it as a
> draft, or approve it after run `<run-id>`?

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| `node class <class> comes from <pack> (<licence>); meta.yaml must ...` | the graph uses a pack the meta does not name | shipped template: report to the maintainer; instance template: register again under a new `--id` with `--node-pack <name>=<licence>` as the operator says |
| `node pack <name> is not in runtime/comfyui.lock.json: its licence (...) ...` | a pack outside the pinned lock | report it; its licence is unverified until someone reads its LICENSE; the operator decides |
| `node pack <name>: licence unverified; read its LICENSE before relying on it` | the meta names the pack as `unverified` | report it to the operator; never fill in a licence yourself |
| `node <id> (<class>): the file name in <input> has no slot; ...` | a model file name is fixed in the graph | the template needs a slot for it; ask the operator for a slot map that names the input |
| `slot '<name>' points at node <n>, which the graph does not have` | the slot map and the graph do not belong together | report it; ask the operator for the matching files |
| `ComfyUI is not reachable at <url>: ...` then `<id>: the offline checks passed; the node check needs a running server (run validate again then)` (exit 3) | no server answers | expected while it is down; do not start one: report to the operator and validate when they say it runs |
| `server: ...` lines, then `<id>: <n> problem(s) on the server; the template is unavailable there until they are fixed` | the server lacks a node class or input the graph uses | report the classes; installing a node pack is the operator's step (at the lock's commit) |
| `workflows/<id> exists; a template is never replaced (pick another --id)` | an instance template has that id | choose another `--id`; never delete the old one |
| `'<id>' is a shipped template's id; pick another with --id` | register cannot reuse a shipped id | choose another `--id` |
| `<run-id> did not use the template '<id>' (its run.json says workflow None)` | recipes do not select a template yet, so no run names it | report it once and stop at this stage; the template stays `draft` |
| `<run-id> is '<status>', not ok: ...` | approval needs a finished, ok run | wait for an ok run with the template; ask the operator |

Not in the table: follow [docs/STUCK.md](../STUCK.md); more ComfyUI symptoms in
[docs/troubleshooting/render.md](../troubleshooting/render.md).

## 7. Record

```
easel journal note "05-workflows: <id> <checked|registered|validated|approved>; packs <packs or none>"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
