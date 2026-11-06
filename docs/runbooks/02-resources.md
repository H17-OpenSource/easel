# 02-resources: know which models are here, what they are and what they work with

Stage map: [README.md](README.md).

## 1. Purpose and who decides

Scan the models folder into the registry (`registry/resources/<sha16>.yaml` plus `INDEX.md`), so every recipe
can say which checkpoint, LoRA or embedding it uses, its licence, its rating and what it is compatible with.

- **The operator decides** ratings that conflict, favourites and rejections, notes on a resource, and which
  of their own LoRAs are self-trained.
- **You:** scan, list, explain, audit and report. You never invent a licence, a rating or a provenance;
  unknown stays unknown, and an unknown rating counts as nsfw (`policy.unknown_rating_is`).

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel next --stage` | `02-resources` |

A step marked planned is not built: report it once and stop at that stage.

## 3. Commands

A header-only scan first (fast, reads no weights), the hashing scan second (reads every file once).

```
easel registry scan
easel registry scan --hash
easel registry list --type lora --family <family>
easel registry list --self-trained
easel registry show <sha16>
easel registry explain <sha16>
easel registry compatible <sha16>
easel registry audit
easel registry dups
easel registry index
easel next
```

- `scan --hash` reads each file once to hash it (several GB per checkpoint; little RAM). Run it when the
  operator is fine with the disk time. A header-only scan lists new files as `found ...: not hashed yet`.
- More model folders: `paths.extra_model_roots` lists folders laid out like `<models_root>`, scanned after it.
  The first root that has a file name wins; the same name under a later root is reported `shadowed` and not
  registered (ComfyUI loads the earlier one). Downloads always land in `<models_root>`.
- `list` prints one line per resource: sha16, name, type, family, rating, status (`--json` for a tool).
- `show <sha16>` names the loader node per type; `explain <sha16>` says where each value came from;
  `compatible <sha16>` lists what works with it (`native`, `works`, `degraded`), from `data/compat.yaml`.
- A scan never infers self-trained: `registry/overrides.yaml` (a human-owned file) or an imported record
  says so, and `list --self-trained` shows the origin.
- Each resource carries a `licence{}` block (name, commercial use, credit, derivatives, relicensing,
  gated, source, checked date). `easel recipe propose --explain` reads it later.
- `registry audit` only reads. It prints, in this order: rating conflicts, unknown ratings, undocumented
  resources (no source link, sha256 or licence), unattributed LoRAs, binding licence terms
  (`commercial_use: none`, no derivatives), LoRAs without a category, resources without a family, resources
  with only family priors (no usage card), records whose file is gone, checkpoints whose family has no graph
  compiler, and pending downloads. Each group names who fixes it. It exits 0 with the report; `--check` exits
  1 on any finding.
- `registry dups` lists files that hold the same bytes; deleting one is the operator's call.
- A note on a resource is the operator's words: `easel note <sha16> "<their words>"`.

## 4. Expected output

`easel registry scan` on the demo (its six model files are registered already):

```
scanned 6 file(s) under <models_root>: 0 new, 0 changed, 6 unchanged, 0 missing, 0 not hashed, 0 skipped, 0 duplicate(s)
unknown licence: 0
rating conflicts: 0
wrote 6 record(s) under registry/resources/, registry/INDEX.md, registry/index.jsonl; 0 changelog line(s)
```

`easel registry list --self-trained`:

```
cf56db17414bcb55 | demo_robot-v1.safetensors | lora | illustrious | sfw | active | recorded: kiln demo_robot run-20300102-001
```

`easel registry compatible bbbabeef2c0d57fe` (the demo's Illustrious checkpoint):

```
checkpoint bbbabeef2c0d57fe demo-anime-xl.safetensors (family illustrious): what works with it
cf56db17414bcb55 | demo_robot-v1.safetensors | lora | illustrious -> illustrious | native | 1.00 | same family
b4d12aa2b3121a5e | demo-xl-vae.safetensors | vae | sdxl -> illustrious | works | 0.90 | shared VAE
60d12866cf6377b0 | demo-style-ink.safetensors | lora | sdxl -> illustrious | degraded | 0.60 | SDXL-base LoRAs lose strength on Illustrious
not listed: 1 untested, 0 broken (--all lists them)
```

`easel registry audit` (exit 0):

```
registry audit: 3 finding(s) over 6 resource(s) and 0 pending: no-usage 3
## no-usage (3): only family priors for weight, cfg, steps, sampler; fix: reviewed runs, collected references, or `easel standards set`
bbbabeef2c0d57fe | checkpoint | illustrious | demo-anime-xl.safetensors | no usage card yet (`easel standards build`)
85306657134cbde8 | checkpoint | sdxl | demo-base-xl.safetensors | no usage card yet (`easel standards build`)
60d12866cf6377b0 | lora | sdxl | demo-style-ink.safetensors | no usage card yet (`easel standards build`)
```

`easel registry dups`:

```
dups: no two files hold the same bytes among 6 file(s) under 1 model root(s)
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. `easel registry list` shows every model file of `<models_root>` (and of each extra root) once.
2. Every rating conflict and every unknown licence the scan and the audit printed has been reported to the
   operator.
3. `easel next --stage` prints `03-render` (or `90-adopt-existing` on an adopted instance).

Ask once, then wait:

> The registry holds `<n>` resources (`<c>` checkpoints, `<l>` LoRAs, `<e>` embeddings). Unknown licence:
> `<sha16 list>`. Rating conflicts: `<sha16 list>`. Which of the LoRAs are your own, and how should the
> conflicts be rated?

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| `found <file> (<type>): not hashed yet; ...` | a header-only scan saw a new file | expected; run `easel registry scan --hash` when the operator agrees to the disk time |
| `changed <file>: its size differs from record <id>; ...` | the file was replaced on disk | `easel registry scan --hash` hashes it again |
| `skipped <file>: not in a known folder of <folders file> and the header names no type; the operator moves it into the right folder` | the file sits outside the type folders and its header gives no type | report it; the operator moves the file; never move it yourself |
| `skipped <file>: an online-only placeholder of a cloud-sync client (not read, not hashed)` | the file is a placeholder of a sync client, not on local disk | ask the operator to make the models folder local; never move files |
| `missing <sha16> <path>: not found under <models_root> (the record stays; status missing)` | the file was moved or deleted | expected; the record stays; ask the operator where the file went |
| `duplicate <file>: the same bytes as <file> (...)` | the same bytes copied into two folders | one resource per sha256; `easel registry dups` lists them; deleting is the operator's call |
| `shadowed <root>/<file>: an earlier model root has the same name, which ComfyUI loads instead; not registered` | the same file name exists under two model roots | expected; tell the operator which copy ComfyUI loads |
| `skipped <root> (paths.extra_model_roots): not a folder` | an extra model root does not exist | report it; the operator fixes `paths.extra_model_roots` |
| `override <sha16> <key>: refused, the record would fail the schema (...)` (exit 1) | a value in `registry/overrides.yaml` has the wrong type or spelling | report the line; the operator fixes that value in `registry/overrides.yaml` |
| `override <sha16>: no such resource in the registry yet (registry/overrides.yaml); ...` | the override names a resource that is not scanned yet | expected; it applies once the file is scanned |
| `models folder not found: <models_root> (paths.models_root) does not exist` | the models folder is not where the instance file says | back to [01-setup.md](01-setup.md); ask the operator for the folder |
| `RATING CONFLICT <sha16> <file>: rated sfw (...) but carries the non-sfw cue <cue>; ...` | rated sfw, but its file name or a marker says otherwise | never use it as sfw; ask the operator to rate it |
| a resource shows `rating: unknown` | no source gave a rating | it counts as nsfw until the operator rates it |
| a LoRA is missing from `compatible` for a checkpoint | its family does not match, or the pair is untested | `easel registry explain <sha16>`; report the family pair; never force it |
| doctor says `registry: <n> resources, <m> record(s) not readable ... warn` | a record file was cut off or edited by hand | `easel registry list` names it; report the file; never repair a record by hand |
| a self-trained LoRA is not marked | a scan cannot know who trained it | the operator says which and records it in `registry/overrides.yaml`; then scan again |

Not in the table: follow [docs/STUCK.md](../STUCK.md).

## 7. Record

```
easel journal note "02-resources: <n> resources, <u> unknown licence, <r> rating conflicts reported"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
