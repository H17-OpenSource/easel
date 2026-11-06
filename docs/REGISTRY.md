# Reading the registry (easel)

easel keeps what it knows as plain files in your instance (`easel config show` names the folder). You read
them through commands, never by opening and editing them: the commands apply the precedence rules, reduce
event logs to their latest state and keep the generated views in step. This page maps the questions you
will be asked to the command that answers them. A step marked planned is not built: report it once and stop
at that stage.

## Question -> command

| question | command |
|---|---|
| Which models do I have? | `easel registry list` (filters: `--type <type>`, `--family <f>`, `--json`) |
| What is this file, what loads it, under which licence? | `easel registry show <sha16>` |
| Why is it classed as this type, family or rating? | `easel registry explain <sha16>` |
| Which LoRAs work with this checkpoint, and how well? | `easel registry compatible <sha16>` |
| Which resources did I train myself? | `easel registry list --type lora --self-trained` |
| May I write prompt text for this resource? | `easel registry show <sha16>`: its `content.class` against `policy.prompt_authoring` |
| Which licence terms bind a resource, and which resources have none recorded? | `easel registry show <sha16>` (the `licence` block); `easel registry audit` |
| Which resources lack a source, a licence or a training record? | `easel registry audit` (`--check` exits 1 on any finding) |
| Do I have the same file twice? | `easel registry dups` |
| Why did a recipe pick this checkpoint, these LoRAs, these numbers? | `easel recipe show <rcp> --explain` |
| Which number wins for a resource (weight, CFG, steps), and where does it come from? | `easel standards explain <sha16>` |
| What has experience taught about each resource? | `easel standards build`, then `easel standards report` |
| What are the current verdicts on a run? | `easel runs reviews <run-id> --current` |
| Did a rerun reproduce the first run? | `easel runs compare <run-id> <run-id>` |
| Is a candidate from a search safe to add? | `easel search vet <url>` |
| Which templates exist, and are they approved? | `easel workflows list` |
| What has been learned, per family? | `easel lessons list --family <family>` |
| Which base do I train a LoRA on? | `kiln runs list --family <f> --status ok`, then `kiln train plan <version> --family <f> --base-from-run <run-id> --steps <n>` (kiln) |
| Continue the existing LoRA or start fresh? | the operator decides; kiln records it with `--continue-from <sha16\|path>` on `kiln train plan` |

Questions about datasets, training runs and checkpoints are kiln's: see kiln's `docs/REGISTRY.md`.
Reference images and the resources they used get their own report later (planned: session-2).

When the answer needs a judgement (is this image good, is this licence acceptable), the command gives the
facts and the operator decides. Never fill the gap yourself.

## Loader node per resource type

`easel registry show <sha16>` names the ComfyUI node that loads each type. All are ComfyUI core nodes.

| type | loader node | how it enters the graph |
|---|---|---|
| checkpoint | `CheckpointLoaderSimple` | gives the model, CLIP and VAE |
| lora | `LoraLoader` | chained after the checkpoint, one weight for the model and one for CLIP |
| vae | `VAELoader` | replaces the checkpoint's VAE; only within the family's `vae.shared_with` list |
| embedding | none: written in the prompt as `embedding:<name>` | positive ones in the prompt, negative ones in the negative prompt |
| upscaler | `UpscaleModelLoader`, then `ImageUpscaleWithModel` | a standalone upscale, or the model hires pass |
| controlnet | `ControlNetLoader` | planned for v0.2; v0.1 renders none |

## Rules the commands keep

- **Current verdict:** the latest verdict per output counts. Changing your mind replaces a verdict; it never
  adds a second one.
- **Experience:** from three current verdicts on, a resource's experience (keep rate, the weight and the
  settings of kept runs) outranks its author's advice. The full precedence is manual > experience > author
  > references > prior; `easel standards explain <sha16>` shows which one won.
- **Compatible:** a LoRA is offered for a checkpoint only when `data/compat.yaml` rates the pair of families
  as working; a degraded pair is shown as degraded, never as working. An instance's own
  `<home>/data/compat.yaml` wins over the shipped file.
- **Self-trained:** a scan never infers it. `registry/overrides.yaml` (a human-owned file) or an imported
  record that names its training run says so, and `list --self-trained` prints that origin.
- **Unknown rating:** a resource no source rated counts as nsfw (`policy.unknown_rating_is`).

## Where the files live (in the instance, never in this repo)

| what | file |
|---|---|
| one resource (identity, provenance, integrity, licence, semantics, usage, relations, lifecycle) | `registry/resources/<sha16>.yaml` |
| its usage text, filtered from the source | `registry/resources/<sha16>.usage.yaml` |
| a resource found by search, not downloaded yet | `registry/pending/civitai-<model>-<version>.yaml` or `registry/pending/huggingface-<owner>--<name>-<commit12>.yaml` |
| human-owned overrides, for example which resources are self-trained | `registry/overrides.yaml` |
| every registry change, and the notes on resources, recipes and templates, append-only (a run's notes live in its `run.json`) | `registry/changelog.jsonl` |
| a usage standard per resource | `standards/cards/<sha16>.yaml` |
| a recipe | `recipes/<rcp>.json` |
| a run, written when it starts | `runs/<run-id>/run.json`, `graph.json`, `manifest.json`, `outputs/<n>` |
| the verdicts on a run, append-only | `runs/<run-id>/reviews.jsonl` |
| a GPU job | `jobs/<job-id>.json`, its log `jobs/<job-id>.log` |
| search results | `candidates/<date>.jsonl` |
| lessons, append-only, and their generated pages | `lessons/entries.jsonl`, `lessons/<family>.md` |
| generated views (rebuilt by `easel registry index` and by every scan; do not edit) | `registry/index.jsonl`, `registry/INDEX.md` |

Three kinds of file: **records** (one YAML or JSON file per thing), **event logs** (append-only `*.jsonl`;
the latest line per key wins, a torn last line is skipped) and **generated views**. Human-owned and
tool-owned fields never share a file. Field by field: [reference/formats.md](reference/formats.md).
