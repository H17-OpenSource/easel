# When to load and apply a resource

Rules of thumb for turning the registry into a recipe: which resource to load, when to add a pass, and which
licence terms come with it. Each rule names its basis:
- **method**: how easel works, by design;
- **prior**: a common community default, not measured on your machine;
- a **date**: a fact recorded on that day.

A usage standard for the resource always wins over a prior on this page. The precedence is manual >
experience (three or more current verdicts) > author > references > prior, and
`easel standards explain <sha16>` shows which one won and why (method). `easel recipe propose --explain`
prints the reasons for every choice in a recipe.

## Checkpoint first (method)

The checkpoint's family decides the rest: graph compiler, sampler defaults, clip skip, prompt style,
baseline negative, hires defaults and which VAEs are interchangeable. All of it is in
`families/<family>.yaml` under `inference`; a child family inherits its parent's blocks. A LoRA is loaded
only when `easel registry compatible <sha16>` lists it for that checkpoint.

## LoRA stacking band

- Start each LoRA at its standard's weight; with no standard, at the family's
  `inference.lora_default_weight` (prior).
- Add one LoRA per revision, so a before/after compares one change (method).
- A subject LoRA (a character, a person-like identity) is never added automatically: it changes who is in the
  picture, so the operator names it (method).
- Band (prior): one subject LoRA plus at most two style or detail LoRAs, with the sum of all LoRA weights at
  or below about 1.5. Above that, colours burn and anatomy warps first. When two LoRAs fight, lower the style
  LoRA before the subject LoRA.

## Hires: `hires.mode`

Every recipe states `hires.mode` (method); `easel recipe propose --help` lists the accepted values.
- `none`: drafts, evaluation grids and any run that judges a LoRA itself, so outputs stay comparable and
  cheap.
- `latent`: upscales the latent and samples again at the family's `inference.hires.denoise`. Adds detail;
  small features may change.
- `model`: decodes, runs an upscale model, encodes and samples again at a low denoise. Keeps the composition
  closer to the first pass.
Scale, steps and denoise come from the family's `inference.hires` block (prior in every shipped family).

## Detailer

- The face detailer re-renders the face region after the first pass. Use it when faces are small in the
  frame (full-body and wide shots) and the operator wants them fixed; not on drafts (prior).
- Never use it on a render that judges a subject LoRA: it repaints what the LoRA learned and hides its faults
  (method).
- It needs ComfyUI-Impact-Pack (GPL-3.0) and ComfyUI-Impact-Subpack (AGPL-3.0, which installs ultralytics,
  AGPL-3.0) inside your ComfyUI; without them the detailer workflow is unavailable. easel drives them over
  HTTP only and never imports them. Their inputs are read from the server's `/object_info` at run time.

## ControlNet

v0.1 renders no ControlNet (templates are planned for v0.2). When they land: one ControlNet per recipe, so a
change in the result has one cause (method).

## Negatives per style

- Start from the family's `inference.negative_baseline`: tag-style families (`inference.prompt.style: tags`)
  take tags, natural-language families take short phrases (prior).
- Negative embeddings go into the negative prompt, positive embeddings into the prompt (method).
- Add a negative for what went wrong in a reviewed run, in the operator's words, one change per revision.
- **Never put adult-marking words such as `mature` or `adult` into a negative:** a negative pushes away from
  what it names, so the subject drifts younger. The subject must read as an adult (method; see
  `policy.subjects` in [AGENTS.md](../AGENTS.md)).

## Upscaler by content

- Choose a model upscaler trained on the same kind of image: photographic models for photographs,
  illustration models for line art and flat colour (prior). The resource's `semantics` block says what it
  was made for; if it says nothing, ask instead of guessing.
- Upscale kept outputs only, after the operator's verdict; an upscale is never the way to decide whether to
  keep an image (method).

## VAE

A separate VAE is loaded only within the family's `vae.shared_with` list and never across
`vae.not_interchangeable_with` (method).

## Licences per resource

Every resource record carries a `licence` block; `easel registry show <sha16>` prints it and
`easel registry audit` lists the resources whose block is missing or binding.

| field | meaning | filled from |
|---|---|---|
| `name`, `spdx`, `url` | the licence and where its text is | the model page or card |
| `commercial_use` | `none`, `images`, `rent` or `sell` | Civitai `allowCommercialUse` |
| `credit_required` | the author must be credited | Civitai `allowNoCredit` (inverted) |
| `derivatives_allowed` | merges and LoRAs trained on it are allowed | Civitai `allowDerivatives` |
| `relicense_allowed` | a derivative may carry a different licence | Civitai `allowDifferentLicense` |
| `redistributable`, `gated` | the file may be shared; a download needs an account or an agreement | Civitai `availability`, the Hugging Face card |
| `source`, `checked_at` | where the facts were read, and when | the tool that read them |

Rules:
- The resource's own block decides. A family's `licence_summary` is a fallback for foundation bases only,
  and every shipped family summary is `unverified` (as shipped on 2026-11-05): read the card at its `url`.
- A missing block, or `status: unknown`, is treated as the most restrictive case and said so; the operator
  decides. Never fill a licence field yourself.
- `commercial_use: none` is named by `easel recipe propose --explain` on every recipe that loads the
  resource.
- Training on a base whose `derivatives_allowed` is false, or whose licence is unknown, blocks the release
  of the LoRA in kiln.
- ComfyUI (GPL-3.0), ComfyUI-Impact-Pack (GPL-3.0) and ComfyUI-Impact-Subpack (AGPL-3.0, with ultralytics,
  AGPL-3.0): licences read from each project's LICENSE on 2026-11-05 and pinned in
  [runtime/comfyui.lock.json](../runtime/comfyui.lock.json). The AGPL-3.0 section 13 duties fall on whoever
  exposes that ComfyUI as a network service. Any other node pack is `unverified` until its LICENSE is read,
  and `easel doctor` says so.

Not legal advice.
