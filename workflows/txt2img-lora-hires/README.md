# txt2img-lora-hires

The `txt2img` graph with one LoRA (on the model and on CLIP) and a latent hires pass: the first sampler's latent
is scaled by `hires.scale` and sampled again at `hires.denoise`. Core ComfyUI nodes only (no node pack).

Slots (`slots.json`): `checkpoint`, `lora` and `positive` are required; the LoRA strengths
(`lora.strength_model`, `lora.strength_clip`), the size, the sampler settings of both passes (`seed`, `steps`,
`cfg`, `sampler`, `scheduler`, and the same under `hires.`), `hires.scale`, `hires.denoise` and
`filename_prefix` have defaults (the `sdxl` family's priors). Pick the LoRA weight from the resource's usage card
(`easel standards explain <sha16>`), not from these defaults.

Check it with `easel workflows check txt2img-lora-hires`; on a running server `easel workflows validate
txt2img-lora-hires`.
