# txt2img

A text-to-image graph built from core ComfyUI nodes only (no node pack): checkpoint, positive and negative
prompt, an empty latent, one sampler pass, decode, save. Made for SDXL-architecture checkpoints (the `sdxl`,
`illustrious`, `pony` and `noobai` families); the defaults in `slots.json` are the `sdxl` family's priors.

Slots (`slots.json`): `checkpoint` and `positive` are required; `negative`, `width`, `height`, `batch`, `seed`,
`steps`, `cfg`, `sampler`, `scheduler` and `filename_prefix` have defaults. The graph holds `<checkpoint>` and
`<prompt>` where a slot fills in the real value, so a template never names a model file.

Check it with `easel workflows check txt2img`; on a running server `easel workflows validate txt2img`.
