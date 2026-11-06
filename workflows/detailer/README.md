# detailer

The `txt2img` graph followed by a face pass: a face detector finds faces in the decoded image and FaceDetailer
re-samples each one at `detailer.denoise`, then the result is saved.

**This template is unavailable without two custom node packs** in the ComfyUI server's `custom_nodes/`, at the
commits pinned in `runtime/comfyui.lock.json`:

| node pack | licence | provides | installs |
|---|---|---|---|
| ComfyUI-Impact-Pack | GPL-3.0 | `FaceDetailer` | |
| ComfyUI-Impact-Subpack | AGPL-3.0 | `UltralyticsDetectorProvider` | ultralytics (AGPL-3.0) into the server's environment |

Both are separate programs. easel drives them over ComfyUI's HTTP API only: it never imports, vendors or copies
their code, and neither easel nor kiln imports ultralytics. AGPL-3.0 section 13 duties fall on whoever exposes
that ComfyUI as a network service. Not legal advice. Without the packs, `easel workflows validate detailer`
names the missing node classes, and renders go without the face pass.

Slots (`slots.json`): `checkpoint`, `positive` and `detector` (a face detector file under the server's
ultralytics folder, for example `bbox/<file>.pt`) are required; the sampler settings of the main pass and of the
face pass (`detailer.seed`, `detailer.steps`, `detailer.cfg`, `detailer.sampler`, `detailer.scheduler`,
`detailer.denoise`) have defaults.

Check it with `easel workflows check detailer` (offline: slots, links, and the packs against the lock); on a
running server `easel workflows validate detailer`.
