# The easel demo instance

A complete, synthetic easel home dated January 2030: six fake model files in the registry (two checkpoints, two
LoRAs of which one is self-trained, a VAE, an upscaler), one recipe, one finished run with the person's
reviews, one saved reference, a lesson, two vetted search results and a job. The tests copy this folder,
generate its media into the copy and run the commands on it; `tests/test_schemas.py` validates every record.

- `registry/resources/<sha16>.yaml` (6), one `<sha16>.usage.yaml`, `registry/changelog.jsonl`,
  `registry/overrides.yaml` (marks the demo_robot LoRA self-trained), `standards/cards/<sha16>.yaml`;
- `families/demo_merge.yaml`: an instance-local family that inherits illustrious; `data/compat.yaml`: a few
  instance-local rules with a planted native pair (the demo_robot LoRA on the Illustrious checkpoint) and a
  planted degraded pair (the SDXL ink-style LoRA on the same checkpoint);
- `recipes/rcp-20300101-001.json`; `runs/run-20300101-001/` with `run.json`, `manifest.json`, `graph.json` and
  `reviews.jsonl` (outputs 1, 2 and 4 kept, 3 rejected; output 4 changed from maybe to keep);
- `references/civitai/9000000001/`, `lessons/entries.jsonl`, `candidates/2030-01-01.jsonl`,
  `jobs/job-20300101-001.json`.

Media: nothing binary is committed. `python tests/fixtures/make_fixtures.py --place <copy> --demo easel` writes
the six model files under `models/`, the four run outputs and the reference image into a copy, byte for byte
matching the sha256 values in the records.
