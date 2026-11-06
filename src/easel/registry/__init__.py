"""The model registry: one YAML record per model file, `<home>/registry/resources/<sha16>.yaml`.

Plain files, so a person can read, diff and back up every record (docs/REGISTRY.md). The modules:
- `classify`: what a file is (type, family, precision, LoRA facts, rating cues) from its folder and header;
- `resources`: reading, overriding, validating and writing the records, the changelog and the generated index;
- `scan`: walking `<models_root>` and turning what it finds into records (`easel registry scan [--hash]`);
- `compat`: the family compatibility rules and the join `easel registry compatible` prints;
- `views`: the read-only questions (`list`, `--self-trained`, `show`, `explain`).
The verbs themselves are the thin argparse layer in `easel/commands/registry.py`.
"""
