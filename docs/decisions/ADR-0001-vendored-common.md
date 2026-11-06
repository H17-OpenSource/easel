# ADR-0001: Vendor the shared code as `_common` in both repositories

Status: accepted, 2026-11-05. Applies to kiln and easel (this file is the same in both).

## Context
kiln (LoRA creation) and easel (image generation) share about 2,500 lines: config, the file store and its locks,
ids, hashing, the header-only image probe, the embedded-metadata reader, the Civitai and Hugging Face clients,
the ComfyUI client and graph compiler, sysinfo, the hidden-process starter, the scheduler interface, the privacy
scanner, the journal and lineage. Each tool must install and test on its own with one `pip install -e`, and the
shared interfaces are young: they will change several times before they settle.

## Decision
- Two repositories, no third library for now.
- The shared code is vendored byte-identically: `src/easel/_common/` is the source of truth, `src/kiln/_common/`
  is a copy. Change it in easel only; never edit kiln's copy by hand.
- `tools/sync_common.py --to ../kiln` copies it. `tools/sync_common.py --check` (run in CI) compares
  `_common/MANIFEST.sha256` and the shared keys of `families/*.yaml` (both repos ship family files generated
  from one source: inference keys in easel, training keys in kiln, the shared keys identical).
- `comfy_graph.py` lives in `_common` only.
- `_common` modules import each other relatively and never import their host package, so the same bytes work
  under both package names.

## Consequences
- Each repo installs, tests and releases alone; there is no version matrix between them.
- A shared fix lands in two commits (easel, then the synced copy in kiln); `--check` fails CI when the copy
  drifts, so a forgotten sync cannot ship.
- Contributors to kiln send `_common` changes to easel first.

## When to revisit
At v0.3 or later, once the shared interfaces have stopped moving: move `_common` into a third package that both
tools depend on, recorded in a new ADR that supersedes this one.

## Not chosen now
- A third library today: an extra package to release and pin while its interface still changes weekly.
- One repository for both tools: couples two tools with different users and release rhythms.
- Git submodules or subtrees: harder for contributors and agents to follow than a plain copy whose checksum
  manifest CI verifies.
