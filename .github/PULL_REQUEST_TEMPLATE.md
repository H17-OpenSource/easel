## What and why

<!-- The behaviour this changes and the reason, in two or three sentences. Link the issue: "Closes #N". -->

## Acceptance criteria

<!-- Copy them from the issue. For each one: the command and its last line, or the test that checks it. -->

## How it was checked

- [ ] The offline suite passes: `python -m pytest -q -m "not gpu and not machine and not comfy_live and not net"` (count: ___ passed)
- [ ] Each new guard was sabotage-checked: broken on purpose, its test failed, then restored. One line each:
  - mutation -> failing test
- [ ] Clean: `python -m ruff check .`, `python tools/privacy_check.py --public .`, `python -m easel qa docs`, `python tools/gen_formats_doc.py --check`, `python tools/sync_common.py --check`
- [ ] The licence gate passes. A new dependency's licence string was added to `ci.yml` after reading the licence, and `THIRD_PARTY_NOTICES.md` was regenerated (`python tools/gen_notices.py`), or: no new dependency
- [ ] The licence of anything new is named in this description (packages, node packs, models, data). No node pack is added without its licence in `meta.yaml` and `runtime/comfyui.lock.json`

## Docs and records

- [ ] The doc section that describes this behaviour (a runbook, `docs/REGISTRY.md`, `docs/reference/`)
- [ ] A dated failure-table or lessons row for anything learned the hard way, or: nothing learned the hard way
- [ ] An ADR for a decision between options (one decision per file), or: no decision
- [ ] One line in `CHANGELOG.md` under "Unreleased"
- [ ] A format change is a new schema version with a reader for the old one and a test of both, or: no format change
- [ ] `_common` was changed here first and the kiln twin pull request is linked, or: `_common` untouched

## Risk

<!-- What could this break: a gate, a refusal, the store or its locks, the scheduler path, privacy? Say how you tried to break it. -->

## Terms

- [ ] I may contribute this under the MIT licence (CONTRIBUTING.md, "Contribution terms").
- [ ] No private value is included: no names, account names, e-mail addresses, machine paths, host ports, tokens, model files or images. Fixtures use the demo subject and 2030 dates.
