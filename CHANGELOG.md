# Changelog

Notable changes to easel, newest first. The format follows Keep a Changelog
(https://keepachangelog.com/en/1.1.0/), and versions will follow Semantic Versioning from v0.1.0 on. Every pull
request adds one line under "Unreleased"; the release commit moves those lines under the new version and its
date.

## Unreleased

### Added
- Packaging, the `easel` command table (a verb that is not built yet prints `planned: <task>` and exits 2), the
  MIT licence, the README and the rule file for AI agents (`AGENTS.md`).
- The shared core in `src/easel/_common/`: configuration with its discovery order and `config show|set`, the file
  store (atomic writes, OS locks, event logs, generated indexes), ids that honour the frozen clock, hashing, the
  journal with its one STUCK template, and the privacy scanner (`easel privacy scan`). It is vendored
  byte-identically into kiln (ADR-0001).
- JSON Schemas for every record type, with examples, and the synthetic demo in `examples/demo/`.
- Model family files with licence summaries and host RAM needs, the compatibility and folder data, and
  `easel families check`.
- The stage table, the first runbooks, `docs/STUCK.md`, `docs/REGISTRY.md`, `docs/WHEN-TO-APPLY.md`, the format
  reference and the render troubleshooting page.
- The contributor kit: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), `SECURITY.md`, issue
  and pull request templates, `CODEOWNERS`, ADR-0002 on secret scanning and `docs/lessons/privacy.md`.
- CI on Linux and Windows with every gate, including the pip-licenses `--allow-only` licence gate (proven by
  `tests/test_licence_gate.py`) and a gitleaks job pinned by version and checksum.
- `tools/pre-push-gate` with `tools/install-hooks.py`, and `tools/gen_notices.py` with `THIRD_PARTY_NOTICES.md`.
