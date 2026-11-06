"""Shared code of easel and kiln, vendored byte-identically into both packages.

`src/easel/_common/` is the source of truth; `src/kiln/_common/` is a copy made by `tools/sync_common.py --to
../kiln`, and `tools/sync_common.py --check` compares both against `MANIFEST.sha256` (ADR-0001). Two rules keep
the same bytes working under both package names:
- modules here import each other relatively (`from . import store`) and never import their host package;
- the tool name ("easel" or "kiln") is always a parameter, never a constant, so environment variables
  (`EASEL_HOME`, `KILN_NOW`, ...) and messages follow the tool that is running.

Exit codes shared by every tool: 0 ok, 1 a check found problems, 2 usage error or refusal, 3 deferred by a guard
(also: a shipped file holds an absolute path), 4 a lock or file is held by another process.
"""
