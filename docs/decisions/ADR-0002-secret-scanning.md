# ADR-0002: Secret and private-value scanning at three points

Status: accepted, 2026-11-06. Applies to kiln and easel (this file is the same in both).

## Context
Two kinds of leak threaten a public repository built out of a private setup. Secrets (API tokens, keys,
passwords) have well-known shapes that general tools detect. Private values (personal and account names, local
folder names, host ports, ids of private records) have no general shape at all: only the person who owns them
can list them, and that list must itself never be published. Neither kind may reach a public remote, and a
check that cannot run must not count as a check that passed.

## Decision
Scan at three points, each failing closed:

1. **Before every push, on the contributor's machine:** `tools/pre-push-gate`, installed as the git `pre-push`
   hook by `tools/install-hooks.py`. It checks the added lines, file paths, commit and tag messages and the
   commit identities of exactly the commits being pushed, against the shipped public patterns, the allowlist
   and the contributor's private word list (kept outside the repository), and refuses binary files. Identities
   must match the `--identity` values given at install time. It also runs the gitleaks CLI over the same commits
   when gitleaks is installed; `--no-gitleaks` skips that with a warning and a line in the gate's log. An
   unreadable private list, a missing identity or a git failure refuses the push (exit 2).
2. **In CI, on every push and pull request:** the shipped scanner with the public patterns over the tree
   (`tools/privacy_check.py --public .`), and the **gitleaks CLI** over the whole history
   (`gitleaks detect --source . --redact --no-banner`). gitleaks is downloaded at a pinned release and verified
   against a sha256 recorded in `.github/workflows/ci.yml`; an empty or wrong checksum fails the job, so an
   unverified binary never runs. The version and the checksum change only in a reviewed pull request, taken
   from the release's own checksums file.
3. **Before a release, by a maintainer:** the scanner with the maintainer's own, freshly refreshed private list
   over the full tree and its history, followed by a person reading the top-level documents
   (`docs/lessons/privacy.md`, lesson 14).

All three print counts per category and place, never the matched text.

## Consequences
- A contributor without a private list cannot push through the installed hook; that is intended. A list with
  only the headings of `data/privacy/private-words.example.txt` is enough to start.
- CI never sees a private list, so it catches shapes only; the gate and the maintainers' sweep catch values.
- The secrets job fails until a maintainer records the gitleaks checksum: the workflow ships the pinned version
  with an empty `GITLEAKS_SHA256`, to be filled from the release's checksums file before the first push.
- gitleaks 8.19 and later prefer `gitleaks git` to `detect`, which remains as a deprecated alias in 8.x; moving
  the pin past 8.18 is the moment to switch the command, in the same pull request.

## Not chosen
- **The gitleaks GitHub Action:** for repositories owned by an organisation it requires a licence key; the
  CLI does the same scan without one.
- **Relying on GitHub's own secret scanning and push protection alone:** worth enabling, but it knows token
  formats only, never private words, and it runs after the push has left the machine.
- **Publishing the private list, or a hashed form of it, so CI could use it:** a list of the values is the leak;
  hashes of short words are reversible by trying candidates.
- **A server-side hook:** a hosted remote does not run custom hooks, and the check belongs before the data
  leaves the machine.

## When to revisit
When the scanner gains a pattern class that general secret scanners cover better, or when a contributor's
workflow cannot run a local hook (for example editing in a browser), in which case CI's public scan is the only
line and the release sweep must cover what the gate would have caught.
