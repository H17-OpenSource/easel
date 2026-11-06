# Security policy (easel)

## Supported versions

easel is in early development (v0.1, not yet released). Only the latest commit on `main` receives fixes; there
are no maintained release branches yet.

## Reporting a vulnerability

Report it privately through GitHub: open this repository's **Security** tab and choose **Report a
vulnerability** (GitHub private vulnerability reporting). Only the maintainers can read the report. Do not open
a public issue, pull request or discussion about a vulnerability, and keep its details out of commit messages.
There is no e-mail address for reports: the private advisory and the issue tracker are the only channels.

A useful report names the affected command or file, the commit, the steps to reproduce on the demo data
(`examples/demo/`) or on synthetic values, what you expected and what happened. Never paste a real private value,
a token or an image into it; describe it by its shape instead ("a drive path inside a PNG text chunk").

## What counts as a vulnerability here

- A way for easel to leak a private value: a false negative of the privacy scanner or the pre-push gate (a path,
  a name, an e-mail address, a token or an id shape that passes), or metadata written into an image, a
  safetensors header or a record that should not be there.
- A way to make easel start GPU work that the scheduler did not grant, to get past a refusal (the `policy` keys,
  the adopted-instance refusals), or to write outside the instance folder.
- Unsafe handling of downloads: a checksum that is not verified, a path built from a remote name or an archive
  entry that escapes its folder, a token that ends up in a log or a record.
- A local server (the review page, when it ships) that listens beyond `127.0.0.1` by default or serves files
  from outside its folder.

Licence questions, content-policy questions and bugs without a security impact belong in the issue tracker.

## What happens next

A maintainer answers on the private advisory, confirms or declines the report, and agrees a fix and a
disclosure date with you there. The fix lands with a line in [CHANGELOG.md](CHANGELOG.md), and the advisory is
published once the fix is out, crediting you unless you ask not to be named. This is a volunteer project with no
fixed response time; the advisory always shows the current state.
