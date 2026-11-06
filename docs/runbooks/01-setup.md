# 01-setup: install, create the instance, point it at ComfyUI and the models

Stage map: [README.md](README.md). Rules for agents: [AGENTS.md](../../AGENTS.md).

## 1. Purpose and who decides

Create the instance folder (`easel.toml`, `registry/`, `runs/`, `recipes/`, `journal/`), point it at the
ComfyUI server and the models folder, and get `easel doctor` to ok/warn on every line.

- **The operator decides** where the instance lives, which ComfyUI server easel talks to, where the models
  are, and every policy key (`[policy]`). The shipped policy defaults are the safe ones; only the operator
  changes them.
- **You:** run the read-only checks, ask the operator once, set the values they gave you with
  `easel config set`, and report every `FAIL` and `warn` line of `doctor` with its `next:` line.

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel --version` | `easel <version>` |
| `easel next --stage` | `01-setup` |

Any version string matches `<version>`, for example `easel 0.1.0.dev0`. A step marked planned is not built:
report it once and stop at that stage.

## 3. Commands

Read-only first. `config set` writes the instance's `easel.toml` atomically and validates it.

```
easel doctor
easel config show --redact-paths
easel families check
easel comfy status
```

**No instance yet** (doctor prints `FAIL: config: no easel.toml found; searched: ...`): ask the operator
where the instance should live (section 5), then create it there and check it:

```
easel init <home>
easel doctor --workspace <home>
```

`init` writes `easel.toml` (from `easel.example.toml`), `registry/`, `runs/`, `recipes/`, `journal/`, a
README and a `.gitignore`. easel finds the instance when you work inside it, when `EASEL_HOME` names it, or
with `--workspace <home>`; the operator decides whether to set `EASEL_HOME`.

**After the operator's answer** (section 5), set what they gave you, then check again:

```
easel config set comfyui.url <the address the operator gave>
easel config set paths.models_root <models_root>
easel config set privacy.wordlist <the operator's private word list>
easel doctor
easel next
```

- `http://127.0.0.1:8188` is ComfyUI's upstream default. The server the operator uses may listen elsewhere;
  the tool's default port is not necessarily the server's. Never set an address the operator did not give.
- `paths.models_root` ships as `${MODELS_ROOT}`. Prefer `easel config set paths.models_root <models_root>`:
  the value stays in the instance file, while a variable set in one shell call is gone in the next.
  Absolute paths belong only in the instance's `easel.toml`.
- `easel comfy status` only reads. Starting a server is the operator's or the scheduler's step (or the host
  runner's on an adopted workspace); you never start, stop or restart one.
- Old variable names of the operator's own setup go into `[env_aliases]` of the instance file
  (`"<OLD_NAME>" = "<section.key>"`), never into this repo.

## 4. Expected output

`easel init <home>` on a new folder:

```
easel home: <home>
created: .gitignore, README.md, registry/, runs/, recipes/, journal/, easel.toml
easel finds it when you work inside it, when EASEL_HOME names it, or with --workspace <home>
next: easel doctor --workspace <home>
```

On the demo (`examples/demo/`, which has no ComfyUI server), `easel doctor` prints:

```
easel: 0.1.0.dev0 (python <python version>) ... ok
config: easel.toml (found by --workspace) ... ok
instance: native ... ok
models: <models_root> exists ... ok
registry: 6 resources ... ok
scheduler: none (this machine alone); free RAM <x> GB ... ok
disk: free <x> GB >= <y> GB floor ... ok
jobs: easel run allowed (jobs.allowed) ... ok
policy: subjects fictional-adults-only, prompt authoring sfw-only, reviewer human-only, render native ... ok
FAIL: comfyui: nothing answers at http://127.0.0.1:8188
  fix: start the server, or point comfyui.url at the one that runs; the tool's default port is not necessarily the server's; set comfyui.url (easel config set comfyui.url <url>)
next: start the server, or point comfyui.url at the one that runs; the tool's default port is not necessarily the server's; set comfyui.url (easel config set comfyui.url <url>)
```

With a server answering, the last lines read `comfyui: up at <url> (version <version>) ... ok`, a
`comfyui version:` line, and `next: easel next`. `easel config show comfyui.url --redact-paths`:

```
file: easel.toml (found by --workspace)
root: <root>
comfyui.url = "http://127.0.0.1:8188"  # file
```

`easel families check` (exit 0; a warning is not a failure):

```
warn: families/anima.yaml: licence_summary.url is null: no public model card recorded yet
warn: families/moss.yaml: licence_summary.url is null: no public model card recorded yet
warn: families/pony.yaml: licence_summary.url is null: no public model card recorded yet
ok: 15 family file(s) checked, 3 warning(s)
```

`easel config set comfyui.url <the address the operator gave>`:

```
set comfyui.url = "<the address>" in easel.toml (found by --workspace)
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. `easel doctor` prints no `FAIL` line and ends with `next: easel next`; every `warn` is reported to the
   operator.
2. `easel families check` exits 0.
3. `easel next --stage` prints `02-resources` or a later stage.

Ask once, after the read-only commands and before any `config set`, then wait:

> easel is installed. `doctor` says: `<the FAIL and warn lines>`. Where should the instance live (`<home>`,
> outside every repository checkout), which ComfyUI address should I use (ComfyUI's own default is
> `http://127.0.0.1:8188`, but your server may listen elsewhere), where are the models (`<models_root>`), and
> where is your private word list (or none)?

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| `FAIL: config: no easel.toml found; searched: ...` | the instance does not exist yet, or easel looks in another folder | ask the operator where it lives; `easel init <home>` for a new one, or `--workspace <home>` for an existing one |
| `refused: <dir> lies inside a repository checkout (...); nothing was written` | an instance holds private files and never lives in a checkout | ask the operator for a folder outside every checkout, for example `~/easel-home` |
| `refused: <dir> is already an easel home (it holds easel.toml); nothing was changed` | the instance exists | expected; use it with `--workspace <dir>` |
| `FAIL: ${MODELS_ROOT} unset (paths.models_root)` | the instance file names the models folder through a variable that is not set | `easel config set paths.models_root <models_root>` with the operator's folder |
| `FAIL: ${PRIVATE_WORDS_FILE} unset (privacy.wordlist)` | the private word list is named through a variable that is not set | ask the operator for the list's path (kept outside every repository); `easel config set privacy.wordlist <file>` |
| `refused: <key> = <value>; easel.toml is unchanged` | unknown key or wrong type for `schemas/config.schema.json` | check the key with `easel config show`; use the documented spelling |
| `FAIL: comfyui: nothing answers at <url>` | no server there, or it listens on another address | do not start one: report to the operator, ask for the address, then `easel config set comfyui.url <address>` |
| `comfyui: not running at <url> (the host runner starts it per job) ... ok` | adopted workspace, server stopped | expected; see [90-adopt-existing.md](90-adopt-existing.md) |
| `comfyui version: unknown until a job has run (lock says <min_version>) ... warn` | the server is not up, so its version is unknown | expected; the version is checked when the server answers |
| doctor warns that the server writes its outputs into its own install folder | a server you did not set up is running | tell the operator; do not render against it |
| `disk: free <x> GB < <y> GB floor (limits.disk_floor_gb); renders and downloads wait ... warn` | less free disk than the instance's floor | tell the operator; freeing space or changing the floor is their call |
| `scheduler: cannot measure free RAM; set limits.skip_ram_check = true to skip ... warn` | free RAM cannot be read on this system | the operator decides; never set the key yourself |
| `jobs: jobs.allowed lacks easel run: easel jobs submit refuses ... warn` | the instance allows no render jobs | the operator decides what the scheduler may start; report it |
| `easel families check` exits 1 | a shipped family file lacks `licence_summary` or `host_ram_gb` | report to the maintainer; never edit `families/` to pass |
| a value refuses an absolute path | absolute paths are allowed only in the instance file | put it in the instance's `easel.toml`, not in a shipped file |
| every write waits 30 s, then exits 4 and names no holder | the instance sits on a folder without file locks (a network share, some mounted drives) | report it; the operator moves the instance to a local disk |

Not in the table: follow [docs/STUCK.md](../STUCK.md); more ComfyUI symptoms in
[docs/troubleshooting/render.md](../troubleshooting/render.md).

## 7. Record

```
easel journal note "01-setup: instance at <home>, comfyui.url set as the operator said, doctor: <n> warn, 0 FAIL"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
