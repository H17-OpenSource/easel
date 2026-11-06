# easel

easel is an agent-friendly, file-based workflow for rendering images with ComfyUI. It keeps a registry of the
models you have (what they are, their licences, what they work with), turns approved recipes into ComfyUI
graphs, records every run with the human verdicts on it, and learns usage standards from those verdicts.
Everything it knows lives as plain YAML, JSON and JSONL files in your own instance folder, outside this
repository, so a person or an AI agent can read, diff and back it up; GPU work goes through a scheduler, only a
human judges images, and subjects are fictional adults only. This is early work (v0.1 in development): a verb
that is not built yet prints `planned: <task>` and exits 2. Agents start at [AGENTS.md](AGENTS.md);
contributors read [CONTRIBUTING.md](CONTRIBUTING.md).

## Install

```
git clone https://github.com/H17-OpenSource/easel && cd easel
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"      # Linux/macOS: .venv/bin/pip
easel init ~/easel-home        # easel.toml registry/ runs/ recipes/ journal/; refuses a folder inside a checkout
easel doctor --workspace ~/easel-home                    # ok|warn|FAIL per line + "next: <command>"
easel config set comfyui.url http://127.0.0.1:8188 --workspace ~/easel-home   # your server's address
```

easel finds the instance when you work inside it, when `EASEL_HOME` names it, or with `--workspace`. The file
is changed only with `easel config set`. `http://127.0.0.1:8188` is ComfyUI's upstream default; use the address
your server listens on.

## Platforms

Developed and tested on Windows 11 with an 8 GB NVIDIA GPU. Linux runs the offline suite in CI; GPU rendering on
Linux is untested until a contributor reports. macOS: metadata and review only.

## Licence

MIT, see [LICENSE](LICENSE). ComfyUI and its node packs are separate programs under their own licences (GPL-3.0,
AGPL-3.0 for the detector pack), driven over HTTP only and never imported or vendored; the pinned versions and
licences are in [runtime/comfyui.lock.json](runtime/comfyui.lock.json). Not legal advice.
