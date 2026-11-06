# Troubleshooting: rendering with ComfyUI

Failures you can meet while easel renders through ComfyUI, with their cause and what to do. They were found on
Windows 11 with an 8 GB NVIDIA GPU; on other systems the messages are the same and the numbers differ. Find the
symptom, do the action once, record the result (AGENTS.md section 6). Anything else:
[docs/STUCK.md](../STUCK.md).

**Rule zero.** GPU work is submitted, never run by an agent: `easel jobs submit` on a native instance, the
host runner's submit line (`easel recipe to-legacy <rcp>`) on an adopted one. `easel run <rcp> --dry-run`
compiles the graph and renders nothing; it is always allowed. An agent never starts, stops, restarts or pauses a
server or any other GPU process: where a fix needs that, the action says "report to the operator".

**How much RAM a render waits for** is not a number on this page: it is `host_ram_gb.eval` in
`families/<family>.yaml`, with a `basis` that says whether it was measured, inherited or a prior. Plan with
that value, not with one reading of free RAM.

## The server

| symptom | cause | action |
|---|---|---|
| easel talks to the wrong server, or to none | `comfyui.url` still holds ComfyUI's upstream default `http://127.0.0.1:8188`, and the server listens elsewhere; one machine can run two servers | ask the operator for the address, then `easel config set comfyui.url <address>`; `easel doctor` prints the address it checked |
| a server started from an agent session is gone soon after the start command returned | on Windows, processes started detached from a session's job can end with that job | report to the operator. easel's own starter tries WMI first and a detached start only as a fallback; `easel comfy status` says `started via wmi` or `started via detached` |
| a node pack fails to import after an update | the server's packs no longer match the versions pinned in `runtime/comfyui.lock.json` | report to the operator: the packs go back to the lock's commits, and the server starts again on their side (an install verb is planned for v0.2) |
| `Errno 22` while the server starts | the server reads online-only placeholder files of a cloud-sync client | report to the operator: the runtime and the models belong on local disk, outside a synced folder |
| model names not found right after models were added | the server lists its model files when it starts | report to the operator; a new server start is their step or the scheduler's; then `easel doctor` |
| doctor warns that the server writes its outputs into its own install folder | a server nobody set up for easel answers on that address | tell the operator; do not render against it |
| `comfyui version: unknown until a job has run (lock says <min_version>) ... warn` | the server is not up, so its version cannot be read | expected; versions are checked once the server answers |
| `FAIL: comfyui version: <version> is older than comfyui.min_version <min>` | the server predates the graphs easel compiles | report to the operator: the server needs the version in `runtime/comfyui.lock.json` |

## Jobs and the scheduler

| symptom | cause | action |
|---|---|---|
| `waiting: free <x> GB < <y> GB needed by family <family>`, every 30 s | the machine has less free RAM than the family's `host_ram_gb.eval` | expected; let it wait; tell the operator if it waits long; never free memory or lower a threshold yourself |
| `family <f>: host_ram_gb.eval is not measured; waiting for limits.host_ram_floor_gb = <n> GB instead` | the family file carries no measured value for renders | expected; the job waits for the instance's floor; a maintainer can measure the family and fill the file |
| `waiting: free disk <x> GB < <y> GB floor (limits.disk_floor_gb)` | less free disk than the instance's floor | tell the operator; freeing space or changing the floor is their call |
| `cannot measure free RAM; set limits.skip_ram_check = true to skip` | the RAM reading failed; nothing starts | the operator decides; never set the key yourself |
| `cannot measure free disk space; nothing starts (...)` | the disk reading failed | report it to the operator |
| `scheduler unreachable: <verb> ...` then `failing closed: nothing starts (scheduler.fail_open = false)` (exit 3) | the external scheduler configured in `[scheduler]` did not answer | report it; never set `scheduler.fail_open` yourself; do not submit again |
| a job went back to `waiting` right after it was granted room | an external scheduler may withdraw a grant that is not started promptly | expected; easel marks a job started right after its grant and waits again otherwise; report it if it repeats |
| a running job stopped and later rendered again | the external scheduler asked it to yield: the stop file `jobs/<job-id>.stop` appeared, the render stopped after its current image, and the job was requeued | expected; it waits for room again by itself |
| `REFUSED: easel run is not in jobs.allowed of this instance (...)` | the instance does not allow render jobs | the operator decides; never widen the list yourself |
| the host runner's submit line prints `REFUSED` | the command is not on the host runner's allowed list | stop and report; never route around it |
| a job ends `failed` | a command exited non-zero | read `easel jobs status <job-id>` once, paste the tail of `jobs/<job-id>.log`, then [docs/STUCK.md](../STUCK.md) |
| a job waits although a server has just stopped | the system has not handed the server's memory back yet | expected; a later check passes |

## The graph and the outputs

| symptom | cause | action |
|---|---|---|
| HTTP 400 on queueing, with the sampler named | an A1111 sampler name (`euler_a`) instead of ComfyUI's (`euler_ancestral`) | revise the recipe to ComfyUI's spelling; easel prints the server's own message instead of the bare status |
| the job log says the GPU ran out of memory | the graph did not fit the GPU | report it; with the operator's yes, revise to a smaller size, `--count 1` or no hires pass |
| a rerun looks different | a setting changed, or a node pack drifted | `easel runs compare <old-run> <new-run>`: the settings diff first, then pixel identity per output |
| two renders of the same pixels have different file hashes | ComfyUI writes the graph into the PNG's text chunks | compare `pixels_sha256` (the pixel data without the text chunks) |
| `RATING CONFLICT` on a resource in a recipe | rated sfw, but its file name or a marker says otherwise | never render it as sfw; the operator rates it |
| the detailer template is unavailable | it needs ComfyUI-Impact-Pack (GPL-3.0) and ComfyUI-Impact-Subpack (AGPL-3.0, which installs ultralytics) inside the operator's ComfyUI | report to the operator (the packs go in at the lock's commits), or render without the detailer |
| a model name differs between the recipe and the server | the server names model files its own way (sub-folders, separators, case) | easel matches names against the server's own list at run time; report a name that matched only by its file name in another folder, since that may be a different model |
