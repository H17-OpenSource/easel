# 06-search: find, vet and add a candidate model

Uses the network: Civitai's public API (`/api/v1/`, downloads through `/api/download/`) and the Hugging Face
API, at most one request per second, backing off on HTTP 429, answers cached for 24 h. Stage map:
[README.md](README.md). `easel next` does not route here; a request does (the routing table of
[AGENTS.md](../../AGENTS.md)).

## 1. Purpose and who decides

Search for a candidate (a checkpoint, LoRA, upscaler or embedding), vet it (licence, rating, gating, size),
and, on the operator's yes, add it to the registry, download it and scan it.

- **The operator decides** whether the network may be used for this, which candidate to add, and whether
  its licence terms are acceptable. Downloading a large file is their yes too (disk).
- **You:** search, vet, report each candidate's class, and run the add and download lines after the yes.
  You never invent a licence or a rating; what the API does not say stays unknown.

## 2. Entry conditions

| command | literal line |
|---|---|
| `easel doctor` | `models: <models_root> exists ... ok` |

You are here because the operator asked for a model that `easel registry list --type <type>` does not show.
Searching also needs the operator's yes for the network in this session. A step marked planned is not built: report
it once and stop at that stage.

## 3. Commands

Search and vet only read; `registry add` writes a pending entry; `download` writes the file.

```
easel search civitai --query "<words>" --type <type>
easel search hf --query "<words>" --type <type>
easel search vet <url>
easel registry add <url>
easel registry download
easel registry download <pending>
easel registry show <sha16>
easel registry register-upstream <sha16> --civitai
```

- `search civitai|hf` prints one line per hit, the newest version's API URL first: that URL is what
  `search vet` takes. `--fresh` skips the cache.
- `search vet <url>` appends one row to `candidates/<date>.jsonl` with a class, the first that applies:
  `rating-conflict`, `gated`, `too-big`, `licence-unclear` or `ok`. A class is a finding for the operator,
  not a decision.
- `registry add <url>` writes `registry/pending/<name>.yaml` and prints `<name>`, which `<pending>` above
  stands for: `civitai-<model>-<version>` or `huggingface-<owner>--<name>-<commit12>` (plus `--<file stem>`
  when the repo holds several files). `registry download` with no name lists the pending entries.
- `registry download <pending>` resumes a `.part` file, checks the sha256 the site publishes (asked again, never
  from the cache), refuses when the disk would fall under `limits.disk_floor_gb`, writes the resource record,
  removes the pending entry and scans. Files land in `<models_root>`, never in an extra model root.
- `register-upstream <sha16> --civitai` (or `--hf <repo> --path <file>`, or `--url <url> --site <site>`) links
  a file you already have to its public source, so its licence and rating come from the site.
- The `licence{}` block is filled from the API's fields (commercial use, credit, derivatives, relicensing,
  availability) with the date it was read.
- **Tokens:** a token is read only from the environment variable that `search.civitai_token_env` names
  (default `CIVITAI_API_TOKEN`) or `search.hf_token_env` (default `HF_TOKEN`). It travels only in a request
  header that is not forwarded when the site redirects to a file host, and never appears in a URL, a file, a
  cache entry or a message. Never write a token into a command line or a file.
- **Cache:** API answers are kept under `<home>/cache/<site>/` for `search.cache_hours` (default 24); file
  names are hashes, so the folder reveals nothing. Downloads are never cached.
- **Exit codes:** 0 ok; 1 a checksum mismatch; 2 a refusal, a login wall or a missing record; 3 the disk
  floor or a rate limit that outlasted the backoff ([STUCK.md](../STUCK.md) has the full table).
- A resource's comments are not read: that site's terms allow automation only through its public API
  ([civitai-rules.md](../reference/civitai-rules.md)).

## 4. Expected output

On the demo, with the synthetic API answers of the offline test suite (`tests/easel/test_search.py`; no
network, ids from 9000000000 up). `easel search civitai --query "upscaler" --type upscaler`:

```
https://civitai.com/api/v1/model-versions/9000000402 | Demo 2x Upscaler | upscaler | Illustrious | rating sfw | licence unverified (commercial use: images)
1 result for "upscaler"
next: easel search vet <url>
```

`easel search vet <url> <url> --query "upscaler"` with the two synthetic versions:

```
ok | Demo 2x Upscaler (upscaler, illustrious, 0.06 GB) | licence unverified (commercial use: images) | rating sfw | https://civitai.com/api/v1/model-versions/9000000402
  sfw by the API rating
  licence fields present
licence-unclear | Demo Line Upscaler (upscaler, illustrious, 0.06 GB) | licence unknown | rating sfw | https://civitai.com/api/v1/model-versions/9000000502
  licence unclear: no licence fields in the API answer
wrote 2 rows to candidates/2030-01-01.jsonl
```

Match the shape, not the numbers (see [README.md](README.md)).

## 5. Gate and operator question

1. The candidate's vet row is `ok`, or the operator accepted its class in their own words.
2. `download` printed `downloaded <models_root>/<path> (<n> bytes, <checksum> verified)` and
   `registered <sha16> ...`, and `easel registry show <sha16>` lists the resource with its licence and rating.

Ask after vetting, once, then wait:

> Found `<n>` candidates for "`<words>`". Best fit: `<name>` (`<type>`, `<family>`, `<size>` GB), licence
> `<licence name>` (`<commercial use>`), class `<class>`. Reply "add" to download it, or pick another.

## 6. Symptom, cause, action

| symptom | cause | action |
|---|---|---|
| HTTP 429, then slower answers | the API's rate limit | the client backs off by itself; record the time; do not loop the search |
| exit 3 after repeated 429 or 503 | the site kept refusing past the backoff | stop; record the time in the journal; try again later, once |
| `licence-unclear` | the API leaves a licence field empty or contradictory | report it; the operator reads the model page and decides |
| `rating-conflict` | the declared rating and the content markers disagree | never treat it as sfw; ask the operator |
| `gated` | the file needs a login or an accepted licence | the operator downloads it themselves into `<models_root>`; then `easel registry scan --hash` |
| `too-big` | the file is over `search.max_file_gb`, or would push the disk under the floor | tell the operator; freeing space or changing the limit is their call |
| `refused: <file> needs <x> GB; ... under the <y> GB floor (limits.disk_floor_gb); ...` (exit 3) | the download would leave less free disk than the floor | tell the operator; never lower the floor yourself |
| `... sha256 mismatch: expected <sha>; the part file was deleted; ...` (exit 1) | a corrupt or changed file | run `download` once more: it starts over; a second mismatch: stop and report |
| `... interrupted at <n> of <total> bytes (...); run the same download again to resume` | the connection dropped | run the same `download` once more; it resumes from the `.part` file |
| `... the connection closed at <n> of <total> bytes; ...` | the site ended the transfer early | same as above: it resumes |
| `... the site sent a web page instead of the file (a login or ...)` (exit 2) | the site asked for a login | same as `gated` |
| `refused: civitai no longer publishes a sha256 for <file>; nothing to verify against` | easel downloads verified files only | report it; the operator decides whether to fetch it by hand |
| `refused: <repo> holds <n> model files (...); give --file <path>` | a repo or version with several weights files | ask the operator which file; `easel registry add <url> --file <name>` |
| `already registered: <sha16> <file> has these bytes; ...` | the file is in the registry already | expected; the source was merged into the existing record |
| exit 4 (or a `PermissionError`) naming the new file right after the bytes arrived | an antivirus scan or a sync client holds the fresh file | wait a minute, then run the same `download` once more: it finishes from the part file |
| a token seems to be ignored | the variable `search.civitai_token_env` names is not set in this shell | tell the operator; never paste a token into a command or a file |
| no network in this session | the session or the tests forbid it | stop; searching needs the operator's yes for the network |

Not in the table: follow [docs/STUCK.md](../STUCK.md).

## 7. Record

```
easel journal note "06-search: '<words>' -> <n> candidates, added <name> (<class>), sha256 ok"
```

A new pitfall: add a row to the table in section 6, dated from `date`, and a journal line, in the same session.
Stuck: `easel journal stuck "<one line>"`, then [docs/STUCK.md](../STUCK.md).
