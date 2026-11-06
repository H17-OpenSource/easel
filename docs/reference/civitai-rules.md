# Civitai: what easel may and may not do

Recorded 2026-11-06. Terms of service change: re-read Civitai's current terms before relying on this page,
and when they differ, the live terms win and this page gets a dated correction. Not legal advice.

## Public API only

- easel talks to Civitai through two URL families and nothing else: the public REST API under
  `https://civitai.com/api/v1/` (search, model and version records) and file downloads under
  `https://civitai.com/api/download/`. A test asserts that no other Civitai URL shape appears in either
  repository.
- No tool logs in, drives a browser, scrapes a web page or calls the internal endpoints the website itself
  uses. In particular, **comments are not supported**: the public API does not offer them, and Civitai's
  Terms of Service, section 11.4, as recorded by this project on 2026-11-05, allow automated access only
  through the public API, Civitai's MCP server or Civitai's own software.
- Post feedback that the operator reads on the site is entered by hand, quoting the operator:
  `easel lessons add --finding "<finding>" --evidence "<evidence>" --rule "<rule>" --source civitai`
  ([07-lessons.md](../runbooks/07-lessons.md)).

## Rate limits: unknown

Civitai's API rate limits were not known to this project on 2026-11-05. easel therefore limits itself:
- at most one request per second (`search.requests_per_s`);
- on HTTP 429 or 503, back off and retry later, never in a tight loop; a refusal that outlasts the backoff
  ends the command with exit 3;
- answers are cached, so a repeated search does not hit the API again (below).
A 429 seen in real use is a fact worth recording: write it in the journal with the time and add it to the
lessons, so this section can say a number with a date.

## The cache

- JSON answers are kept under `<home>/cache/<site>/` for `search.cache_hours` (default 24). `--fresh` on a
  search skips the cache once.
- Each cache file is named by a hash of the request (the URL and whether a token was sent, never the token
  itself), so the folder's names reveal nothing.
- Files are never cached, and a download asks the site for the checksum again instead of trusting the cache.

## Tokens

- A token is optional for public records and needed for some downloads. easel reads it only from the
  environment variable that `search.civitai_token_env` names (default `CIVITAI_API_TOKEN`); for Hugging Face,
  `search.hf_token_env` (default `HF_TOKEN`).
- The token travels only in a request header that is not forwarded when the site redirects a download to a
  file host. It never appears in a URL, a cache file, a record, a message or a log line.
- Never write a token into a command line, the instance file or a repository file.

## Downloads

- `easel registry download` fetches a file only after `easel registry add` put it in `registry/pending/`,
  resumes an interrupted download from its `.part` file, checks the sha256 the site publishes, and refuses to
  go below the disk floor (`limits.disk_floor_gb`).
- A checksum mismatch deletes the part file: the next run starts over. A second mismatch is a stop: report it.
- Some files need an account to download. easel detects the login page instead of a file and stops with a
  message; it does not log in. The operator downloads such a file by hand into `<models_root>`, then
  `easel registry scan --hash` registers it.
- The record keeps ids and hashes, never the uploader's user name, the description HTML or the preview
  images.

## Exit codes of the network verbs

| code | when |
|---|---|
| 0 | done |
| 1 | the downloaded bytes do not match the published checksum |
| 2 | a refusal (a URL outside the public API, a file without a published checksum), a record the site does not have, or a login wall |
| 3 | the disk floor would be crossed, or the rate limit outlasted the backoff: wait, then run it once more |

## Licences and ratings

- The licence flags of a model version (`allowCommercialUse`, `allowNoCredit`, `allowDerivatives`,
  `allowDifferentLicense`, `availability`) are copied into the resource's `licence` block with the date they
  were read; [WHEN-TO-APPLY.md](../WHEN-TO-APPLY.md) says what each field means.
- A rating easel cannot establish is treated as not safe for work (`policy.unknown_rating_is = "nsfw"`).
  The content markers in `data/content_markers.yaml` are three short, hand-written, generic lists; Civitai's
  own term lists are not shipped.

## Publishing is by hand, by the operator

These rules hold for every model that kiln or easel helped to make:
- No tool and no agent uploads, posts or edits anything on Civitai. The operator uploads by hand.
- Upload only after the operator's explicit go for that model and version, in their own words.
- Upload only the files of a release package that its privacy check called clean, never an original render
  or a training checkpoint: those carry workflows, local paths, ids and dataset statistics.
- The subject is a fictional adult. Never tick a box saying the model depicts a minor, and answer the
  real-person question with no; if either answer would be untrue, the model is not published.
- Only the operator chooses and judges the showcase images.
