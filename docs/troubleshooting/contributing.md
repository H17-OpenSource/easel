# Troubleshooting: contributing code, tests and docs

Pitfalls that contributors (people and agents) have already paid for while writing easel and kiln, each with
the rule that avoids it. Most hold for both repositories, because `src/easel/_common/` is shared. Rendering
failures are in [render.md](render.md); the contribution process is [CONTRIBUTING.md](../../CONTRIBUTING.md). A
new pitfall gets a row here (or in the runbook a user would hit it in) in the same pull request.

## Tests and fixtures

| symptom | cause | rule |
|---|---|---|
| two test files with the same name break collection | `tests/` has no `__init__.py`, so pytest imports both under one module name | every test file name is unique across all subfolders of `tests/` (`easel qa tests` checks it) |
| a demo sha256 differs between your machine and CI | image writers and zlib builds compress differently | fixture media that a record hashes are written byte by byte (stored deflate, sorted JSON), never through a compressor |
| a test passes alone and fails beside another | it edited `examples/demo/` or depended on files another change regenerates | tests copy `examples/demo/` into a temporary folder at test time and change only the copy |
| a planted copy of a record carries an edit the test made earlier | the copy was taken after the original was edited | plant copies first, then edit the originals |
| a test's expected number is wrong while the command is right | the number was worked out by hand | assert that the command agrees with an independent reference first, and take literal numbers from that reference |
| a test asserts a placeholder that has just been replaced by a real verb | the stub's assertion outlived the stub | whoever replaces a planned stub removes exactly that stub's assertion in the same change and says so |
| `Popen.pid` is not the pid of the Python child on Windows | a venv's `python.exe` is a launcher that starts the real interpreter | a child that must be identified prints its own `os.getpid()` |
| a refused loopback connection takes about two seconds per test on Windows | Windows retries a refused connection before it gives up | for a fast "unreachable" test, use a listening socket that never answers plus a short client timeout |
| an HTTP header seems missing in a test server | `urllib` sends header names in its own capitalisation | compare header names case-insensitively (keep the message object, not a plain dict) |
| a check for a dropped record key also matches a legal value | a bare word can be both a key and a value | check dropped keys as `"key":`, never as a bare word |
| a fixture that reads a flag behaves differently than the test expects | the fixture row carries a flag the test did not read | read the fixture's flags before asserting on them |
| a fake lock in a test silently breaks the journal's own lock | the monkeypatch replaced the lock for every store root | a fake raises only for the one root under test and hands every other root to the real lock |
| the whole suite stops at collection on someone else's half-written module | one import error ends collection for everybody | run the test paths you changed, report the broken module to its owner, never edit it to get past |
| piping the test run into `tail` reports success after a failure | the pipe returns `tail`'s exit code, not pytest's | send the output to a file and read pytest's own exit code |
| a test copied to the sister repository breaks | it names the other tool literally, and the copy renames easel to kiln | tests use a made-up tool name, and sync tests take the names from `tools/sync_common.py` |
| a CLI test fails only when the repository's schema changes | the test validated against the live schema file | tests pin their own schema; `config.load()` validates only when an instance file exists |
| a fixture file's line endings doubled | `Path.write_text` on Windows writes CRLF | fixtures whose bytes matter are written with `write_bytes` |

## Privacy scanner and needles

| symptom | cause | rule |
|---|---|---|
| the privacy scan flags your test | a test that checks a value is absent contains the value | build every hazardous needle at run time from fragments: paths, e-mail addresses, placeholders, the frozen-clock warning, real-year ids |
| a low decision number in a demo or a test is a private hit | real stores hold low decision numbers, and the private list harvests them | demo, docs and tests use high numbers (`dec-9001` and up) and 2030 dates |
| a percent-encoded slash or space in a test's expected URL is flagged | the scanner flags encoded separators everywhere | build encoded strings with `urllib.parse.quote` at run time and assert decoded values |
| a synthetic model-page URL in a test is flagged | page URLs of the model site are a scanned class | build such URLs from fragments, even in private tests |
| drive-letter test data is flagged | a drive path is a public pattern | use placeholders such as `<tools dir>` in path test data |
| the scanner matches its own pattern file | a literal regex matches the line it is written on | break literals with a one-character class, and keep pattern ids free of the words they detect |
| a YAML key `on:` turned into `true` | YAML 1.1 reads a bare `on` as a boolean | never use `on`, `off`, `yes` or `no` as keys; loaders reject unknown keys |
| a case-scoped entry of the private list never matches | it was run only on casefolded text | `re:` entries see the text with its case kept; only literals are casefolded |
| a scan from a subfolder misses the repository root | a relative root was walked without making it absolute | absolutize with `os.path.abspath` before walking up, and never follow symlinks for repository paths |
| a placeholder in a repo file matches the public placeholder pattern | the CODEOWNERS placeholder shape (the word handle in angle brackets) is a pattern | name the value instead, for example `<author>` |
| an error message shows an absolute path | the text of a lower-level exception carried it | say what is wrong in your own words; never insert an exception's text that may hold a path |
| a docstring names a product the code only handles generically | it was carried over from an older source | describe behaviour generically ("a cloud placeholder file"), never by product name |

## Editing files

| symptom | cause | rule |
|---|---|---|
| a patch with backslash escapes changed meaning or failed to apply | a shell heredoc or inline script collapsed or expanded the escapes, or broke on nested quotes | change lines that hold backslashes with an editor (or a script file written by one), never through a heredoc |
| an editor refuses to change a file you just changed from the shell | it still holds the version it read before | read the region again before the next edit |
| a long `grep` with several `-e` patterns aborted in Git Bash | that build of grep fails on some long pattern sets | search with one pattern per call, or with ripgrep |
| invisible zero-width or bidi characters appeared in a file | an editing tool turned backslash-u escapes into the characters | build such characters from code points with `chr()` |
| backtick spans in a shell-appended line ran as commands | an unquoted heredoc expands backticks | append text with a quoted heredoc; compute values such as the time first |
| a command blocked until its timeout | a stdin-reading command came before the heredoc that was meant to feed another | write patch files with an editor; never put a stdin reader ahead of a heredoc |
| a Python rewrite left CRLF line endings in a source file | `Path.write_text` on Windows writes CRLF | edit repository files with an editor; after a scripted rewrite, check the line endings |
| `import re` fails with an odd error inside a helper script | the working folder was inside a package, so a module there shadowed the standard library | run helper scripts from the repository root or a scratch folder, with `python -I` |
| `ruff` reports dozens of syntax errors in a YAML file | the YAML file was passed to ruff explicitly | run `ruff check .` or pass `.py` files only |

## Schemas, data and records

| symptom | cause | rule |
|---|---|---|
| a frozen schema rejects files that already exist | only the schema's own examples were validated | before freezing a schema, validate every file in the tree that names it; for a format that will hold imported records, map a set of real records in memory first and count the errors by path |
| a contributed type cannot be named in a record | the record schema lists the shipped values as an enum | let a record schema check its own `type` field with a pattern; keep folder-name rules for the files that need them |
| a record with a key set to `null` fails validation | the schema does not allow null there | leave a key out when it does not apply, unless the schema allows null |
| a YAML flow mapping grew an extra key | a plain value with a comma was split at the comma | quote every flow-mapping value that holds a comma, or use block style |
| `gen_formats_doc.py --check` fails after a schema change | the formats page renders whatever `schemas/` holds | a new or changed schema ships with its regenerated `docs/reference/formats.md` in the same change; prefer an existing format |
| a demo record names a command that does not exist | a `by` field was written from memory | `by` fields name only verbs that `cli.py` lists, or the field is left out |
| a reader crashed on one broken YAML record | `yaml.YAMLError` is not a `ValueError` | readers that skip bad records catch `(OSError, ValueError, yaml.YAMLError)` and report the file |
| exploration output showed a private value | a free-text column or a path component was printed | exploration prints only enumerations you chose and counts; test a classifier on a synthetic path first |
| a `PRAGMA` was used to read a host database's columns | some `PRAGMA` statements change the connection or the file, so a read-only reader allows none | list columns with `SELECT * FROM <table> LIMIT 0` and the cursor description, never a `PRAGMA`, on a database opened with `mode=ro` |
| printed output carries a real date | a dated fact from a data file was echoed | printed output carries no real dates; dated facts stay in the data file |
| a note on a resource has nowhere to go in `registry/overrides.yaml` | an override entry needs at least one field to set | resource, recipe and template notes are changelog lines; run notes go into the run's `run.json` |

## CLI and planned verbs

| symptom | cause | rule |
|---|---|---|
| an agent gets argparse's "invalid choice" for a subcommand a doc names | the verb was built, but a later subcommand was not registered | when you build a verb, register every subcommand the docs name as a placeholder with `set_defaults(planned="<task>")` that prints the planned line and exits 2; grep `docs/` and `schemas/` for the verb first |
| a placeholder subcommand rejects `--help` or a leading flag | `argparse.REMAINDER` does not swallow a leading option | a placeholder subparser takes every word as plain text (no help option, a prefix character no word uses, `nargs="*"`) |
| a doc's command line names a planned verb | the verb is not built yet | the fenced line ends with `# planned: <task>`, the task the CLI prints; drop the marker in the change that builds it (`easel qa docs` checks both) |
| a runbook command fails although `qa docs` passed | the command left out an option the verb requires | a command line carries every option the verb requires (`qa docs` checks fenced lines) |
| `qa docs` demands an option on a fenced line that the verb does not need | the option's help text starts with `required:` | start a help text with `required:` only when the verb refuses without the option |
| help text in parentheses reads badly | a planned verb prints its help inside parentheses after `planned: <task>` | CLI help strings carry no parentheses; use a colon or a semicolon |
| a note line printed before the verb's answer | the scheduler's notes were sent to stdout | a verb that returns at once gives the scheduler a stderr writer; stdout carries the answer only |
| an empty optional block was treated as absent | `{}` is falsy | test optional blocks with `is not None` when `{}` means "all defaults" |
| `importlib.util.find_spec("pkg.sub.mod")` raises | a dotted lookup raises `ModuleNotFoundError` when the parent is missing | wrap a dotted `find_spec` in `try/except ModuleNotFoundError` |

## Network clients

| symptom | cause | rule |
|---|---|---|
| a dropped download looks finished | `http.client` returns empty bytes at an early end of stream instead of raising | compare the bytes received with the announced total (`Content-Length` or `Content-Range`) before verifying |
| a token reached a file host | ordinary headers are forwarded on a redirect | send tokens only with `add_unredirected_header`, never in a URL |
| a repository's own `config.json` was refused as a login page | a JSON or HTML content type was treated as a site message | exempt files whose own suffix is `.json` or `.html` from the message and login-page check |
| a ComfyUI history entry was read as done too early | `status_str` said success while `completed` was false | an entry is done when `completed` is true; `status_str` decides only when that flag is absent |

## Docs

| symptom | cause | rule |
|---|---|---|
| an entry condition names a line that no command prints | the output of an unbuilt verb was guessed | entry conditions use `easel next --stage`, a doctor line or another literal a built command prints; never invent output |
| `qa docs` flags a path in a guide | the guide names a folder that does not exist yet | name only paths that exist; describe a planned module by its task |
| `qa docs` flags a path that exists in kiln | paths are checked in this repository | write "kiln's `<path>`": a span right after the sister's name is not checked |
| the docs show the literal pending marker and CI fails | `qa docs` counts the marker wherever it appears, even in prose | describe the marker in words; only an unfilled output block carries it |
| a public lesson points back to one release | it combined a site, a version count and a month | public lessons carry no counts, exact dates or details that identify one release |
| an inline command was not checked | it was split over two lines inside one code span | keep each inline command on one line |
| an output block shows a real date or the frozen-clock warning | it was pasted from a real session | generate it from the demo under the frozen clock with `--redact-paths`, stdout only |

## Locks and files

| symptom | cause | rule |
|---|---|---|
| a waiting writer cannot read who holds a lock | on Windows `msvcrt.locking` blocks reads of the locked byte range | lock one byte far past the diagnostics and keep the diagnostics at the start of the lock file |
| a replace fails with `PermissionError` right after a large write | an antivirus scan, a backup or a sync client holds the new file for a moment | atomic replaces go through the store's retry loop, which ends in exit 4 naming the file, never a raw traceback |
| every writer waits 30 s and then names no holder | the folder (a network share or some FUSE mounts) does not support file locks | keep instances on a local disk; a lock error that is not contention must be reported as such, not as a held lock |

## The licence gate and the hook

| symptom | cause | rule |
|---|---|---|
| `pip install -e .` fails offline | the build backend downloads `setuptools>=77` into an isolated environment | offline, install `setuptools>=77` into the venv first and add `--no-build-isolation` |
| the licence gate refuses a reviewed package | `pip-licenses --allow-only` compares whole licence strings, and an expression such as `Apache-2.0 OR BSD-2-Clause` is one string | put each reviewed expression on the allow list exactly; never add `--partial-match` |
| the installed hook runs without the venv's packages | the hook recorded a resolved interpreter path, which on Linux is the base interpreter | record interpreter paths with `os.path.abspath(sys.executable)`, never `Path(sys.executable).resolve()` |
| the copyright placeholder check fires on the MIT text | the licence text itself says COPYRIGHT HOLDERS | the pattern matches only the bracketed placeholder form; keep the brackets when you touch it |
| a folder you added is never committed | `.gitignore` matches `instance/`, `journal/`, `build/` and `dist/` at any depth | never name a source, test or docs folder that way |
| the gate refuses your commit for its date | the commit carries a local UTC offset | commit in UTC (`TZ=UTC`, or `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` ending in `+0000`), then amend or rebase |
