"""`easel registry scan|list|show|explain|compatible|index|add|download|audit|dups|register-upstream`.

The argparse layer over `easel.registry`. Read verbs (`list`, `show`, `explain`, `compatible`, `audit`, `dups`)
never write. `scan`, `index`, `add`, `download` and `register-upstream` write through `_common.store` under the
registry's lock. `add`, `download` and `register-upstream --civitai|--hf` use the network (the public Civitai and
Hugging Face APIs only, one request per second, answers cached for 24 h); nothing else does. Output names paths
relative to the instance (`<models_root>/...`, `registry/...`), never an absolute path, so it can go into a
runbook as it is.
Exit codes: 0 ok; 1 a scan left problems, a download failed its checksum, a source did not match, or (`audit
--check`) a finding; 2 usage, configuration or data error, an unknown id, or a refusal; 3 deferred (the site
asks to slow down, or the disk floor would be crossed): retry later; 4 the registry lock or a file is held by
another process.
"""

from __future__ import annotations

import argparse
import json
import sys

HELP = ("the model registry: scan, list, show, explain, compatible, index, add, download, audit, dups, "
        "register-upstream")
TYPES = ("checkpoint", "lora", "lycoris", "vae", "text_encoder", "clip", "clip_vision", "controlnet", "ipadapter",
         "upscaler", "embedding", "detector", "sam", "style_model", "unet", "model_patch")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `registry` and its actions."""
    from easel._common.config import add_workspace_arg

    parser = subparsers.add_parser(
        "registry", help=HELP,
        description="The model registry: one record per model file in <home>/registry/resources/<sha16>.yaml. "
                    "Read verbs never write; scan and index rebuild registry/INDEX.md and registry/index.jsonl.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    scan = actions.add_parser(
        "scan", help="walk <models_root>, classify every model file and write its record",
        description="Walk paths.models_root. Without --hash: classify every file from its folder and header, "
                    "update the records of known files and list new ones (a new file's id is its hash). With "
                    "--hash: also hash new or changed files (reads each once; several GB per checkpoint) and "
                    "write their records. Records are never deleted: a file that is gone turns missing.")
    scan.add_argument("--hash", action="store_true", help="hash new or changed files and register them")
    scan.add_argument("--rehash", action="store_true", help="with --hash: hash every file, even unchanged ones")
    scan.add_argument("--quiet", action="store_true", help="print the summary only, not one line per file")
    add_workspace_arg(scan)
    scan.set_defaults(func=_scan)

    lst = actions.add_parser(
        "list", help="one line per resource: sha16 | name | type | family | rating | status",
        description="One line per resource: sha16 | name | type | family | rating | status. --self-trained adds "
                    "whether a training record names the file.")
    lst.add_argument("--type", choices=TYPES, help="only this resource type")
    lst.add_argument("--family", metavar="<family>", help="only this family id (exact, for example illustrious)")
    lst.add_argument("--self-trained", action="store_true", help="only resources whose source class is self-trained")
    lst.add_argument("--json", action="store_true", help="a JSON list instead of lines")
    add_workspace_arg(lst)
    lst.set_defaults(func=_list)

    show = actions.add_parser(
        "show", help="one resource: loader node, provenance, licence, content class against the prompt policy",
        description="Show one resource with the person's overrides applied: the ComfyUI loader node of its type, "
                    "provenance, integrity, the licence block, and whether prompt text may be written for it.")
    show.add_argument("sha16", help="the resource id (16 hex digits, or a unique prefix of at least 6)")
    show.add_argument("--json", action="store_true", help="the whole record as JSON")
    add_workspace_arg(show)
    show.set_defaults(func=_show)

    explain = actions.add_parser(
        "explain", help="why a resource is classed as its type, family, rating and source class",
        description="Say where each classifying value came from (folder, header, record, overrides.yaml, cue "
                    "words) next to what the folder and the header say today.")
    explain.add_argument("sha16", help="the resource id (16 hex digits, or a unique prefix of at least 6)")
    add_workspace_arg(explain)
    explain.set_defaults(func=_explain)

    compatible = actions.add_parser(
        "compatible", help="what works with a checkpoint, or which checkpoints a LoRA (or other add-on) works on",
        description="Join the registry with data/compat.yaml (and <home>/data/compat.yaml, which wins): native, "
                    "works and degraded pairs; --all adds untested and broken.")
    compatible.add_argument("sha16", help="the resource id (16 hex digits, or a unique prefix of at least 6)")
    compatible.add_argument("--all", action="store_true", help="also list untested and broken pairs")
    compatible.add_argument("--json", action="store_true", help="a JSON list instead of lines")
    add_workspace_arg(compatible)
    compatible.set_defaults(func=_compatible)

    index = actions.add_parser(
        "index", help="rebuild registry/INDEX.md and registry/index.jsonl from the records",
        description="Rebuild the generated views of the registry from the records (overrides applied). They carry "
                    "no timestamp, so an unchanged registry rebuilds byte-identically.")
    add_workspace_arg(index)
    index.set_defaults(func=_index)

    _register_write_actions(actions, add_workspace_arg)


def _register_write_actions(actions: argparse._SubParsersAction, add_workspace_arg) -> None:
    """add, download, audit, dups, register-upstream (task T2b)."""
    from easel.registry.audit import CHECKS

    add = actions.add_parser(
        "add", help="record a candidate from its model link as registry/pending/<name>.yaml (no download yet)",
        description="Read a model from the public API of civitai.com or huggingface.co (a model page link is "
                    "read for its ids only; the page is never fetched) and write registry/pending/"
                    "<source>-<model>-<version>.yaml: type, base model, size, licence block, rating. Network: the "
                    "public APIs only. Refused: a file without a published checksum, a type easel has no folder "
                    "for (give --type), several model files without --file. Next: `easel registry download "
                    "<name>`.")
    add.add_argument("url", metavar="<url>", help="the model's link: a Civitai model page or API URL (with or "
                                                  "without ?modelVersionId=), or a Hugging Face repo or file link")
    add.add_argument("--file", metavar="<name>", help="the file to take when the version or repo has several")
    add.add_argument("--type", choices=TYPES, help="the resource type, when the site's type has no easel equivalent")
    add.add_argument("--family", metavar="<family>", help="the family id, when the site's base model names none")
    add.add_argument("--replace", action="store_true", help="rewrite an existing pending entry from the site")
    add_workspace_arg(add)
    add.set_defaults(func=_net(_add))

    download = actions.add_parser(
        "download", help="download a pending entry: sha256-verified, disk floor kept, registered, then scanned",
        description="Download the file of registry/pending/<pending>.yaml into <models_root>/<folder of its "
                    "type>/: resumes a .part file, refuses a login page or a disk under limits.disk_floor_gb, "
                    "checks the sha256 the site publishes (asked again, never from the cache), writes "
                    "registry/resources/<sha16>.yaml, removes the pending entry and runs `registry scan`. Without "
                    "<pending> it lists the pending entries. Network: the public APIs only.")
    download.add_argument("pending", nargs="?", metavar="<pending>",
                          help="the pending entry's name (as `registry add` printed it)")
    add_workspace_arg(download)
    download.set_defaults(func=_net(_download))

    audit = actions.add_parser(
        "audit", help="list gaps: unknown ratings and licences, missing provenance or attribution, binding terms",
        description="Read-only. One section per check with who fixes it: " + ", ".join(CHECKS) + ". "
                    "`undocumented` = no source link, sha256 or licence; `unattributed` = a LoRA with no training "
                    "record and no source; `licence-terms` flags commercial_use: none and derivatives not "
                    "allowed. Resources that are not sfw are named by id only. Exit 0 with the report; --check "
                    "exits 1 when there is any finding.")
    audit.add_argument("--only", metavar="<checks>", help="comma-separated checks: " + ", ".join(CHECKS))
    audit.add_argument("--json", action="store_true", help="counts, findings and fixes as JSON")
    audit.add_argument("--check", action="store_true", help="exit 1 when there is any finding")
    add_workspace_arg(audit)
    audit.set_defaults(func=_guarded(_audit))

    dups = actions.add_parser(
        "dups", help="list model files with the same bytes under two names",
        description="Walk the model roots and group files with the same bytes: same size first, then a "
                    "record's sha256 or 8 sampled chunks; --full confirms every sampled group with a sha256 "
                    "(reads those files once). Read-only: deleting a copy is the operator's call.")
    dups.add_argument("--full", action="store_true", help="hash every sampled match so each group is confirmed")
    dups.add_argument("--json", action="store_true", help="the groups as JSON")
    add_workspace_arg(dups)
    dups.set_defaults(func=_guarded(_dups))

    upstream = actions.add_parser(
        "register-upstream", help="record where a resource came from when easel did not download it",
        description="Record the source of a registered file: --civitai looks its sha256 up in Civitai's public "
                    "API; --hf <repo> --path <file> compares the repo's checksum with the file (a mismatch is "
                    "refused, exit 1, unless --force); --url <url> --site <site> records a source with nothing to "
                    "check against (status partial). Fills the licence, rating and family only where the record "
                    "has none (--force replaces the source and the licence). Self-trained resources are refused.")
    upstream.add_argument("sha16", metavar="<sha16>", help="the resource id (16 hex digits, or a unique prefix of 6)")
    source = upstream.add_mutually_exclusive_group(required=True)
    source.add_argument("--civitai", action="store_true", help="find the version by the file's sha256")
    source.add_argument("--hf", metavar="<repo>", help="a Hugging Face repo id, owner/name")
    source.add_argument("--url", metavar="<url>", help="any other source link (no check possible)")
    upstream.add_argument("--path", metavar="<file>", help="with --hf: the file's path inside the repo")
    upstream.add_argument("--revision", metavar="<rev>", help="with --hf: a branch, tag or commit (default main)")
    upstream.add_argument("--site", metavar="<site>", help="with --url: the site's name in lower case")
    upstream.add_argument("--licence-name", metavar="<name>", help="the licence's name, when the source gives none")
    upstream.add_argument("--author", metavar="<author>", help="the author's name at the source")
    upstream.add_argument("--title", metavar="<title>", help="the model's title at the source")
    upstream.add_argument("--force", action="store_true",
                          help="replace a recorded source or licence; with --hf, record despite a checksum mismatch")
    add_workspace_arg(upstream)
    upstream.set_defaults(func=_net(_register_upstream))


# --- shared plumbing --------------------------------------------------------------------------------------------------

def _err(lines) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _open(args: argparse.Namespace):
    """(config, registry) for the instance; ConfigError propagates."""
    from easel._common import config
    from easel.registry import resources as res

    cfg = config.load(args.tool, args.workspace, strict=True)
    return cfg, res.open_registry(cfg, args.tool)


def _guarded(handler):
    """Map the known failures to their exit codes and lines; a held lock is journalled, then re-raised (exit 4)."""

    def run(args: argparse.Namespace) -> int:
        from easel._common import store
        from easel._common.config import ConfigError
        from easel.registry.scan import ScanRefused
        from easel.registry.shipped import DataError

        try:
            return handler(args)
        except ConfigError as exc:
            _err(exc.lines)
            return exc.code
        except (DataError, ScanRefused) as exc:
            _err(exc.lines)
            return 2
        except LookupError as exc:
            _err([str(exc)])
            return 2
        except store.Held as held:
            _journal_held(args, held)
            raise

    run.__doc__ = handler.__doc__
    return run


def _journal_held(args: argparse.Namespace, held) -> None:
    """Best effort: the exit-4 line into today's journal of the instance (never raises)."""
    try:
        from easel._common import config, journal

        cfg = config.load(args.tool, args.workspace, strict=False)
        journal.note(f"stuck: {held.line}", tool=args.tool, journal_dir=cfg.path("paths.journal"), timeout=2.0)
    except Exception:  # noqa: BLE001  (journalling a held lock must never hide the exit 4)
        pass


def _effective(reg):
    """(effective records, overrides, warnings) for a read verb."""
    from easel.registry import resources as res

    loaded = res.load_records(reg)
    overrides = res.load_overrides(reg)
    return res.effective(loaded.records, overrides), overrides, [f"warning: {p}" for p in loaded.problems]


# --- handlers ---------------------------------------------------------------------------------------------------------

@_guarded
def _scan(args: argparse.Namespace) -> int:
    from easel.registry import classify as cls
    from easel.registry import scan as scanner

    cfg, reg = _open(args)
    if args.rehash and not args.hash:
        _err(["--rehash goes with --hash"])
        return 2
    folders, markers = cls.load_folders(cfg.root), cls.load_markers(cfg.root)
    by = "easel registry scan" + (" --hash" if args.hash else "")
    report = scanner.scan(reg, folders, markers, do_hash=args.hash, rehash=args.rehash, by=by,
                          progress=None if args.quiet else (lambda line: print(line, file=sys.stderr, flush=True)))
    for line in report.lines:
        if not args.quiet or line.startswith("RATING CONFLICT"):
            print(line)
    for line in scanner.summary_lines(report, reg, do_hash=args.hash):
        print(line)
    _err(f"problem: {p}" for p in report.problems)
    return 1 if report.problems else 0


@_guarded
def _list(args: argparse.Namespace) -> int:
    from easel.registry import views

    _cfg, reg = _open(args)
    records, _overrides, warnings = _effective(reg)
    _err(warnings)
    rows = views.select(records, rtype=args.type, family=args.family, self_trained=args.self_trained)
    if args.json:
        print(json.dumps([views.list_json_row(r, self_trained=args.self_trained) for r in rows], indent=2,
                         ensure_ascii=False))
        return 0
    for row in rows:
        print(views.list_line(row, self_trained=args.self_trained))
    if not rows:
        _err([f"no resource matches (the registry holds {len(records)}; `easel registry scan --hash` fills it)"])
    return 0


@_guarded
def _show(args: argparse.Namespace) -> int:
    from easel.registry import resources as res
    from easel.registry import views

    cfg, reg = _open(args)
    records, overrides, warnings = _effective(reg)
    _err(warnings)
    sha16 = res.resolve_id(records, args.sha16)
    record = records[sha16]
    if args.json:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        return 0
    authoring = str(cfg.get("policy.prompt_authoring", "sfw-only"))
    for line in views.show_lines(record, authoring=authoring, override=overrides.get(sha16)):
        print(line)
    return 0


@_guarded
def _explain(args: argparse.Namespace) -> int:
    from easel.registry import classify as cls
    from easel.registry import resources as res
    from easel.registry import views

    cfg, reg = _open(args)
    records, overrides, warnings = _effective(reg)
    _err(warnings)
    sha16 = res.resolve_id(records, args.sha16)
    folders, markers = cls.load_folders(cfg.root), cls.load_markers(cfg.root)
    for line in views.explain_lines(records[sha16], reg, folders, markers, overrides.get(sha16)):
        print(line)
    return 0


@_guarded
def _compatible(args: argparse.Namespace) -> int:
    from easel.registry import compat
    from easel.registry import resources as res

    cfg, reg = _open(args)
    records, _overrides, warnings = _effective(reg)
    rule_warnings: list[str] = []
    rules = compat.load_rules(cfg.root, rule_warnings)
    _err(warnings + [f"warning: {w}" for w in rule_warnings])
    sha16 = res.resolve_id(records, args.sha16)
    target = records[sha16]
    rows = compat.for_resource(sha16, records, rules)
    shown = rows if args.all else [p for p in rows if p.verdict in compat.OFFERED]
    if args.json:
        print(json.dumps([p.as_dict() for p in shown], indent=2, ensure_ascii=False))
        return 0
    is_checkpoint = res.get_field(target, "identity.type") == "checkpoint"
    name = res.get_field(target, "identity.file_name")
    if is_checkpoint:
        print(f"checkpoint {sha16} {name} (family {res.get_field(target, 'identity.family') or 'unknown'}): "
              "what works with it")
    else:
        print(f"{res.get_field(target, 'identity.type')} {sha16} {name} (source family "
              f"{compat.source_family(target) or 'unknown'}): the checkpoints it works on")
    for p in shown:
        other, other_file, status = ((p.resource, p.file_name, p.status) if is_checkpoint
                                     else (p.checkpoint, p.checkpoint_file, p.checkpoint_status))
        other_type = p.artifact_type if is_checkpoint else "checkpoint"
        flag = "" if status in (None, "active") else f" ({status})"
        print(f"{other} | {other_file}{flag} | {other_type} | {p.source_family or 'unknown'} -> "
              f"{p.checkpoint_family or 'unknown'} | {p.verdict} | {p.confidence:.2f} | {p.reason or '-'}")
    hidden = [p for p in rows if p not in shown]
    if hidden:
        counts = {v: sum(1 for p in hidden if p.verdict == v) for v in ("untested", "broken")}
        print(f"not listed: {counts['untested']} untested, {counts['broken']} broken (--all lists them)")
    if not shown:
        print("nothing rated native, works or degraded for this resource")
    return 0


@_guarded
def _index(args: argparse.Namespace) -> int:
    from easel.registry import resources as res

    _cfg, reg = _open(args)
    records, _overrides, warnings = _effective(reg)
    _err(warnings)
    idx, md = res.rebuild_index(reg, records)
    print(f"wrote {reg.shown(md)} and {reg.shown(idx)} ({len(records)} resource(s))")
    return 0


# --- the write actions (T2b) ------------------------------------------------------------------------------------------

def _net(handler):
    """`_guarded`, plus the network errors of `_common.civitai`/`hf`: their lines to stderr, their exit code."""

    def run(args: argparse.Namespace) -> int:
        from easel._common.civitai import NetError

        try:
            return handler(args)
        except NetError as exc:
            _err(exc.lines)
            return exc.code

    run.__doc__ = handler.__doc__
    return _guarded(run)


def _add(args: argparse.Namespace) -> int:
    from easel.registry import add as adding

    cfg, reg = _open(args)
    done = adding.add(reg, cfg, args.url, file=args.file, rtype=args.type, family=args.family, replace=args.replace,
                      by="easel registry add")
    for line in done.lines:
        print(line)
    _err(f"warning: {w}" if not w.startswith("RATING CONFLICT") else w for w in done.warnings)
    return 0


def _download(args: argparse.Namespace) -> int:
    from easel.registry import add as adding
    from easel.registry import download as downloading

    cfg, reg = _open(args)
    if not args.pending:
        names = [p.stem for p in adding.pending_files(reg)]
        for name in names:
            print(name)
        print(f"{len(names)} pending entr{'y' if len(names) == 1 else 'ies'} in {reg.shown(reg.pending_dir)}/"
              + ("; next: easel registry download <pending>" if names else "; `easel registry add <url>` adds one"))
        return 0
    done = downloading.download(reg, cfg, args.pending, by="easel registry download",
                                say=lambda line: print(line, file=sys.stderr, flush=True))
    for line in done.lines:
        print(line)
    if done.scan_report is not None:
        _err(f"problem: {p}" for p in done.scan_report.problems)
    return 0


def _audit(args: argparse.Namespace) -> int:
    from easel.registry import add as adding
    from easel.registry import audit as auditing

    cfg, reg = _open(args)
    records, _overrides, warnings = _effective(reg)
    _err(warnings)
    only = [c.strip() for c in args.only.split(",") if c.strip()] if args.only else None
    unknown = sorted(set(only or []) - set(auditing.CHECKS))
    if unknown:
        _err([f"unknown check(s): {', '.join(unknown)}; the checks: {', '.join(auditing.CHECKS)}"])
        return 2
    findings = auditing.audit(reg, cfg, records, only=only)
    if args.json:
        print(json.dumps(auditing.as_json(findings), indent=2, ensure_ascii=False))
    else:
        for line in auditing.lines(findings, len(records), len(adding.pending_files(reg))):
            print(line)
    return 1 if args.check and findings else 0


def _dups(args: argparse.Namespace) -> int:
    from easel.registry import dups as duplicates

    _cfg, reg = _open(args)
    records, _overrides, warnings = _effective(reg)
    _err(warnings)
    report = duplicates.find(reg, records, full=args.full,
                             progress=lambda line: print(line, file=sys.stderr, flush=True))
    _err(report.notes)
    if args.json:
        print(json.dumps([g.as_dict() for g in report.groups], indent=2, ensure_ascii=False))
        return 0
    for line in duplicates.lines(report):
        print(line)
    return 0


def _register_upstream(args: argparse.Namespace) -> int:
    from easel.registry import upstream

    cfg, reg = _open(args)
    if args.path and not args.hf or args.revision and not args.hf:
        _err(["--path and --revision go with --hf"])
        return 2
    if args.site and not args.url:
        _err(["--site goes with --url"])
        return 2
    done = upstream.register(reg, cfg, args.sha16, civitai_hash=args.civitai, hf_repo=args.hf, hf_path=args.path,
                             revision=args.revision, url=args.url, site=args.site, licence_name=args.licence_name,
                             author=args.author, title=args.title, force=args.force)
    for line in done.lines:
        print(line)
    return 0
