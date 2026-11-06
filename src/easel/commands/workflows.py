"""`easel workflows check|list|validate|register|approve`: the argparse layer over `easel.workflows`.

`check` and `list` work without an instance (then they read the shipped templates only), so CI can run
`easel workflows check` on a fresh clone.
Exit codes: 0 ok; 1 a template has problems; 2 usage, configuration or data error, or a refusal; 3 `validate`
could not reach the ComfyUI server (the offline checks still ran); 4 a lock is held.
"""

from __future__ import annotations

import argparse
import sys

HELP = "register, validate, approve and check workflow templates"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `workflows` and its actions."""
    from easel._common.config import add_workspace_arg
    from easel.workflows import DOMAINS, TASKS

    parser = subparsers.add_parser(
        "workflows", help=HELP,
        description="Workflow templates: workflows/<id>/ with graph.json (ComfyUI API format), slots.json and "
                    "meta.yaml. Shipped templates come with the repository; an instance keeps its own under "
                    "<home>/workflows/. Every custom node pack a graph needs is named with its licence.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    chk = actions.add_parser(
        "check", help="offline checks of every template (or the ones named): schema, slots, links, node packs",
        description="Check templates without a server: the meta and slot schemas, slots pointing at real nodes, "
                    "links, no model file left without a slot, and each node pack against "
                    "runtime/comfyui.lock.json (licence, commit, what it installs). Exit 1 on any problem.")
    chk.add_argument("ids", nargs="*", metavar="<id>", help="only these templates (default: all)")
    add_workspace_arg(chk)
    chk.set_defaults(func=_check)

    lst = actions.add_parser("list", help="one line per template: id | status | task | family | origin | packs",
                             description="List the shipped and the instance's templates.")
    add_workspace_arg(lst)
    lst.set_defaults(func=_list)

    val = actions.add_parser(
        "validate", help="check a template, then ask the ComfyUI server whether it has every node class and input",
        description="The offline checks, then GET /object_info for the graph's node classes. Records "
                    "validated_at on an instance template. Exit 3 when the server cannot be reached.")
    val.add_argument("id", help="the template id")
    add_workspace_arg(val)
    val.set_defaults(func=_validate)

    reg = actions.add_parser(
        "register", help="add a graph and its slot map as a new instance template (status draft)",
        description="Copy an API-format graph (Export (API) in ComfyUI) and its slot map into "
                    "<home>/workflows/<id>/ with a meta.yaml, status draft. Node packs the lock knows are filled "
                    "in from runtime/comfyui.lock.json; name any other with --node-pack.")
    reg.add_argument("graph", help="the API-format graph JSON file")
    reg.add_argument("--slots", required=True, help="the slot map (easel/workflow-slots/1, or just its slots list)")
    reg.add_argument("--task", required=True, choices=TASKS)
    reg.add_argument("--domain", default="image", choices=DOMAINS)
    reg.add_argument("--family", metavar="<family>", help="the model family the graph is built for")
    reg.add_argument("--id", dest="wid", metavar="<id>", help="the template id (default: from the graph file name)")
    reg.add_argument("--name", help="a short human name")
    reg.add_argument("--node-pack", action="append", default=[], metavar="NAME=LICENCE",
                     help="a custom node pack the graph needs and its licence (SPDX id or unverified); repeatable")
    reg.add_argument("--notes", help="what the template is for")
    add_workspace_arg(reg)
    reg.set_defaults(func=_register)

    app = actions.add_parser(
        "approve", help="mark a template approved after a run with it finished ok (the operator's yes)",
        description="Approve a template once a run that used it finished ok. A shipped template is copied into "
                    "the instance with status approved; the repository's copy is never changed.")
    app.add_argument("id", help="the template id")
    app.add_argument("--run", required=True, metavar="<run-id>", help="the ok run that used this template")
    app.add_argument("--note", required=True, help="the operator's words")
    add_workspace_arg(app)
    app.set_defaults(func=_approve)


def _err(lines) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _guarded(handler):
    def run(args: argparse.Namespace) -> int:
        from easel._common import store
        from easel._common.config import ConfigError
        from easel.registry.shipped import DataError
        from easel.workflows import WorkflowError

        try:
            return handler(args)
        except ConfigError as exc:
            _err(exc.lines)
            return exc.code
        except (WorkflowError, DataError) as exc:
            _err(exc.lines)
            return 2
        except store.Held as held:
            try:
                from easel._common import config, journal

                cfg = config.load(args.tool, args.workspace, strict=False)
                journal.note(f"stuck: {held.line}", tool=args.tool, journal_dir=cfg.path("paths.journal"), timeout=2.0)
            except Exception:  # noqa: BLE001  (journalling a held lock must never hide the exit 4)
                pass
            raise

    run.__doc__ = handler.__doc__
    return run


def _optional_config(args: argparse.Namespace):
    """The instance config when there is one; None on a fresh clone (shipped templates only)."""
    from easel._common import config

    cfg = config.load(args.tool, args.workspace, strict=False)
    return cfg if cfg.file is not None else None


def _required_config(args: argparse.Namespace):
    from easel._common import config

    return config.load(args.tool, args.workspace, strict=True)


def _print_report(report) -> None:
    from easel import workflows

    t = report.template
    head = "ok" if not report.problems else "FAIL"
    nodes, slots = len(t.graph), len(t.slots.get("slots") or [])
    print(f"{head} {workflows.summary(t)} ({nodes} nodes, {slots} slots)")
    for line in report.problems:
        print(f"  problem: {line}")
    for line in report.warnings:
        print(f"  warn: {line}")


@_guarded
def _check(args: argparse.Namespace) -> int:
    from easel import workflows

    cfg = _optional_config(args)
    templates = workflows.discover(cfg)
    unknown = [i for i in args.ids if i not in templates]
    if unknown:
        _err([f"no workflow template {', '.join(unknown)}; have {', '.join(sorted(templates)) or 'none'}"])
        return 2
    lock = workflows.load_lock()
    reports = [workflows.check(templates[i], lock) for i in (args.ids or sorted(templates))]
    for report in reports:
        _print_report(report)
    bad = sum(1 for r in reports if r.problems)
    print(f"workflows: {len(reports)} checked, {bad} with problems")
    return 1 if bad else 0


@_guarded
def _list(args: argparse.Namespace) -> int:
    from easel import workflows

    templates = workflows.discover(_optional_config(args))
    for wid in sorted(templates):
        print(workflows.summary(templates[wid]))
    print(f"{len(templates)} template{'' if len(templates) == 1 else 's'}")
    return 0


@_guarded
def _validate(args: argparse.Namespace) -> int:
    from easel import workflows
    from easel._common import comfy_client, comfy_graph

    cfg = _required_config(args)
    t = workflows.get(cfg, args.id)
    report = workflows.check(t)
    _print_report(report)
    if report.problems:
        return 1
    client = comfy_client.Client.from_config(cfg)
    try:
        version = client.version()
        info = client.object_info_for(comfy_graph.node_classes(t.graph))
    except comfy_client.ComfyError as exc:
        _err(exc.lines)
        _err([f"{t.id}: the offline checks passed; the node check needs a running server (run validate again then)"])
        return 3
    problems = comfy_graph.check(t.graph, info)
    for line in problems:
        print(f"  server: {line}")
    if problems:
        print(f"{t.id}: {len(problems)} problem(s) on the server; the template is unavailable there until they are fixed")
        return 1
    note = f"ComfyUI {version or 'unknown version'}: every node class and input of the graph exists"
    recorded = workflows.record_validation(cfg, t, note, tool=args.tool)
    print(f"{t.id}: valid on the server ({note})" + ("" if recorded else "; shipped template, nothing recorded"))
    return 0


@_guarded
def _register(args: argparse.Namespace) -> int:
    from pathlib import Path

    from easel import workflows
    from easel.registry.shipped import shown

    cfg = _required_config(args)
    packs = []
    for item in args.node_pack:
        name, sep, licence = item.partition("=")
        if not sep or not name.strip() or not licence.strip():
            _err([f"--node-pack takes NAME=LICENCE (SPDX id or unverified), not {item!r}"])
            return 2
        packs.append((name.strip(), licence.strip()))
    report = workflows.register(cfg, Path(args.graph), Path(args.slots), task=args.task, wid=args.wid, name=args.name,
                                domain=args.domain, family=args.family, packs=packs, notes=args.notes, tool=args.tool)
    _print_report(report)
    if report.problems:
        print("not registered: fix the problems above")
        return 1
    print(f"registered {report.template.id} as draft in {shown(report.template.folder, cfg.root)}")
    print(f"next: easel workflows validate {report.template.id}")
    return 0


@_guarded
def _approve(args: argparse.Namespace) -> int:
    from easel import workflows
    from easel.registry.shipped import shown

    cfg = _required_config(args)
    t = workflows.approve(cfg, args.id, args.run, args.note, tool=args.tool)
    print(f"approved {t.id} after {args.run} ({shown(t.folder, cfg.root)})")
    return 0
