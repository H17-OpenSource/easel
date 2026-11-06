"""`easel search civitai|hf|vet`: the argparse layer over `easel.search`.

Uses the network (the public APIs only). Exit codes: 0 ok (every vet class is a result, not an error); 1 a site or
network problem, or a checksum mismatch; 2 usage, configuration, an unknown URL, a refused request or a record the
site does not have; 3 the site asks to slow down (retry later, do not loop); 4 a lock is held.
"""

from __future__ import annotations

import argparse
import sys

HELP = "search Civitai and Hugging Face for candidates; vet a candidate"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `search` and its actions."""
    from easel._common.civitai import QUERY_TYPES
    from easel._common.config import add_workspace_arg

    types = sorted(QUERY_TYPES)
    parser = subparsers.add_parser(
        "search", help=HELP,
        description="Search the public Civitai and Hugging Face APIs (one request per second, cached for 24 h) and "
                    "vet a candidate before the operator decides to add it. Comments are never read.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    civ = actions.add_parser(
        "civitai", help="search Civitai's public API; one line per hit with the URL to vet",
        description="Search Civitai's /api/v1/models. By default the site hides models it marks nsfw. Prints the "
                    "newest version's API URL, name, type, base model, rating and licence of each hit.")
    civ.add_argument("--query", required=True, metavar="<words>", help="the search words")
    civ.add_argument("--type", choices=types, help="only this resource type")
    civ.add_argument("--base-model", action="append", default=[], metavar="<name>",
                     help="only this base model as the site names it (repeatable)")
    civ.add_argument("--limit", type=int, default=10, help="how many hits (1-100, default 10)")
    civ.add_argument("--nsfw", action="store_true", help="also list models the site marks nsfw (the operator's call)")
    civ.add_argument("--fresh", action="store_true", help="ask the site again instead of using a cached answer")
    add_workspace_arg(civ)
    civ.set_defaults(func=_civitai)

    hug = actions.add_parser(
        "hf", help="search the Hugging Face model API; one line per hit with the URL to vet",
        description="Search Hugging Face's /api/models. --type maps to a tag filter where one exists (lora, "
                    "checkpoint, embedding, controlnet).")
    hug.add_argument("--query", required=True, metavar="<words>", help="the search words")
    hug.add_argument("--type", choices=types, help="only this resource type, where the site has a tag for it")
    hug.add_argument("--limit", type=int, default=10, help="how many hits (1-100, default 10)")
    hug.add_argument("--fresh", action="store_true", help="ask the site again instead of using a cached answer")
    add_workspace_arg(hug)
    hug.set_defaults(func=_hf)

    vet = actions.add_parser(
        "vet", help="vet candidates: ok | licence-unclear | rating-conflict | gated | too-big, into candidates/<date>.jsonl",
        description="Read each URL's model version from the public API, class it (rating-conflict, gated, too-big, "
                    "licence-unclear or ok, the first that applies) and append one row per URL to "
                    "<home>/candidates/<date>.jsonl. A class is a finding for the operator, not a decision.")
    vet.add_argument("urls", nargs="+", metavar="<url>",
                     help="a Civitai /api/v1/ model, version or by-hash URL, a download URL or a model page link; a "
                          "Hugging Face repo URL")
    vet.add_argument("--query", metavar="<words>", help="the search words that found it (recorded in the row)")
    add_workspace_arg(vet)
    vet.set_defaults(func=_vet)


def _err(lines) -> None:
    for line in lines:
        print(line, file=sys.stderr)


def _guarded(handler):
    def run(args: argparse.Namespace) -> int:
        from easel._common import store
        from easel._common.civitai import NetError
        from easel._common.config import ConfigError
        from easel.registry.shipped import DataError
        from easel.search import SearchError

        try:
            return handler(args)
        except ConfigError as exc:
            _err(exc.lines)
            return exc.code
        except NetError as exc:
            _err(exc.lines)
            return exc.code
        except (SearchError, DataError) as exc:
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


def _config(args: argparse.Namespace):
    from easel._common import config

    return config.load(args.tool, args.workspace, strict=False)


def _limit(value: int) -> int:
    from easel.search import SearchError

    if not 1 <= value <= 100:
        raise SearchError(f"--limit must be 1..100, not {value}")
    return value


@_guarded
def _civitai(args: argparse.Namespace) -> int:
    from easel import search

    cfg = _config(args)
    client, _hf = search.make_clients(cfg)
    page = client.search(args.query, type_=args.type, limit=_limit(args.limit), nsfw=None if args.nsfw else False,
                         base_models=args.base_model, fresh=args.fresh)
    for line in search.civitai_lines(page):
        print(line)
    print(f"{len(page.items)} result{'' if len(page.items) == 1 else 's'} for \"{args.query}\""
          + ("; more exist (raise --limit)" if page.next_cursor else ""))
    if page.items:
        print("next: easel search vet <url>")
    return 0


@_guarded
def _hf(args: argparse.Namespace) -> int:
    from easel import search

    cfg = _config(args)
    _civ, client = search.make_clients(cfg)
    hits = client.search(args.query, limit=_limit(args.limit), filters=search.HF_FILTERS.get(args.type or "", ()),
                         fresh=args.fresh)
    for line in search.hf_lines(hits):
        print(line)
    print(f"{len(hits)} result{'' if len(hits) == 1 else 's'} for \"{args.query}\"")
    if hits:
        print("next: easel search vet <url>")
    return 0


@_guarded
def _vet(args: argparse.Namespace) -> int:
    from easel import search
    from easel.registry import classify
    from easel.registry.shipped import shown

    cfg = _config(args)
    if cfg.file is None:
        from easel.cli import instance_not_found, new_instance_hint

        _err([instance_not_found(), f"vet rows go into <home>/candidates/: {new_instance_hint()}"])
        return 2
    for url in args.urls:  # refuse a bad URL before any request is made
        search.parse_url(url)
    civ, hug = search.make_clients(cfg)
    markers = classify.load_markers(cfg.root)
    limits = search.limits_for(cfg, civ)
    entries = [search.vet(url, civitai_client=civ, hf_client=hug, markers=markers, limits=limits, query=args.query,
                          tool=args.tool) for url in args.urls]
    path = search.append(search.candidates_dir(cfg), entries, tool=args.tool)
    for entry in entries:
        print(search.vet_line(entry))
        for reason in entry["reasons"]:
            print(f"  {reason}")
    print(f"wrote {len(entries)} row{'' if len(entries) == 1 else 's'} to {shown(path, cfg.root)}")
    return 0
