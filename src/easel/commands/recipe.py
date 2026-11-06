"""`easel recipe propose|revise|like|show|list|approve|to-legacy`: the argparse layer over `easel.studio.recipe`.

`propose`, `revise` and `like` write a new `recipes/<rcp-id>.json` (status proposed) and print it; `approve`
records the operator's yes. `to-legacy` (an adopted workspace) prints the host's own command lines; it lives in
`easel.legacy` and is imported only when it runs. Exit codes: 0 ok; 2 a refusal (policy.subjects, no hires
mode, an unknown id or key) or a usage error, with nothing written; 4 the recipes lock is held.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from easel.studio.bridge import register_bridge
from easel.studio.common import EXIT_USAGE, StudioError, add_workspace_arg, guarded, open_studio

HELP = "propose, revise, like, show, list and approve recipes; to-legacy on an adopted instance"
HIRES = ("none", "latent", "model")


def _request_args(parser: argparse.ArgumentParser, *, revise: bool) -> None:
    parser.add_argument("--text", help="the scene, in the operator's words (sfw requests: your plain reading of "
                                       "them); kept verbatim" + ("; default the parent's" if revise else ""))
    if not revise:
        parser.add_argument("--family", metavar="<family>", help="only checkpoints of this family (for example "
                                                                  "illustrious)")
    parser.add_argument("--checkpoint", metavar="<id|name>", help="use this checkpoint (sha16, a unique prefix, or "
                                                                   "a unique part of its file name)")
    parser.add_argument("--lora", action="append", default=[], metavar="<id|name>[:weight]",
                        help="add this LoRA (repeatable); without a weight: its usage card, else the family default")
    parser.add_argument("--size", metavar="<preset|WxH>", help="a preset of the family (square, portrait, landscape, "
                                                               "tall, wide) or WxH")
    parser.add_argument("--count", type=int, metavar="<n>", help="images per run (default 1"
                                                                  + (", or the parent's)" if revise else ")"))
    parser.add_argument("--seed", type=int, metavar="<n>", help="the first seed (default: derived from the text"
                                                                 + (", or the parent's)" if revise else ")"))
    parser.add_argument("--hires", choices=HIRES, help="required: the second pass, none, latent or model (see "
                                                       "docs/WHEN-TO-APPLY.md)" + ("; default the parent's" if revise
                                                                                   else ""))
    parser.add_argument("--upscaler", metavar="<id|name>", help="the upscale model of --hires model (default: the "
                                                                 "only active one)")
    parser.add_argument("--name", help="a short label for the recipe")
    if not revise:
        parser.add_argument("--rating", choices=("sfw", "nsfw"), default=None,
                            help="the content class asked for (default sfw); not sfw means only the operator's "
                                 "words go into the prompt")
    parser.add_argument("--negative-extra", metavar="<text>", help="appended to the negative prompt")
    parser.add_argument("--set", action="append", default=[], metavar="<key>=<value>",
                        help="override one value: sampler, scheduler, steps, cfg, width, height, clip_skip, seed, "
                             "count, denoise, hires.mode, hires.scale, hires.denoise, hires.steps")
    parser.add_argument("--explain", action="store_true", help="also print why each choice was made and the "
                                                               "licence terms of each resource")
    parser.add_argument("--json", action="store_true", help="print the recipe record as JSON")
    add_workspace_arg(parser)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add `recipe` and its actions."""
    parser = subparsers.add_parser(
        "recipe", help=HELP,
        description="Recipes: everything needed to render an image again, with the reason for every choice. "
                    "propose, revise and like write recipes/<rcp-id>.json (status proposed); approve records the "
                    "operator's yes; to-legacy prints the host's lines on an adopted workspace.")
    actions = parser.add_subparsers(dest="action", metavar="<action>", title="actions")

    propose = actions.add_parser(
        "propose", help="propose a recipe for a request and write it (status proposed)",
        description="Assemble a recipe from the registry, the usage cards and the family defaults, and write it. "
                    "--hires is required (none, latent or model). Under policy.subjects a request with an age-down "
                    "or real-person term is refused and nothing is written.")
    _request_args(propose, revise=False)
    propose.add_argument("--from-json", metavar="<file>", help="the request as JSON (keys as the options: text, "
                                                               "family, checkpoint, loras, size, count, seed, hires, "
                                                               "upscaler, name, rating, negative_extra, set)")
    propose.set_defaults(func=_propose)

    revise = actions.add_parser(
        "revise", help="a new recipe from an old one that changes only what you name",
        description="Keep the parent's checkpoint, LoRAs and weights, seed, size and hires mode; change only what "
                    "is named, so the next render compares one change. Writes a new recipe with parent set.")
    revise.add_argument("recipe", metavar="<rcp>", help="the parent recipe id")
    _request_args(revise, revise=True)
    revise.add_argument("--lora-weight", action="append", default=[], metavar="<lora id>=<weight>",
                        help="change the weight of one of the parent's LoRAs")
    revise.add_argument("--drop-lora", action="append", default=[], metavar="<lora id>",
                        help="remove one of the parent's LoRAs")
    revise.set_defaults(func=_revise)

    like = actions.add_parser(
        "like", help="a new recipe that repeats a run (with one output's seed)",
        description="Propose the recipe of a run again, with the seed of one output (--output) or the run's, so "
                    "more images like a kept one can be rendered. Writes a new recipe with parent set.")
    like.add_argument("run", metavar="<run-id>", help="the run to repeat")
    like.add_argument("--output", type=int, metavar="<n>", help="take this output's seed")
    like.add_argument("--count", type=int, metavar="<n>", help="images per run (default the parent's)")
    like.add_argument("--seed", type=int, metavar="<n>", help="the first seed (default the output's or the run's)")
    like.add_argument("--explain", action="store_true", help="also print why each choice was made")
    like.add_argument("--json", action="store_true", help="print the recipe record as JSON")
    add_workspace_arg(like)
    like.set_defaults(func=_like)

    show = actions.add_parser("show", help="print one recipe; --explain adds the why and the licences",
                              description="Print one recipe. The prompt of a recipe that is not sfw is not echoed.")
    show.add_argument("recipe", metavar="<rcp>", help="the recipe id")
    show.add_argument("--explain", action="store_true", help="also print why each choice was made and the licences")
    show.add_argument("--json", action="store_true", help="print the record as JSON")
    add_workspace_arg(show)
    show.set_defaults(func=_show)

    lst = actions.add_parser("list", help="one line per recipe: id, status, family, content class, name",
                             description="One line per recipe, newest first.")
    lst.add_argument("--status", choices=("proposed", "approved", "run", "archived"), help="only this status")
    lst.add_argument("--limit", type=int, default=20, metavar="<n>", help="at most this many lines (default 20)")
    lst.add_argument("--json", action="store_true", help="a JSON list of the records")
    add_workspace_arg(lst)
    lst.set_defaults(func=_list)

    approve = actions.add_parser("approve", help="record the operator's yes for a recipe (after they said it)",
                                 description="Set a proposed recipe to approved. Only after the operator's yes for "
                                             "that recipe; it is checked against policy.subjects once more.")
    approve.add_argument("recipe", metavar="<rcp>", help="the recipe id")
    add_workspace_arg(approve)
    approve.set_defaults(func=_approve)

    register_bridge(actions, "to-legacy", "add_to_legacy_args", "run_to_legacy",
                     "print the host's propose, dry-run, submit and status lines for a recipe (adopted workspace)")


# --- handlers -------------------------------------------------------------------------------------------------------

def _print_recipe(recipe: dict, records: dict, *, explain: bool, as_json: bool, created: bool) -> None:
    from easel.studio import recipe as recipes

    if as_json:
        print(json.dumps(recipe, indent=2, ensure_ascii=False))
        return
    for line in recipes.describe(recipe, records, explain=explain):
        print(line)
    if created:
        print(f"next: show the operator this recipe; on their yes: easel recipe approve {recipe['id']}")


def _from_json(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        from easel.registry.shipped import reason

        raise StudioError(EXIT_USAGE, [f"--from-json {Path(path).name}: not readable JSON ({reason(exc)})"]) from exc
    if not isinstance(data, dict):
        raise StudioError(EXIT_USAGE, [f"--from-json {path}: the request is a JSON object"])
    return data


@guarded
def _propose(args: argparse.Namespace) -> int:
    from easel.studio import recipe as recipes

    studio = open_studio(args.tool, args.workspace)
    asked = _from_json(args.from_json) if args.from_json else {}
    sets = recipes.parse_sets([f"{k}={v if isinstance(v, str) else json.dumps(v)}"
                               for k, v in dict(asked.get("set") or {}).items()])
    sets.update(recipes.parse_sets(args.set))
    text = args.text if args.text is not None else asked.get("text")
    if not text or not str(text).strip():
        raise StudioError(EXIT_USAGE, ["--text is required: the scene, in the operator's words"])
    req = recipes.Request(
        text=str(text), family=args.family or asked.get("family"), checkpoint=args.checkpoint or asked.get("checkpoint"),
        loras=list(args.lora) or [str(x) for x in asked.get("loras") or []], size=args.size or asked.get("size"),
        count=args.count if args.count is not None else asked.get("count"),
        seed=args.seed if args.seed is not None else asked.get("seed"), hires=args.hires or asked.get("hires"),
        upscaler=args.upscaler or asked.get("upscaler"), name=args.name or asked.get("name"),
        rating=args.rating or asked.get("rating") or "sfw",
        negative_extra=args.negative_extra or asked.get("negative_extra"), sets=sets)
    recipe, ctx = recipes.propose(studio, req)
    _print_recipe(recipe, ctx.records, explain=args.explain, as_json=args.json, created=True)
    return 0


@guarded
def _revise(args: argparse.Namespace) -> int:
    from easel.studio import recipe as recipes

    studio = open_studio(args.tool, args.workspace)
    parent = recipes.load(studio, args.recipe)
    req = recipes.revise_request(parent, text=args.text, checkpoint=args.checkpoint, loras=args.lora,
                                 lora_weights=args.lora_weight, drop=args.drop_lora, seed=args.seed, size=args.size,
                                 count=args.count, hires=args.hires, upscaler=args.upscaler, name=args.name,
                                 negative_extra=args.negative_extra, sets=recipes.parse_sets(args.set))
    recipe, ctx = recipes.propose(studio, req)
    _print_recipe(recipe, ctx.records, explain=args.explain, as_json=args.json, created=True)
    return 0


@guarded
def _like(args: argparse.Namespace) -> int:
    from easel.studio import ledger
    from easel.studio import recipe as recipes

    studio = open_studio(args.tool, args.workspace)
    run = ledger.load_run(studio, args.run)
    if not run.get("recipe"):
        raise StudioError(EXIT_USAGE, [f"{args.run} has no recipe to repeat (an imported run?)"])
    parent = recipes.load(studio, str(run["recipe"]))
    seed = args.seed
    if seed is None and args.output is not None:
        match = [o for o in run.get("outputs") or [] if int(o["n"]) == args.output]
        if not match:
            raise StudioError(EXIT_USAGE, [f"--output {args.output}: {args.run} has no such output "
                                           f"(easel runs show {args.run})"])
        seed = match[0].get("seed")
    if seed is None:
        seed = run.get("seed")
    req = recipes.revise_request(parent, seed=seed, count=args.count, kind="like", from_run=args.run,
                                 from_output=args.output)
    recipe, ctx = recipes.propose(studio, req)
    _print_recipe(recipe, ctx.records, explain=args.explain, as_json=args.json, created=True)
    return 0


@guarded
def _show(args: argparse.Namespace) -> int:
    from easel.studio import recipe as recipes

    studio = open_studio(args.tool, args.workspace)
    recipe = recipes.load(studio, args.recipe)
    records = {} if args.json else recipes.load_context(studio).records
    _print_recipe(recipe, records, explain=args.explain, as_json=args.json, created=False)
    return 0


@guarded
def _list(args: argparse.Namespace) -> int:
    from easel.studio import recipe as recipes

    studio = open_studio(args.tool, args.workspace)
    rows = [r for r in recipes.all_recipes(studio) if not args.status or r.get("status") == args.status]
    rows = rows[: max(0, args.limit)]
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for row in rows:
        print(recipes.list_line(row))
    if not rows:
        print("no recipes" + (f" with status {args.status}" if args.status else ""))
    return 0


@guarded
def _approve(args: argparse.Namespace) -> int:
    from easel.studio import recipe as recipes

    studio = open_studio(args.tool, args.workspace)
    recipe, line = recipes.approve(studio, args.recipe)
    print(line)
    if studio.legacy_runner:
        print(f"next: easel recipe to-legacy {recipe['id']}")
    else:
        print(f"next: easel run {recipe['id']} --dry-run, then easel jobs submit {recipe['id']} --count "
              f"{(recipe.get('params') or {}).get('count', 1)}")
    return 0
