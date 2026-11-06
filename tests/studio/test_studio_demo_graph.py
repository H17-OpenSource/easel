"""The demo run's graph.json is what the compiler makes of the demo recipe, and the manifest hashes that file.

`easel run` records the first prompt's graph (file prefix `easel/<run-id>/001`). If the compiler changes (a new
node such as CLIPSetLastLayer for clip_skip 2, or the `_meta` titles), the demo must be regenerated with it,
or `easel run --dry-run` and the demo disagree and a weaker model reads two different graphs for one recipe.
"""

from __future__ import annotations

import json

from studio_support import RECIPE, RUN


def test_the_demo_graph_is_the_compilers_output(demo_workspace, frozen_clock):
    from easel._common import hashing
    from easel.studio import common, render
    from easel.studio import recipe as recipes

    studio = common.open_studio("easel")
    plan = render.make_plan(studio, recipes.load(studio, RECIPE))
    compiled = render.compile_one(plan, plan.seeds[0], f"easel/{RUN}/001")
    folder = demo_workspace / "runs" / RUN
    text = (folder / "graph.json").read_bytes().decode("utf-8")
    assert json.loads(text) == compiled
    assert text == common.dump(compiled) + "\n"  # byte for byte as the renderer writes it
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["graph_sha256"] == hashing.sha256_text(text)
    assert manifest["seeds"] == plan.seeds
    classes = {node["class_type"] for node in compiled.values()}
    assert "CLIPSetLastLayer" in classes  # the demo recipe sets clip_skip 2
    assert all(node.get("_meta", {}).get("title") for node in compiled.values())
