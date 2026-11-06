"""_common.comfy_graph, node by node, with the family defaults from the shipped families/ and the demo's.

The server side is the conftest `mock_comfy`: its /object_info answers are built here from input-name tuples
only (a placeholder type for every input, the model-name lists for the loaders); no text from any node pack.
"""

import json
from pathlib import Path

import pytest

from easel._common import comfy_client as cc
from easel._common import comfy_graph as cg

REPO = Path(__file__).resolve().parents[2]
FAMILY_DIRS = [REPO / "examples" / "demo" / "families", REPO / "families"]
CKPT = "illustrious/demo-anime-xl.safetensors"
LORA = "demo_robot-v1.safetensors"
STYLE = "styles/demo-style.safetensors"
VAE = "demo-vae.safetensors"
UPSCALER = "4x-demo-upscaler.pth"
PREFIX = "easel/run-20300101-001"

# Input names per node class: (required, optional, hidden). ComfyUI core nodes first, then the face pass's.
NODE_INPUTS = {
    "CheckpointLoaderSimple": (("ckpt_name",), (), ()),
    "VAELoader": (("vae_name",), (), ()),
    "CLIPSetLastLayer": (("clip", "stop_at_clip_layer"), (), ()),
    "LoraLoader": (("model", "clip", "lora_name", "strength_model", "strength_clip"), (), ()),
    "LoraLoaderModelOnly": (("model", "lora_name", "strength_model"), (), ()),
    "CLIPTextEncode": (("text", "clip"), (), ()),
    "EmptyLatentImage": (("width", "height", "batch_size"), (), ()),
    "KSampler": (("model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative",
                  "latent_image", "denoise"), (), ()),
    "VAEDecode": (("samples", "vae"), (), ()),
    "VAEEncode": (("pixels", "vae"), (), ()),
    "SaveImage": (("images", "filename_prefix"), (), ("prompt", "extra_pnginfo")),
    "LatentUpscaleBy": (("samples", "upscale_method", "scale_by"), (), ()),
    "UpscaleModelLoader": (("model_name",), (), ()),
    "ImageUpscaleWithModel": (("upscale_model", "image"), (), ()),
    "ImageScale": (("image", "upscale_method", "width", "height", "crop"), (), ()),
    "ImageScaleBy": (("image", "upscale_method", "scale_by"), (), ()),
    "LoadImage": (("image",), (), ()),
    "RepeatLatentBatch": (("samples", "amount"), (), ()),
    "UNETLoader": (("unet_name", "weight_dtype"), (), ()),
    "CLIPLoader": (("clip_name", "type"), ("device",), ()),
    "EmptyFlux2LatentImage": (("width", "height", "batch_size"), (), ()),
    "UltralyticsDetectorProvider": (("model_name",), (), ()),
    "SAMLoader": (("model_name", "device_mode"), (), ()),
    "FaceDetailer": (cg.FACE_DETAILER_REQUIRED, cg.FACE_DETAILER_OPTIONAL, ()),
}
# What the server lists for the model-name inputs (its own spelling: forward slashes).
SERVER_NAMES = {
    ("CheckpointLoaderSimple", "ckpt_name"): [CKPT, "demo-base-xl.safetensors"],
    ("LoraLoader", "lora_name"): [LORA, STYLE],
    ("LoraLoaderModelOnly", "lora_name"): [LORA],
    ("VAELoader", "vae_name"): [VAE, "moss/demo-moss-vae.safetensors"],
    ("UpscaleModelLoader", "model_name"): [UPSCALER],
    ("UNETLoader", "unet_name"): ["moss/demo-moss.safetensors"],
    ("CLIPLoader", "clip_name"): ["moss/demo-moss-encoder.safetensors"],
    ("UltralyticsDetectorProvider", "model_name"): ["bbox/face_yolov8m.pt", "segm/face_yolov8m.pt"],
    ("SAMLoader", "model_name"): ["sam_demo.pth"],
}


def info_entry(cls, table=NODE_INPUTS):
    required, optional, hidden = table[cls]

    def spec(name):
        names = SERVER_NAMES.get((cls, name))
        return [list(names)] if names is not None else ["*"]

    return {"input": {"required": {n: spec(n) for n in required}, "optional": {n: spec(n) for n in optional},
                      "hidden": {n: ["*"] for n in hidden}},
            "input_order": {"required": list(required), "optional": list(optional)}}


def object_info(table=NODE_INPUTS, drop=()):
    return {cls: info_entry(cls, table) for cls in table if cls not in drop}


def serve(mock, table=NODE_INPUTS, drop=()):
    """Answer /object_info/<class> on the mock for every class of `table` but `drop` (unknown ones answer {})."""
    for cls in table:
        if cls not in drop:
            mock.routes[("GET", f"/object_info/{cls}")] = (200, {cls: info_entry(cls, table)})


@pytest.fixture(scope="module")
def illustrious():
    return cg.inference_defaults("illustrious", FAMILY_DIRS)


def spec(**over):
    """The demo recipe as a resolved spec (file names instead of resource ids)."""
    base = {"task": "txt2img", "checkpoint": CKPT,
            "loras": [{"name": LORA, "weight": 0.8, "slot": "lora.1"}],
            "prompt": "dmrbt, robot, standing in a tidy workshop", "negative": "lowres, blurry",
            "params": {"sampler": "euler_ancestral", "scheduler": "normal", "steps": 28, "cfg": 5.5, "width": 832,
                       "height": 1216, "clip_skip": 2, "seed": 1001, "hires": {"mode": "none"}}}
    params = over.pop("params", {})
    base.update(over)
    base["params"] = {**base["params"], **params}
    return base


def classes(graph):
    return [node["class_type"] for node in graph.values()]


def by_title(graph, title):
    hits = [nid for nid, node in graph.items() if (node.get("_meta") or {}).get("title") == title]
    assert len(hits) == 1, f"{title!r}: {hits}"
    return hits[0]


def node(graph, title):
    return graph[by_title(graph, title)]


# --- the family's defaults ----------------------------------------------------------------------------------------

def test_a_demo_family_inherits_its_parents_inference_block():
    merged = cg.inference_defaults("demo_merge", FAMILY_DIRS)
    assert merged == cg.inference_defaults("illustrious", FAMILY_DIRS)
    assert merged["graph_compiler"] == "sdxl_txt2img" and merged["clip_skip"] == 2


def test_inheritance_is_a_deep_merge_where_the_child_wins(tmp_path):
    (tmp_path / "parent.yaml").write_text("schema: easel/family/1\nid: parent\ninherits: null\ninference:\n"
                                          "  sampler: euler\n  steps: 20\n  hires: {scale: 1.5, steps: 10}\n")
    (tmp_path / "child.yaml").write_text("schema: easel/family/1\nid: child\ninherits: parent\ninference:\n"
                                         "  steps: 30\n  hires: {steps: 12}\n")
    assert cg.inference_defaults("child", [tmp_path]) == {"sampler": "euler", "steps": 30,
                                                           "hires": {"scale": 1.5, "steps": 12}}


def test_a_missing_family_and_a_cycle_are_errors(tmp_path):
    with pytest.raises(cg.GraphError, match="no nosuch.yaml"):
        cg.inference_defaults("nosuch", [tmp_path])
    (tmp_path / "a.yaml").write_text("id: a\ninherits: b\ninference: {}\n")
    (tmp_path / "b.yaml").write_text("id: b\ninherits: a\ninference: {}\n")
    with pytest.raises(cg.GraphError, match="cycle"):
        cg.inference_defaults("a", [tmp_path])


def test_unset_params_come_from_the_family(illustrious):
    graph = cg.compile_graph({"checkpoint": CKPT, "prompt": "dmrbt", "negative": None,
                              "params": {"seed": 7, "clip_skip": None, "hires": {"mode": "none"}}}, inference=illustrious)
    ks = node(graph, "sampler")["inputs"]
    assert (ks["sampler_name"], ks["scheduler"], ks["steps"], ks["cfg"]) == ("euler_ancestral", "normal", 28, 5.5)
    assert node(graph, "latent")["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
    assert node(graph, "clip skip 2")["inputs"]["stop_at_clip_layer"] == -2
    assert node(graph, "negative")["inputs"]["text"] == illustrious["negative_baseline"]


def test_every_shipped_family_with_a_compiler_compiles():
    for path in sorted((REPO / "families").glob("*.yaml")):
        inference = cg.inference_defaults(path.stem, FAMILY_DIRS)
        name = inference.get("graph_compiler")
        base = {"checkpoint": CKPT, "prompt": "dmrbt", "negative": None, "params": {"seed": 1, "hires": {"mode": "none"}}}
        if name is None:
            with pytest.raises(cg.GraphError, match="graph_compiler: null"):
                cg.compile_graph(base, inference=inference)
            continue
        if name == "moss_txt2img":
            base.update(text_encoder="moss/demo-moss-encoder.safetensors", vae="moss/demo-moss-vae.safetensors")
        graph = cg.compile_graph(base, inference=inference)
        assert classes(graph)[-1] == "SaveImage", path.name
        assert cg.check(graph, object_info()) == [], path.name


# --- txt2img node by node -------------------------------------------------------------------------------------------

def test_txt2img_node_by_node(illustrious):
    graph = cg.compile_graph(spec(), inference=illustrious, filename_prefix=PREFIX)
    assert classes(graph) == ["CheckpointLoaderSimple", "CLIPSetLastLayer", "LoraLoader", "CLIPTextEncode",
                              "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"]
    assert list(graph) == [str(i) for i in range(1, 10)]
    assert graph["1"]["inputs"] == {"ckpt_name": CKPT}
    assert graph["2"]["inputs"] == {"stop_at_clip_layer": -2, "clip": ["1", 1]}
    assert graph["3"] == {"class_type": "LoraLoader", "_meta": {"title": "lora.1"},
                          "inputs": {"lora_name": LORA, "strength_model": 0.8, "strength_clip": 0.8,
                                     "model": ["1", 0], "clip": ["2", 0]}}
    assert graph["4"]["inputs"] == {"text": "dmrbt, robot, standing in a tidy workshop", "clip": ["3", 1]}
    assert graph["5"]["inputs"] == {"text": "lowres, blurry", "clip": ["3", 1]}
    assert graph["6"]["inputs"] == {"width": 832, "height": 1216, "batch_size": 1}
    assert graph["7"]["inputs"] == {"seed": 1001, "steps": 28, "cfg": 5.5, "sampler_name": "euler_ancestral",
                                    "scheduler": "normal", "denoise": 1.0, "model": ["3", 0], "positive": ["4", 0],
                                    "negative": ["5", 0], "latent_image": ["6", 0]}
    assert graph["8"]["inputs"] == {"samples": ["7", 0], "vae": ["1", 2]}
    assert graph["9"]["inputs"] == {"filename_prefix": PREFIX, "images": ["8", 0]}
    assert cg.check(graph, object_info()) == []
    assert cg.save_nodes(graph) == {"base": "9"}


def test_the_graph_is_plain_json_and_deterministic(illustrious):
    one = json.dumps(cg.compile_graph(spec(), inference=illustrious), sort_keys=True)
    assert one == json.dumps(cg.compile_graph(spec(), inference=illustrious), sort_keys=True)


def test_clip_skip_one_adds_no_node(illustrious):
    graph = cg.compile_graph(spec(params={"clip_skip": 1}), inference=illustrious)
    assert "CLIPSetLastLayer" not in classes(graph)
    assert node(graph, "lora.1")["inputs"]["clip"] == [by_title(graph, "checkpoint"), 1]


def test_a_lora_chain_links_each_lora_to_the_previous_one(illustrious):
    loras = [{"name": LORA, "weight": 0.8, "slot": "lora.1"}, {"name": STYLE, "weight": None, "clip_weight": 0.3}]
    graph = cg.compile_graph(spec(loras=loras, params={"clip_skip": 1}), inference=illustrious)
    first, second = node(graph, "lora.1"), node(graph, "lora.2")
    assert second["inputs"]["model"] == [by_title(graph, "lora.1"), 0]
    assert second["inputs"]["clip"] == [by_title(graph, "lora.1"), 1]
    assert (second["inputs"]["strength_model"], second["inputs"]["strength_clip"]) == (0.8, 0.3)  # family default
    assert first["inputs"]["strength_clip"] == 0.8
    assert node(graph, "positive")["inputs"]["clip"] == [by_title(graph, "lora.2"), 1]
    assert node(graph, "sampler")["inputs"]["model"] == [by_title(graph, "lora.2"), 0]


def test_a_lora_without_any_weight_is_an_error():
    with pytest.raises(cg.GraphError, match="no weight"):
        cg.compile_graph(spec(loras=[{"name": LORA}]), inference={"graph_compiler": "sdxl_txt2img"})


def test_a_vae_replaces_the_checkpoints_for_decoding(illustrious):
    graph = cg.compile_graph(spec(vae=VAE), inference=illustrious)
    assert node(graph, "vae") == {"class_type": "VAELoader", "inputs": {"vae_name": VAE}, "_meta": {"title": "vae"}}
    assert node(graph, "decode")["inputs"]["vae"] == [by_title(graph, "vae"), 0]


def test_embeddings_go_into_the_prompts_as_text(illustrious):
    embeddings = [{"name": "demo_good.safetensors", "target": "positive"},
                  {"name": "negatives\\demo_bad", "weight": 1.2, "target": "negative"}]
    graph = cg.compile_graph(spec(embeddings=embeddings), inference=illustrious)
    assert node(graph, "positive")["inputs"]["text"].endswith(", embedding:demo_good.safetensors")
    assert node(graph, "negative")["inputs"]["text"] == "lowres, blurry, (embedding:negatives/demo_bad:1.2)"
    assert "embedding" not in " ".join(classes(graph)).lower()  # no loader node


@pytest.mark.parametrize("name", ["demo bad", "demo(1)", "a,b", ""])
def test_an_embedding_name_that_cannot_be_written_into_a_prompt_is_refused(name):
    with pytest.raises(cg.GraphError, match="cannot go into a prompt"):
        cg.embedding_token(name)


def test_the_seed_override_and_a_missing_seed(illustrious):
    graph = cg.compile_graph(spec(), inference=illustrious, seed=1003)
    assert node(graph, "sampler")["inputs"]["seed"] == 1003
    with pytest.raises(cg.GraphError, match="no seed"):
        cg.compile_graph(spec(params={"seed": None}), inference=illustrious)
    with pytest.raises(cg.GraphError, match="whole number"):
        cg.compile_graph(spec(), inference=illustrious, seed=-1)


@pytest.mark.parametrize("bad", [{"width": 1004}, {"height": 32}])
def test_a_size_off_the_latent_grid_is_refused(illustrious, bad):
    with pytest.raises(cg.GraphError, match="multiple of 8"):
        cg.compile_graph(spec(params=bad), inference=illustrious)


def test_txt2img_refuses_a_denoise_below_one(illustrious):
    with pytest.raises(cg.GraphError, match="applies to img2img"):
        cg.compile_graph(spec(params={"denoise": 0.5}), inference=illustrious)
    assert cg.compile_graph(spec(params={"denoise": 1.0}), inference=illustrious)


# --- hires ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("hires", [None, {}, {"mode": "auto"}, {"scale": 1.5}])
def test_hires_mode_is_required(illustrious, hires):
    with pytest.raises(cg.GraphError, match="params.hires.mode is required"):
        cg.compile_graph(spec(params={"hires": hires}), inference=illustrious)


def test_latent_hires_node_by_node(illustrious):
    graph = cg.compile_graph(spec(params={"hires": {"mode": "latent"}}), inference=illustrious, filename_prefix=PREFIX)
    assert classes(graph)[-5:] == ["KSampler", "LatentUpscaleBy", "KSampler", "VAEDecode", "SaveImage"]
    up = node(graph, "hires upscale")["inputs"]
    assert up == {"upscale_method": "nearest-exact", "scale_by": 1.5, "samples": [by_title(graph, "sampler"), 0]}
    second = node(graph, "hires sampler")["inputs"]
    assert (second["steps"], second["denoise"], second["seed"]) == (15, 0.35, 1001)  # family hires defaults
    assert second["latent_image"] == [by_title(graph, "hires upscale"), 0]
    assert second["positive"] == node(graph, "sampler")["inputs"]["positive"]
    assert node(graph, "hires decode")["inputs"]["samples"] == [by_title(graph, "hires sampler"), 0]
    assert cg.save_nodes(graph) == {"hires": by_title(graph, "save hires")}
    assert node(graph, "save hires")["inputs"]["filename_prefix"] == PREFIX
    assert cg.check(graph, object_info()) == []


def test_model_hires_node_by_node(illustrious):
    hires = {"mode": "model", "upscaler": UPSCALER, "scale": 1.5, "denoise": 0.3, "steps": 12}
    graph = cg.compile_graph(spec(params={"hires": hires}), inference=illustrious)
    assert classes(graph)[6:] == ["KSampler", "VAEDecode", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScale",
                                  "VAEEncode", "KSampler", "VAEDecode", "SaveImage"]
    assert node(graph, "upscale model")["inputs"] == {"model_name": UPSCALER}
    assert node(graph, "model upscale")["inputs"] == {"upscale_model": [by_title(graph, "upscale model"), 0],
                                                      "image": [by_title(graph, "decode"), 0]}
    scaled = node(graph, "to x1.5")["inputs"]
    assert (scaled["width"], scaled["height"], scaled["upscale_method"], scaled["crop"]) == (1248, 1824, "lanczos", "disabled")
    assert node(graph, "hires encode")["inputs"]["pixels"] == [by_title(graph, "to x1.5"), 0]
    second = node(graph, "hires sampler")["inputs"]
    assert (second["steps"], second["denoise"]) == (12, 0.3)
    assert cg.check(graph, object_info()) == []


def test_model_hires_needs_an_upscaler(illustrious):
    with pytest.raises(cg.GraphError, match="upscaler names no upscale model"):
        cg.compile_graph(spec(params={"hires": {"mode": "model"}}), inference=illustrious)


def test_a_hires_value_neither_spec_nor_family_gives_is_an_error():
    inference = {"graph_compiler": "sdxl_txt2img", "hires": None}
    with pytest.raises(cg.GraphError, match="params.hires.scale is not set"):
        cg.compile_graph(spec(params={"hires": {"mode": "latent"}}), inference=inference)


def test_hires_size_rounds_down_to_the_grid():
    assert cg.hires_size(832, 1216, 1.5) == (1248, 1824)
    assert cg.hires_size(1000, 1000, 1.3, multiple=16) == (1296, 1296)


def test_saving_several_stages_suffixes_the_prefix(illustrious):
    hires = {"mode": "latent"}
    graph = cg.compile_graph(spec(params={"hires": hires}, save=["base", "hires"]), inference=illustrious,
                             filename_prefix=PREFIX)
    saves = cg.save_nodes(graph)
    assert set(saves) == {"base", "hires"}
    assert graph[saves["base"]]["inputs"]["filename_prefix"] == f"{PREFIX}_base"
    assert graph[saves["hires"]]["inputs"]["filename_prefix"] == f"{PREFIX}_hires"
    with pytest.raises(cg.GraphError, match="save 'hires' needs a second pass"):
        cg.compile_graph(spec(save=["hires"]), inference=illustrious)
    with pytest.raises(cg.GraphError, match="unknown stage"):
        cg.compile_graph(spec(save=["final"]), inference=illustrious)


def test_the_graph_stops_at_the_last_stage_asked_for(illustrious):
    graph = cg.compile_graph(spec(params={"hires": {"mode": "latent"}}, save=["base"]), inference=illustrious)
    assert "LatentUpscaleBy" not in classes(graph)


# --- img2img and upscale -----------------------------------------------------------------------------------------------

def test_img2img_node_by_node(illustrious):
    graph = cg.compile_graph(spec(task="img2img", init_image="init.png", params={"batch": 2}), inference=illustrious)
    assert "EmptyLatentImage" not in classes(graph)
    assert node(graph, "init image")["inputs"] == {"image": "init.png"}
    assert node(graph, "fit to size")["inputs"] == {"upscale_method": "lanczos", "width": 832, "height": 1216,
                                                    "crop": "disabled", "image": [by_title(graph, "init image"), 0]}
    assert node(graph, "encode")["inputs"] == {"pixels": [by_title(graph, "fit to size"), 0], "vae": ["1", 2]}
    assert node(graph, "batch")["inputs"] == {"samples": [by_title(graph, "encode"), 0], "amount": 2}
    ks = node(graph, "sampler")["inputs"]
    assert ks["latent_image"] == [by_title(graph, "batch"), 0] and ks["denoise"] == cg.IMG2IMG_DENOISE
    assert cg.check(graph, object_info()) == []
    given = cg.compile_graph(spec(task="img2img", init_image="init.png", params={"denoise": 0.4}), inference=illustrious)
    assert node(given, "sampler")["inputs"]["denoise"] == 0.4 and "RepeatLatentBatch" not in classes(given)


def test_img2img_needs_the_uploaded_image(illustrious):
    with pytest.raises(cg.GraphError, match="init_image"):
        cg.compile_graph(spec(task="img2img"), inference=illustrious)


def test_upscale_node_by_node():
    graph = cg.compile_upscale("keep.png", UPSCALER, 2.0, filename_prefix=PREFIX)
    assert classes(graph) == ["LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "ImageScaleBy", "SaveImage"]
    assert graph["3"]["inputs"] == {"upscale_model": ["2", 0], "image": ["1", 0]}
    assert graph["4"]["inputs"] == {"upscale_method": "lanczos", "scale_by": 0.5, "image": ["3", 0]}
    assert graph["5"]["inputs"] == {"filename_prefix": PREFIX, "images": ["4", 0]}
    assert cg.check(graph, object_info()) == []
    native = cg.compile_graph({"task": "upscale", "image": "keep.png", "upscaler": UPSCALER})
    assert "ImageScaleBy" not in classes(native)


@pytest.mark.parametrize(("name", "factor"), [("4x-demo.pth", 4), ("demo_x2.safetensors", 2),
                                              ("upscalers/2x_demo.pth", 2), ("demo.pth", 4), ("demo-v12x.pth", 4)])
def test_upscale_factor_from_the_file_name(name, factor):
    assert cg.upscale_factor(name) == factor


def test_tasks_without_a_compiler(illustrious):
    with pytest.raises(cg.GraphError, match="workflow template"):
        cg.compile_graph(spec(task="inpaint"), inference=illustrious)
    with pytest.raises(cg.GraphError, match="no graph compiler"):
        cg.compile_graph(spec())
    with pytest.raises(cg.GraphError, match="unknown graph compiler"):
        cg.compile_graph(spec(compiler="nosuch"), inference=illustrious)


# --- the face pass ------------------------------------------------------------------------------------------------

def test_face_pass_offline_uses_the_constant_input_names(illustrious):
    graph = cg.compile_graph(spec(detailer={"denoise": 0.3}), inference=illustrious, filename_prefix=PREFIX)
    assert classes(graph)[-3:] == ["UltralyticsDetectorProvider", "FaceDetailer", "SaveImage"]
    assert node(graph, "face detector")["inputs"] == {"model_name": "bbox/face_yolov8m.pt"}
    fd = node(graph, "face pass")["inputs"]
    assert set(cg.FACE_DETAILER_REQUIRED) <= set(fd)
    assert fd["image"] == [by_title(graph, "decode"), 0] and fd["bbox_detector"] == [by_title(graph, "face detector"), 0]
    assert fd["model"] == [by_title(graph, "lora.1"), 0] and fd["clip"] == [by_title(graph, "lora.1"), 1]
    assert (fd["denoise"], fd["seed"], fd["sampler_name"], fd["cfg"]) == (0.3, 1001, "euler_ancestral", 5.5)
    assert "sam_model_opt" not in fd
    assert cg.save_nodes(graph) == {"detailed": by_title(graph, "save detailed")}
    assert node(graph, "save detailed")["inputs"]["filename_prefix"] == PREFIX


def test_face_pass_after_a_model_hires_pass_with_sam(illustrious):
    hires = {"mode": "model", "upscaler": UPSCALER}
    graph = cg.compile_graph(spec(params={"hires": hires}, detailer={"sam": "sam_demo.pth", "detector": "face_yolov8m.pt"},
                                  save=["base", "detailed"]), inference=illustrious, filename_prefix=PREFIX)
    fd = node(graph, "face pass")["inputs"]
    assert fd["image"] == [by_title(graph, "hires decode"), 0]
    assert fd["sam_model_opt"] == [by_title(graph, "sam"), 0]
    assert node(graph, "sam")["inputs"] == {"model_name": "sam_demo.pth", "device_mode": "AUTO"}
    assert set(cg.save_nodes(graph)) == {"base", "detailed"}
    assert cg.check(graph, object_info()) == []


def test_unknown_detailer_settings_are_refused(illustrious):
    with pytest.raises(cg.GraphError, match="unknown setting"):
        cg.compile_graph(spec(detailer={"strength": 1}), inference=illustrious)


def live_detailer(mock, illustrious, table, detailer=None):
    """Compile offline, fetch the graph's classes from the mock, compile again with the server's inputs."""
    serve(mock, table)
    client = cc.Client(mock.url, timeout=5)
    offline = cg.compile_graph(spec(detailer=detailer or {}), inference=illustrious)
    info = client.object_info_for(cg.node_classes(offline))
    return cg.compile_graph(spec(detailer=detailer or {}), inference=illustrious, object_info=info), info


def test_face_pass_reads_its_inputs_from_the_server(mock_comfy, no_network, illustrious):
    graph, info = live_detailer(mock_comfy, illustrious, NODE_INPUTS)
    assert set(info) == set(cg.node_classes(graph))
    assert set(node(graph, "face pass")["inputs"]) == set(cg.FACE_DETAILER_REQUIRED) | {"inpaint_model", "noise_mask_feather"}
    assert cg.resolve_names(graph, info) == [] and cg.check(graph, info) == []
    asked = {p for _, p, _ in mock_comfy.requests}
    assert "/object_info/FaceDetailer" in asked and "/object_info" not in asked  # only the classes in use


def test_a_default_the_server_does_not_offer_is_dropped(mock_comfy, no_network, illustrious):
    table = dict(NODE_INPUTS, FaceDetailer=(cg.FACE_DETAILER_REQUIRED, ("sam_model_opt", "inpaint_model"), ()))
    graph, _ = live_detailer(mock_comfy, illustrious, table)
    assert "noise_mask_feather" not in node(graph, "face pass")["inputs"]


def test_a_setting_the_server_does_not_offer_is_refused(mock_comfy, no_network, illustrious):
    table = dict(NODE_INPUTS, FaceDetailer=(cg.FACE_DETAILER_REQUIRED, ("sam_model_opt", "inpaint_model"), ()))
    with pytest.raises(cg.GraphError, match="'noise_mask_feather' has no FaceDetailer input"):
        live_detailer(mock_comfy, illustrious, table, detailer={"noise_mask_feather": 10})


def test_a_new_required_input_is_named(mock_comfy, no_network, illustrious):
    table = dict(NODE_INPUTS, FaceDetailer=(cg.FACE_DETAILER_REQUIRED + ("demo_new_input",), cg.FACE_DETAILER_OPTIONAL, ()))
    with pytest.raises(cg.GraphError, match="demo_new_input"):
        live_detailer(mock_comfy, illustrious, table)


@pytest.mark.parametrize("missing", ["FaceDetailer", "UltralyticsDetectorProvider"])
def test_without_the_node_packs_the_face_pass_is_unavailable(illustrious, missing):
    with pytest.raises(cg.GraphError) as err:
        cg.compile_graph(spec(detailer={}), inference=illustrious, object_info=object_info(drop=(missing,)))
    assert f"has no {missing} node" in err.value.lines[0]
    assert "ComfyUI-Impact-Pack (GPL-3.0)" in err.value.lines[1] and "AGPL-3.0" in err.value.lines[1]


# --- MOSS-Image (experimental) ------------------------------------------------------------------------------------------

def moss_spec(**over):
    base = {"checkpoint": "moss/demo-moss.safetensors", "text_encoder": "moss/demo-moss-encoder.safetensors",
            "vae": "moss/demo-moss-vae.safetensors", "loras": [{"name": LORA, "weight": 1.0}],
            "prompt": "a demo robot in a workshop", "negative": None,
            "params": {"seed": 5, "width": 1024, "height": 1024, "hires": {"mode": "none"}}}
    base.update(over)
    return base


@pytest.fixture(scope="module")
def moss():
    return cg.inference_defaults("moss", FAMILY_DIRS)


def test_moss_node_by_node(moss):
    graph = cg.compile_graph(moss_spec(), inference=moss, filename_prefix=PREFIX)
    assert classes(graph) == ["UNETLoader", "LoraLoaderModelOnly", "CLIPLoader", "CLIPTextEncode", "CLIPTextEncode",
                              "EmptyFlux2LatentImage", "KSampler", "VAELoader", "VAEDecode", "SaveImage"]
    assert graph["1"]["inputs"] == {"unet_name": "moss/demo-moss.safetensors", "weight_dtype": "default"}
    assert graph["2"]["inputs"] == {"model": ["1", 0], "lora_name": LORA, "strength_model": 1.0}
    assert graph["3"]["inputs"] == {"clip_name": "moss/demo-moss-encoder.safetensors", "type": "moss_image",
                                    "device": "default"}
    assert graph["4"]["inputs"]["clip"] == ["3", 0]
    assert graph["5"]["inputs"]["text"] == moss["negative_baseline"]
    ks = graph["7"]["inputs"]
    assert (ks["model"], ks["sampler_name"], ks["scheduler"], ks["steps"], ks["cfg"]) == (["2", 0], "er_sde", "simple", 25, 3.0)
    assert graph["9"]["inputs"] == {"samples": ["7", 0], "vae": ["8", 0]}
    assert cg.check(graph, object_info()) == []
    assert "moss_txt2img" in cg.EXPERIMENTAL


def test_moss_model_hires_keeps_the_16_grid(moss):
    hires = {"mode": "model", "upscaler": UPSCALER, "scale": 1.3, "denoise": 0.3, "steps": 10}
    graph = cg.compile_graph(moss_spec(params={"seed": 5, "width": 1024, "height": 1024, "hires": hires}), inference=moss)
    scaled = node(graph, "to x1.3")["inputs"]
    assert (scaled["width"], scaled["height"]) == (1328, 1328)


@pytest.mark.parametrize(("over", "match"), [
    ({"params": {"seed": 5, "clip_skip": 2, "hires": {"mode": "none"}}}, "no clip skip"),
    ({"vae": None}, "missing"),
    ({"embeddings": [{"name": "demo"}]}, "no embeddings"),
    ({"task": "img2img"}, "txt2img only"),
    ({"params": {"seed": 5, "hires": {"mode": "latent"}}}, "none or model"),
    ({"params": {"seed": 5, "width": 1000, "height": 1024, "hires": {"mode": "none"}}}, "multiple of 16"),
])
def test_moss_refusals(moss, over, match):
    with pytest.raises(cg.GraphError, match=match):
        cg.compile_graph(moss_spec(**over), inference=moss)


# --- names and checks against the server ------------------------------------------------------------------------------

def test_resolve_names_maps_onto_the_servers_spelling(illustrious):
    graph = cg.compile_graph(spec(checkpoint="Illustrious\\Demo-Anime-XL.safetensors"), inference=illustrious)
    assert cg.resolve_names(graph, object_info()) == []
    assert node(graph, "checkpoint")["inputs"]["ckpt_name"] == CKPT  # separator and case: the same path


def test_a_file_name_only_match_is_a_problem_never_a_silent_swap(illustrious):
    """The spec asks for one folder's file; the server has a file of that name in another folder only."""
    wanted = "other-run/" + STYLE.rsplit("/", 1)[-1]
    graph = cg.compile_graph(spec(loras=[{"name": wanted, "weight": 0.5}]), inference=illustrious)
    problems = cg.resolve_names(graph, object_info())
    assert problems == [f"node {by_title(graph, 'lora.1')} (LoraLoader): lora_name {wanted!r} is not on the server "
                        f"by that path; the server has {STYLE!r}, which only shares the file name and may be another "
                        "file; name the file as the server lists it, or report it to the operator"]
    assert node(graph, "lora.1")["inputs"]["lora_name"] == wanted  # left as asked, never swapped
    assert cg.match_name_how(wanted, [STYLE]) == (STYLE, cg.MATCH_BASENAME)


def test_a_caller_may_allow_file_name_matches_and_hears_about_each(illustrious):
    graph = cg.compile_graph(spec(loras=[{"name": "demo-style.safetensors", "weight": 0.5}]), inference=illustrious)
    notes: list[str] = []
    assert cg.resolve_names(graph, object_info(), allow_basename=True, notes=notes) == []
    assert node(graph, "lora.1")["inputs"]["lora_name"] == STYLE
    assert notes == [f"node {by_title(graph, 'lora.1')} (LoraLoader): lora_name 'demo-style.safetensors' matched by "
                     f"file name only: using {STYLE!r}"]


def test_resolve_names_reports_a_missing_or_ambiguous_name(illustrious):
    graph = cg.compile_graph(spec(checkpoint="nosuch.safetensors"), inference=illustrious)
    problems = cg.resolve_names(graph, object_info())
    assert problems == ["node 1 (CheckpointLoaderSimple): ckpt_name 'nosuch.safetensors' is not on the server; "
                        "report it to the operator (the file is missing from the server's models folder for its "
                        "type)"]
    assert not any("restart" in p for p in problems)  # an agent never restarts the server
    assert cg.match_name("face_yolov8m.pt", ["bbox/face_yolov8m.pt", "segm/face_yolov8m.pt"]) is None
    assert cg.detector_name("face_yolov8m.pt") == "bbox/face_yolov8m.pt"


def test_allowed_names_reads_both_combo_shapes():
    classic = {"VAELoader": {"input": {"required": {"vae_name": [[VAE], {}]}}}}
    combo = {"UpscaleModelLoader": {"input": {"required": {"model_name": ["COMBO", {"options": [UPSCALER]}]}}}}
    assert cg.allowed_names(classic, "VAELoader", "vae_name") == [VAE]
    assert cg.allowed_names(combo, "UpscaleModelLoader", "model_name") == [UPSCALER]
    assert cg.allowed_names(combo, "VAELoader", "vae_name") == []


def test_check_names_each_mismatch():
    graph = {"1": {"class_type": "NoSuchNode", "inputs": {}},
             "2": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "extra": 1}}}
    assert cg.check(graph, object_info()) == [
        "node 1: the server has no node class NoSuchNode (is its node pack installed?)",
        "node 2 (VAEDecode): the server's VAEDecode has no input 'extra'",
        "node 2 (VAEDecode): required input 'vae' is not set",
    ]


def test_check_against_the_mock_server(mock_comfy, no_network, illustrious):
    serve(mock_comfy, drop=("LoraLoader",))
    graph = cg.compile_graph(spec(), inference=illustrious)
    info = cc.Client(mock_comfy.url, timeout=5).object_info_for(cg.node_classes(graph))
    assert cg.check(graph, info) == ["node 3: the server has no node class LoraLoader (is its node pack installed?)"]


# --- templates ------------------------------------------------------------------------------------------------------

TEMPLATE = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ""}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}}}
SLOTS = {"schema": "easel/workflow-slots/1", "workflow": "demo",
         "slots": [{"name": "checkpoint", "node": "1", "input": "ckpt_name", "kind": "resource", "required": True},
                   {"name": "prompt", "node": "2", "input": "text", "kind": "text", "required": False,
                    "default": "dmrbt"}]}


def test_fill_template_fills_slots_and_defaults_and_leaves_the_template_alone():
    filled = cg.fill_template(TEMPLATE, SLOTS, {"checkpoint": CKPT})
    assert filled["1"]["inputs"]["ckpt_name"] == CKPT and filled["2"]["inputs"]["text"] == "dmrbt"
    assert TEMPLATE["1"]["inputs"]["ckpt_name"] == ""
    assert cg.fill_template(TEMPLATE, SLOTS["slots"], {"checkpoint": CKPT, "prompt": "x"})["2"]["inputs"]["text"] == "x"


@pytest.mark.parametrize(("slots", "values", "match"), [
    (SLOTS, {}, "required"),
    (SLOTS, {"checkpoint": CKPT, "seed": 1}, "no slot"),
    ({"slots": [{"name": "x", "node": "9", "input": "a", "kind": "int", "required": False}]}, {"x": 1}, "node 9"),
])
def test_fill_template_refusals(slots, values, match):
    with pytest.raises(cg.GraphError, match=match):
        cg.fill_template(TEMPLATE, slots, values)
