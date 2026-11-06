"""_common.embedded_meta: safetensors headers written byte by byte, PNG text chunks and the recipes they carry."""

import io
import json
import struct
import zlib

import pytest

from easel._common import embedded_meta

TENSORS = {
    "lora_unet_input_blocks_4_1_proj_in.alpha": ("F32", []),
    "lora_unet_input_blocks_4_1_proj_in.lora_down.weight": ("F16", [4, 8]),
    "lora_unet_input_blocks_4_1_proj_in.lora_up.weight": ("F16", [8, 4]),
    "lora_te1_text_model_encoder_layers_0_mlp_fc1.lora_down.weight": ("BF16", [4, 2]),
}
SIZES = {"F32": 4, "F16": 2, "BF16": 2}
META = {"ss_network_module": "networks.lora", "ss_network_dim": "4", "ss_output_name": "demo_robot-v1",
        "ss_tag_frequency": json.dumps({"demo": {"dmrbt": 3, "robot": 3}})}


def safetensors_bytes(tensors=TENSORS, metadata=META, pad_to=8, cut=0) -> bytes:
    """A complete safetensors file: u64 header length, JSON header, then zeroed tensor data."""
    header: dict = {"__metadata__": metadata} if metadata is not None else {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        size = count * SIZES[dtype]
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % pad_to)
    data = struct.pack("<Q", len(raw)) + raw + b"\x00" * offset
    return data[:len(data) - cut] if cut else data


def raw_header_file(tmp_path, header_bytes: bytes, tail: bytes = b"") -> object:
    path = tmp_path / "x.safetensors"
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + tail)
    return path


# --- safetensors ------------------------------------------------------------------------------------------------

def test_header_summary(tmp_path):
    path = tmp_path / "demo_robot-v1.safetensors"
    path.write_bytes(safetensors_bytes())
    head = embedded_meta.safetensors_header(path)
    assert head["metadata"]["ss_network_module"] == "networks.lora"
    assert head["tensor_count"] == 4
    assert head["param_count"] == 1 + 32 + 32 + 8
    assert head["dtypes"] == ["BF16", "F16", "F32"]
    assert head["precision"] == "fp16"  # the dtype holding the most parameters
    assert head["prefixes"] == {"lora_te1": 1, "lora_unet": 3}
    assert head["tensor_names"] == sorted(TENSORS)
    assert head["tensors"]["lora_unet_input_blocks_4_1_proj_in.lora_up.weight"] == {"dtype": "F16", "shape": [8, 4]}
    assert head["data_bytes"] == 4 + 64 + 64 + 16
    assert head["file_bytes"] == path.stat().st_size == 8 + head["header_bytes"] + head["data_bytes"]


def test_metadata_value_decodes_json_text(tmp_path):
    path = tmp_path / "a.safetensors"
    path.write_bytes(safetensors_bytes())
    meta = embedded_meta.safetensors_header(path)["metadata"]
    assert embedded_meta.metadata_value(meta, "ss_tag_frequency") == {"demo": {"dmrbt": 3, "robot": 3}}
    assert embedded_meta.metadata_value(meta, "ss_network_dim") == "4"
    assert embedded_meta.metadata_value(meta, "absent") is None


def test_no_metadata_block_and_non_string_values(tmp_path):
    path = tmp_path / "a.safetensors"
    path.write_bytes(safetensors_bytes(metadata=None))
    assert embedded_meta.safetensors_header(path)["metadata"] == {}
    path.write_bytes(safetensors_bytes(metadata={"steps": 800}))
    assert embedded_meta.safetensors_header(path)["metadata"] == {"steps": "800"}


def test_checkpoint_style_prefixes(tmp_path):
    tensors = {"model.diffusion_model.input_blocks.0.0.weight": ("F16", [2]),
               "first_stage_model.decoder.conv_in.weight": ("F16", [2]),
               "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight": ("F16", [2])}
    path = tmp_path / "c.safetensors"
    path.write_bytes(safetensors_bytes(tensors=tensors, metadata={"modelspec.title": "Demo Base XL"}))
    assert embedded_meta.safetensors_header(path)["prefixes"] == {"conditioner": 1, "first_stage_model": 1, "model": 1}


def test_tensor_data_is_never_read(tmp_path, monkeypatch):
    path = tmp_path / "big.safetensors"
    path.write_bytes(safetensors_bytes())
    sizes = []
    real_open = type(path).open

    class Spy:
        """A file handle that records every read size."""

        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def read(self, n=-1):
            sizes.append(n)
            return self.handle.read(n)

    monkeypatch.setattr(type(path), "open", lambda self, *a, **k: Spy(real_open(self, *a, **k)))
    head = embedded_meta.safetensors_header(path)
    assert sizes == [8, head["header_bytes"]]  # the length, then the header: nothing after it


@pytest.mark.parametrize("data, needle", [
    (b"", "shorter than the 8-byte"),
    (b"\x01\x02\x03", "shorter than the 8-byte"),
    (struct.pack("<Q", 0), "implausible header length"),
    (struct.pack("<Q", 10**12) + b"{}", "implausible header length"),
    (struct.pack("<Q", 64) + b"{}", "shorter than its header length"),
])
def test_bad_length_fields(tmp_path, data, needle):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(data)
    with pytest.raises(ValueError, match=needle):
        embedded_meta.safetensors_header(path)


@pytest.mark.parametrize("header, needle", [
    (b"not json at all", "not JSON"),
    (b"\xff\xfe{}", "not JSON"),
    (b"[1, 2, 3]", "not a JSON object"),
    (json.dumps({"__metadata__": ["x"]}).encode(), "__metadata__ is not an object"),
    (json.dumps({"t": "F16"}).encode(), "not an object"),
    (json.dumps({"t": {"dtype": "F16", "shape": "2", "data_offsets": [0, 4]}}).encode(), "dtype or shape"),
    (json.dumps({"t": {"dtype": "F16", "shape": [2], "data_offsets": [4, 0]}}).encode(), "data_offsets"),
    (json.dumps({"t": {"dtype": "F16", "shape": [2], "data_offsets": [0, 6]}}).encode(), "do not match"),
])
def test_malformed_headers(tmp_path, header, needle):
    path = raw_header_file(tmp_path, header, tail=b"\x00" * 16)
    with pytest.raises(ValueError, match=needle):
        embedded_meta.safetensors_header(path)


def test_partial_download_is_a_bad_file(tmp_path):
    path = tmp_path / "part.safetensors"
    path.write_bytes(safetensors_bytes(cut=10))
    with pytest.raises(ValueError, match="partial download"):
        embedded_meta.safetensors_header(path)


# --- PNG text and recipes ---------------------------------------------------------------------------------------

SIG = b"\x89PNG\r\n\x1a\n"


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def png_with(*chunks: bytes) -> bytes:
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    return SIG + ihdr + b"".join(chunks) + chunk(b"IDAT", b"\x00" * 16) + chunk(b"IEND", b"")


def text(key: str, value: str) -> bytes:
    return chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))


API_GRAPH = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "demo-anime-xl.safetensors"}},
    "2": {"class_type": "LoraLoader", "inputs": {"lora_name": "demo_robot-v1.safetensors", "strength_model": 0.8,
                                                 "strength_clip": 0.8, "model": ["1", 0], "clip": ["1", 1]}},
    "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "dmrbt, robot, standing in a tidy workshop", "clip": ["2", 1]}},
    "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, blurry", "clip": ["2", 1]}},
    "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
    "6": {"class_type": "KSampler", "inputs": {"seed": 1001, "steps": 28, "cfg": 5.5, "sampler_name": "euler_ancestral",
                                               "scheduler": "normal", "denoise": 1.0, "model": ["2", 0],
                                               "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
    "7": {"class_type": "CLIPSetLastLayer", "inputs": {"stop_at_clip_layer": -2, "clip": ["1", 1]}},
}
UI_GRAPH = {"last_node_id": 2, "nodes": [{"id": 1, "type": "KSampler"}], "links": []}
A1111 = ("dmrbt, robot, <lora:demo_robot-v1:0.7> front view\n"
         "Negative prompt: lowres, blurry\n"
         'Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 6.5, Seed: 1234, Size: 832x1216, '
         'Model hash: abc123, Model: demo-anime-xl, Clip skip: 2, Denoising strength: 0.4, '
         'Lora hashes: "demo_robot-v1: 0123456789ab", Version: v1.10')


def test_png_text_reads_text_ztxt_and_itxt(tmp_path):
    itxt_plain = b"note\x00\x00\x00en\x00Note\x00demo text \xc3\xa9"
    itxt_zip = b"zipped\x00\x01\x00\x00\x00" + zlib.compress("compressed é".encode())
    data = png_with(text("prompt", json.dumps(API_GRAPH)),
                    chunk(b"zTXt", b"parameters\x00\x00" + zlib.compress(A1111.encode("latin-1"))),
                    chunk(b"iTXt", itxt_plain), chunk(b"iTXt", itxt_zip))
    path = tmp_path / "a.png"
    path.write_bytes(data)
    fields = embedded_meta.png_text(path)
    assert json.loads(fields["prompt"]) == API_GRAPH
    assert fields["parameters"] == A1111
    assert fields["note"] == "demo text é"
    assert fields["zipped"] == "compressed é"


def test_png_text_skips_undecodable_and_oversized_chunks(tmp_path, monkeypatch):
    bomb = zlib.compress(b"a" * 4096)
    path = tmp_path / "b.png"
    path.write_bytes(png_with(chunk(b"zTXt", b"broken\x00\x00not zlib"), chunk(b"zTXt", b"bomb\x00\x00" + bomb),
                              text("ok", "fine")))
    monkeypatch.setattr(embedded_meta, "MAX_TEXT_BYTES", 1024)
    assert embedded_meta.png_text(path) == {"ok": "fine"}


def test_png_text_stops_at_a_chunk_that_claims_more_than_the_file(tmp_path):
    """A truncated file whose text chunk declares 8 MiB: the earlier chunks are kept, nothing big is read."""
    lying = struct.pack(">I", 8 * 2**20) + b"tEXt" + b"late\x00cut short"
    path = tmp_path / "cut.png"
    path.write_bytes(SIG + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)) + text("ok", "fine") + lying)
    assert embedded_meta.png_text(path) == {"ok": "fine"}


def test_png_text_of_a_non_png_is_empty(tmp_path):
    path = tmp_path / "c.png"
    path.write_bytes(b"GIF89a....")
    assert embedded_meta.png_text(path) == {}
    assert embedded_meta.media_kind(path) == "unknown"


def test_comfy_graphs_and_recipe_from_a_comfyui_png(tmp_path):
    path = tmp_path / "001.png"
    path.write_bytes(png_with(text("prompt", json.dumps(API_GRAPH)), text("workflow", json.dumps(UI_GRAPH))))
    fields = embedded_meta.read_fields(path)
    api, ui = embedded_meta.comfy_graphs(fields)
    assert api == API_GRAPH and ui == UI_GRAPH
    found = embedded_meta.recipe(fields)
    assert found["format"] == "comfy"
    assert (found["seed"], found["steps"], found["cfg"], found["sampler"], found["scheduler"], found["denoise"]) == \
        (1001, 28, 5.5, "euler_ancestral", "normal", 1.0)
    assert found["prompt"].startswith("dmrbt, robot") and found["negative"] == "lowres, blurry"
    assert found["model"] == "demo-anime-xl.safetensors"
    assert found["loras"] == [{"name": "demo_robot-v1.safetensors", "hash": None, "weight": 0.8}]
    assert (found["width"], found["height"], found["clip_skip"]) == (832, 1216, 2)


def test_comfy_graphs_rejects_lookalikes_and_reads_a_nested_comment():
    assert embedded_meta.comfy_graphs({"prompt": "dmrbt, robot", "workflow": "{}"}) == (None, None)
    assert embedded_meta.comfy_graphs({"prompt": json.dumps({"a": 1})}) == (None, None)
    nested = {"comment": json.dumps({"prompt": API_GRAPH, "workflow": json.dumps(UI_GRAPH)})}
    assert embedded_meta.comfy_graphs(nested) == (API_GRAPH, UI_GRAPH)


def test_sampler_inputs_that_are_links_are_not_values():
    graph = {"1": {"class_type": "KSamplerAdvanced", "inputs": {"noise_seed": 7, "steps": ["9", 0], "cfg": 4}}}
    found = embedded_meta.parse_comfy(graph)
    assert (found["seed"], found["steps"], found["cfg"], found["prompt"]) == (7, None, 4, "")


def test_a1111_block(tmp_path):
    found = embedded_meta.recipe({"parameters": A1111})
    assert found["format"] == "a1111"
    assert found["prompt"] == "dmrbt, robot, <lora:demo_robot-v1:0.7> front view"
    assert found["negative"] == "lowres, blurry"
    assert (found["steps"], found["sampler"], found["scheduler"], found["cfg"], found["seed"]) == \
        (20, "Euler a", "Karras", 6.5, 1234)
    assert (found["width"], found["height"], found["clip_skip"], found["denoise"]) == (832, 1216, 2, 0.4)
    assert (found["model"], found["model_hash"]) == ("demo-anime-xl", "abc123")
    assert found["loras"] == [{"name": "demo_robot-v1", "hash": "0123456789ab", "weight": 0.7}]


def test_a1111_without_negative_prompt_and_unknown_fields():
    assert embedded_meta.recipe({"x": "no recipe here"}) == {"format": "none"}
    found = embedded_meta.parse_a1111("a robot\nSteps: 12, Sampler: DPM++ 2M, Seed: abc")
    assert (found["prompt"], found["negative"], found["steps"], found["seed"]) == ("a robot", "", 12, None)
    assert found["loras"] == [] and found["width"] is None
    bare = embedded_meta.parse_a1111("just a prompt\nNegative prompt: blurry")
    assert (bare["prompt"], bare["negative"], bare["steps"]) == ("just a prompt", "blurry", None)


def test_exif_text_of_a_jpeg_via_the_header_reader(tmp_path):
    from PIL import Image

    image = Image.new("RGB", (8, 8), (40, 80, 120))
    exif = Image.Exif()
    exif[0x010E] = "demo description"
    exif.get_ifd(0x8769)[0x9286] = b"UNICODE\x00" + A1111.encode("utf-16-be")
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", exif=exif)
    path = tmp_path / "a.jpg"
    path.write_bytes(buffer.getvalue())
    assert embedded_meta.media_kind(path) == "jpeg"
    fields = embedded_meta.read_fields(path)
    assert fields["ImageDescription"] == "demo description"
    assert fields["UserComment"] == A1111
    assert embedded_meta.recipe(fields)["seed"] == 1234


def test_user_comment_charsets():
    body = "Steps: 20"
    assert embedded_meta.user_comment(b"UNICODE\x00" + body.encode("utf-16-le")) == body
    assert embedded_meta.user_comment(b"UNICODE\x00" + body.encode("utf-16-be")) == body
    assert embedded_meta.user_comment(b"ASCII\x00\x00\x00" + body.encode()) == body
    assert embedded_meta.user_comment(body) == body
    assert embedded_meta.user_comment(None) is None


def test_probe_keys_prints_no_values(tmp_path):
    path = tmp_path / "001.png"
    path.write_bytes(png_with(text("prompt", json.dumps(API_GRAPH)), text("parameters", A1111)))
    summary = embedded_meta.probe_keys(path)
    assert summary == {"kind": "png", "keys": ["parameters", "prompt"], "comfy_api_graph": True,
                       "comfy_workflow": False, "a1111": True}
    assert "dmrbt" not in repr(summary)


def test_media_kind_of_video_containers(tmp_path):
    mp4 = tmp_path / "a.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8)
    webm = tmp_path / "a.webm"
    webm.write_bytes(b"\x1a\x45\xdf\xa3" + b"\x00" * 12)
    assert embedded_meta.media_kind(mp4) == embedded_meta.media_kind(webm) == "video"
    assert embedded_meta.read_fields(mp4) == {}
