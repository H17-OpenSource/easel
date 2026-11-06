"""Write the test media: 24 deterministic 8x8 PNGs and 6 fake model files, into tests/fixtures/_gen/.

Why: the repository never tracks a binary (no image, no model file), yet the demo instance in examples/demo/
refers to images and models by sha256. This script makes those files, byte for byte the same on every machine,
so the hashes in the demo records stay true. The same file ships in both repositories (kiln and easel), so a
kiln demo image imported from the easel demo run has the easel output's sha256.

How the bytes stay identical everywhere: PNG chunks are written by hand and the image data uses "stored"
(uncompressed) deflate blocks, so no zlib version or compression level can change a byte; the fake models are
safetensors files whose JSON header is serialised with sorted keys.

What is made:
- png/fix-01.png .. png/fix-24.png: 8x8 RGB, one colour pattern each (every sha256 differs). fix-17 .. fix-20
  carry ComfyUI's two tEXt keys `prompt` (API graph) and `workflow` (UI graph); fix-21 .. fix-24 carry `prompt`
  only; the rest carry no text.
- models/<folder>/<file>.safetensors: six header-only models (two checkpoints, two LoRAs, a VAE, an upscaler)
  with realistic tensor names and kohya-style metadata, for the registry scan tests.

usage:
    python tests/fixtures/make_fixtures.py                        # write into tests/fixtures/_gen/
    python tests/fixtures/make_fixtures.py --out <dir>            # write somewhere else
    python tests/fixtures/make_fixtures.py --place <home> --demo kiln|easel
                                                                  # also copy them where the demo expects them
Running it again changes nothing (files whose bytes already match are left alone). Exit 0, or 2 on bad usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "_gen"
PNG_COUNT = 24
SIZE = 8
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

EASEL_RUN = "run-20300101-001"   # the easel demo's render run (recipe rcp-20300101-001)
KILN_RUN = "run-20300102-001"    # the kiln demo's training run

# --- PNG ---------------------------------------------------------------------------------------------------------


def _chunk(kind: bytes, data: bytes) -> bytes:
    """One PNG chunk: length, type, data, CRC over type + data."""
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _stored_zlib(raw: bytes) -> bytes:
    """A zlib stream of uncompressed deflate blocks: the same bytes from every zlib build."""
    out = bytearray(b"\x78\x01")
    blocks = [raw[i:i + 65535] for i in range(0, len(raw), 65535)] or [b""]
    for i, block in enumerate(blocks):
        final = 1 if i == len(blocks) - 1 else 0
        out += bytes([final]) + struct.pack("<HH", len(block), len(block) ^ 0xFFFF) + block
    out += struct.pack(">I", zlib.adler32(raw) & 0xFFFFFFFF)
    return bytes(out)


def _pixels(n: int) -> bytes:
    """Raw scanlines (filter byte 0) of fixture n: a base colour from n plus a small x/y gradient."""
    base = ((n * 37) % 256, (n * 91) % 256, (n * 53) % 256)
    rows = bytearray()
    for y in range(SIZE):
        rows.append(0)
        for x in range(SIZE):
            rows += bytes(((base[0] + 8 * x) % 256, (base[1] + 8 * y) % 256, base[2] ^ (x * y)))
    return bytes(rows)


def _api_graph(n: int) -> dict:
    """A small ComfyUI API graph, as ComfyUI stores it under the `prompt` key."""
    if 17 <= n <= 20:  # the easel demo run's four outputs
        ckpt, lora, seed = "demo-anime-xl.safetensors", "demo_robot-v1.safetensors", 1000 + (n - 16)
        prefix = f"easel/{EASEL_RUN}"
        text = "dmrbt, robot, standing in a tidy workshop, soft daylight, full body"
    else:  # the kiln demo's eval renders of the training run
        ckpt, lora, seed = "demo-anime-xl.safetensors", "demo_robot-s800.safetensors", 1000 + (n - 20)
        prefix = f"kiln/{KILN_RUN}"
        text = "dmrbt, robot, front view, plain background"
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoraLoader", "inputs": {"lora_name": lora, "strength_model": 0.8, "strength_clip": 0.8,
                                                     "model": ["1", 0], "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "lowres, blurry, text, watermark", "clip": ["2", 1]}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 1216, "batch_size": 1}},
        "6": {"class_type": "KSampler", "inputs": {"seed": seed, "steps": 28, "cfg": 5.5, "sampler_name": "euler_ancestral",
                                                   "scheduler": "normal", "denoise": 1.0, "model": ["2", 0],
                                                   "positive": ["3", 0], "negative": ["4", 0], "latent_image": ["5", 0]}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"filename_prefix": prefix, "images": ["7", 0]}},
    }


def _ui_graph(n: int) -> dict:
    """A minimal ComfyUI UI workflow, as ComfyUI stores it under the `workflow` key."""
    types = [node["class_type"] for _, node in sorted(_api_graph(n).items(), key=lambda kv: int(kv[0]))]
    return {"last_node_id": len(types), "last_link_id": 0, "version": 0.4, "links": [],
            "nodes": [{"id": i + 1, "type": t, "pos": [40 * i, 40], "size": [200, 80]} for i, t in enumerate(types)]}


def png_text(n: int) -> list[tuple[str, str]]:
    """The tEXt chunks of fixture n, in file order."""
    def dump(obj: dict) -> str:
        return json.dumps(obj, separators=(",", ":"), sort_keys=True)

    if 17 <= n <= 20:
        return [("prompt", dump(_api_graph(n))), ("workflow", dump(_ui_graph(n)))]
    if 21 <= n <= 24:
        return [("prompt", dump(_api_graph(n)))]
    return []


def png_bytes(n: int) -> bytes:
    """The complete PNG file of fixture n (1-based)."""
    if not 1 <= n <= PNG_COUNT:
        raise ValueError(f"fixture number must be 1..{PNG_COUNT}")
    ihdr = struct.pack(">IIBBBBB", SIZE, SIZE, 8, 2, 0, 0, 0)  # 8-bit RGB, no interlace
    out = PNG_SIGNATURE + _chunk(b"IHDR", ihdr)
    for key, value in png_text(n):
        out += _chunk(b"tEXt", key.encode("latin-1") + b"\x00" + value.encode("latin-1"))
    return out + _chunk(b"IDAT", _stored_zlib(_pixels(n))) + _chunk(b"IEND", b"")


def png_name(n: int) -> str:
    return f"fix-{n:02d}.png"


# --- fake models (safetensors headers) ---------------------------------------------------------------------------

# (folder, file name, tensor names, metadata). Values are synthetic; the tensor names follow the real layouts so a
# header-only classifier can tell the types apart.
MODELS: tuple[tuple[str, str, tuple[str, ...], dict[str, str]], ...] = (
    ("checkpoints", "demo-base-xl.safetensors",
     ("model.diffusion_model.input_blocks.0.0.weight", "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
      "conditioner.embedders.1.model.ln_final.weight", "first_stage_model.decoder.conv_in.weight"),
     {"modelspec.architecture": "stable-diffusion-xl-v1-base", "modelspec.title": "Demo Base XL"}),
    ("checkpoints", "demo-anime-xl.safetensors",
     ("model.diffusion_model.input_blocks.0.0.weight", "conditioner.embedders.0.transformer.text_model.final_layer_norm.weight",
      "conditioner.embedders.1.model.ln_final.weight", "first_stage_model.decoder.conv_in.weight"),
     {"modelspec.architecture": "stable-diffusion-xl-v1-base", "modelspec.title": "Demo Anime XL"}),
    ("loras", "demo-style-ink.safetensors",
     ("lora_unet_input_blocks_4_1_proj_in.alpha", "lora_unet_input_blocks_4_1_proj_in.lora_down.weight",
      "lora_unet_input_blocks_4_1_proj_in.lora_up.weight"),
     {"ss_base_model_version": "sdxl_base_v1-0", "ss_network_alpha": "8", "ss_network_dim": "16",
      "ss_network_module": "networks.lora", "ss_output_name": "demo-style-ink"}),
    ("loras", "demo_robot-v1.safetensors",
     ("lora_unet_input_blocks_4_1_proj_in.alpha", "lora_unet_input_blocks_4_1_proj_in.lora_down.weight",
      "lora_unet_input_blocks_4_1_proj_in.lora_up.weight"),
     {"ss_base_model_version": "sdxl_base_v1-0", "ss_network_alpha": "32", "ss_network_dim": "32",
      "ss_network_module": "networks.lora", "ss_output_name": "demo_robot-v1", "ss_steps": "800",
      "ss_training_comment": f"kiln demo_robot {KILN_RUN} step 800"}),
    ("vae", "demo-xl-vae.safetensors",
     ("decoder.conv_in.weight", "encoder.conv_in.weight", "quant_conv.weight"),
     {"modelspec.title": "Demo XL VAE"}),
    ("upscale_models", "demo-4x.safetensors",
     ("conv_first.weight", "body.0.rdb1.conv1.weight", "conv_last.weight"),
     {"modelspec.title": "Demo 4x upscaler", "scale": "4"}),
)


def model_bytes(index: int) -> bytes:
    """The safetensors file of MODELS[index]: 8-byte header length, sorted JSON header, one fp16 value per tensor."""
    _folder, _name, tensors, metadata = MODELS[index]
    header: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    for i, tensor in enumerate(tensors):
        header[tensor] = {"dtype": "F16", "shape": [1], "data_offsets": [2 * i, 2 * i + 2]}
    raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    raw += b" " * (-len(raw) % 8)  # the format allows trailing spaces; 8-byte alignment as the reference writer does
    return struct.pack("<Q", len(raw)) + raw + b"\x00\x3c" * len(tensors)  # fp16 1.0 per tensor


def model_relpath(index: int) -> str:
    folder, name, _tensors, _meta = MODELS[index]
    return f"{folder}/{name}"


# --- writing -----------------------------------------------------------------------------------------------------


def all_files() -> dict[str, bytes]:
    """Every generated file, keyed by its path relative to the output folder (forward slashes)."""
    files = {f"png/{png_name(n)}": png_bytes(n) for n in range(1, PNG_COUNT + 1)}
    files.update({f"models/{model_relpath(i)}": model_bytes(i) for i in range(len(MODELS))})
    return files


def sha256_of(rel: str) -> str:
    """The sha256 of one generated file, without writing anything."""
    return hashlib.sha256(all_files()[rel]).hexdigest()


def _write_if_changed(path: Path, data: bytes) -> bool:
    """Write `data` atomically unless the file already holds exactly these bytes. True when written."""
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def generate(out: Path = DEFAULT_OUT) -> dict[str, Path]:
    """Write every fixture file under `out` (idempotent) and return {relative path: absolute path}."""
    written = {}
    for rel, data in all_files().items():
        target = out / rel
        _write_if_changed(target, data)
        written[rel] = target
    return written


# Where each demo expects the generated files, relative to the demo home (examples/demo/ or a copy of it).
_KILN_IMAGE_FILES = {
    1: "raw/demo_robot-01.png", 2: "raw/demo_robot-02.png", 3: "raw/demo_robot-03.png", 4: "raw/demo_robot-04.png",
    5: "raw/demo_robot-05.png", 6: "raw/demo_robot-06.png", 7: "raw/demo_robot-07.png", 8: "raw/demo_robot-08.png",
    9: "raw/demo_robot-09.png", 10: "raw/demo_robot-10.png", 11: "raw/demo_robot-11.png",
    12: "crops/demo_robot-03-a.png", 13: "raw/demo_robot-13.png", 14: "raw/demo_robot-14.png",
    15: "raw/demo_robot-15.png", 16: "raw/demo_robot-16.png",
}
DEMO_MEDIA: dict[str, dict[str, str]] = {
    "kiln": {
        **{f"png/{png_name(n)}": f"dataset-images/{rel}" for n, rel in _KILN_IMAGE_FILES.items()},
        # the three kept outputs of the easel demo run, imported into the project (output 3 was rejected)
        f"png/{png_name(17)}": f"projects/demo_robot/images/{EASEL_RUN}/001.png",
        f"png/{png_name(18)}": f"projects/demo_robot/images/{EASEL_RUN}/002.png",
        f"png/{png_name(20)}": f"projects/demo_robot/images/{EASEL_RUN}/004.png",
        # eval renders of the training run's step-800 checkpoint, prompts E01/E02, seeds 1001/1002
        f"png/{png_name(21)}": f"projects/demo_robot/evals/renders/{KILN_RUN}/s800-E01-1001.png",
        f"png/{png_name(22)}": f"projects/demo_robot/evals/renders/{KILN_RUN}/s800-E01-1002.png",
        f"png/{png_name(23)}": f"projects/demo_robot/evals/renders/{KILN_RUN}/s800-E02-1001.png",
        f"png/{png_name(24)}": f"projects/demo_robot/evals/renders/{KILN_RUN}/s800-E02-1002.png",
    },
    "easel": {
        **{f"png/{png_name(n)}": f"runs/{EASEL_RUN}/outputs/{n - 16:03d}.png" for n in range(17, 21)},
        f"png/{png_name(24)}": "references/civitai/9000000001/media.png",
        **{f"models/{model_relpath(i)}": f"models/{model_relpath(i)}" for i in range(len(MODELS))},
    },
}


def place(home: Path, demo: str, out: Path = DEFAULT_OUT) -> list[Path]:
    """Generate the fixtures, then copy the ones `demo` uses into `home` at the paths DEMO_MEDIA names."""
    if demo not in DEMO_MEDIA:
        raise ValueError(f"demo must be one of {sorted(DEMO_MEDIA)}")
    made = generate(out)
    placed = []
    for rel, dest_rel in DEMO_MEDIA[demo].items():
        dest = home / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not (dest.is_file() and dest.read_bytes() == made[rel].read_bytes()):
            shutil.copyfile(made[rel], dest)
        placed.append(dest)
    return placed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the deterministic test media into tests/fixtures/_gen/.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder; default tests/fixtures/_gen")
    parser.add_argument("--place", type=Path, metavar="HOME", help="also copy the demo's media into this demo home")
    parser.add_argument("--demo", choices=sorted(DEMO_MEDIA), help="which demo --place fills: kiln or easel")
    args = parser.parse_args(argv)
    if (args.place is None) != (args.demo is None):
        parser.print_usage(sys.stderr)
        print("make_fixtures: --place and --demo go together", file=sys.stderr)
        return 2
    made = generate(args.out)
    print(f"{len(made)} fixture files in {args.out.name}/ ({PNG_COUNT} PNGs, {len(MODELS)} models)")
    if args.place is not None:
        placed = place(args.place, args.demo, args.out)
        print(f"{len(placed)} files placed for the {args.demo} demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
