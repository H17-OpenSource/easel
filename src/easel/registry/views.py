"""The read-only questions of the registry: `list`, `list --self-trained`, `show`, `explain`.

Each answers one question from the records with the person's overrides applied. `--self-trained` lists every
resource whose source class is `self-trained`, with whether a training record names it (`provenance.trained_by`:
tool, project, run).
Under `policy.prompt_authoring = "sfw-only"`, `show` withholds the descriptive text (title, description,
trigger words, when-to-use) of a resource whose class is not sfw: such a resource is named by id only.
"""

from __future__ import annotations

from collections.abc import Mapping

from . import classify as cls
from . import resources as res
from .resources import get_field
from .shipped import reason

LOADERS = {
    "checkpoint": "CheckpointLoaderSimple (gives the model, CLIP and VAE)",
    "lora": "LoraLoader (chained after the checkpoint; one weight for the model, one for CLIP)",
    "lycoris": "LoraLoader (chained after the checkpoint; one weight for the model, one for CLIP)",
    "vae": "VAELoader (replaces the checkpoint's VAE; only within the family's vae.shared_with list)",
    "embedding": "none: written in the prompt as embedding:<name>",
    "upscaler": "UpscaleModelLoader, then ImageUpscaleWithModel",
    "controlnet": "ControlNetLoader (planned for v0.2; v0.1 renders none)",
    "text_encoder": "CLIPLoader or DualCLIPLoader",
    "clip": "CLIPLoader or DualCLIPLoader",
    "clip_vision": "CLIPVisionLoader",
    "unet": "UNETLoader",
    "style_model": "StyleModelLoader",
    "model_patch": "ModelPatchLoader",
    "ipadapter": "the IP-Adapter node pack's model loader (not in the v0.1 graphs)",
    "detector": "the detailer node packs' detector provider (see workflows/ for the licences)",
    "sam": "the detailer node packs' SAM loader (see workflows/ for the licences)",
}
BASIS_WORDS = {
    "source": "the model's source (site metadata)",
    "markers": "cue words of data/content_markers.yaml",
    "manual": "the person",
    "default": "nothing known: unknown, which counts as not sfw (policy.unknown_rating_is)",
}


def _yes(value: object) -> str:
    return "-" if value is None else ("yes" if value else "no")


def _dash(value: object) -> str:
    if value is None or value == [] or value == "":
        return "-"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


# --- list -------------------------------------------------------------------------------------------------------------

def select(records: Mapping[str, Mapping], *, rtype: str | None = None, family: str | None = None,
           self_trained: bool = False) -> list[dict]:
    """The records that pass the filters, in list order (type, then file name)."""
    rows = []
    for rec in records.values():
        if rtype and get_field(rec, "identity.type") != rtype:
            continue
        if family and get_field(rec, "identity.family") != family:
            continue
        if self_trained and get_field(rec, "provenance.source_class") != "self-trained":
            continue
        rows.append(dict(rec))
    return sorted(rows, key=res.sort_key)


def self_trained_row(record: Mapping) -> dict:
    """The `--self-trained` row: identity, whether a training record names the file, and that record's ids."""
    trained = get_field(record, "provenance.trained_by") or {}
    return {"id": record.get("id"), "file_name": get_field(record, "identity.file_name"),
            "type": get_field(record, "identity.type"), "family": get_field(record, "identity.family"),
            "training_record": "recorded" if trained.get("run") else "no-record",
            "tool": trained.get("tool"), "project": trained.get("project"), "run": trained.get("run"),
            "checkpoint_step": trained.get("checkpoint_step"), "snapshot": trained.get("snapshot"),
            "network_dim": get_field(record, "lora.network_dim")}


def list_line(record: Mapping, *, self_trained: bool = False) -> str:
    """One `list` line: sha16 | name | type | family | rating | status (+ the training record with --self-trained)."""
    cells = [record.get("id"), get_field(record, "identity.file_name"), get_field(record, "identity.type"),
             get_field(record, "identity.family"), get_field(record, "semantics.content.rating"),
             get_field(record, "lifecycle.install_status")]
    line = " | ".join(_dash(c) for c in cells)
    if self_trained:
        row = self_trained_row(record)
        if row["training_record"] == "recorded":
            line += f" | recorded: {_dash(row['tool'])} {_dash(row['project'])} {row['run']}"
        else:
            line += " | no-record"
    return line


def list_json_row(record: Mapping, *, self_trained: bool = False) -> dict:
    row = res.index_row(record)
    if self_trained:
        row.update({k: v for k, v in self_trained_row(record).items() if k not in row})
    return row


# --- show -------------------------------------------------------------------------------------------------------------

def prompt_policy_line(record: Mapping, authoring: str) -> str:
    """May the assistant write prompt text for this resource? The answer `show` gives, policy by policy."""
    klass = get_field(record, "semantics.content.class") or "unknown"
    if authoring == "user-text-only":
        return "prompt text: only the person's words (policy.prompt_authoring = user-text-only)"
    if klass == "sfw":
        return f"prompt text: allowed under policy.prompt_authoring = {authoring} (class sfw)"
    why = "unknown counts as nsfw" if klass == "unknown" else "class nsfw"
    return (f"prompt text: refused under policy.prompt_authoring = {authoring} ({why}); name this resource by "
            "its id only")


def show_lines(record: Mapping, *, authoring: str, override: res.Override | None) -> list[str]:
    """What `registry show` prints: identity and loader, provenance, integrity, content and policy, licence, usage."""
    rtype = get_field(record, "identity.type")
    klass = get_field(record, "semantics.content.class") or "unknown"
    withhold = authoring == "sfw-only" and klass != "sfw"
    held = f"withheld (class {klass} under policy.prompt_authoring = sfw-only)"
    lines = [f"{record.get('id')} {get_field(record, 'identity.file_name')}",
             f"path: <models_root>/{_dash(get_field(record, 'identity.path'))} "
             f"(status {get_field(record, 'lifecycle.install_status')})",
             f"type: {rtype}; loader: {LOADERS.get(str(rtype), 'not used by the v0.1 graphs')}"]
    family = get_field(record, "identity.family")
    trained_on = get_field(record, "lora.trained_on_family")
    lines.append(f"family: {_dash(family)}" + (f"; trained on: {trained_on}" if trained_on else ""))
    lines.append(f"title: {held if withhold else _dash(get_field(record, 'identity.title'))}")
    prov = record.get("provenance") or {}
    text = f"provenance: {prov.get('source_class')}, site {_dash(prov.get('site'))}, status {prov.get('status')}"
    if prov.get("author"):
        text += f", author {prov['author']}"
    if prov.get("model_id") or prov.get("version_id"):
        text += f", model {_dash(prov.get('model_id'))} version {_dash(prov.get('version_id'))}"
    trained = prov.get("trained_by") or {}
    if trained:
        text += (f"; trained by {trained.get('tool')} project {trained.get('project')} run {trained.get('run')}"
                 + (f" step {trained['checkpoint_step']}" if trained.get("checkpoint_step") is not None else "")
                 + (f" snapshot {trained['snapshot']}" if trained.get("snapshot") else ""))
    lines.append(text)
    integ = record.get("integrity") or {}
    lines.append(f"integrity: sha256 {_dash(integ.get('sha256'))}, {_dash(integ.get('bytes'))} bytes, "
                 f"{_dash(integ.get('tensor_count'))} tensors, {_dash(integ.get('precision'))}, "
                 f"quantization {_dash(integ.get('quantization'))}, hashed {_dash(integ.get('hashed_at'))}")
    lora = record.get("lora")
    if isinstance(lora, Mapping):
        weight = lora.get("recommended_weight") or {}
        triggers = held if withhold else _dash(lora.get("trigger_words"))
        lines.append(f"lora: dim {_dash(lora.get('network_dim'))}, alpha {_dash(lora.get('network_alpha'))}, "
                     f"module {_dash(lora.get('network_module'))}, {_dash(lora.get('lora_type'))}; "
                     f"triggers {triggers}; weight "
                     + (f"{weight.get('min')}-{weight.get('max')}" if weight else "-")
                     + f"; clip skip {_dash(lora.get('clip_skip'))}")
    content = get_field(record, "semantics.content") or {}
    lines.append(f"content: rating {content.get('rating')}, class {klass} (basis {content.get('basis', '-')})")
    lines.append(prompt_policy_line(record, authoring))
    if not withhold and get_field(record, "semantics.description"):
        lines.append(f"description: {get_field(record, 'semantics.description')}")
    lic = record.get("licence") or {}
    lines.append(f"licence: {lic.get('status')} - {_dash(lic.get('name'))}; commercial use "
                 f"{_dash(lic.get('commercial_use'))}; credit required {_yes(lic.get('credit_required'))}; "
                 f"derivatives {_yes(lic.get('derivatives_allowed'))}; relicensing {_yes(lic.get('relicense_allowed'))}; "
                 f"redistributable {_yes(lic.get('redistributable'))}; gated {_yes(lic.get('gated'))}; "
                 f"source {_dash(lic.get('source'))}; checked {_dash(lic.get('checked_at'))}")
    if lic.get("note"):
        lines.append(f"  licence note: {lic['note']}")
    if lic.get("commercial_use") == "none":
        lines.append("  licence flag: commercial_use none (no commercial use of any kind)")
    when = record.get("when_to_use")
    if isinstance(when, Mapping) and any(when.values()):
        lines.append("when to use: " + (held if withhold else
                                         f"{_dash(when.get('summary'))}; good for {_dash(when.get('good_for'))}; "
                                         f"avoid {_dash(when.get('avoid'))}"))
    usage = record.get("usage") or {}
    lines.append(f"usage: card {_dash(usage.get('card'))}; author notes {_dash(usage.get('author_notes'))}")
    life = record.get("lifecycle") or {}
    lines.append(f"lifecycle: {life.get('install_status')}; tested {_dash(life.get('tested_at'))} "
                 f"{_dash(life.get('test_verdict'))}; favorite {_yes(life.get('favorite'))}")
    for rel in record.get("relations") or []:
        lines.append(f"relation: {rel.get('relation')} {rel.get('to')}" + (f" ({rel['note']})" if rel.get("note") else ""))
    if override is not None:
        lines.append(f"overrides: {', '.join(override.values)} (registry/overrides.yaml"
                     + (f": {override.note})" if override.note else ")"))
    return lines


# --- explain ----------------------------------------------------------------------------------------------------------

def _header_for(record: Mapping, reg: res.Registry) -> tuple[cls.Header | None, str]:
    rel = get_field(record, "identity.path")
    if not rel:
        return None, "no file yet (pending)"
    path = reg.model_file(str(rel))
    if not path.is_file():
        return None, "the file is not under <models_root> any more"
    suffix = path.suffix.lower()
    if suffix not in cls.SAFETENSORS_SUFFIXES:
        return None, "not a safetensors file: the header is not read"
    try:
        return cls.read_header(path), "safetensors header read"
    except cls.HeaderUnavailable as exc:
        return None, str(exc)
    except (ValueError, OSError) as exc:
        return None, f"header unreadable: {reason(exc)}"


def _origin(dotted: str, override: res.Override | None) -> str | None:
    if override is not None and dotted in override.values:
        return "registry/overrides.yaml" + (f" ({override.note})" if override.note else "")
    return None


def explain_lines(record: Mapping, reg: res.Registry, folders: cls.Folders, markers: cls.Markers,
                  override: res.Override | None) -> list[str]:
    """Where each classifying value of a record came from, next to what the folder and header say today."""
    rel = str(get_field(record, "identity.path") or "")
    header, note = _header_for(record, reg)
    fresh = cls.classify(reg.model_file(rel), rel, folders, markers, header, note) if rel else None
    lines = [f"{record.get('id')} {get_field(record, 'identity.file_name')}", f"header: {note}"]

    rtype = get_field(record, "identity.type")
    origin = _origin("identity.type", override)
    if origin:
        lines.append(f"type: {rtype} <- {origin}")
    elif fresh and fresh.type and fresh.type.value == rtype:
        header_agrees = fresh.header_type is not None and fresh.header_type.value == rtype
        agrees = f"; the header agrees ({fresh.header_type.reason})" if header_agrees and fresh.folder_type else ""
        lines.append(f"type: {rtype} <- {fresh.type.reason}{agrees}")
    else:
        seen = f"; today the folder and header say {fresh.type.value} ({fresh.type.reason})" \
            if fresh and fresh.type else ""
        lines.append(f"type: {rtype} <- the record (an import, the person or an earlier scan){seen}")
    if fresh and fresh.header_type and fresh.header_type.value != rtype:
        lines.append(f"  the header alone suggests {fresh.header_type.value} ({fresh.header_type.reason})")

    family = get_field(record, "identity.family")
    origin = _origin("identity.family", override)
    detected = fresh.family if fresh else None
    if origin:
        lines.append(f"family: {family} <- {origin}")
    elif family is None:
        lines.append("family: unknown <- nothing in the header or the file name names one; the operator says which")
    elif detected and detected.value == family:
        lines.append(f"family: {family} <- {detected.reason}")
    else:
        lines.append(f"family: {family} <- the record (an import, the person or an earlier scan; a scan never "
                     "replaces a family)")
    if detected and detected.value != family:
        lines.append(f"  the header suggests {detected.value} ({detected.reason})")
    source = get_field(record, "lora.trained_on_family") or family
    if rtype != "checkpoint":
        how = "lora.trained_on_family" if get_field(record, "lora.trained_on_family") else "identity.family"
        lines.append(f"compat source family: {_dash(source)} ({how}); pairs: `easel registry compatible "
                     f"{record.get('id')}`")
    else:
        lines.append(f"compat target family: {_dash(family)}; pairs: `easel registry compatible {record.get('id')}`")

    content = get_field(record, "semantics.content") or {}
    origin = _origin("semantics.content.rating", override) or _origin("semantics.content.class", override)
    basis = content.get("basis") or "default"
    cues = markers.hits(markers.non_sfw, cls.marker_texts(str(get_field(record, "identity.file_name") or ""),
                                                          header, record))
    good = markers.hits(markers.sfw, cls.marker_texts(str(get_field(record, "identity.file_name") or ""),
                                                      header, record))
    lines.append(f"rating: {content.get('rating')}, class {content.get('class')} <- "
                 + (origin or f"basis {basis}: {BASIS_WORDS.get(basis, basis)}"))
    lines.append(f"  cue words: non-sfw {_dash(cues)}; sfw {_dash(good)} ({markers.source})")
    if content.get("class") == "sfw" and cues:
        lines.append("  RATING CONFLICT: rated sfw but a non-sfw cue is present; never use it as sfw; ask the operator")

    source_class = get_field(record, "provenance.source_class")
    origin = _origin("provenance.source_class", override)
    if origin:
        lines.append(f"source class: {source_class} <- {origin}")
    elif source_class == "unknown":
        lines.append("source class: unknown <- nothing recorded; a scan never infers self-trained or downloaded")
    else:
        lines.append(f"source class: {source_class} <- the record (an import or a download)")
    lic = record.get("licence") or {}
    lines.append(f"licence: {lic.get('status')} <- source {_dash(lic.get('source'))}, checked "
                 f"{_dash(lic.get('checked_at'))}" + ("" if lic.get("status") != "unknown"
                                                      else "; unknown: never invent one, ask the operator"))
    if override is not None:
        others = [k for k in override.values if k not in ("identity.type", "identity.family",
                                                          "semantics.content.rating", "semantics.content.class",
                                                          "provenance.source_class")]
        if others:
            lines.append(f"other overrides: {', '.join(others)} <- registry/overrides.yaml")
    return lines

