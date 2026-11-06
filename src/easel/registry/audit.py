"""`easel registry audit`: the gaps that make recipes worse or unsafe, each with who fixes it and how.

Read-only. Checks, in the order they are printed:
- rating-conflict: rated sfw, but the file name or the record carries a non-sfw cue (data/content_markers.yaml);
- rating-unknown: a checkpoint, LoRA or embedding with no rating: default-deny, no prompt text may be written;
- undocumented: a resource that is not self-trained and lacks a source link (`provenance.url` or
  `download_url`), a sha256, or any licence information (`licence.status: unknown`);
- unattributed: a LoRA or LyCORIS with no training record (`provenance.trained_by`) and no download source that
  attributes it (a downloaded LoRA with a source link is attributed by its page);
- licence-terms: binding terms recorded in the licence block: `commercial_use: none`, derivatives not allowed;
- no-category: a LoRA without `semantics.category` (a recipe cannot tell a style LoRA from a character LoRA);
- no-family: a resource of a family-bound type without a family (compatibility unknowable);
- no-usage: a checkpoint or LoRA whose card has no value beyond the family priors for weight, cfg, steps, sampler;
- missing-file: a record whose file is not under the model roots any more;
- no-compiler: a checkpoint whose family names no `inference.graph_compiler`;
- pending: an entry of `registry/pending/` that is not downloaded yet.
A resource whose class is not sfw is named by id only. Exit 0 with the report; `--check` exits 1 on any finding.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

from . import add as adding
from . import classify as cls
from . import resources as res
from .shipped import reason

CHECKS = ("rating-conflict", "rating-unknown", "undocumented", "unattributed", "licence-terms", "no-category",
          "no-family", "no-usage", "missing-file", "no-compiler", "pending")
WHAT = {
    "rating-conflict": "rated sfw, but a cue word says otherwise; fix: the operator rates it "
                       "(registry/overrides.yaml semantics.content.*)",
    "rating-unknown": "no rating, so default-deny; fix: the operator rates it (registry/overrides.yaml)",
    "undocumented": "no source link, sha256 or licence; fix: "
                    "`easel registry register-upstream <sha16> --civitai` (or --hf / --url)",
    "unattributed": "a LoRA with no training record and no source; fix: the "
                    "operator says who trained it (overrides.yaml provenance.*) or `register-upstream`",
    "licence-terms": "licence terms that bind; fix: none, they are the author's: respect them in recipes and "
                     "releases",
    "no-category": "a LoRA without a category; fix: the operator names it (overrides.yaml semantics.category)",
    "no-family": "no family, so compatibility is unknowable; fix: overrides.yaml identity.family",
    "no-usage": "only family priors for weight, cfg, steps, sampler; fix: reviewed runs, collected references, "
                "or `easel standards set`",
    "missing-file": "registered but not on disk; fix: `easel registry scan`, or the operator restores the file",
    "no-compiler": "its family names no graph compiler; fix: a family file or workflow template that renders it",
    "pending": "added but not downloaded; fix: `easel registry download <pending>` on the operator's yes",
}
CORE_TYPES = ("checkpoint", "lora", "lycoris", "embedding")
LORA_TYPES = ("lora", "lycoris")
FAMILY_FREE_TYPES = ("upscaler", "detector", "sam")
USAGE_FIELDS = ("weight", "cfg", "steps", "sampler")
EVIDENCE_SOURCES = ("manual", "experience", "author", "references")


@dataclass(frozen=True)
class Finding:
    """One gap: the check, the resource (or pending entry), and what is missing."""

    check: str
    id: str
    name: str
    type: str
    family: str | None
    detail: str


def _name(record: Mapping) -> str:
    klass = res.get_field(record, "semantics.content.class")
    return str(res.get_field(record, "identity.file_name")) if klass == "sfw" else f"(name withheld: class {klass})"


def _has_source(record: Mapping) -> bool:
    return bool(res.get_field(record, "provenance.url") or res.get_field(record, "provenance.download_url"))


def undocumented(record: Mapping) -> list[str]:
    """What the `undocumented` check finds missing on one record (nothing for a self-trained resource)."""
    if res.get_field(record, "provenance.source_class") == "self-trained":
        return []
    gaps = []
    if not _has_source(record):
        gaps.append("no source link")
    if not res.get_field(record, "integrity.sha256"):
        gaps.append("no sha256")
    if res.get_field(record, "licence.status") in (None, "unknown"):
        gaps.append("licence unknown")
    return gaps


def unattributed(record: Mapping) -> str | None:
    """Why a LoRA counts as unattributed (no training record and no attributing source), or None."""
    if res.get_field(record, "identity.type") not in LORA_TYPES or res.get_field(record, "provenance.trained_by"):
        return None
    source_class = res.get_field(record, "provenance.source_class")
    if source_class == "self-trained":
        return "self-trained, but no training record names it (provenance.trained_by)"
    if source_class == "downloaded" and _has_source(record):
        return None
    return f"no training record and no source link (source class {source_class})"


def licence_terms(record: Mapping) -> list[str]:
    terms = []
    if res.get_field(record, "licence.commercial_use") == "none":
        terms.append("commercial_use: none (no commercial use of the model or its images)")
    if res.get_field(record, "licence.derivatives_allowed") is False:
        terms.append("derivatives not allowed (no merges or LoRAs built on it may be shared)")
    return terms


def _card_has_evidence(reg: res.Registry, sha16: str) -> bool | None:
    from easel.standards import cards

    try:
        card = cards.load_card(reg, sha16)
    except Exception:  # noqa: BLE001  (a broken card is the standards verb's to report; here it counts as none)
        return None
    if card is None:
        return None
    return any(v.source in EVIDENCE_SOURCES and v.field in USAGE_FIELDS for v, _at in cards.card_values(card))


def audit(reg: res.Registry, cfg, records: Mapping[str, Mapping], *, only: Iterable[str] | None = None
          ) -> list[Finding]:
    """Every finding over the effective records and the pending entries, in CHECKS order."""
    from easel.standards.sources import load_families

    markers = cls.load_markers(cfg.root)
    families = load_families(cfg.root)
    found: list[Finding] = []

    def add(check: str, rec: Mapping, detail: str, ident: str | None = None) -> None:
        found.append(Finding(check, ident or str(rec.get("id")), _name(rec), str(res.get_field(rec, "identity.type")),
                             res.get_field(rec, "identity.family"), detail))

    for rec in sorted(records.values(), key=res.sort_key):
        rtype = res.get_field(rec, "identity.type")
        klass = res.get_field(rec, "semantics.content.class")
        cues = markers.hits(markers.non_sfw, cls.marker_texts(str(res.get_field(rec, "identity.file_name")), None, rec))
        if klass == "sfw" and cues:
            add("rating-conflict", rec, f"rated sfw (basis {res.get_field(rec, 'semantics.content.basis')}), but "
                                        "its name or record carries a non-sfw cue")
        if rtype in CORE_TYPES and (klass == "unknown" or res.get_field(rec, "semantics.content.rating") == "unknown"):
            add("rating-unknown", rec, "default-deny until the operator rates it")
        if gaps := undocumented(rec):
            add("undocumented", rec, ", ".join(gaps))
        if why := unattributed(rec):
            add("unattributed", rec, why)
        if terms := licence_terms(rec):
            add("licence-terms", rec, "; ".join(terms))
        if rtype in LORA_TYPES and not str(res.get_field(rec, "semantics.category") or "").strip():
            add("no-category", rec, "style / character / concept / utility / anatomy / quality")
        if not res.get_field(rec, "identity.family") and rtype not in FAMILY_FREE_TYPES:
            add("no-family", rec, res.get_field(rec, "identity.base_model_label") or "no base model label")
        if rtype in ("checkpoint", *LORA_TYPES):
            evidence = _card_has_evidence(reg, str(rec["id"]))
            if not evidence:
                add("no-usage", rec, "no usage card yet (`easel standards build`)" if evidence is None
                    else "only family priors")
        status = res.get_field(rec, "lifecycle.install_status")
        rel = res.get_field(rec, "identity.path")
        if status == "missing" or (status == "active" and rel and not reg.model_file(str(rel)).is_file()):
            add("missing-file", rec, f"{rel} (status {status})")
        if rtype == "checkpoint":
            fid = res.get_field(rec, "identity.family")
            fam = families.get(str(fid), ({}, ""))[0] if fid else {}
            inf = fam.get("inference") if isinstance(fam.get("inference"), Mapping) else {}
            if fid and not inf.get("graph_compiler"):
                add("no-compiler", rec, f"family {fid}")
    for path in adding.pending_files(reg):
        try:
            pending = adding.load_pending(reg, path)
        except Exception as exc:  # noqa: BLE001  (an unreadable entry is still a finding, with its first line)
            found.append(Finding("pending", path.stem, "-", "-", None, reason(exc)))
            continue
        add("pending", pending, f"`easel registry download {path.stem}`", ident=path.stem)
    keep = set(only) if only else set(CHECKS)
    return [f for f in sorted(found, key=lambda f: CHECKS.index(f.check)) if f.check in keep]


def lines(findings: list[Finding], resources: int, pending: int) -> list[str]:
    """The report: a summary line, then one section per check with its fix and one line per finding."""
    counts = {c: sum(1 for f in findings if f.check == c) for c in CHECKS}
    summary = ", ".join(f"{c} {counts[c]}" for c in CHECKS if counts[c])
    out = [f"registry audit: {len(findings)} finding(s) over {resources} resource(s) and {pending} pending"
           + (f": {summary}" if summary else "")]
    for check in CHECKS:
        rows = [f for f in findings if f.check == check]
        if not rows:
            continue
        out.append(f"## {check} ({len(rows)}): {WHAT[check]}")
        out += [f"{f.id} | {f.type} | {f.family or '-'} | {f.name} | {f.detail}" for f in rows]
    return out


def as_json(findings: list[Finding]) -> dict:
    counts = {c: sum(1 for f in findings if f.check == c) for c in CHECKS}
    return {"counts": {c: n for c, n in counts.items() if n}, "findings": [asdict(f) for f in findings],
            "fixes": WHAT}
