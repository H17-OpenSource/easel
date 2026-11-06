"""Family compatibility: the rules of data/compat.yaml and the join `easel registry compatible` prints.

The join pairs every resource that is not a checkpoint with every checkpoint; the rule is looked up by (source
family, target family, resource type), where the source family is the family the resource was trained on
(`lora.trained_on_family`) and otherwise the family it is filed under, and the target family is the checkpoint's.
A pair without a rule (or with a family unknown on either side) is `untested` with confidence 0. Resources are
paired whatever their install status; the status travels in the row so the caller can say a file is missing.

Rules: the shipped data/compat.yaml, then `<home>/data/compat.yaml`, whose rule wins for the same key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .resources import get_field
from .shipped import DATA, DataError, load_yaml, schema_errors

VERDICT_ORDER = ("native", "works", "degraded", "untested", "broken")
OFFERED = ("native", "works", "degraded")  # what `compatible` lists without --all
RuleKey = tuple[str, str, str]


@dataclass(frozen=True)
class Rule:
    source_family: str
    target_family: str
    artifact_type: str
    verdict: str
    confidence: float
    reason: str | None
    origin: str  # data/compat.yaml or <home>/data/compat.yaml


@dataclass(frozen=True)
class Pair:
    """One row of the join: a resource, a checkpoint, and the rule's verdict for them."""

    resource: str
    file_name: str
    artifact_type: str
    source_family: str | None
    checkpoint: str
    checkpoint_file: str
    checkpoint_family: str | None
    verdict: str
    confidence: float
    reason: str | None
    origin: str | None
    status: str | None
    checkpoint_status: str | None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _read_rules(path: Path, origin: str, warnings: list[str]) -> dict[RuleKey, Rule]:
    data = load_yaml(path, origin)
    problems = schema_errors("compat", data)
    if problems or not isinstance(data, dict):
        raise DataError([f"{origin} is not a valid easel/compat/1 file:"] + [f"  {p}" for p in problems])
    rules: dict[RuleKey, Rule] = {}
    for item in data.get("rules") or []:
        key = (item["source_family"], item["target_family"], item["artifact_type"])
        if key in rules:
            warnings.append(f"{origin}: two rules for {' -> '.join(key[:2])} {key[2]}; the later one counts")
        rules[key] = Rule(*key, verdict=item["verdict"], confidence=float(item["confidence"]),
                          reason=item.get("reason"), origin=origin)
    return rules


def load_rules(home: Path | None = None, warnings: list[str] | None = None) -> dict[RuleKey, Rule]:
    """The shipped rules overlaid with the instance's; duplicate keys inside one file go to `warnings`."""
    warnings = [] if warnings is None else warnings
    rules = _read_rules(DATA / "compat.yaml", "data/compat.yaml", warnings)
    local = home / "data" / "compat.yaml" if home is not None else None
    if local is not None and local.is_file():
        rules.update(_read_rules(local, "<home>/data/compat.yaml", warnings))
    return rules


def source_family(record: Mapping) -> str | None:
    """COALESCE(trained_on_family, family): the family a resource brings to a pairing."""
    return get_field(record, "lora.trained_on_family") or get_field(record, "identity.family")


def pair(resource: Mapping, checkpoint: Mapping, rules: Mapping[RuleKey, Rule]) -> Pair:
    """The verdict for one resource on one checkpoint."""
    src, dst = source_family(resource), get_field(checkpoint, "identity.family")
    rtype = get_field(resource, "identity.type")
    rule = rules.get((src, dst, rtype)) if src and dst else None
    return Pair(resource=str(resource.get("id")), file_name=str(get_field(resource, "identity.file_name")),
                artifact_type=str(rtype), source_family=src, checkpoint=str(checkpoint.get("id")),
                checkpoint_file=str(get_field(checkpoint, "identity.file_name")), checkpoint_family=dst,
                verdict=rule.verdict if rule else "untested", confidence=rule.confidence if rule else 0.0,
                reason=rule.reason if rule else None, origin=rule.origin if rule else None,
                status=get_field(resource, "lifecycle.install_status"),
                checkpoint_status=get_field(checkpoint, "lifecycle.install_status"))


def join(records: Iterable[Mapping], rules: Mapping[RuleKey, Rule]) -> list[Pair]:
    """The whole join: every non-checkpoint resource x every checkpoint."""
    records = list(records)
    checkpoints = [r for r in records if get_field(r, "identity.type") == "checkpoint"]
    others = [r for r in records if get_field(r, "identity.type") != "checkpoint"]
    return [pair(a, c, rules) for a in others for c in checkpoints]


def for_resource(sha16: str, records: Mapping[str, Mapping], rules: Mapping[RuleKey, Rule]) -> list[Pair]:
    """The rows of the join that involve one resource: as the checkpoint, or as the add-on."""
    rows = [p for p in join(records.values(), rules) if sha16 in (p.resource, p.checkpoint)]
    is_checkpoint = get_field(records[sha16], "identity.type") == "checkpoint"

    def key(p: Pair) -> tuple:
        other = p.file_name if is_checkpoint else p.checkpoint_file
        rank = VERDICT_ORDER.index(p.verdict) if p.verdict in VERDICT_ORDER else len(VERDICT_ORDER)
        return (rank, -p.confidence, p.artifact_type, other.lower())

    return sorted(rows, key=key)
