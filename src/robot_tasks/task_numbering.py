"""Explicit translation of pre-publication records, never an implicit CLI alias.

Old L1-03 meant the adapter task; current L1-03 means the hat. Delivered USD
directories retain their original names and are resolved through manifests.
"""
from functools import lru_cache
import json
import re

from .paths import PROJECT_ROOT

SCHEME = "paper_alignment_20261026"
PREVIOUS_SCHEME = "sequential_20261021"
RECLASSIFIED_SCHEME = "level_reclassification_20261024"
PAPER_RENUMBERED = {"L3-04": "L3-03", "L3-05": "L3-04"}
RECLASSIFIED = {"L3-03": "L2-07", **PAPER_RENUMBERED}


@lru_cache(maxsize=1)
def numbering():
    path = PROJECT_ROOT / "configs/benchmark/task_numbering.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _translate(task_id, mapping):
    for suffix in ("_variant01", "_v3", ""):
        if not suffix or task_id.endswith(suffix):
            base = task_id.removesuffix(suffix) if suffix else task_id
            if base not in mapping:
                raise ValueError(f"Unmapped task ID: {task_id}")
            return mapping[base] + suffix


def from_legacy(task_id):
    """Call only when the input is explicitly a pre-renumbering identity."""
    return _translate(task_id, {r["legacy_id"]: r["id"] for r in numbering()["tasks"]})


def to_legacy(task_id):
    """Resolve a current ID to its immutable delivered-asset/history identity."""
    return _translate(task_id, {r["id"]: r["legacy_id"] for r in numbering()["tasks"]})


def previous_task_id(task_id, scheme=PREVIOUS_SCHEME):
    """Resolve a current ID in an explicitly identified prior numbering scheme."""
    if scheme not in (PREVIOUS_SCHEME, RECLASSIFIED_SCHEME):
        raise ValueError(f"Unsupported prior numbering scheme: {scheme}")
    changes = RECLASSIFIED if scheme == PREVIOUS_SCHEME else PAPER_RENUMBERED
    return _translate(task_id, {
        row["id"]: next((old for old, new in changes.items()
                         if new == row["id"]), row["id"])
        for row in numbering()["tasks"]
    })


def historical_setup_record(record, scheme=None):
    """Translate identity annotations after verifying the archived byte hash."""
    if isinstance(record, dict):
        return {k: historical_setup_record(v, scheme) for k, v in record.items()}
    if isinstance(record, list):
        return [historical_setup_record(v, scheme) for v in record]
    if isinstance(record, str):
        ids = (RECLASSIFIED if scheme == PREVIOUS_SCHEME else
               PAPER_RENUMBERED if scheme == RECLASSIFIED_SCHEME else
               {r["legacy_id"]: r["id"] for r in numbering()["tasks"]})
        return re.sub(r"(?<![\w/-])L[123]-\d{2}(?!\d)",
                      lambda m: ids.get(m[0], m[0]), record)
    return record
