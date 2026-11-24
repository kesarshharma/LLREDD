"""Loader for original replication package datasets (standard-library only)."""
from __future__ import annotations
import json
from pathlib import Path

_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "replication"

EXPECTED = {
    "adb_evaluation": "adb_evaluation_annotation.json",
    "adb_pilot": "adb_pilot_annotation.json",
    "apa": "apa_annotation.json",
    "tja": "tja_annotation.json",
}

INTERNAL_LABELS = {"Requires", "Implements", "Conflicts", "Details", "Is similar", "No_dependency"}
ALLOWED_ORIGINAL_LABELS = INTERNAL_LABELS | {"Implemented by"}

EXPECTED_COUNTS = {
    "adb_evaluation": 313,
    "adb_pilot": 100,
    "apa": 200,
    "tja": 200,
}


def load_replication_annotation(dataset="adb_evaluation", *, data_root: str | Path | None = None):
    if dataset not in EXPECTED:
        raise ValueError(f"Unknown dataset: {dataset}. Expected one of {list(EXPECTED)}")
    root = Path(data_root) if data_root is not None else _DATA_ROOT
    path = root / EXPECTED[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Replication annotation file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{dataset}: top-level JSON must be a list, got {type(records).__name__}")
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"{dataset}[{idx}]: record is not a dict")
        required = {"pair_id", "req_a_text", "req_b_text", "dependency_status_original",
                    "normalized_internal_label", "normalized_mapping_unresolved", "provenance"}
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(f"{dataset}[{idx}] missing required keys: {missing}")
        # Type and non-empty checks for required string fields
        for k in ("pair_id", "req_a_text", "req_b_text", "dependency_status_original", "normalized_internal_label"):
            v = rec.get(k)
            if not isinstance(v, str) or v.strip() == "":
                raise ValueError(f"{dataset}[{idx}]: {k} must be non-empty string")
        # Type check for mapping-unresolved flag
        v_unresolved = rec.get("normalized_mapping_unresolved")
        if not isinstance(v_unresolved, bool):
            raise ValueError(f"{dataset}[{idx}]: normalized_mapping_unresolved must be bool")
        # Provenance must be a dict
        prov = rec.get("provenance")
        if prov is None:
            raise ValueError(f"{dataset}[{idx}]: provenance missing")
        if not isinstance(prov, dict):
            raise ValueError(f"{dataset}[{idx}]: provenance must be a dict")
        if prov.get("source") != "imported":
            raise ValueError(f"{dataset}[{idx}]: provenance source must be 'imported', got {prov.get('source')}")
        if prov.get("status") != "pending_review":
            raise ValueError(f"{dataset}[{idx}]: provenance status must be 'pending_review', got {prov.get('status')}")
        if prov.get("annotator_id") is not None:
            raise ValueError(f"{dataset}[{idx}]: annotator_id must be None for unavailable original annotation")
        original_label = rec.get("dependency_status_original")
        if original_label not in ALLOWED_ORIGINAL_LABELS:
            raise ValueError(f"{dataset}[{idx}]: unsupported original dependency label: {original_label}")
        if original_label == "Implemented by":
            if rec.get("normalized_internal_label") != "Implemented by":
                raise ValueError(f"{dataset}[{idx}]: silent remap of 'Implemented by' detected")
            if rec.get("normalized_mapping_unresolved") is not True:
                raise ValueError(f"{dataset}[{idx}]: mapping_unresolved must be True for 'Implemented by'")
    # Unique synthetic IDs
    pair_ids = [r["pair_id"] for r in records]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError(f"{dataset}: duplicate synthetic pair_ids detected")
    expected_n = EXPECTED_COUNTS.get(dataset)
    if expected_n is not None and len(records) != expected_n:
        raise ValueError(f"{dataset}: expected {expected_n} records, got {len(records)}")
    return records


def test_load_all_datasets():
    for key in EXPECTED:
        records = load_replication_annotation(dataset=key)
        assert isinstance(records, list)
        assert len(records) > 0
        for r in records:
            if r.get("dependency_status_original") == "Implemented by":
                assert r.get("normalized_internal_label") == "Implemented by"
                assert r.get("normalized_mapping_unresolved") is True
