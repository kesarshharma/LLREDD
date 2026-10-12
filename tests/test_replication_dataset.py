"""Tests for replication dataset integration (read-only, no openpyxl required)."""
import json
import os
from pathlib import Path

# Project root is parent of tests/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED = {
    "adb_evaluation_annotation.json": 313,
    "adb_pilot_annotation.json": 100,
    "apa_annotation.json": 200,
    "tja_annotation.json": 200,
}

REQUIRED_KEYS = {
    "pair_id",
    "req_a_text",
    "req_b_text",
    "dependency_status_original",
    "normalized_internal_label",
    "normalized_mapping_unresolved",
    "provenance",
}

PROVENANCE_KEYS = {
    "source",
    "status",
    "annotator_id",
    "review_notes",
    "worksheet",
    "original_row_number",
    "spreadsheet_file",
}


def _load(name: str):
    path = _PROJECT_ROOT / "data" / "replication" / name
    assert path.exists(), f"Missing replication file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestReplicationDataset:
    """Validation of integrated replication package datasets."""

    def test_all_four_files_present_and_counts(self):
        for name, expected in EXPECTED.items():
            records = _load(name)
            assert isinstance(records, list), f"{name} is not a list"
            assert len(records) == expected, f"{name}: expected {expected}, got {len(records)}"

    def test_all_records_have_required_fields(self):
        for name in EXPECTED:
            records = _load(name)
            for idx, rec in enumerate(records):
                missing = REQUIRED_KEYS - set(rec.keys())
                assert not missing, f"{name}[{idx}]: missing keys {missing}"
                prov = rec.get("provenance", {})
                missing_prov = PROVENANCE_KEYS - set(prov.keys())
                assert not missing_prov, f"{name}[{idx}]: provenance missing {missing_prov}"
                # Synthetic ID format verified
                expected_prefix_map = {
                    "adb_evaluation_annotation.json": "ADB_EVAL_ROW_",
                    "adb_pilot_annotation.json": "ADB_PILOT_ROW_",
                    "apa_annotation.json": "APA_ROW_",
                    "tja_annotation.json": "TJA_ROW_"
                }
                assert rec["pair_id"].startswith(expected_prefix_map[name]), f"Unexpected synthetic ID format: {rec['pair_id']}"
                # Original label preserved (not silent remapped)
                assert rec["dependency_status_original"] in {
                    "Implemented by", "Requires", "Conflicts", "Details", "Is similar", "No_dependency"
                }, f"{name}[{idx}]: unexpected original label {rec['dependency_status_original']}"

    def test_implemented_by_preserved_exactly(self):
        for name in EXPECTED:
            records = _load(name)
            implemented_by = [r for r in records if r.get("dependency_status_original") == "Implemented by"]
            # Must exist in original where present; do not assert exact count per file
            if name == "adb_evaluation_annotation.json":
                assert len(implemented_by) == 18, f"ADB_EVAL Implemented by count expected 18, got {len(implemented_by)}"
            # Preserve original label; do not silently remap to Implemen...
            for r in implemented_by:
                assert r["dependency_status_original"] == "Implemented by"
                assert r["normalized_internal_label"] == "Implemented by", f"{name}: silent remap to Implemen... detected"
                assert r["normalized_mapping_unresolved"] is True

    def test_original_row_numbers_preserved_and_unique(self):
        for name in EXPECTED:
            records = _load(name)
            rows = [r["provenance"]["original_row_number"] for r in records]
            # All must be positive integers (original worksheet rows, 2-based for first data)
            for r in rows:
                assert isinstance(r, int) and r > 1, f"{name}: expected row > 1, got {r}"
            # No duplicate row numbers within file
            assert len(rows) == len(set(rows)), f"{name}: duplicate original_row_numbers found"

    def test_source_worksheet_and_spreadsheet_file_recorded(self):
        for name in EXPECTED:
            records = _load(name)
            for r in records:
                prov = r.get("provenance", {})
                assert prov.get("source") == "imported"
                assert prov.get("status") == "pending_review"
                assert prov.get("annotator_id") is None, f"{name}[{r['pair_id']}]: annotator_id unavailable for imported replication data"
                assert prov.get("worksheet") in ("Sheet1", "Annotations")
                assert "spreadsheet_file" in prov

    def test_deterministic_reload_identity(self):
        """Re-loading should produce identical structures (same counts, same first/last IDs)."""
        for name in EXPECTED:
            first = _load(name)
            second = _load(name)
            assert len(first) == len(second)
            assert first[0]["pair_id"] == second[0]["pair_id"]
            assert first[-1]["pair_id"] == second[-1]["pair_id"]
            assert first[0]["dependency_status_original"] == second[0]["dependency_status_original"]

    def test_no_modification_of_existing_reports_or_fixtures(self):
        # Ensure existing evaluation reports and fixtures are unchanged from integration
        # This is a minimal guard; full verification requires external inspection
        report_path = Path("data/evaluation/system_disjoint_report.json")
        if report_path.exists():
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()
            assert "LEREDD_Hybrid" in content, "Existing evaluation report altered unexpectedly"
