"""Replication import contract tests; private source spreadsheets stay optional."""
import json
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SOURCE_ROOT = _PROJECT_ROOT / "data" / "replication"

EXPECTED = {
    "adb_evaluation_annotation.json": 313,
    "adb_pilot_annotation.json": 100,
    "apa_annotation.json": 200,
    "tja_annotation.json": 200,
}

PREFIXES = {
    "adb_evaluation_annotation.json": "ADB_EVAL_ROW_",
    "adb_pilot_annotation.json": "ADB_PILOT_ROW_",
    "apa_annotation.json": "APA_ROW_",
    "tja_annotation.json": "TJA_ROW_",
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

_LOCAL_SOURCE_FILES_PRESENT = all((_SOURCE_ROOT / name).is_file() for name in EXPECTED)


def _load(name: str, root: Path = _SOURCE_ROOT):
    path = root / name
    assert path.exists(), f"Missing replication file: {path}"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def dataset_roots(tmp_path):
    """Always test the schema with synthetic rows; also inspect private local rows when present."""
    synthetic_root = tmp_path / "synthetic_replication"
    synthetic_root.mkdir()
    implemented_by_counts = {
        "adb_evaluation_annotation.json": 18,
        "adb_pilot_annotation.json": 6,
        "apa_annotation.json": 3,
        "tja_annotation.json": 10,
    }
    supported_labels = ["Requires", "Conflicts", "Details", "Is similar", "No_dependency"]

    for name, count in EXPECTED.items():
        records = []
        for index in range(count):
            is_implemented_by = index < implemented_by_counts[name]
            label = "Implemented by" if is_implemented_by else supported_labels[index % len(supported_labels)]
            records.append(
                {
                    "pair_id": f"{PREFIXES[name]}{index + 1:04d}",
                    "req_a_text": f"Synthetic requirement A {index + 1}.",
                    "req_b_text": f"Synthetic requirement B {index + 1}.",
                    "dependency_status_original": label,
                    "normalized_internal_label": label,
                    "normalized_mapping_unresolved": is_implemented_by,
                    "provenance": {
                        "source": "imported",
                        "status": "pending_review",
                        "annotator_id": None,
                        "review_notes": "synthetic fixture; not human annotation",
                        "worksheet": "Annotations",
                        "original_row_number": index + 2,
                        "spreadsheet_file": f"SYNTHETIC_{name.removesuffix('.json')}.xlsx",
                    },
                }
            )
        (synthetic_root / name).write_text(json.dumps(records), encoding="utf-8")

    roots = [synthetic_root]
    if _LOCAL_SOURCE_FILES_PRESENT:
        roots.append(_SOURCE_ROOT)
    return roots


@pytest.mark.skipif(
    not _LOCAL_SOURCE_FILES_PRESENT,
    reason="Imported replication assets are local-only; redistribution rights are unresolved.",
)
def test_all_four_local_source_files_present_and_counts():
    """When supplied locally, check the actual imported files and their exact counts."""
    for name, expected in EXPECTED.items():
        records = _load(name)
        assert isinstance(records, list), f"{name} is not a list"
        assert len(records) == expected, f"{name}: expected {expected}, got {len(records)}"


class TestReplicationDataset:
    """Validation of record structure and metadata without requiring source assets."""

    def test_all_records_have_required_fields(self, dataset_roots):
        for root in dataset_roots:
            for name in EXPECTED:
                records = _load(name, root)
                for idx, record in enumerate(records):
                    missing = REQUIRED_KEYS - set(record)
                    assert not missing, f"{name}[{idx}]: missing keys {missing}"
                    provenance = record.get("provenance", {})
                    missing_provenance = PROVENANCE_KEYS - set(provenance)
                    assert not missing_provenance, f"{name}[{idx}]: provenance missing {missing_provenance}"
                    assert record["pair_id"].startswith(PREFIXES[name])
                    assert record["dependency_status_original"] in {
                        "Implemented by", "Requires", "Conflicts", "Details", "Is similar", "No_dependency"
                    }

    def test_implemented_by_preserved_exactly(self, dataset_roots):
        for root in dataset_roots:
            for name in EXPECTED:
                records = _load(name, root)
                implemented_by = [
                    record for record in records
                    if record.get("dependency_status_original") == "Implemented by"
                ]
                for record in implemented_by:
                    assert record["dependency_status_original"] == "Implemented by"
                    assert record["normalized_internal_label"] == "Implemented by"
                    assert record["normalized_mapping_unresolved"] is True
            if root == _SOURCE_ROOT:
                adb_evaluation = _load("adb_evaluation_annotation.json", root)
                assert sum(
                    record["dependency_status_original"] == "Implemented by"
                    for record in adb_evaluation
                ) == 18

    def test_original_row_numbers_preserved_and_unique(self, dataset_roots):
        for root in dataset_roots:
            for name in EXPECTED:
                records = _load(name, root)
                rows = [record["provenance"]["original_row_number"] for record in records]
                assert all(isinstance(row, int) and row > 1 for row in rows)
                assert len(rows) == len(set(rows))

    def test_source_worksheet_and_spreadsheet_file_recorded(self, dataset_roots):
        for root in dataset_roots:
            for name in EXPECTED:
                for record in _load(name, root):
                    provenance = record["provenance"]
                    assert provenance["source"] == "imported"
                    assert provenance["status"] == "pending_review"
                    assert provenance["annotator_id"] is None
                    assert provenance["worksheet"] in ("Sheet1", "Annotations")
                    assert provenance["spreadsheet_file"]

    def test_deterministic_reload_identity(self, dataset_roots):
        for root in dataset_roots:
            for name in EXPECTED:
                first = _load(name, root)
                second = _load(name, root)
                assert len(first) == len(second)
                assert first[0]["pair_id"] == second[0]["pair_id"]
                assert first[-1]["pair_id"] == second[-1]["pair_id"]
                assert first[0]["dependency_status_original"] == second[0]["dependency_status_original"]

    def test_no_modification_of_existing_reports_or_fixtures(self):
        report_path = _PROJECT_ROOT / "data" / "evaluation" / "system_disjoint_report.json"
        if report_path.exists():
            content = report_path.read_text(encoding="utf-8")
            assert "LEREDD_Hybrid" in content, "Existing evaluation report altered unexpectedly"
