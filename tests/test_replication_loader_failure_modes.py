"""Failure-mode tests for the replication loader using isolated temporary data."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

_spec = importlib.util.spec_from_file_location(
    "integrate_replication",
    str(_project_root / "scripts" / "integrate_replication.py"),
)
_integrate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integrate_mod)

load_replication_annotation = _integrate_mod.load_replication_annotation
_EXPECTED = _integrate_mod.EXPECTED


def _temp_dataset(tmp_path: Path, dataset: str, content) -> Path:
    """Write one synthetic dataset under pytest's per-test temporary directory."""
    path = tmp_path / _EXPECTED[dataset]
    if isinstance(content, str):
        path.write_text(content, encoding="utf-8")
    else:
        path.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def _valid_record() -> dict:
    return {
        "pair_id": "test-pid",
        "req_a_text": "Req A",
        "req_b_text": "Req B",
        "dependency_status_original": "Requires",
        "normalized_internal_label": "Requires",
        "normalized_mapping_unresolved": False,
        "provenance": {
            "source": "imported",
            "status": "pending_review",
            "annotator_id": None,
        },
    }


def _valid_implemented_by() -> dict:
    return {
        "pair_id": "test-impl",
        "req_a_text": "Req A",
        "req_b_text": "Req B",
        "dependency_status_original": "Implemented by",
        "normalized_internal_label": "Implemented by",
        "normalized_mapping_unresolved": True,
        "provenance": {
            "source": "imported",
            "status": "pending_review",
            "annotator_id": None,
        },
    }


class TestReplicationLoaderFailureModes:
    # 1. Malformed JSON syntax
    def test_malformed_json_syntax(self, tmp_path):
        data_root = _temp_dataset(tmp_path, "adb_evaluation", "{bad json")
        with pytest.raises(json.JSONDecodeError):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 2. Top-level JSON object instead of a list
    def test_top_level_json_object_instead_of_list(self, tmp_path):
        # The loader explicitly rejects non-list JSON with ValueError.
        data_root = _temp_dataset(tmp_path, "adb_evaluation", _valid_record())
        with pytest.raises(ValueError, match="top-level JSON must be a list"):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 3. Non-object record within a list
    def test_non_object_record_within_list(self, tmp_path):
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [1])
        with pytest.raises(ValueError):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 4. Missing required fields
    def test_missing_required_fields(self, tmp_path):
        rec = _valid_record()
        del rec["pair_id"]
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "missing required keys" in str(exc_info.value)

    # 5a. Empty required string field
    def test_empty_string_required_field(self, tmp_path):
        rec = _valid_record()
        rec["pair_id"] = ""
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 5b. Non-string required field
    def test_non_string_required_field(self, tmp_path):
        rec = _valid_record()
        rec["pair_id"] = 99999
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 6. Unsupported original label
    def test_unsupported_original_label(self, tmp_path):
        rec = _valid_record()
        rec["dependency_status_original"] = "UnsupportedLabel"
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 7. Incorrect dataset record count
    def test_incorrect_dataset_record_count(self, tmp_path):
        # adb_evaluation expects 313 records; supplying one valid record is rejected.
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [_valid_record()])
        with pytest.raises(ValueError, match="expected 313 records, got 1"):
            load_replication_annotation("adb_evaluation", data_root=data_root)

    # 8. Duplicate pair IDs
    def test_duplicate_pair_ids(self, tmp_path):
        rec = _valid_record()
        rec["pair_id"] = "dup-id"
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec, rec.copy()])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "duplicate synthetic pair_ids" in str(exc_info.value)

    # 9. Missing or invalid provenance
    def test_missing_provenance(self, tmp_path):
        rec = _valid_record()
        rec["provenance"] = None
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "provenance missing" in str(exc_info.value)

    def test_invalid_provenance_source(self, tmp_path):
        rec = _valid_record()
        rec["provenance"] = {
            "source": "bad_source",
            "status": "pending_review",
            "annotator_id": None,
        }
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "provenance source must be 'imported'" in str(exc_info.value)

    def test_invalid_provenance_status(self, tmp_path):
        rec = _valid_record()
        rec["provenance"] = {
            "source": "imported",
            "status": "bad_status",
            "annotator_id": None,
        }
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "provenance status must be 'pending_review'" in str(exc_info.value)

    def test_invalid_provenance_annotator_id(self, tmp_path):
        rec = _valid_record()
        rec["provenance"]["annotator_id"] = "some_annotator"
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "annotator_id must be None" in str(exc_info.value)

    # 10. Implemented by record with silent remap or incorrect unresolved flag
    def test_implemented_by_silent_remap(self, tmp_path):
        rec = _valid_implemented_by()
        rec["normalized_internal_label"] = "Requires"  # silently remapped
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "silent remap" in str(exc_info.value)

    def test_implemented_by_unresolved_incorrect(self, tmp_path):
        rec = _valid_implemented_by()
        rec["normalized_mapping_unresolved"] = False  # should be True
        data_root = _temp_dataset(tmp_path, "adb_evaluation", [rec])
        with pytest.raises(ValueError) as exc_info:
            load_replication_annotation("adb_evaluation", data_root=data_root)
        assert "mapping_unresolved must be True" in str(exc_info.value)
