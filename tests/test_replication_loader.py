"""Loader contract tests using synthetic datasets, independent of private source data."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_spec = importlib.util.spec_from_file_location(
    "integrate_replication",
    str(_PROJECT_ROOT / "scripts" / "integrate_replication.py"),
)
_integrate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integrate_mod)
load_replication_annotation = _integrate_mod.load_replication_annotation


@pytest.fixture
def synthetic_dataset_root(tmp_path):
    """Build expected-size, synthetic imports under pytest's isolated directory."""
    original_labels = ["Requires", "Conflicts", "Details", "Is similar", "No_dependency"]
    for dataset, filename in _integrate_mod.EXPECTED.items():
        records = []
        count = _integrate_mod.EXPECTED_COUNTS[dataset]
        for index in range(count):
            is_implemented_by = index < 6
            original_label = "Implemented by" if is_implemented_by else original_labels[index % len(original_labels)]
            records.append(
                {
                    "pair_id": f"SYNTH_{dataset.upper()}_{index + 1:04d}",
                    "req_a_text": f"Synthetic requirement A {index + 1}.",
                    "req_b_text": f"Synthetic requirement B {index + 1}.",
                    "dependency_status_original": original_label,
                    "normalized_internal_label": original_label,
                    "normalized_mapping_unresolved": is_implemented_by,
                    "provenance": {
                        "source": "imported",
                        "status": "pending_review",
                        "annotator_id": None,
                    },
                }
            )
        (tmp_path / filename).write_text(json.dumps(records), encoding="utf-8")
    return tmp_path


class TestReplicationLoader:
    def test_load_all_datasets(self, synthetic_dataset_root):
        for name in _integrate_mod.EXPECTED:
            records = load_replication_annotation(name, data_root=synthetic_dataset_root)
            assert isinstance(records, list)
            assert len(records) > 0

    def test_implemented_by_not_silently_remapped(self, synthetic_dataset_root):
        for name in _integrate_mod.EXPECTED:
            for record in load_replication_annotation(name, data_root=synthetic_dataset_root):
                if record["dependency_status_original"] == "Implemented by":
                    assert record["normalized_internal_label"] == "Implemented by"
                    assert record["normalized_mapping_unresolved"] is True

    def test_provenance_imported_pending_unavailable_annotator(self, synthetic_dataset_root):
        for name in _integrate_mod.EXPECTED:
            for record in load_replication_annotation(name, data_root=synthetic_dataset_root):
                provenance = record["provenance"]
                assert provenance["source"] == "imported"
                assert provenance["status"] == "pending_review"
                assert provenance["annotator_id"] is None

    def test_all_expected_counts_are_enforced(self, synthetic_dataset_root):
        for name, expected_count in _integrate_mod.EXPECTED_COUNTS.items():
            assert len(load_replication_annotation(name, data_root=synthetic_dataset_root)) == expected_count

    def test_deterministic_reload_identity(self, synthetic_dataset_root):
        for name in _integrate_mod.EXPECTED:
            first = load_replication_annotation(name, data_root=synthetic_dataset_root)
            second = load_replication_annotation(name, data_root=synthetic_dataset_root)
            assert len(first) == len(second)
            assert first[0]["pair_id"] == second[0]["pair_id"]
            assert first[0]["dependency_status_original"] == second[0]["dependency_status_original"]

    def test_default_root_remains_the_configured_replication_directory(self):
        assert _integrate_mod._DATA_ROOT == _PROJECT_ROOT / "data" / "replication"
