"""Focused loader tests for integrated replication dataset JSON files."""
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib.util, sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_spec = importlib.util.spec_from_file_location("integrate_replication", str(_PROJECT_ROOT / "scripts" / "integrate_replication.py"))
_integrate_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_integrate_mod)
load_replication_annotation = _integrate_mod.load_replication_annotation
test_load_all_datasets = _integrate_mod.test_load_all_datasets

class TestReplicationLoader:
    def test_load_all_datasets(self):
        for name in ["adb_evaluation", "adb_pilot", "apa", "tja"]:
            records = load_replication_annotation(dataset=name)
            assert isinstance(records, list)
            assert len(records) > 0

    def test_implemented_by_not_silently_remapped(self):
        for name in ["adb_evaluation", "adb_pilot", "apa", "tja"]:
            for r in load_replication_annotation(dataset=name):
                if r.get("dependency_status_original") == "Implemented by":
                    assert r.get("normalized_internal_label") == "Implemented by"
                    assert r.get("normalized_mapping_unresolved") is True

    def test_provenance_imported_pending_unavailable_annotator(self):
        for name in ["adb_evaluation", "adb_pilot", "apa", "tja"]:
            for r in load_replication_annotation(dataset=name):
                prov = r.get("provenance")
                assert prov is not None
                assert prov.get("source") == "imported"
                assert prov.get("status") == "pending_review"
                assert prov.get("annotator_id") is None

    def test_all_counts_813(self):
        assert len(load_replication_annotation("adb_evaluation")) == 313
        assert len(load_replication_annotation("adb_pilot")) == 100
        assert len(load_replication_annotation("apa")) == 200
        assert len(load_replication_annotation("tja")) == 200

    def test_deterministic_reload_identity(self):
        for name in ["adb_evaluation", "adb_pilot", "apa", "tja"]:
            a = load_replication_annotation(dataset=name)
            b = load_replication_annotation(dataset=name)
            assert len(a) == len(b)
            assert a[0]["pair_id"] == b[0]["pair_id"]
            assert a[0]["dependency_status_original"] == b[0]["dependency_status_original"]
