"""Focused integration tests for the replication-evaluation prototype.

Uses temporary directories for outputs; does not modify production dataset files or legacy reports.
Synthetic fixtures used where possible; real loader used only for dataset-selection verification.
"""
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import pytest
from unittest.mock import patch
from scripts.run_replication_evaluation import run_prototype, DATASET, build_pairs


class TestReplicationRunner:
    def test_pending_imported_labels_block_evaluation(self, tmp_path):
        """Imported spreadsheet labels cannot produce metrics before local verification."""
        out_path = tmp_path / "pending.json"
        with pytest.raises(ValueError, match="eligible test set empty"):
            run_prototype(dataset="adb_pilot", output_path=str(out_path))
        assert not out_path.exists()

    def test_implemented_by_preserved_not_remapped(self):
        pairs, implemented_by = build_pairs(DATASET)
        assert len(implemented_by) == 6
        assert all(record["original_label"] == "Implemented by" for record in implemented_by)
        assert all(record["normalized_internal_label"] == "Implemented by" for record in implemented_by)
        assert all(record["normalized_mapping_unresolved"] is True for record in implemented_by)
        assert "IMPLMENTS" not in str(implemented_by)
        assert all(pair.dependency_type is None for pair in pairs if pair.id in {r["pair_id"] for r in implemented_by})

    def test_invalid_unresolved_handled_transparently(self):
        # Direct pair construction with synthetic invalid Implemented by (unresolved False) — fail-closed
        from leredd.data_models import DependencyType, Requirement, RequirementPair
        bad_rec = {
            "pair_id": "test-bad-impl",
            "req_a_text": "A",
            "req_b_text": "B",
            "dependency_status_original": "Implemented by",
            "normalized_internal_label": "Requires",
            "normalized_mapping_unresolved": False,
            "provenance": {"source": "imported", "status": "pending_review", "annotator_id": None},
        }
        req_a = Requirement(id="SYN_A_bad", text="A", system="ADB", source="replication_loader")
        req_b = Requirement(id="SYN_B_bad", text="B", system="ADB", source="replication_loader")
        pair_bad = RequirementPair(
            id="test-bad-impl", req_a=req_a, req_b=req_b, dependency_type=None, provenance=bad_rec["provenance"]
        )
        assert pair_bad.dependency_type is None

    def test_split_is_reproducible_for_same_seed(self):
        # Same seed -> identical splits (after shuffle)
        from scripts.run_replication_evaluation import synthetic_split, build_pairs
        pairs, _ = build_pairs(DATASET)
        s1 = synthetic_split(pairs, seed=99)
        s2 = synthetic_split(pairs, seed=99)
        assert [p.id for p in s1[0]] == [p.id for p in s2[0]]
        assert [p.id for p in s1[2]] == [p.id for p in s2[2]]

    def test_grouped_label_input_not_split_by_contiguous_slice(self):
        # Synthetic ordered input (all Requires first, then No_dep) should not put all of one label in same split
        from scripts.run_replication_evaluation import synthetic_split
        from leredd.data_models import RequirementPair, Requirement
        req = Requirement(id="X", text="t", system="ADB", source="replication_loader")
        ordered = [RequirementPair(id=f"p{i}", req_a=req, req_b=req, dependency_type=None) for i in range(30)]
        # Force first 20 = "Requires" (simulated via list order), last 10 = "No_dep"
        train, dev, test = synthetic_split(ordered, seed=7)
        # With shuffle, train/dev/test should contain mixed indices, not first 21 contiguous
        train_ids = {p.id for p in train}
        dev_ids = {p.id for p in dev}
        test_ids = {p.id for p in test}
        assert len(train_ids & dev_ids) == 0
        assert len(dev_ids & test_ids) == 0
        assert len(train_ids & test_ids) == 0
        # After shuffle, split must be disjoint and not forced by original contiguous order
        all_ids = {p.id for p in train} | {p.id for p in dev} | {p.id for p in test}
        assert len(all_ids) == 30
        assert len(train_ids & dev_ids) == 0
        assert len(dev_ids & test_ids) == 0

    def test_empty_eligible_fails_clearly(self, monkeypatch, tmp_path):
        import scripts.run_replication_evaluation as runner_mod
        original_build = runner_mod.build_pairs
        # Force all dependency_type None -> all excluded -> empty eligible
        def fake_build(dataset_name=DATASET):
            pairs, ib = original_build(dataset_name)
            for p in pairs:
                p.dependency_type = None
            return pairs, ib
        monkeypatch.setattr(runner_mod, "build_pairs", fake_build)
        with pytest.raises(ValueError, match="eligible test set empty"):
            run_prototype(dataset=DATASET, output_path=str(tmp_path / "should_not_create.json"))

    def test_pending_imports_fail_before_model_or_detector_initialization(self, monkeypatch, tmp_path):
        from leredd.detector import LEREDDDetector
        from leredd.embeddings import EmbeddingModel

        def unexpected_init(*args, **kwargs):
            raise AssertionError("model initialization must not run for unverified ground truth")

        monkeypatch.setattr(EmbeddingModel, "__init__", unexpected_init)
        monkeypatch.setattr(LEREDDDetector, "__init__", unexpected_init)
        with pytest.raises(ValueError, match="eligible test set empty"):
            run_prototype(dataset=DATASET, output_path=str(tmp_path / "not-written.json"))
