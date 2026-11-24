"""Comprehensive tests for Part 8: Controlled Evaluation & Six-Way Ablation Experiments."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from leredd.data_models import (
    AnnotationProvenance,
    AnnotationSource,
    DependencyType,
    EvaluationReport,
    EvaluationMetric,
    Requirement,
    RequirementPair,
    VerificationStatus,
)
from leredd.embeddings import EmbeddingModel
from leredd.evaluation import SplitOverlapError, summarize_candidate_recall_vs_classifier
from scripts.run_full_evaluation import run_evaluation


class _FastOfflineEmbedder(EmbeddingModel):
    def __init__(self, model_name: str | None = None, cache_dir: str | None = None):
        self.model_name = model_name or "offline-deterministic-test"
        self.cache_dir = cache_dir or "."
        self._model = None
        self._fallback_hasher = None

    def encode(self, texts):
        return np.asarray(
            [
                [
                    float(len(t) % 31 + 1),
                    float(sum(map(ord, t)) % 197 + 1),
                    float(t.lower().count("shall") + 1),
                ]
                + [0.0] * 381
                for t in texts
            ],
            dtype=np.float32,
        )


def _verified_pair(
    pair_id: str,
    req_a_id: str,
    req_b_id: str,
    text_a: str,
    text_b: str,
    label: DependencyType,
    *,
    system: str,
    doc: str,
    component: str = "Controller",
) -> dict:
    return RequirementPair(
        id=pair_id,
        req_a=Requirement(
            id=req_a_id,
            text=text_a,
            system=system,
            component=component,
            source=doc,
            metadata={"document": doc},
        ),
        req_b=Requirement(
            id=req_b_id,
            text=text_b,
            system=system,
            component=component,
            source=doc,
            metadata={"document": doc},
        ),
        dependency_type=label,
        provenance=AnnotationProvenance(
            source=AnnotationSource.HUMAN,
            status=VerificationStatus.VERIFIED,
            annotator_id="reviewer-alpha",
            reviewer_id="reviewer-alpha",
            accepted_label=label,
        ),
    ).model_dump(mode="json")


def test_part8_six_controlled_configurations_dev_tuning_and_candidate_recall_separation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("scripts.run_full_evaluation.EmbeddingModel", _FastOfflineEmbedder)

    split_dir = tmp_path / "doc_disjoint_split"
    split_dir.mkdir()

    train_records = [
        _verified_pair(
            "TR-1", "REQ-TR-1", "REQ-TR-2",
            "The brake module requires radar distance input.",
            "The radar module shall publish object distance.",
            DependencyType.REQUIRES, system="SYS_A", doc="doc_train.md",
        ),
        _verified_pair(
            "TR-2", "REQ-TR-3", "REQ-TR-4",
            "Valet mode conflicts with highway speed settings.",
            "Highway mode shall allow 120 km/h.",
            DependencyType.CONFLICTS, system="SYS_A", doc="doc_train.md",
        ),
        _verified_pair(
            "TR-3", "REQ-TR-5", "REQ-TR-6",
            "Radio tuner shall decode FM stations.",
            "Wiper motor shall park at bottom stop.",
            DependencyType.NO_DEPENDENCY, system="SYS_A", doc="doc_train.md",
        ),
    ]
    dev_records = [
        _verified_pair(
            "DV-1", "REQ-DV-1", "REQ-DV-2",
            "Steering assist requires torque sensor input.",
            "Torque sensor shall sample at 100 Hz.",
            DependencyType.REQUIRES, system="SYS_A", doc="doc_dev.md",
        ),
        _verified_pair(
            "DV-2", "REQ-DV-3", "REQ-DV-4",
            "Cabin lamp shall turn on.",
            "Fuel pump shall prime.",
            DependencyType.NO_DEPENDENCY, system="SYS_A", doc="doc_dev.md",
        ),
    ]
    test_records = [
        _verified_pair(
            "TS-1", "REQ-TS-1", "REQ-TS-2",
            "Cruise control requires wheel speed telemetry from REQ-TS-2.",
            "Wheel speed sensor shall report velocity.",
            DependencyType.REQUIRES, system="SYS_A", doc="doc_test.md",
        ),
        _verified_pair(
            "TS-2", "REQ-TS-3", "REQ-TS-4",
            "Bluetooth module shall pair phones.",
            "Rear defroster shall heat glass.",
            DependencyType.NO_DEPENDENCY, system="SYS_A", doc="doc_test.md",
        ),
    ]

    (split_dir / "train_pairs.json").write_text(json.dumps(train_records), encoding="utf-8")
    (split_dir / "dev_pairs.json").write_text(json.dumps(dev_records), encoding="utf-8")
    (split_dir / "test_pairs.json").write_text(json.dumps(test_records), encoding="utf-8")
    (split_dir / "split_metadata.json").write_text(
        json.dumps({"summary": {"strategy": "document", "seed": 123}}), encoding="utf-8"
    )

    cand_manifest = tmp_path / "candidate_manifest.json"
    cand_manifest.write_text(
        json.dumps(
            {
                "strategy": "hybrid",
                "pair_space_recall": 0.42,
                "uncovered_pair_count": 29,
                "exhaustive_pair_count": 50,
            }
        ),
        encoding="utf-8",
    )

    out_report = tmp_path / "eval_out" / "report.json"
    report = run_evaluation(
        split_dir=str(split_dir),
        output_path=str(out_report),
        seed=123,
        bert_model_type="rf",
        candidate_manifest_path=str(cand_manifest),
    )

    # 1. Reproducible configuration & no test-set tuning
    cfg = report["configuration"]
    assert cfg["seed"] == 123
    assert cfg["test_set_tuning"] is False
    assert len(cfg["prompt_template_hash"]) == 16
    assert report["split_integrity"]["checked"] is True
    assert report["split_integrity"]["strategy"] == "document"
    assert report["split_integrity"]["is_disjoint"] is True

    # 2. Dev-set-only threshold tuning executed
    dev_tuning = report["dev_threshold_tuning"]
    assert dev_tuning["status"] == "tuned_on_dev_only"
    assert dev_tuning["partition_used"] == "DEV"
    assert dev_tuning["test_set_touched"] is False

    # 3. All 6 controlled configurations evaluated with full metrics
    six_way = report["ablation_results"]["controlled_six_way_comparison"]
    expected_six_keys = {
        "1_ZeroShot",
        "2_Semantic_Only",
        "3_BM25_Only",
        "4_Existing_Hybrid",
        "5_Metadata_Aware_Hybrid",
        "6_Context_Enhanced_RAG",
    }
    assert set(six_way.keys()) == expected_six_keys
    for key, entry in six_way.items():
        assert "macro_f1" in entry
        assert "overall_accuracy" in entry
        assert "per_class_metrics" in entry
        assert "confusion_matrix" in entry
        for dt in DependencyType:
            cls_m = entry["per_class_metrics"][dt.value]
            assert {"precision", "recall", "f1_score", "support"} <= set(cls_m.keys())

    # 4. Candidate recall reported separately from classifier accuracy
    sep = report["candidate_recall_separation"]
    assert sep["candidate_generation_stage"]["measured_separately"] is True
    assert sep["candidate_generation_stage"]["pair_space_recall"] == pytest.approx(0.42)
    assert sep["classifier_evaluation_stage"]["conditioned_on_supplied_pairs"] is True


def test_part8_document_disjoint_violation_aborts_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("scripts.run_full_evaluation.EmbeddingModel", _FastOfflineEmbedder)
    split_dir = tmp_path / "bad_doc_split"
    split_dir.mkdir()

    shared_doc_train = [
        _verified_pair(
            "TR-1", "R1", "R2", "Req 1", "Req 2",
            DependencyType.REQUIRES, system="S1", doc="shared_srs.md",
        )
    ]
    shared_doc_test = [
        _verified_pair(
            "TS-1", "R3", "R4", "Req 3", "Req 4",
            DependencyType.REQUIRES, system="S2", doc="shared_srs.md",
        )
    ]
    (split_dir / "train_pairs.json").write_text(json.dumps(shared_doc_train), encoding="utf-8")
    (split_dir / "dev_pairs.json").write_text("[]", encoding="utf-8")
    (split_dir / "test_pairs.json").write_text(json.dumps(shared_doc_test), encoding="utf-8")
    (split_dir / "split_metadata.json").write_text(
        json.dumps({"summary": {"strategy": "document"}}), encoding="utf-8"
    )

    with pytest.raises(SplitOverlapError, match="shared_srs.md"):
        run_evaluation(
            split_dir=str(split_dir),
            output_path=str(tmp_path / "never_created.json"),
        )


def test_part8_candidate_recall_helper_keeps_metrics_distinct():
    dummy_report = EvaluationReport(
        system_name="TJA",
        model_name="LEREDD-Hybrid-Metadata",
        overall_accuracy=0.85,
        macro_f1=0.78,
        weighted_f1=0.84,
        per_class_metrics={
            dt.value: EvaluationMetric(precision=0.8, recall=0.8, f1_score=0.8, support=2)
            for dt in DependencyType
        },
    )
    summary = summarize_candidate_recall_vs_classifier(
        dummy_report,
        candidate_manifest={"strategy": "bm25", "pair_space_recall": 0.12, "uncovered_pair_count": 88},
        coverage_report={"weighted_positive_recall": 0.74, "confidence_interval_95": [0.61, 0.87]},
    )
    assert summary["candidate_generation_stage"]["pair_space_recall"] == 0.12
    assert summary["candidate_generation_stage"]["weighted_positive_dependency_recall"] == 0.74
    assert summary["classifier_evaluation_stage"]["overall_accuracy"] == 0.85
    assert summary["classifier_evaluation_stage"]["macro_f1"] == 0.78
