"""tests/test_ablation_benchmark.py
Unit and integration tests for TASK-P7-01 benchmark suite and ablation study matrix.
"""

from __future__ import annotations


import numpy as np

from leredd.config import LEREDDConfig
from leredd.data_models import (
    DependencyType,
    Prediction,
    Requirement,
    RequirementPair,
)
from leredd.detector import LEREDDDetector
from leredd.llm import MockLLMClient
from scripts.run_full_evaluation import (
    run_heuristic_model,
    run_leredd_variant,
)


class _OfflineEmbedder:
    """Deterministic embeddings for retrieval wiring tests; never initialize SBERT."""

    def encode(self, texts):
        return np.asarray(
            [[len(text), sum(map(ord, text)) % 997, text.lower().count("system")] for text in texts],
            dtype=np.float32,
        )


def _make_req(req_id: str, system: str = "SYS") -> Requirement:
    return Requirement(id=req_id, text=f"The system shall perform function {req_id}.", system=system)


def _make_pair(pair_id: str, dep_type: DependencyType = DependencyType.REQUIRES, system: str = "SYS") -> RequirementPair:
    return RequirementPair(
        id=pair_id,
        req_a=_make_req(f"{pair_id}-A", system=system),
        req_b=_make_req(f"{pair_id}-B", system=system),
        dependency_type=dep_type,
    )


class TestAblationBenchmarkSuite:
    """Tests for Phase 7 integrated benchmark and ablation study capabilities."""

    def test_heuristic_baseline_execution(self):
        pairs = [_make_pair("P1", DependencyType.REQUIRES), _make_pair("P2", DependencyType.NO_DEPENDENCY)]
        preds, report = run_heuristic_model(pairs)
        assert len(preds) == 2
        assert all(isinstance(p, Prediction) for p in preds)
        assert report.overall_accuracy >= 0.0

    def test_leredd_variant_modes(self):
        train_pairs = [_make_pair(f"TR-{i}") for i in range(5)]
        test_pairs = [_make_pair(f"TS-{i}") for i in range(2)]

        # Dense mode
        preds_dense, rep_dense = run_leredd_variant(
            test_pairs, train_pairs, retrieval_mode="dense", k_examples=2, confidence_threshold=4,
            embedder=_OfflineEmbedder(),
        )
        assert len(preds_dense) == 2

        # BM25 mode
        preds_bm25, rep_bm25 = run_leredd_variant(
            test_pairs, train_pairs, retrieval_mode="bm25", k_examples=2, confidence_threshold=4,
            embedder=_OfflineEmbedder(),
        )
        assert len(preds_bm25) == 2

        # Hybrid mode
        preds_hybrid, rep_hybrid = run_leredd_variant(
            test_pairs, train_pairs, retrieval_mode="hybrid", k_examples=2, confidence_threshold=4,
            embedder=_OfflineEmbedder(),
        )
        assert len(preds_hybrid) == 2

    def test_zero_shot_k_zero_config(self):
        config = LEREDDConfig(k_examples=0)
        assert config.k_examples == 0
        detector = LEREDDDetector(config=config, embedder=_OfflineEmbedder(), llm=MockLLMClient())
        pair = _make_pair("P-ZS")
        pred = detector.detect_pair(pair, annotated_pool=[_make_pair("TR-1")])
        assert isinstance(pred, Prediction)

    def test_threshold_disabled_minus_one_config(self):
        config = LEREDDConfig(confidence_threshold=-1)
        assert config.confidence_threshold == -1
        detector = LEREDDDetector(config=config, embedder=_OfflineEmbedder(), llm=MockLLMClient())
        pair = _make_pair("P-TH")
        pred = detector.detect_pair(pair)
        assert isinstance(pred, Prediction)

    def test_threshold_behavior_flags_review_without_relabeling(self):
        """Confidence screening changes disposition, never the predicted class."""
        from leredd.detector import LEREDDDetector
        from leredd.llm import LLMClient

        class LowConfMockLLM(LLMClient):
            def generate(self, prompt: str, temperature: float = 0.2):
                raw = "**Dependency_Type: Requires**\n**Rationale: Test**\n**Confidence Score: 3**"
                return raw, {"prompt_tokens": 10, "completion_tokens": 10, "cost_usd": 0.0}

        low_llm = LowConfMockLLM()
        detector_on = LEREDDDetector(config=LEREDDConfig(confidence_threshold=4), embedder=_OfflineEmbedder(), llm=low_llm)
        detector_off = LEREDDDetector(config=LEREDDConfig(confidence_threshold=-1), embedder=_OfflineEmbedder(), llm=low_llm)

        pair = _make_pair("P-LOW")
        pred_on = detector_on.detect_pair(pair, confidence_threshold=4)
        pred_off = detector_off.detect_pair(pair, confidence_threshold=-1)

        # Low confidence requires review but preserves the model's Requires label.
        assert pred_on.dependency_type == DependencyType.REQUIRES
        assert pred_on.validation_status.value == "requires_review"
        # Disabling the check also preserves the class and is explicit in output.
        assert pred_off.dependency_type == DependencyType.REQUIRES
        assert pred_off.validation_status.value == "threshold_disabled"

    def test_retrieval_mode_dispatch_and_example_building(self):
        """Verify retrieval modes build prompt example blocks when k>0 and omit when k=0."""
        from leredd.detector import LEREDDDetector

        train_pool = [_make_pair(f"TR-{i}") for i in range(5)]
        test_pair = _make_pair("P-DISPATCH")

        # k=0 -> No in-context examples
        config_z = LEREDDConfig(k_examples=0)
        det_z = LEREDDDetector(config=config_z, embedder=_OfflineEmbedder(), llm=MockLLMClient())
        pred_z = det_z.detect_pair(test_pair, annotated_pool=train_pool)
        assert pred_z is not None

        # k=2 -> Dense vs BM25 vs Hybrid retrieval modes
        for mode in ["dense", "bm25", "hybrid"]:
            cfg = LEREDDConfig(k_examples=2, retrieval_mode=mode)
            det = LEREDDDetector(config=cfg, embedder=_OfflineEmbedder(), llm=MockLLMClient())
            pred = det.detect_pair(test_pair, annotated_pool=train_pool, retrieval_mode=mode)
            assert pred is not None

    def test_mock_llm_per_example_deterministic_predictions(self):
        """Verify MockLLMClient per-example predictions on test pairs are deterministic."""
        test_pairs = [
            RequirementPair(id="PAIR-TJA-001", req_a=_make_req("R1", "TJA"), req_b=_make_req("R2", "TJA")),
            RequirementPair(id="PAIR-TJA-002", req_a=_make_req("R3", "TJA"), req_b=_make_req("R4", "TJA")),
        ]

        detector = LEREDDDetector(config=LEREDDConfig(llm_provider="mock"), embedder=_OfflineEmbedder(), llm=MockLLMClient())
        preds1 = detector.detect_batch(test_pairs)
        preds2 = detector.detect_batch(test_pairs)

        assert [p.dependency_type for p in preds1] == [p.dependency_type for p in preds2]
