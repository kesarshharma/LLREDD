"""
scripts/run_full_evaluation.py
==============================
Safe, reproducible evaluation & controlled ablation runner for a leakage-checked split.

Compares six controlled LEREDD configurations (Part 8):
  1. Zero-shot (k=0, RAG disabled)
  2. Semantic-only retrieval (SBERT dense)
  3. BM25-only retrieval (lexical)
  4. Existing hybrid retrieval (RRF: SBERT + BM25, metadata_weight=0)
  5. Metadata-aware hybrid retrieval (RRF: SBERT + BM25 + structured metadata)
  6. Context-enhanced retrieval (Metadata-aware hybrid + SRS RAG context)

Enforces:
  - Fixed prompts (recorded via SHA-256 template hash), fixed model versions, fixed seeds
  - Development-set-only confidence threshold tuning (never touching test pairs)
  - System-disjoint or document-disjoint split verification
  - Strict rejection of synthetic, provisional, pending, or rejected ground truth
  - Explicit separation of candidate-generation recall from classifier accuracy
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from leredd.baselines.bert_classifier import BERTBaseline
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.config import LEREDDConfig
from leredd.data_models import (
    DependencyType,
    EvaluationReport,
    Prediction,
    RequirementPair,
)
from leredd.detector import LEREDDDetector
from leredd.embeddings import EmbeddingModel
from leredd.evaluation import (
    ContaminationError,
    Evaluator,
    GroundTruthViolationError,
    build_safe_retrieval_pool,
    check_ground_truth_eligibility,
    summarize_candidate_recall_vs_classifier,
    tune_confidence_threshold_on_dev,
    validate_document_disjoint_splits,
    validate_system_disjoint_splits,
)
from leredd.llm import LLMClientFactory, MockLLMClient
from leredd.prompts import PromptBuilder
from leredd.provisional_labeler import HeuristicLabeler
from leredd.utils import load_pairs_json

SCRIPT_VERSION = "2.1.0"

MOCK_ABLATION_VALIDITY = {
    "backend": "MockLLMClient",
    "is_empirical_model_evidence": False,
    "retrieval_strategy": {
        "status": "not_informative_with_mock_backend",
        "reason": (
            "MockLLMClient classifies the target text with fixed keyword rules and does not "
            "interpret retrieved examples. Dense, BM25, and hybrid metrics are pipeline checks, "
            "not evidence of retrieval strategy effects."
        ),
    },
    "in_context_examples": {
        "status": "not_informative_with_mock_backend",
        "reason": (
            "MockLLMClient does not interpret the prompt's retrieved-example block. "
            "Zero-shot versus k-example metrics cannot estimate the effect of ICL."
        ),
    },
    "confidence_thresholding": {
        "status": "not_informative_with_mock_backend",
        "reason": (
            "MockLLMClient returns fixed confidence 5, so the configured threshold of 4 "
            "never marks examples for review. Threshold-on/off class metrics cannot estimate "
            "the effect of confidence-based review routing."
        ),
    },
    "required_for_empirical_ablation": (
        "Use a real authorized LLM or a separately validated controlled test double that responds "
        "to retrieved examples and emits variable confidence. Do not interpret these mock outputs "
        "as empirical effects."
    ),
}

__all__ = [
    "ContaminationError",
    "GroundTruthViolationError",
    "_serialize_report",
    "build_safe_retrieval_pool",
    "check_ground_truth_eligibility",
    "run_evaluation",
    "run_heuristic_model",
    "run_leredd_variant",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_distribution(pairs: List[RequirementPair]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for p in pairs:
        label = p.dependency_type.value if p.dependency_type else "None"
        counter[label] += 1
    return dict(counter)


def _unique_systems(pairs: List[RequirementPair]) -> List[str]:
    systems: Set[str] = set()
    for p in pairs:
        systems.add(p.req_a.system)
        systems.add(p.req_b.system)
    return sorted(systems)


def _system_description(pairs: List[RequirementPair]) -> str:
    return ", ".join(_unique_systems(pairs)) or "unspecified systems"


def _build_srs_context_for_test_pairs(
    test_pairs: List[RequirementPair],
    srs_context_path: Optional[str] = None,
) -> str:
    """Load an explicit SRS document or assemble unlabeled requirement statements only.

    Never includes ground-truth dependency labels so RAG context cannot leak labels.
    """
    if srs_context_path:
        path = Path(srs_context_path)
        if path.exists():
            return path.read_text(encoding="utf-8")

    seen_reqs: Dict[str, str] = {}
    for pair in test_pairs:
        for req in (pair.req_a, pair.req_b):
            if req.id not in seen_reqs:
                seen_reqs[req.id] = f"[{req.id}] {req.text}"
    return "\n".join(seen_reqs.values())


def _ground_truth_report_summary(raw_partitions, eligible_partitions) -> Dict[str, Any]:
    """Build report provenance counts and wording from the supplied split."""
    raw_counts = {name: len(pairs) for name, pairs in raw_partitions.items()}
    eligible_counts = {name: len(pairs) for name, pairs in eligible_partitions.items()}
    all_raw = [pair for pairs in raw_partitions.values() for pair in pairs]
    all_eligible = [pair for pairs in eligible_partitions.values() for pair in pairs]
    provenance_free = sum(pair.provenance is None for pair in all_raw)
    eligible_with_provenance = sum(pair.provenance is not None for pair in all_eligible)
    total = len(all_raw)
    if provenance_free:
        provenance_note = (
            f"{provenance_free} of {total} supplied partition records have no provenance; "
            "legacy labels are accepted conditionally only when otherwise eligible. "
            f"{len(all_eligible)} of {total} records passed eligibility checks."
        )
    elif eligible_with_provenance == len(all_eligible):
        provenance_note = (
            f"{eligible_with_provenance} eligible records carried provenance and passed the "
            f"configured checks ({len(all_eligible)} of {total} input records were eligible). "
            "Actor identities are self-reported and are not authenticated."
        )
    else:
        provenance_note = (
            f"{len(all_eligible)} of {total} supplied records passed eligibility checks; "
            f"{eligible_with_provenance} eligible records carried provenance. "
            "Actor identities are self-reported and are not authenticated."
        )
    return {
        "raw_counts": raw_counts,
        "eligible_counts": eligible_counts,
        "provenance_free_count": provenance_free,
        "eligible_provenance_count": eligible_with_provenance,
        "provenance_note": provenance_note,
    }


def _load_split_partition(split_dir: Path, partition: str) -> List[RequirementPair]:
    path = split_dir / f"{partition}_pairs.json"
    if not path.exists():
        logger.warning(f"Partition file not found: {path}")
        return []
    return load_pairs_json(str(path))


def _serialize_report(
    report: EvaluationReport,
    model_tag: str,
    note: str,
    predictions: List[Prediction],
    n_test: int,
) -> Dict[str, Any]:
    pred_map = {p.pair_id: p.dependency_type.value for p in predictions}
    k_correct = int(round(report.overall_accuracy * n_test))
    ci = Evaluator.compute_wilson_score_interval(k_correct, n_test)
    review_statuses = {"passed_threshold", "requires_review"}
    screened_predictions = [
        prediction
        for prediction in predictions
        if prediction.validation_status.value in review_statuses
    ]
    review_required_count = sum(
        prediction.validation_status.value == "requires_review"
        for prediction in screened_predictions
    )
    threshold_values = {
        prediction.validation_threshold
        for prediction in predictions
        if prediction.validation_threshold is not None
    }
    return {
        "model": model_tag,
        "note": note,
        "overall_accuracy": report.overall_accuracy,
        "macro_f1": report.macro_f1,
        "weighted_f1": report.weighted_f1,
        "macro_precision": report.macro_precision,
        "macro_recall": report.macro_recall,
        "confidence_interval_95": {
            "method": "Wilson score interval",
            "point_estimate": ci["point_estimate"],
            "lower_bound": ci["lower_bound"],
            "upper_bound": ci["upper_bound"],
            "margin_pp": ci["margin_pp"],
            "interpretation_note": (
                f"For sample size n={n_test}, 95% Wilson CI is [{ci['lower_bound']:.4f}, {ci['upper_bound']:.4f}] "
                f"(~+-{ci['margin_pp']:.1f}pp margin)."
            ),
        },
        "predictions": pred_map,
        "confidence_screening": {
            "threshold": next(iter(threshold_values)) if len(threshold_values) == 1 else None,
            "screened_count": len(screened_predictions),
            "passed_threshold_count": len(screened_predictions) - review_required_count,
            "review_required_count": review_required_count,
            "review_required_rate": (
                review_required_count / len(screened_predictions)
                if screened_predictions else None
            ),
            "not_evaluated_count": sum(
                prediction.validation_status.value == "not_evaluated"
                for prediction in predictions
            ),
            "threshold_disabled_count": sum(
                prediction.validation_status.value == "threshold_disabled"
                for prediction in predictions
            ),
            "interpretation": (
                "Review routing is based on self-reported Likert confidence. "
                "It does not change predicted classes, establish annotation truth, "
                "or represent calibrated probability."
            ),
        },
        "per_class_metrics": {
            cls: {
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "support": m.support,
            }
            for cls, m in report.per_class_metrics.items()
        },
        "confusion_matrix": report.confusion_matrix,
    }


def run_heuristic_model(test_pairs: List[RequirementPair]) -> Tuple[List[Prediction], EvaluationReport]:
    """Evaluate HeuristicLabeler baseline on test pairs."""
    preds = []
    for pair in test_pairs:
        res = HeuristicLabeler.evaluate_pair(pair)
        label = res.suggested_label if res.suggested_label is not None else DependencyType.NO_DEPENDENCY
        preds.append(
            Prediction(
                pair_id=pair.id,
                dependency_type=label,
                rationale=res.evidence,
                confidence=int(round(res.confidence * 5)),
            )
        )
    report = Evaluator.evaluate(
        test_pairs,
        preds,
        system_name=_system_description(test_pairs),
        model_name="HeuristicLabeler",
    )
    return preds, report


def run_leredd_variant(
    test_pairs: List[RequirementPair],
    retrieval_pool: Optional[List[RequirementPair]],
    retrieval_mode: str = "hybrid",
    k_examples: int = 6,
    confidence_threshold: int = 4,
    model_name: str = "LEREDD-Hybrid",
    embedder: Optional[EmbeddingModel] = None,
    *,
    metadata_weight: float = 1.0,
    use_rag: bool = False,
    srs_text: Optional[str] = None,
    llm_provider: str = "mock",
    llm_model: str = "mock",
) -> Tuple[List[Prediction], EvaluationReport]:
    """Evaluate a controlled LEREDD detector variant on test pairs."""
    config = LEREDDConfig(
        llm_provider=llm_provider,  # type: ignore[arg-type]
        llm_model=llm_model,
        retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
        k_examples=k_examples,
        confidence_threshold=confidence_threshold,
        metadata_weight=metadata_weight,
        use_rag=use_rag,
        rag_deduplicate_chunks=True if use_rag else False,
        rag_cite_id_boost=0.15 if use_rag else 0.0,
    )
    llm_client = (
        MockLLMClient(model=llm_model)
        if llm_provider == "mock"
        else LLMClientFactory.create(config)
    )
    detector = LEREDDDetector(config=config, llm=llm_client, embedder=embedder)
    pool = retrieval_pool if k_examples > 0 else None
    preds = detector.detect_batch(
        test_pairs,
        annotated_pool=pool,
        srs_text=srs_text if use_rag else None,
        max_workers=1,
        retrieval_mode=retrieval_mode,
        confidence_threshold=confidence_threshold,
        use_rag=use_rag,
    )
    report = Evaluator.evaluate(
        test_pairs,
        preds,
        system_name=_system_description(test_pairs),
        model_name=model_name,
    )
    return preds, report


# ---------------------------------------------------------------------------
# Main Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    split_dir: str = "data/splits/system_disjoint",
    output_path: str = "data/evaluation/system_disjoint_report.json",
    seed: int = 42,
    bert_model_type: str = "rf",
    strict_contamination: bool = True,
    strict_gt: bool = True,
    *,
    llm_provider: str = "mock",
    llm_model: str = "mock",
    srs_context_path: Optional[str] = None,
    candidate_manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full safe evaluation & controlled ablation pipeline and return the report dict."""
    run_start = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info("LEREDD Integrated Benchmark & Controlled Ablation Evaluation")
    logger.info(f"Split dir  : {split_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Seed       : {seed}")
    logger.info(f"BERT type  : {bert_model_type}")
    logger.info(f"LLM backend: {llm_provider} ({llm_model})")
    logger.info("=" * 70)

    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    meta_path = split_path / "split_metadata.json"
    split_metadata: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            split_metadata = json.load(f)

    # Step 1: Load split partitions
    logger.info("Step 1: Loading split partitions ...")
    train_pairs_raw = _load_split_partition(split_path, "train")
    dev_pairs_raw = _load_split_partition(split_path, "dev")
    test_pairs_raw = _load_split_partition(split_path, "test")
    logger.info(
        f"  Raw: train={len(train_pairs_raw)}, dev={len(dev_pairs_raw)}, test={len(test_pairs_raw)}"
    )

    if not test_pairs_raw:
        raise ValueError(
            "Test partition is empty. Ensure system-disjoint or document-disjoint split has been generated."
        )

    split_strategy = str(
        split_metadata.get("summary", {}).get("strategy", "system")
    ).casefold().split(".")[-1]
    split_integrity: Dict[str, Any] = {"checked": False, "strategy": split_strategy}
    if split_strategy == "system":
        split_integrity.update(
            validate_system_disjoint_splits(
                train_pairs=train_pairs_raw,
                test_pairs=test_pairs_raw,
                dev_pairs=dev_pairs_raw,
            )
        )
        split_integrity["checked"] = True
        logger.info("  System-disjoint split integrity check passed.")
    elif split_strategy == "document":
        split_integrity.update(
            validate_document_disjoint_splits(
                train_pairs=train_pairs_raw,
                test_pairs=test_pairs_raw,
                dev_pairs=dev_pairs_raw,
            )
        )
        split_integrity["checked"] = True
        logger.info("  Document-disjoint split integrity check passed.")

    # Step 2: Ground-truth eligibility
    logger.info("Step 2: Checking ground-truth eligibility ...")
    all_gt_warnings: List[str] = []
    train_pairs, w = check_ground_truth_eligibility(train_pairs_raw, "TRAIN", strict=strict_gt)
    all_gt_warnings.extend(w)
    dev_pairs, w = check_ground_truth_eligibility(dev_pairs_raw, "DEV", strict=strict_gt)
    all_gt_warnings.extend(w)
    test_pairs, w = check_ground_truth_eligibility(test_pairs_raw, "TEST", strict=strict_gt)
    all_gt_warnings.extend(w)

    for msg in all_gt_warnings:
        logger.info(f"  GT-CHECK: {msg}")
    logger.info(f"  Eligible: train={len(train_pairs)}, dev={len(dev_pairs)}, test={len(test_pairs)}")

    if not test_pairs:
        raise ValueError("No eligible test pairs after ground-truth filtering.")

    # Step 3: Contamination guard
    logger.info("Step 3: Building contamination-safe retrieval pool ...")
    safe_retrieval_pool, leakage_audit = build_safe_retrieval_pool(
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        dev_pairs=dev_pairs,
        strict=strict_contamination,
    )

    # Step 4: BERTBaseline (fixed seed)
    cls_desc = (
        f"RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state={seed})"
        if bert_model_type == "rf"
        else f"LogisticRegression(max_iter=1000, class_weight='balanced', random_state={seed})"
    )
    logger.info(f"Step 4: BERTBaseline (classifier={bert_model_type.upper()}: {cls_desc}) ...")
    shared_embedder = EmbeddingModel()
    bert = BERTBaseline(embedder=shared_embedder, model_type=bert_model_type, random_state=seed)
    bert.fit(train_pairs)
    bert_preds = bert.predict_batch(test_pairs)
    bert_report = Evaluator.evaluate(
        test_pairs,
        bert_preds,
        system_name=_system_description(test_pairs),
        model_name=f"BERTBaseline-{bert_model_type.upper()}",
    )
    logger.info(f"  Accuracy={bert_report.overall_accuracy:.4f}, Macro-F1={bert_report.macro_f1:.4f}")

    # Step 5: TFIDF_LSA (fixed seed)
    logger.info("Step 5: TFIDF_LSABaseline ...")
    tfidf_lsa = TFIDF_LSABaseline(random_state=seed)
    tfidf_lsa.fit(train_pairs)
    tfidf_preds = tfidf_lsa.predict_batch(test_pairs)
    tfidf_report = Evaluator.evaluate(
        test_pairs, tfidf_preds, system_name=_system_description(test_pairs), model_name="TFIDF_LSA"
    )
    logger.info(f"  Accuracy={tfidf_report.overall_accuracy:.4f}, Macro-F1={tfidf_report.macro_f1:.4f}")

    # Step 6: Heuristic Rule Classifier Baseline
    logger.info("Step 6: Heuristic Rule Classifier Baseline ...")
    heuristic_preds, heuristic_report = run_heuristic_model(test_pairs)
    logger.info(f"  Accuracy={heuristic_report.overall_accuracy:.4f}, Macro-F1={heuristic_report.macro_f1:.4f}")

    # Step 6b: Development-set-only threshold tuning (NEVER touches test_pairs)
    logger.info("Step 6b: Tuning confidence screening threshold on DEV partition only ...")
    if dev_pairs:
        dev_pool, _ = build_safe_retrieval_pool(
            train_pairs=train_pairs,
            test_pairs=dev_pairs,
            dev_pairs=[],
            strict=False,
        )
        dev_preds, _ = run_leredd_variant(
            dev_pairs,
            dev_pool,
            retrieval_mode="hybrid_metadata",
            k_examples=6,
            confidence_threshold=4,
            model_name="LEREDD-DevTuning",
            embedder=shared_embedder,
            llm_provider=llm_provider,
            llm_model=llm_model,
        )
        dev_threshold_tuning = tune_confidence_threshold_on_dev(
            dev_pairs, dev_preds, partition_label="DEV", default_threshold=4
        )
    else:
        dev_threshold_tuning = tune_confidence_threshold_on_dev(
            [], [], partition_label="DEV", default_threshold=4
        )
    dev_tuned_threshold = int(dev_threshold_tuning["selected_threshold"])

    # Step 7: Six Controlled LEREDD Configurations & Ablations (Part 8)
    logger.info("Step 7: Controlled LEREDD Configurations & Ablation Matrix ...")
    srs_context_text = _build_srs_context_for_test_pairs(test_pairs, srs_context_path)

    # C1. Zero-Shot (k=0, RAG disabled)
    leredd_zeroshot_preds, leredd_zeroshot_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="dense",
        k_examples=0,
        confidence_threshold=4,
        model_name="LEREDD-ZeroShot",
        embedder=shared_embedder,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # C2. Semantic-only retrieval (SBERT dense)
    leredd_dense_preds, leredd_dense_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="dense",
        k_examples=6,
        confidence_threshold=4,
        model_name="LEREDD-Dense",
        embedder=shared_embedder,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # C3. BM25-only retrieval (lexical)
    leredd_bm25_preds, leredd_bm25_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="bm25",
        k_examples=6,
        confidence_threshold=4,
        model_name="LEREDD-BM25",
        embedder=shared_embedder,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # C4. Existing hybrid retrieval (SBERT + BM25 RRF without metadata)
    leredd_hybrid_preds, leredd_hybrid_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="hybrid_no_metadata",
        k_examples=6,
        confidence_threshold=4,
        model_name="LEREDD-Hybrid",
        embedder=shared_embedder,
        metadata_weight=0.0,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # C5. Metadata-aware hybrid retrieval (SBERT + BM25 + structured metadata RRF)
    leredd_meta_preds, leredd_meta_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="hybrid_metadata",
        k_examples=6,
        confidence_threshold=4,
        model_name="LEREDD-Hybrid-Metadata",
        embedder=shared_embedder,
        metadata_weight=1.0,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # C6. Context-enhanced retrieval (Metadata-aware hybrid + SRS RAG context)
    leredd_context_preds, leredd_context_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="hybrid_metadata",
        k_examples=6,
        confidence_threshold=4,
        model_name="LEREDD-Context-RAG",
        embedder=shared_embedder,
        metadata_weight=1.0,
        use_rag=True,
        srs_text=srs_context_text,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    # Thresholding Disabled (-1) & Dev-Tuned Threshold
    leredd_nothresh_preds, leredd_nothresh_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="hybrid_no_metadata",
        k_examples=6,
        confidence_threshold=-1,
        model_name="LEREDD-Hybrid-NoThreshold",
        embedder=shared_embedder,
        metadata_weight=0.0,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )
    leredd_devthresh_preds, leredd_devthresh_report = run_leredd_variant(
        test_pairs,
        safe_retrieval_pool,
        retrieval_mode="hybrid_metadata",
        k_examples=6,
        confidence_threshold=dev_tuned_threshold,
        model_name="LEREDD-Hybrid-DevTunedThreshold",
        embedder=shared_embedder,
        metadata_weight=1.0,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )

    logger.info(f"  Proposed Hybrid Accuracy={leredd_hybrid_report.overall_accuracy:.4f}, Macro-F1={leredd_hybrid_report.macro_f1:.4f}")
    logger.info(f"  Zero-Shot LLM Accuracy={leredd_zeroshot_report.overall_accuracy:.4f}, Macro-F1={leredd_zeroshot_report.macro_f1:.4f}")

    # Step 8: Statistical tests (Exploratory due to small n)
    logger.info("Step 8: McNemar test assessment ...")
    n_test = len(test_pairs)
    mcnemar_notice = (
        f"McNemar test applied to n_test={n_test} examples. "
        + (
            "This small sample has low statistical power; p-values are exploratory and demonstrative only."
            if n_test < 10 else
            "Interpret p-values as exploratory; they do not establish research significance by themselves."
        )
    )
    if n_test < 10:
        logger.warning(f"  [EXPLORATORY TEST] {mcnemar_notice}")

    training_systems = _system_description(train_pairs)
    test_systems = _system_description(test_pairs)
    provenance_summary = _ground_truth_report_summary(
        {"train": train_pairs_raw, "dev": dev_pairs_raw, "test": test_pairs_raw},
        {"train": train_pairs, "dev": dev_pairs, "test": test_pairs},
    )
    n_classes = len(_label_distribution(test_pairs))

    mcnemar_results: Dict[str, Any] = {
        "status": "exploratory",
        "exploratory_notice": mcnemar_notice,
        "leredd_vs_bert": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, bert_preds),
        "leredd_vs_tfidf": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, tfidf_preds),
        "proposed_hybrid_vs_bert": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, bert_preds),
        "proposed_hybrid_vs_tfidf": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, tfidf_preds),
        "proposed_hybrid_vs_heuristic": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, heuristic_preds),
        "proposed_hybrid_vs_zeroshot": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, leredd_zeroshot_preds),
        "proposed_hybrid_vs_dense": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, leredd_dense_preds),
        "proposed_hybrid_vs_bm25": Evaluator.analyze_mcnemar(test_pairs, leredd_hybrid_preds, leredd_bm25_preds),
        "metadata_hybrid_vs_existing_hybrid": Evaluator.analyze_mcnemar(test_pairs, leredd_meta_preds, leredd_hybrid_preds),
        "context_rag_vs_metadata_hybrid": Evaluator.analyze_mcnemar(test_pairs, leredd_context_preds, leredd_meta_preds),
    }

    # Honest delta summary (including zero or negative deltas!)
    zs_f1 = leredd_zeroshot_report.macro_f1
    zs_acc = leredd_zeroshot_report.overall_accuracy
    ablation_deltas = {}
    for cfg_key, cfg_rep in (
        ("ZeroShot", leredd_zeroshot_report),
        ("Semantic_Only_Dense", leredd_dense_report),
        ("BM25_Only", leredd_bm25_report),
        ("Existing_Hybrid", leredd_hybrid_report),
        ("Metadata_Aware_Hybrid", leredd_meta_report),
        ("Context_Enhanced_RAG", leredd_context_report),
    ):
        delta_f1 = round(cfg_rep.macro_f1 - zs_f1, 6)
        delta_acc = round(cfg_rep.overall_accuracy - zs_acc, 6)
        ablation_deltas[cfg_key] = {
            "macro_f1": cfg_rep.macro_f1,
            "overall_accuracy": cfg_rep.overall_accuracy,
            "delta_macro_f1_vs_zeroshot": delta_f1,
            "delta_accuracy_vs_zeroshot": delta_acc,
            "outcome": (
                "improved" if delta_f1 > 0 else ("degraded_negative_result" if delta_f1 < 0 else "no_change")
            ),
        }

    candidate_manifest_data = None
    if candidate_manifest_path and Path(candidate_manifest_path).exists():
        candidate_manifest_data = json.loads(
            Path(candidate_manifest_path).read_text(encoding="utf-8")
        )
    candidate_separation = summarize_candidate_recall_vs_classifier(
        leredd_meta_report,
        candidate_manifest=candidate_manifest_data,
    )

    prompt_hash = PromptBuilder().template_hash()
    is_mock = llm_provider == "mock"
    ablation_validity = (
        MOCK_ABLATION_VALIDITY
        if is_mock
        else {
            "backend": llm_provider,
            "model": llm_model,
            "is_empirical_model_evidence": True,
            "note": (
                "Live LLM provider invoked with fixed prompt template hash and system/document-disjoint "
                "test partition. Interpret significance alongside test sample size n_test."
            ),
        }
    )

    report = {
        "schema_version": "2.1",
        "script_version": SCRIPT_VERSION,
        "run_timestamp": run_start,
        "configuration": {
            "split_dir": str(split_dir),
            "output_path": str(output_path),
            "seed": seed,
            "prompt_template_hash": prompt_hash,
            "embedding_model": shared_embedder.model_name,
            "bert_classifier_backend": bert_model_type,
            "bert_classifier_description": cls_desc,
            "split_strategy": split_metadata.get("summary", {}).get("strategy", "system"),
            "leredd_provider": llm_provider,
            "leredd_model": llm_model,
            "test_set_tuning": False,
            "dev_tuned_threshold": dev_tuned_threshold,
            "leredd_note": (
                "MockLLMClient — deterministic keyword heuristics. NOT a real LLM evaluation."
                if is_mock
                else f"Live LLM evaluation via provider={llm_provider!r}, model={llm_model!r}."
            ),
        },
        "split_integrity": split_integrity,
        "split_inventory": {
            "train": {
                "size": len(train_pairs),
                "systems": _unique_systems(train_pairs),
                "label_distribution": _label_distribution(train_pairs),
            },
            "dev": {
                "size": len(dev_pairs),
                "systems": _unique_systems(dev_pairs),
                "label_distribution": _label_distribution(dev_pairs),
            },
            "test": {
                "size": len(test_pairs),
                "systems": _unique_systems(test_pairs),
                "label_distribution": _label_distribution(test_pairs),
            },
        },
        "ground_truth_eligibility": {
            "warnings": all_gt_warnings,
            **provenance_summary,
        },
        "leakage_audit": leakage_audit,
        "dev_threshold_tuning": dev_threshold_tuning,
        "candidate_recall_separation": candidate_separation,
        "model_results": {
            "BERTBaseline": _serialize_report(
                bert_report,
                model_tag=f"BERTBaseline-{bert_model_type.upper()}",
                note=(
                    f"SBERT ({shared_embedder.model_name}) + {cls_desc}. "
                    "Supervised cross-system evaluation trained on "
                    f"{training_systems} (n={len(train_pairs)}) "
                    f"and tested on {test_systems} (n={n_test}). Not a zero-shot model."
                ),
                predictions=bert_preds,
                n_test=n_test,
            ),
            "TFIDF_LSA": _serialize_report(
                tfidf_report,
                model_tag="TFIDF_LSA",
                note=(
                    "TF-IDF + LSA cosine similarity thresholding. "
                    f"Fitted vocabulary and SVD on {training_systems} training data, "
                    f"tested on {test_systems}. "
                    "Unsupervised / fitted cross-system baseline."
                ),
                predictions=tfidf_preds,
                n_test=n_test,
            ),
            "HeuristicLabeler": _serialize_report(
                heuristic_report,
                model_tag="HeuristicLabeler",
                note=(
                    "Rule-based deterministic baseline checking explicit citations, "
                    "structural keywords, and Jaccard lexical overlap."
                ),
                predictions=heuristic_preds,
                n_test=n_test,
            ),
            "LEREDD_ZeroShot": _serialize_report(
                leredd_zeroshot_report,
                model_tag="LEREDD-ZeroShot",
                note="Zero-shot LLM inference without in-context dynamic examples (k=0).",
                predictions=leredd_zeroshot_preds,
                n_test=n_test,
            ),
            "LEREDD_Hybrid": _serialize_report(
                leredd_hybrid_report,
                model_tag="LEREDD-Hybrid (Proposed System)",
                note=(
                    "Proposed LEREDD system using Hybrid RRF retrieval (SBERT + BM25, rrf_k=60), "
                    "k=6 dynamic example pairs, and a confidence screen that flags low-confidence "
                    "predictions for review without changing their class."
                ),
                predictions=leredd_hybrid_preds,
                n_test=n_test,
            ),
            "LEREDD_Hybrid_Metadata": _serialize_report(
                leredd_meta_report,
                model_tag="LEREDD-Hybrid-Metadata",
                note=(
                    "Metadata-aware Hybrid RRF retrieval fusing SBERT, BM25, and structured "
                    "component/subsystem/module/system overlap (k=6)."
                ),
                predictions=leredd_meta_preds,
                n_test=n_test,
            ),
            "LEREDD_Context_RAG": _serialize_report(
                leredd_context_report,
                model_tag="LEREDD-Context-RAG",
                note=(
                    "Context-enhanced LEREDD combining Metadata-aware Hybrid ICL retrieval "
                    "with deduplicated, cited-ID-boosted SRS RAG context chunks."
                ),
                predictions=leredd_context_preds,
                n_test=n_test,
            ),
        },
        "ablation_results": {
            "controlled_six_way_comparison": {
                "1_ZeroShot": _serialize_report(
                    leredd_zeroshot_report,
                    model_tag="1. Zero-Shot (k=0, no RAG)",
                    note="Zero-shot baseline with fixed taxonomy prompt only.",
                    predictions=leredd_zeroshot_preds,
                    n_test=n_test,
                ),
                "2_Semantic_Only": _serialize_report(
                    leredd_dense_report,
                    model_tag="2. Semantic-Only Retrieval (SBERT)",
                    note="Dense SBERT embedding retrieval only (k=6).",
                    predictions=leredd_dense_preds,
                    n_test=n_test,
                ),
                "3_BM25_Only": _serialize_report(
                    leredd_bm25_report,
                    model_tag="3. BM25-Only Retrieval (Lexical)",
                    note="Lexical BM25Plus retrieval only (k=6).",
                    predictions=leredd_bm25_preds,
                    n_test=n_test,
                ),
                "4_Existing_Hybrid": _serialize_report(
                    leredd_hybrid_report,
                    model_tag="4. Existing Hybrid Retrieval (SBERT + BM25)",
                    note="Reciprocal Rank Fusion over SBERT and BM25 only (metadata_weight=0).",
                    predictions=leredd_hybrid_preds,
                    n_test=n_test,
                ),
                "5_Metadata_Aware_Hybrid": _serialize_report(
                    leredd_meta_report,
                    model_tag="5. Metadata-Aware Hybrid Retrieval",
                    note="Reciprocal Rank Fusion over SBERT, BM25, and structured metadata (metadata_weight=1).",
                    predictions=leredd_meta_preds,
                    n_test=n_test,
                ),
                "6_Context_Enhanced_RAG": _serialize_report(
                    leredd_context_report,
                    model_tag="6. Context-Enhanced Retrieval (Hybrid + RAG)",
                    note="Metadata-aware Hybrid ICL plus SRS RAG context chunks.",
                    predictions=leredd_context_preds,
                    n_test=n_test,
                ),
            },
            "retrieval_strategy": {
                "LEREDD_Dense": _serialize_report(
                    leredd_dense_report,
                    model_tag="LEREDD-Dense (SBERT Only)",
                    note="Dense vector retrieval only via SBERT embeddings.",
                    predictions=leredd_dense_preds,
                    n_test=n_test,
                ),
                "LEREDD_BM25": _serialize_report(
                    leredd_bm25_report,
                    model_tag="LEREDD-BM25 (Lexical Only)",
                    note="Lexical BM25Plus retrieval only.",
                    predictions=leredd_bm25_preds,
                    n_test=n_test,
                ),
                "LEREDD_Hybrid": _serialize_report(
                    leredd_hybrid_report,
                    model_tag="LEREDD-Hybrid (RRF Dense+Lexical)",
                    note="Hybrid RRF fusion combining SBERT and BM25Plus (metadata_weight=0).",
                    predictions=leredd_hybrid_preds,
                    n_test=n_test,
                ),
                "LEREDD_Hybrid_Metadata": _serialize_report(
                    leredd_meta_report,
                    model_tag="LEREDD-Hybrid-Metadata (RRF Dense+Lexical+Metadata)",
                    note="Hybrid RRF fusion combining SBERT, BM25Plus, and structured metadata.",
                    predictions=leredd_meta_preds,
                    n_test=n_test,
                ),
                "LEREDD_Context_RAG": _serialize_report(
                    leredd_context_report,
                    model_tag="LEREDD-Context-RAG (Metadata Hybrid + SRS RAG)",
                    note="Metadata-aware Hybrid RRF plus SRS RAG context chunks.",
                    predictions=leredd_context_preds,
                    n_test=n_test,
                ),
            },
            "in_context_examples": {
                "ZeroShot": _serialize_report(
                    leredd_zeroshot_report,
                    model_tag="Zero-Shot (k=0)",
                    note="Prompt without retrieved in-context examples.",
                    predictions=leredd_zeroshot_preds,
                    n_test=n_test,
                ),
                "Hybrid_k6": _serialize_report(
                    leredd_hybrid_report,
                    model_tag="Hybrid Dynamic Examples (k=6)",
                    note="Prompt augmented with 6 dynamic examples per dependency type.",
                    predictions=leredd_hybrid_preds,
                    n_test=n_test,
                ),
            },
            "confidence_thresholding": {
                "Threshold_Off": _serialize_report(
                    leredd_nothresh_report,
                    model_tag="Confidence Screening Disabled (Off)",
                    note="Raw LLM prediction; confidence review screening disabled.",
                    predictions=leredd_nothresh_preds,
                    n_test=n_test,
                ),
                "Threshold_On": _serialize_report(
                    leredd_hybrid_report,
                    model_tag="Confidence Screening Enabled (<= 4: review cue)",
                    note=(
                        "Predictions with self-reported confidence <= 4 retain their class "
                        "and are marked requires_review."
                    ),
                    predictions=leredd_hybrid_preds,
                    n_test=n_test,
                ),
                "Threshold_Dev_Tuned": _serialize_report(
                    leredd_devthresh_report,
                    model_tag=f"Dev-Tuned Screening (<= {dev_tuned_threshold}: review cue)",
                    note=(
                        f"Threshold {dev_tuned_threshold} selected exclusively on the development "
                        "partition without touching test pairs."
                    ),
                    predictions=leredd_devthresh_preds,
                    n_test=n_test,
                ),
            },
        },
        "ablation_deltas_vs_zeroshot": ablation_deltas,
        "ablation_validity": ablation_validity,
        "statistical_tests": {"mcnemar": mcnemar_results},
        "evaluation_limitations": [
            f"Test sample size n_test={n_test}; test systems: {test_systems}. "
            + ("Small-sample metrics have high uncertainty. " if n_test < 10 else ""),
            "BERTBaseline and TFIDF_LSABaseline fit on the train partition "
            f"({training_systems}, n={len(train_pairs)}) "
            f"and score the held-out test partition ({test_systems}); they are not zero-shot models.",
            *(
                [
                    "LEREDD evaluated with MockLLMClient — NOT a real LLM evaluation.",
                    "MockLLMClient does not interpret retrieved examples and returns fixed confidence 5; retrieval-strategy, ICL-example, and confidence-threshold ablations are not informative about model effects.",
                ]
                if is_mock
                else []
            ),
            "Zero or negative ablation deltas are preserved unedited; proposed retrieval/context variants do not guarantee accuracy gains on every split.",
            "Candidate-generation recall is distinct from classifier accuracy and is reported separately under candidate_recall_separation.",
            provenance_summary["provenance_note"],
            f"Test labels cover {n_classes} of 6 dependency classes.",
            f"Evaluated split strategy: {split_strategy}. Threshold tuning used the development partition only (test_set_tuning=False).",
            mcnemar_notice,
        ],
        "split_metadata": split_metadata,
    }

    # Write primary report
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to: {out_path}")

    # Write dedicated ablation report file
    ablation_out_path = out_path.parent / "ablation_report.json"
    with open(ablation_out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": "2.1",
                "run_timestamp": run_start,
                "n_test": n_test,
                "configuration": report["configuration"],
                "dev_threshold_tuning": report["dev_threshold_tuning"],
                "candidate_recall_separation": report["candidate_recall_separation"],
                "ablation_results": report["ablation_results"],
                "ablation_deltas_vs_zeroshot": report["ablation_deltas_vs_zeroshot"],
                "ablation_validity": report["ablation_validity"],
                "statistical_tests": report["statistical_tests"],
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Dedicated ablation report saved to: {ablation_out_path}")

    print("\n" + "=" * 78)
    print("LEREDD Safe Evaluation — Primary Model Benchmark Summary")
    print("=" * 78)
    print(f"{'Model':<38} {'Accuracy':>10} {'Macro-F1':>10} {'WtdF1':>10}")
    print("-" * 78)
    for res in report["model_results"].values():
        print(
            f"{res['model']:<38} "
            f"{res['overall_accuracy']:>10.4f} "
            f"{res['macro_f1']:>10.4f} "
            f"{res['weighted_f1']:>10.4f}"
        )
    print("=" * 78)

    if is_mock:
        print("[WARNING] LEREDD uses MockLLMClient — NOT a real LLM evaluation.")
    print(f"Report: {out_path}\n")

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LEREDD Safe Evaluation & Controlled Ablation Study Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-dir", default="data/splits/system_disjoint",
                        help="Directory containing system-disjoint or document-disjoint split partition JSON files.")
    parser.add_argument("--output", default="data/evaluation/system_disjoint_report.json",
                        help="Output JSON report file path.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed random seed for baseline model fitting.")
    parser.add_argument("--bert-classifier-type", "--model-type", dest="bert_classifier_type",
                        default="rf", choices=["rf", "lr"],
                        help="Classification backend head for BERTBaseline: 'rf' or 'lr'.")
    parser.add_argument("--llm-provider", default="mock", choices=["mock", "openai", "ollama", "gemini"],
                        help="LLM backend provider (defaults safely to 'mock').")
    parser.add_argument("--llm-model", default="mock",
                        help="Fixed LLM model version identifier.")
    parser.add_argument("--srs-context", default=None,
                        help="Optional path to SRS document text for context-enhanced RAG ablation.")
    parser.add_argument("--candidate-manifest", default=None,
                        help="Optional candidate generation manifest JSON to report candidate recall separately.")
    parser.add_argument("--no-strict", action="store_true",
                        help="Log warnings instead of raising errors for contamination/GT violations.")
    args = parser.parse_args()

    output_path = Path(args.output).expanduser().resolve()
    ablation_path = output_path.parent / "ablation_report.json"
    if output_path.exists() or ablation_path.exists():
        parser.error(
            "Refusing to overwrite an existing evaluation report; choose a fresh output directory."
        )
    project_data = (Path.cwd() / "data").resolve()
    protected_roots = [
        project_data / name for name in ("annotated", "raw", "replication", "splits")
    ]
    if any(output_path == root or root in output_path.parents for root in protected_roots):
        parser.error("Evaluation report output cannot be written inside source dataset directories.")

    try:
        run_evaluation(
            split_dir=args.split_dir,
            output_path=str(output_path),
            seed=args.seed,
            bert_model_type=args.bert_classifier_type,
            strict_contamination=not args.no_strict,
            strict_gt=not args.no_strict,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            srs_context_path=args.srs_context,
            candidate_manifest_path=args.candidate_manifest,
        )
    except (ContaminationError, GroundTruthViolationError, ValueError, FileNotFoundError) as exc:
        logger.error(f"Evaluation aborted due to safety violation: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
