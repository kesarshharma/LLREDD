"""
scripts/run_full_evaluation.py
==============================
Safe, reproducible evaluation script for a supplied leakage-checked split.

Per-run reports summarize the actual input partition sizes, systems, provenance
eligibility, configuration, and limitations. LEREDD variants in this runner use
MockLLMClient; they are demonstrations and do not measure remote-LLM quality.

Usage:
  python scripts/run_full_evaluation.py [--split-dir data/splits/system_disjoint]
                                        [--output data/evaluation/system_disjoint_report.json]
                                        [--seed 42]
                                        [--model-type rf]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from leredd.config import LEREDDConfig
from leredd.data_models import (
    DependencyType,
    Prediction,
    RequirementPair,
)
from leredd.evaluation import (
    ContaminationError,
    Evaluator,
    ground_truth_provenance_problem,
    GroundTruthViolationError,
    build_safe_retrieval_pool,
    check_ground_truth_eligibility,
    validate_ground_truth,
    validate_predictions,
    validate_system_disjoint_splits,
)
from leredd.utils import load_pairs_json
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.detector import LEREDDDetector
from leredd.llm import MockLLMClient
from leredd.provisional_labeler import HeuristicLabeler
from leredd.embeddings import EmbeddingModel
from leredd.synthetic import is_synthetic

SCRIPT_VERSION = "2.0.0"

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


# ---------------------------------------------------------------------------
# Ground-Truth Safety Guards
# ---------------------------------------------------------------------------

class GroundTruthViolationError(ValueError):
    """Raised when provisional or test-fixture records are supplied as ground truth."""


def check_ground_truth_eligibility(
    pairs: List[RequirementPair],
    partition_label: str,
    strict: bool = True,
) -> Tuple[List[RequirementPair], List[str]]:
    """
    Check each pair for ground-truth eligibility. Returns (eligible, warnings).

    Rules (fail-closed):
      - provenance.status != VERIFIED     -> REJECTED (always)
      - dependency_type is None           -> SKIPPED (cannot evaluate)
      - provenance is None (legacy)       -> Conditionally accepted with warning
    """
    eligible: List[RequirementPair] = []
    warnings_found: List[str] = []

    for pair in pairs:
        if is_synthetic(pair):
            msg = (
                f"[{partition_label}] REJECTED pair {pair.id}: "
                "synthetic/demo records cannot be used as ground truth."
            )
            warnings_found.append(msg)
            if strict:
                raise GroundTruthViolationError(msg)
            continue
        if pair.dependency_type is None:
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} has dependency_type=None — skipped."
            )
            continue

        if pair.provenance is not None:
            status_val = pair.provenance.status.value if pair.provenance.status else None
            provenance_problem = ground_truth_provenance_problem(pair)
            if provenance_problem:
                msg = (
                    f"[{partition_label}] REJECTED pair {pair.id}: "
                    f"{provenance_problem}."
                )
                warnings_found.append(msg)
                if strict:
                    raise GroundTruthViolationError(msg)
                continue
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} provenance status={status_val} — accepted."
            )
        else:
            # Legacy annotated record — no provenance field
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} has NO provenance metadata "
                f"(legacy annotated record). Accepted conditionally."
            )

        eligible.append(pair)

    return eligible, warnings_found


# ---------------------------------------------------------------------------
# Contamination Guard
# ---------------------------------------------------------------------------

class ContaminationError(RuntimeError):
    """Raised when test/dev requirement IDs leak into the retrieval pool."""


def build_safe_retrieval_pool(
    train_pairs: List[RequirementPair],
    test_pairs: List[RequirementPair],
    dev_pairs: Optional[List[RequirementPair]] = None,
    strict: bool = True,
) -> Tuple[List[RequirementPair], Dict]:
    """
    Build a retrieval pool strictly isolated from test and dev requirement IDs.

    Returns (safe_pool, leakage_audit_report).
    Raises ContaminationError if leakage is detected and strict=True.
    """
    excluded_req_ids: Set[str] = set()
    excluded_pair_ids: Set[str] = set()

    for pair in test_pairs:
        excluded_req_ids.add(pair.req_a.id)
        excluded_req_ids.add(pair.req_b.id)
        excluded_pair_ids.add(pair.id)

    for pair in (dev_pairs or []):
        excluded_req_ids.add(pair.req_a.id)
        excluded_req_ids.add(pair.req_b.id)
        excluded_pair_ids.add(pair.id)

    contaminated_pairs: List[str] = []
    excluded_synthetic_pair_ids: List[str] = []
    safe_pool: List[RequirementPair] = []

    for pair in train_pairs:
        pair_req_ids = {pair.req_a.id, pair.req_b.id}
        leaked_ids = pair_req_ids & excluded_req_ids
        if leaked_ids or pair.id in excluded_pair_ids:
            contaminated_pairs.append(
                f"Pair {pair.id} (req_ids: {sorted(pair_req_ids)}) "
                f"overlaps with test/dev excluded IDs: {sorted(leaked_ids or {pair.id})}"
            )
        elif is_synthetic(pair):
            excluded_synthetic_pair_ids.append(pair.id)
        else:
            safe_pool.append(pair)

    leakage_audit = {
        "excluded_req_ids": sorted(excluded_req_ids),
        "excluded_pair_ids": sorted(excluded_pair_ids),
        "train_pool_size": len(train_pairs),
        "contaminated_pairs": contaminated_pairs,
        "contaminated_count": len(contaminated_pairs),
        "safe_pool_size": len(safe_pool),
        "excluded_synthetic_pair_ids": sorted(excluded_synthetic_pair_ids),
        "excluded_synthetic_count": len(excluded_synthetic_pair_ids),
        "contamination_detected": len(contaminated_pairs) > 0,
    }

    if contaminated_pairs:
        msg = (
            f"CONTAMINATION DETECTED: {len(contaminated_pairs)} training pair(s) "
            f"contain requirement IDs present in test/dev set. "
            f"Details: {contaminated_pairs}"
        )
        logger.warning(msg)
        if strict:
            raise ContaminationError(msg)

    logger.info(
        f"Contamination guard: {len(train_pairs)} train -> "
        f"{len(contaminated_pairs)} removed -> {len(safe_pool)} safe retrieval pool."
    )
    return safe_pool, leakage_audit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _label_distribution(pairs: List[RequirementPair]) -> Dict[str, int]:
    counter: Counter = Counter()
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


def _ground_truth_report_summary(raw_partitions, eligible_partitions) -> Dict:
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


def _serialize_report(report, model_tag: str, note: str, predictions: List[Prediction], n_test: int) -> Dict:
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


def run_heuristic_model(test_pairs: List[RequirementPair]) -> Tuple[List[Prediction], Any]:
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
) -> Tuple[List[Prediction], Any]:
    """Evaluate LEREDD detector variant on test pairs."""
    config = LEREDDConfig(
        llm_provider="mock",
        retrieval_mode=retrieval_mode,
        k_examples=k_examples,
        confidence_threshold=confidence_threshold,
    )
    mock_llm = MockLLMClient(model="mock")
    detector = LEREDDDetector(config=config, llm=mock_llm, embedder=embedder)
    pool = retrieval_pool if k_examples > 0 else None
    preds = detector.detect_batch(
        test_pairs,
        annotated_pool=pool,
        max_workers=1,
        retrieval_mode=retrieval_mode,
        confidence_threshold=confidence_threshold,
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
) -> Dict:
    """Run the full safe evaluation pipeline and return the report dict."""
    run_start = datetime.now(timezone.utc).isoformat()
    logger.info("=" * 70)
    logger.info("LEREDD Integrated Benchmark & Ablation Evaluation")
    logger.info(f"Split dir  : {split_dir}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Seed       : {seed}")
    logger.info(f"BERT type  : {bert_model_type}")
    logger.info("=" * 70)

    split_path = Path(split_dir)
    if not split_path.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    meta_path = split_path / "split_metadata.json"
    split_metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            split_metadata = json.load(f)

    # Step 1: Load split partitions
    logger.info("Step 1: Loading split partitions ...")
    train_pairs_raw = _load_split_partition(split_path, "train")
    dev_pairs_raw   = _load_split_partition(split_path, "dev")
    test_pairs_raw  = _load_split_partition(split_path, "test")
    logger.info(
        f"  Raw: train={len(train_pairs_raw)}, dev={len(dev_pairs_raw)}, test={len(test_pairs_raw)}"
    )

    if not test_pairs_raw:
        raise ValueError(
            "Test partition is empty. Ensure system-disjoint split has been generated."
        )

    split_strategy = str(
        split_metadata.get("summary", {}).get("strategy", "system")
    ).casefold().split(".")[-1]
    split_integrity = {"checked": False, "strategy": split_strategy}
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

    # Step 4: BERTBaseline
    cls_desc = (
        "RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)"
        if bert_model_type == "rf"
        else "LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)"
    )
    logger.info(f"Step 4: BERTBaseline (classifier={bert_model_type.upper()}: {cls_desc}) ...")
    bert = BERTBaseline(model_type=bert_model_type)
    bert.fit(train_pairs)
    bert_preds = bert.predict_batch(test_pairs)
    bert_report = Evaluator.evaluate(
        test_pairs, bert_preds, system_name=_system_description(test_pairs), model_name=f"BERTBaseline-{bert_model_type.upper()}"
    )
    logger.info(f"  Accuracy={bert_report.overall_accuracy:.4f}, Macro-F1={bert_report.macro_f1:.4f}")

    # Step 5: TFIDF_LSA
    logger.info("Step 5: TFIDF_LSABaseline ...")
    tfidf_lsa = TFIDF_LSABaseline()
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

    # Step 7: Zero-Shot LLM & Proposed Hybrid System & Ablation Matrix
    logger.info("Step 7: LEREDD Variants & Ablation Matrix (MockLLM) ...")

    # Reuse a single real EmbeddingModel instance across all LEREDD evaluation variants
    shared_embedder = EmbeddingModel()

    # Primary LEREDD Proposed System (Hybrid RRF + Examples k=6 + Confidence Threshold=4)
    leredd_hybrid_preds, leredd_hybrid_report = run_leredd_variant(
        test_pairs, safe_retrieval_pool, retrieval_mode="hybrid", k_examples=6, confidence_threshold=4, model_name="LEREDD-Hybrid", embedder=shared_embedder
    )

    # Primary LEREDD Zero-Shot (k=0 examples)
    leredd_zeroshot_preds, leredd_zeroshot_report = run_leredd_variant(
        test_pairs, safe_retrieval_pool, retrieval_mode="dense", k_examples=0, confidence_threshold=4, model_name="LEREDD-ZeroShot", embedder=shared_embedder
    )

    # Ablation Variants:
    # A1. Dense Retrieval Only (SBERT)
    leredd_dense_preds, leredd_dense_report = run_leredd_variant(
        test_pairs, safe_retrieval_pool, retrieval_mode="dense", k_examples=6, confidence_threshold=4, model_name="LEREDD-Dense", embedder=shared_embedder
    )

    # A2. Lexical Retrieval Only (BM25)
    leredd_bm25_preds, leredd_bm25_report = run_leredd_variant(
        test_pairs, safe_retrieval_pool, retrieval_mode="bm25", k_examples=6, confidence_threshold=4, model_name="LEREDD-BM25", embedder=shared_embedder
    )

    # A3. Thresholding Disabled (Off: threshold=-1)
    leredd_nothresh_preds, leredd_nothresh_report = run_leredd_variant(
        test_pairs, safe_retrieval_pool, retrieval_mode="hybrid", k_examples=6, confidence_threshold=-1, model_name="LEREDD-Hybrid-NoThreshold", embedder=shared_embedder
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

    mcnemar_results: Dict = {
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
    }

    # Assemble report
    report = {
        "schema_version": "2.0",
        "script_version": SCRIPT_VERSION,
        "run_timestamp": run_start,
        "configuration": {
            "split_dir": str(split_dir),
            "output_path": str(output_path),
            "seed": seed,
            "bert_classifier_backend": bert_model_type,
            "bert_classifier_description": cls_desc,
            "split_strategy": split_metadata.get("summary", {}).get("strategy", "system"),
            "leredd_provider": "mock",
            "leredd_note": "MockLLMClient — deterministic keyword heuristics. NOT a real LLM evaluation.",
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
        "model_results": {
            "BERTBaseline": _serialize_report(
                bert_report,
                model_tag=f"BERTBaseline-{bert_model_type.upper()}",
                note=(
                    f"SBERT (all-MiniLM-L6-v2) + {cls_desc}. "
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
                note=(
                    "Zero-shot LLM inference without in-context dynamic examples (k=0)."
                ),
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
        },
        "ablation_results": {
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
                    note="Hybrid RRF fusion combining SBERT and BM25Plus.",
                    predictions=leredd_hybrid_preds,
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
            },
        },
        "ablation_validity": MOCK_ABLATION_VALIDITY,
        "statistical_tests": {"mcnemar": mcnemar_results},

        "evaluation_limitations": [
            f"Test sample size n_test={n_test}; test systems: {test_systems}. "
            + ("Small-sample metrics have high uncertainty. " if n_test < 10 else ""),
            "BERTBaseline and TFIDF_LSABaseline fit on the train partition "
            f"({training_systems}, n={len(train_pairs)}) "
            f"and score the held-out test partition ({test_systems}); they are not zero-shot models.",
            "LEREDD evaluated with MockLLMClient — NOT a real LLM evaluation.",
            "MockLLMClient does not interpret retrieved examples and returns fixed confidence 5; retrieval-strategy, ICL-example, and confidence-threshold ablations are not informative about model effects.",
            provenance_summary["provenance_note"],
            f"Test labels cover {n_classes} of 6 dependency classes.",
            "This report evaluates only the supplied split; a separate requirement-disjoint evaluation was not run.",
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
                "schema_version": "2.0",
                "run_timestamp": run_start,
                "n_test": n_test,
                "ablation_results": report["ablation_results"],
                "ablation_validity": report["ablation_validity"],
                "statistical_tests": report["statistical_tests"],
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"Dedicated ablation report saved to: {ablation_out_path}")

    # Print summary table
    print("\n" + "=" * 78)
    print("LEREDD Safe Evaluation — Primary Model Benchmark Summary")
    print("=" * 78)
    print(f"{'Model':<35} {'Accuracy':>10} {'Macro-F1':>10} {'WtdF1':>10}")
    print("-" * 78)
    for key, res in report["model_results"].items():
        print(
            f"{res['model']:<35} "
            f"{res['overall_accuracy']:>10.4f} "
            f"{res['macro_f1']:>10.4f} "
            f"{res['weighted_f1']:>10.4f}"
        )
    print("=" * 78)

    print("\n" + "=" * 78)
    print("LEREDD Ablation Study Matrix Summary")
    print("=" * 78)
    for cat_name, cat_dict in report["ablation_results"].items():
        print(f"\n--- Dimension: {cat_name.upper()} ---")
        print(f"{'Variant':<35} {'Accuracy':>10} {'Macro-F1':>10} {'WtdF1':>10}")
        print("-" * 78)
        for var_key, var_res in cat_dict.items():
            print(
                f"{var_res['model']:<35} "
                f"{var_res['overall_accuracy']:>10.4f} "
                f"{var_res['macro_f1']:>10.4f} "
                f"{var_res['weighted_f1']:>10.4f}"
            )
    print("=" * 78)

    print(f"\n[WARNING] n_test={len(test_pairs)} — results are demonstrative only.")
    print("[WARNING] LEREDD uses MockLLMClient — NOT a real LLM evaluation.")
    print(f"Report: {out_path}\n")

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LEREDD Safe Evaluation & Ablation Study Benchmark Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--split-dir", default="data/splits/system_disjoint",
                        help="Directory containing system-disjoint split partition JSON files.")
    parser.add_argument("--output", default="data/evaluation/system_disjoint_report.json",
                        help="Output JSON report file path.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for baseline model fitting.")
    parser.add_argument("--bert-classifier-type", "--model-type", dest="bert_classifier_type",
                        default="rf", choices=["rf", "lr"],
                        help="Classification backend head for BERTBaseline: 'rf' (RandomForestClassifier) or 'lr' (LogisticRegression). Note: All baselines and ablation variants are evaluated in sequence.")
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
        )
    except (ContaminationError, GroundTruthViolationError, FileNotFoundError, ValueError) as e:
        logger.error(f"Evaluation aborted: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
