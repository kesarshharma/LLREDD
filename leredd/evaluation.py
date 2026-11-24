"""Evaluation suite for computing metrics, confusion matrices, and statistical significance tests.

Provides strict ground truth guardrails, prediction alignment verification,
contamination detection, system/document-disjoint split validation, development-set-only
threshold tuning, and explicit separation of candidate recall from classifier accuracy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.stats import fisher_exact
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from statsmodels.stats.contingency_tables import mcnemar

from leredd.data_models import (
    AnnotationSource,
    DependencyType,
    EvaluationMetric,
    EvaluationReport,
    Prediction,
    RequirementPair,
    VerificationStatus,
)
from leredd.synthetic import is_synthetic

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guardrail Exceptions
# ---------------------------------------------------------------------------

class EvaluationGuardrailError(ValueError):
    """Base exception for evaluation guardrail violations."""

    def __init__(self, message: str, invalid_ids: Optional[List[str]] = None):
        super().__init__(message)
        self.invalid_ids: List[str] = invalid_ids or []


class GroundTruthViolationError(EvaluationGuardrailError):
    """Raised when ground truth contains unverified, provisional, synthetic, or unlabeled records."""


class PredictionAlignmentError(EvaluationGuardrailError):
    """Raised when predictions have duplicate IDs or missing coverage for ground truth."""


class SplitOverlapError(EvaluationGuardrailError):
    """Raised when train/test/dev partitions have overlapping systems, documents, requirements, or pairs."""


class ContaminationError(RuntimeError):
    """Raised when test/dev requirement IDs leak into the retrieval pool."""


# ---------------------------------------------------------------------------
# Ground Truth and Split Guardrail Functions
# ---------------------------------------------------------------------------

def validate_ground_truth(
    pairs: List[RequirementPair],
    partition_label: str = "EVAL",
    strict: bool = True,
    allow_legacy: bool = True,
    reject_unlabeled: bool = True,
) -> Tuple[List[RequirementPair], List[str]]:
    """Validate ground truth records ensuring strictly verified, non-synthetic, and labeled pairs.

    Enforces fail-closed evaluation guardrails:
      - Empty dataset raises GroundTruthViolationError when strict=True.
      - Synthetic fixtures (synthetic=True or source_identity='synthetic-review-tests')
        are strictly rejected and must never enter benchmark evaluations.
      - Pairs with dependency_type=None are rejected when reject_unlabeled=True.
      - Pairs with non-verified provenance status (PROVISIONAL, PENDING_REVIEW,
        HUMAN_REVIEWED, REJECTED) are rejected; only explicit VerificationStatus.VERIFIED
        is accepted.
      - Legacy pairs (no provenance) are accepted conditionally when allow_legacy=True,
        or rejected when allow_legacy=False.
    """
    if not pairs:
        msg = f"[{partition_label}] Ground truth dataset is empty."
        logger.warning(msg)
        if strict:
            raise GroundTruthViolationError(msg, invalid_ids=[])
        return [], [msg]

    eligible: List[RequirementPair] = []
    warnings_found: List[str] = []
    synthetic_ids: List[str] = []
    unlabeled_ids: List[str] = []
    unverified_ids: List[str] = []
    legacy_ids: List[str] = []

    for pair in pairs:
        if is_synthetic(pair):
            synthetic_ids.append(pair.id)
            warnings_found.append(
                f"[{partition_label}] REJECTED pair {pair.id}: Synthetic record detected. "
                "Synthetic data must never enter benchmark evaluation."
            )
            continue

        if pair.dependency_type is None:
            unlabeled_ids.append(pair.id)
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} has dependency_type=None (unlabeled)."
            )
            continue

        if pair.provenance is not None:
            status = pair.provenance.status
            if status != VerificationStatus.VERIFIED:
                unverified_ids.append(pair.id)
                warnings_found.append(
                    f"[{partition_label}] REJECTED pair {pair.id}: VerificationStatus={status.value if status else None}. "
                    "Only verified pairs (VerificationStatus.VERIFIED) are eligible as ground truth."
                )
                continue
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} provenance status={status.value} — accepted."
            )
        else:
            if not allow_legacy:
                legacy_ids.append(pair.id)
                warnings_found.append(
                    f"[{partition_label}] REJECTED pair {pair.id}: Legacy record with no provenance "
                    "rejected under strict verification mode (allow_legacy=False)."
                )
                continue
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} has NO provenance metadata "
                "(legacy annotated record). Accepted conditionally."
            )

        eligible.append(pair)

    if strict:
        if synthetic_ids:
            raise GroundTruthViolationError(
                f"[{partition_label}] Synthetic records rejected from ground truth ({len(synthetic_ids)} pairs); "
                f"synthetic/demo records are ineligible: {synthetic_ids}",
                invalid_ids=synthetic_ids,
            )
        if reject_unlabeled and unlabeled_ids:
            raise GroundTruthViolationError(
                f"[{partition_label}] Unlabeled pairs rejected ({len(unlabeled_ids)}): {unlabeled_ids}; "
                "ground truth contains unlabeled or unverified records.",
                invalid_ids=unlabeled_ids,
            )
        if unverified_ids:
            raise GroundTruthViolationError(
                f"[{partition_label}] Ground truth contains unlabeled or unverified pairs; "
                f"unverified pairs rejected ({len(unverified_ids)}): {unverified_ids}",
                invalid_ids=unverified_ids,
            )
        if legacy_ids:
            raise GroundTruthViolationError(
                f"[{partition_label}] Legacy pairs without provenance rejected ({len(legacy_ids)} pairs): {legacy_ids}",
                invalid_ids=legacy_ids,
            )

    return eligible, warnings_found


def ground_truth_provenance_problem(pair: RequirementPair) -> str | None:
    """Describe provenance that cannot substantiate a verified ground-truth label."""
    if is_synthetic(pair):
        return "synthetic/demo records cannot be used as ground truth"
    provenance = pair.provenance
    if provenance is None:
        return None
    if pair.dependency_type is None:
        return "dependency_type is missing"
    if provenance.status != VerificationStatus.VERIFIED:
        return f"status={provenance.status.value} (explicit VERIFIED status is required)"
    if provenance.source != AnnotationSource.HUMAN:
        return f"verified source={provenance.source.value!r} is not human"
    actors = [
        actor.strip()
        for actor in (provenance.reviewer_id, provenance.annotator_id)
        if actor and actor.strip()
    ]
    if not actors:
        return "verified human annotation is missing reviewer_id and annotator_id"
    if all(actor.casefold() in {"human_reviewer_1", "human_cli"} for actor in actors):
        return "verified human annotation uses only a generic placeholder identity"
    if provenance.accepted_label is None:
        return "verified human annotation is missing accepted_label"
    if provenance.accepted_label != pair.dependency_type:
        return (
            f"accepted_label={provenance.accepted_label.value!r} does not match "
            f"dependency_type={pair.dependency_type.value!r}"
        )
    return None


def check_ground_truth_eligibility(
    pairs: List[RequirementPair],
    partition_label: str = "EVAL",
    strict: bool = True,
    allow_legacy: bool = True,
    reject_unlabeled: bool = False,
) -> Tuple[List[RequirementPair], List[str]]:
    """Check each pair for ground-truth eligibility with full provenance coherence checks.

    Rules (fail-closed):
      - synthetic/demo records           -> REJECTED (always)
      - provenance problem / != VERIFIED -> REJECTED (always)
      - dependency_type is None          -> SKIPPED when reject_unlabeled=False; REJECTED when True
      - provenance is None (legacy)      -> Conditionally accepted with warning when allow_legacy=True
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
                raise GroundTruthViolationError(msg, invalid_ids=[pair.id])
            continue

        if pair.dependency_type is None:
            msg = f"[{partition_label}] Pair {pair.id} has dependency_type=None — skipped."
            warnings_found.append(msg)
            if strict and reject_unlabeled:
                raise GroundTruthViolationError(
                    f"[{partition_label}] Unlabeled pairs rejected: {[pair.id]}",
                    invalid_ids=[pair.id],
                )
            continue

        if pair.provenance is not None:
            status_val = pair.provenance.status.value if pair.provenance.status else None
            provenance_problem = ground_truth_provenance_problem(pair)
            if provenance_problem:
                msg = f"[{partition_label}] REJECTED pair {pair.id}: {provenance_problem}."
                warnings_found.append(msg)
                if strict:
                    raise GroundTruthViolationError(msg, invalid_ids=[pair.id])
                continue
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} provenance status={status_val} — accepted."
            )
        else:
            if not allow_legacy:
                msg = (
                    f"[{partition_label}] REJECTED pair {pair.id}: "
                    "Legacy record with no provenance rejected (allow_legacy=False)."
                )
                warnings_found.append(msg)
                if strict:
                    raise GroundTruthViolationError(msg, invalid_ids=[pair.id])
                continue
            warnings_found.append(
                f"[{partition_label}] Pair {pair.id} has NO provenance metadata "
                "(legacy annotated record). Accepted conditionally."
            )

        eligible.append(pair)

    return eligible, warnings_found


def validate_predictions(
    ground_truth: List[RequirementPair],
    predictions: List[Prediction],
) -> None:
    """Validate that predictions cover all ground truth pairs with no duplicates."""
    seen_ids: Set[str] = set()
    duplicate_ids: List[str] = []
    for pred in predictions:
        if pred.pair_id in seen_ids:
            duplicate_ids.append(pred.pair_id)
        seen_ids.add(pred.pair_id)

    if duplicate_ids:
        unique_dups = sorted(set(duplicate_ids))
        raise PredictionAlignmentError(
            f"Duplicate prediction IDs detected ({len(unique_dups)}): {unique_dups}",
            invalid_ids=unique_dups,
        )

    gt_ids = {pair.id for pair in ground_truth}
    missing_ids = sorted(gt_ids - seen_ids)
    if missing_ids:
        raise PredictionAlignmentError(
            f"Missing prediction coverage; predictions missing ground-truth pair IDs ({len(missing_ids)}): {missing_ids}",
            invalid_ids=missing_ids,
        )


def validate_system_disjoint_splits(
    train_pairs: List[RequirementPair],
    test_pairs: List[RequirementPair],
    dev_pairs: Optional[List[RequirementPair]] = None,
) -> Dict[str, Any]:
    """Verify that train, test, and optional dev partitions are strictly system-disjoint."""
    dev_list = dev_pairs or []

    def get_systems(pairs: List[RequirementPair]) -> Set[str]:
        systems = set()
        for p in pairs:
            if p.req_a.system:
                systems.add(p.req_a.system)
            if p.req_b.system:
                systems.add(p.req_b.system)
        return systems

    def get_req_ids(pairs: List[RequirementPair]) -> Set[str]:
        req_ids = set()
        for p in pairs:
            req_ids.add(p.req_a.id)
            req_ids.add(p.req_b.id)
        return req_ids

    def get_pair_ids(pairs: List[RequirementPair]) -> Set[str]:
        return {p.id for p in pairs}

    train_systems = get_systems(train_pairs)
    test_systems = get_systems(test_pairs)
    dev_systems = get_systems(dev_list)

    train_reqs = get_req_ids(train_pairs)
    test_reqs = get_req_ids(test_pairs)
    dev_reqs = get_req_ids(dev_list)

    train_pairs_ids = get_pair_ids(train_pairs)
    test_pairs_ids = get_pair_ids(test_pairs)
    dev_pairs_ids = get_pair_ids(dev_list)

    overlap_systems = (train_systems & test_systems) | (train_systems & dev_systems) | (test_systems & dev_systems)
    overlap_reqs = (train_reqs & test_reqs) | (train_reqs & dev_reqs) | (test_reqs & dev_reqs)
    overlap_pairs = (train_pairs_ids & test_pairs_ids) | (train_pairs_ids & dev_pairs_ids) | (test_pairs_ids & dev_pairs_ids)

    audit_report = {
        "train_size": len(train_pairs),
        "test_size": len(test_pairs),
        "dev_size": len(dev_list),
        "train_systems": sorted(train_systems),
        "test_systems": sorted(test_systems),
        "dev_systems": sorted(dev_systems),
        "overlapping_systems": sorted(overlap_systems),
        "overlapping_req_ids": sorted(overlap_reqs),
        "overlapping_pair_ids": sorted(overlap_pairs),
        "is_disjoint": not (overlap_systems or overlap_reqs or overlap_pairs),
    }

    if overlap_systems or overlap_reqs or overlap_pairs:
        invalid_ids = sorted(overlap_systems | overlap_reqs | overlap_pairs)
        details = []
        if overlap_systems:
            details.append(f"systems: {sorted(overlap_systems)}")
        if overlap_reqs:
            details.append(f"req_ids: {sorted(overlap_reqs)}")
        if overlap_pairs:
            details.append(f"pair_ids: {sorted(overlap_pairs)}")
        msg = f"System-disjoint split violation detected ({', '.join(details)})."
        logger.warning(msg)
        raise SplitOverlapError(msg, invalid_ids=invalid_ids)

    return audit_report


def validate_document_disjoint_splits(
    train_pairs: List[RequirementPair],
    test_pairs: List[RequirementPair],
    dev_pairs: Optional[List[RequirementPair]] = None,
) -> Dict[str, Any]:
    """Verify that train, test, and optional dev partitions are strictly document-disjoint."""
    dev_list = dev_pairs or []

    def _doc_key(req: Any) -> str:
        meta = getattr(req, "metadata", None) or {}
        doc = meta.get("document") or getattr(req, "module", None) or meta.get("module") or req.source or req.system
        return str(doc)

    def get_docs(pairs: List[RequirementPair]) -> Set[str]:
        docs = set()
        for p in pairs:
            docs.add(_doc_key(p.req_a))
            docs.add(_doc_key(p.req_b))
        return docs

    def get_req_ids(pairs: List[RequirementPair]) -> Set[str]:
        return {r_id for p in pairs for r_id in (p.req_a.id, p.req_b.id)}

    def get_pair_ids(pairs: List[RequirementPair]) -> Set[str]:
        return {p.id for p in pairs}

    train_docs = get_docs(train_pairs)
    test_docs = get_docs(test_pairs)
    dev_docs = get_docs(dev_list)

    train_reqs = get_req_ids(train_pairs)
    test_reqs = get_req_ids(test_pairs)
    dev_reqs = get_req_ids(dev_list)

    train_pids = get_pair_ids(train_pairs)
    test_pids = get_pair_ids(test_pairs)
    dev_pids = get_pair_ids(dev_list)

    overlap_docs = (train_docs & test_docs) | (train_docs & dev_docs) | (test_docs & dev_docs)
    overlap_reqs = (train_reqs & test_reqs) | (train_reqs & dev_reqs) | (test_reqs & dev_reqs)
    overlap_pairs = (train_pids & test_pids) | (train_pids & dev_pids) | (test_pids & dev_pids)

    audit_report = {
        "train_size": len(train_pairs),
        "test_size": len(test_pairs),
        "dev_size": len(dev_list),
        "train_documents": sorted(train_docs),
        "test_documents": sorted(test_docs),
        "dev_documents": sorted(dev_docs),
        "overlapping_documents": sorted(overlap_docs),
        "overlapping_req_ids": sorted(overlap_reqs),
        "overlapping_pair_ids": sorted(overlap_pairs),
        "is_disjoint": not (overlap_docs or overlap_reqs or overlap_pairs),
    }

    if overlap_docs or overlap_reqs or overlap_pairs:
        invalid_ids = sorted(overlap_docs | overlap_reqs | overlap_pairs)
        details = []
        if overlap_docs:
            details.append(f"documents: {sorted(overlap_docs)}")
        if overlap_reqs:
            details.append(f"req_ids: {sorted(overlap_reqs)}")
        if overlap_pairs:
            details.append(f"pair_ids: {sorted(overlap_pairs)}")
        msg = f"Document-disjoint split violation detected ({', '.join(details)})."
        logger.warning(msg)
        raise SplitOverlapError(msg, invalid_ids=invalid_ids)

    return audit_report


def build_safe_retrieval_pool(
    train_pairs: List[RequirementPair],
    test_pairs: List[RequirementPair],
    dev_pairs: Optional[List[RequirementPair]] = None,
    strict: bool = True,
) -> Tuple[List[RequirementPair], Dict[str, Any]]:
    """Build a retrieval pool strictly isolated from test/dev IDs and synthetic records."""
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
        "excluded_synthetic_pair_ids": sorted(excluded_synthetic_pair_ids),
        "excluded_synthetic_count": len(excluded_synthetic_pair_ids),
        "safe_pool_size": len(safe_pool),
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


def tune_confidence_threshold_on_dev(
    dev_pairs: List[RequirementPair],
    dev_predictions: List[Prediction],
    *,
    partition_label: str = "DEV",
    default_threshold: int = 4,
    min_coverage: float = 0.25,
    allow_legacy: bool = True,
) -> Dict[str, Any]:
    """Tune the confidence review cutoff strictly on the development set (Part 4 & Part 8).

    Evaluates candidate cutoffs t in [-1, 0, 1, 2, 3, 4] on dev predictions:
      - Predictions with confidence <= t are routed for human review.
      - Predictions with confidence > t (or all predictions when t == -1) are auto-accepted.
      - Selects t maximizing selective Macro-F1 on auto-accepted dev pairs subject to
        auto-accepted coverage >= min_coverage. Ties prefer default_threshold.

    Raises:
        EvaluationGuardrailError: if partition_label indicates a test split.
    """
    if "test" in partition_label.casefold():
        raise EvaluationGuardrailError(
            f"Threshold tuning on partition {partition_label!r} is strictly forbidden; "
            "tune thresholds on the development set only.",
            invalid_ids=[p.id for p in dev_pairs],
        )

    if not dev_pairs or not dev_predictions:
        return {
            "status": "skipped_empty_dev_partition",
            "partition_used": partition_label,
            "test_set_touched": False,
            "selected_threshold": default_threshold,
            "candidates": [],
        }

    eligible_dev, _ = validate_ground_truth(
        dev_pairs,
        partition_label=partition_label,
        strict=True,
        allow_legacy=allow_legacy,
        reject_unlabeled=True,
    )
    validate_predictions(eligible_dev, dev_predictions)

    pred_by_id = {p.pair_id: p for p in dev_predictions}
    labels = [dt.value for dt in DependencyType]
    n_dev = len(eligible_dev)
    candidates: List[Dict[str, Any]] = []

    for threshold in (-1, 0, 1, 2, 3, 4):
        accepted_gt: List[str] = []
        accepted_pred: List[str] = []
        routed_count = 0
        for pair in eligible_dev:
            pred = pred_by_id[pair.id]
            if threshold >= 0 and pred.confidence <= threshold:
                routed_count += 1
            else:
                assert pair.dependency_type is not None
                accepted_gt.append(pair.dependency_type.value)
                accepted_pred.append(pred.dependency_type.value)

        coverage = len(accepted_gt) / n_dev if n_dev > 0 else 0.0
        review_rate = routed_count / n_dev if n_dev > 0 else 0.0
        if accepted_gt:
            sel_acc = float(accuracy_score(accepted_gt, accepted_pred))
            _, _, sel_macro_f1, _ = precision_recall_fscore_support(
                accepted_gt, accepted_pred, labels=labels, average="macro", zero_division=0
            )
            sel_f1 = float(sel_macro_f1)
        else:
            sel_acc = 0.0
            sel_f1 = 0.0

        candidates.append(
            {
                "threshold": threshold,
                "auto_accepted_count": len(accepted_gt),
                "review_routed_count": routed_count,
                "coverage": round(coverage, 4),
                "review_rate": round(review_rate, 4),
                "selective_accuracy": round(sel_acc, 4),
                "selective_macro_f1": round(sel_f1, 4),
                "meets_min_coverage": coverage >= min_coverage,
            }
        )

    feasible = [c for c in candidates if c["meets_min_coverage"]] or candidates
    best = max(
        feasible,
        key=lambda c: (
            c["selective_macro_f1"],
            c["selective_accuracy"],
            1 if c["threshold"] == default_threshold else 0,
            c["threshold"],
        ),
    )

    return {
        "status": "tuned_on_dev_only",
        "partition_used": partition_label,
        "test_set_touched": False,
        "n_dev": n_dev,
        "min_coverage": min_coverage,
        "selected_threshold": int(best["threshold"]),
        "selected_metrics": best,
        "candidates": candidates,
    }


def summarize_candidate_recall_vs_classifier(
    classifier_report: EvaluationReport,
    *,
    candidate_manifest: Optional[Dict[str, Any]] = None,
    coverage_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Explicitly separate candidate-generation recall from classifier accuracy (Part 8)."""
    candidate_stage: Dict[str, Any] = {
        "measured_separately": True,
        "pair_space_recall": None,
        "uncovered_pair_count": None,
        "weighted_positive_dependency_recall": None,
        "note": (
            "Candidate generation recall measures what fraction of the requirement pair space "
            "(or true positive dependencies in a stratified sample) is retained prior to "
            "classification. It must never be conflated with classifier accuracy on the "
            "supplied evaluation split."
        ),
    }
    if isinstance(candidate_manifest, dict):
        candidate_stage["strategy"] = candidate_manifest.get("strategy")
        candidate_stage["pair_space_recall"] = candidate_manifest.get("pair_space_recall")
        candidate_stage["uncovered_pair_count"] = candidate_manifest.get("uncovered_pair_count")
        candidate_stage["exhaustive_pair_count"] = candidate_manifest.get("exhaustive_pair_count")

    if isinstance(coverage_report, dict):
        candidate_stage["weighted_positive_dependency_recall"] = coverage_report.get(
            "weighted_positive_recall"
        )
        candidate_stage["coverage_confidence_interval_95"] = coverage_report.get(
            "confidence_interval_95"
        )

    return {
        "candidate_generation_stage": candidate_stage,
        "classifier_evaluation_stage": {
            "model_name": classifier_report.model_name,
            "system_name": classifier_report.system_name,
            "overall_accuracy": classifier_report.overall_accuracy,
            "macro_f1": classifier_report.macro_f1,
            "weighted_f1": classifier_report.weighted_f1,
            "conditioned_on_supplied_pairs": True,
        },
    }


# ---------------------------------------------------------------------------
# Evaluator Class
# ---------------------------------------------------------------------------

class Evaluator:
    """Computes classification metrics, guardrail validations, and statistical tests."""

    validate_ground_truth = staticmethod(validate_ground_truth)
    check_ground_truth_eligibility = staticmethod(check_ground_truth_eligibility)
    validate_predictions = staticmethod(validate_predictions)
    validate_system_disjoint_splits = staticmethod(validate_system_disjoint_splits)
    validate_document_disjoint_splits = staticmethod(validate_document_disjoint_splits)
    build_safe_retrieval_pool = staticmethod(build_safe_retrieval_pool)
    tune_confidence_threshold_on_dev = staticmethod(tune_confidence_threshold_on_dev)
    summarize_candidate_recall_vs_classifier = staticmethod(summarize_candidate_recall_vs_classifier)

    @staticmethod
    def evaluate(
        ground_truth: List[RequirementPair],
        predictions: List[Prediction],
        system_name: str = "EvaluationSystem",
        model_name: str = "LEREDD",
        strict_ground_truth: bool = True,
        allow_legacy: bool = True,
        validate_alignment: bool = True,
    ) -> EvaluationReport:
        """Compute Accuracy, Precision, Recall, F1 (per-class & macro/weighted) and confusion matrix."""
        eligible_gt, _ = validate_ground_truth(
            ground_truth,
            partition_label=system_name,
            strict=strict_ground_truth,
            allow_legacy=allow_legacy,
            reject_unlabeled=True,
        )

        provenance_problems = {
            pair.id: problem
            for pair in eligible_gt
            if (problem := ground_truth_provenance_problem(pair)) is not None
        }
        if provenance_problems:
            invalid_ids = sorted(provenance_problems)
            details = "; ".join(
                f"{pair_id}: {provenance_problems[pair_id]}" for pair_id in invalid_ids
            )
            raise GroundTruthViolationError(
                f"Ground truth provenance is invalid: {details}",
                invalid_ids=invalid_ids,
            )

        if validate_alignment:
            validate_predictions(eligible_gt, predictions)

        pred_map = {p.pair_id: p.dependency_type for p in predictions}

        y_true: List[str] = []
        y_pred: List[str] = []

        for pair in eligible_gt:
            gt_type = pair.dependency_type.value if pair.dependency_type else DependencyType.NO_DEPENDENCY.value
            pred_type = pred_map.get(pair.id, DependencyType.NO_DEPENDENCY).value
            y_true.append(gt_type)
            y_pred.append(pred_type)

        labels = [dt.value for dt in DependencyType]

        acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        p_per, r_per, f1_per, supp_per = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        _, _, f1_weight, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        per_class_metrics: Dict[str, EvaluationMetric] = {}
        for idx, label in enumerate(labels):
            per_class_metrics[label] = EvaluationMetric(
                precision=float(p_per[idx]),
                recall=float(r_per[idx]),
                f1_score=float(f1_per[idx]),
                support=int(supp_per[idx]),
            )

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_dict: Dict[str, Dict[str, int]] = {}
        for i, label_true in enumerate(labels):
            cm_dict[label_true] = {}
            for j, label_pred in enumerate(labels):
                cm_dict[label_true][label_pred] = int(cm[i, j])

        return EvaluationReport(
            system_name=system_name,
            model_name=model_name,
            overall_accuracy=acc,
            macro_f1=float(f1_macro),
            weighted_f1=float(f1_weight),
            macro_precision=float(p_macro),
            macro_recall=float(r_macro),
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm_dict,
        )

    @staticmethod
    def run_mcnemar_test(
        ground_truth: List[RequirementPair],
        preds_a: List[Prediction],
        preds_b: List[Prediction],
    ) -> Tuple[float, float]:
        """Perform McNemar's test comparing overall predictive accuracy of two models."""
        map_a = {p.pair_id: p.dependency_type for p in preds_a}
        map_b = {p.pair_id: p.dependency_type for p in preds_b}

        n00 = n01 = n10 = n11 = 0

        for pair in ground_truth:
            gt = pair.dependency_type or DependencyType.NO_DEPENDENCY
            pa = map_a.get(pair.id, DependencyType.NO_DEPENDENCY)
            pb = map_b.get(pair.id, DependencyType.NO_DEPENDENCY)

            correct_a = pa == gt
            correct_b = pb == gt

            if correct_a and correct_b:
                n11 += 1
            elif correct_a and not correct_b:
                n10 += 1
            elif not correct_a and correct_b:
                n01 += 1
            else:
                n00 += 1

        table = [[n11, n10], [n01, n00]]
        try:
            result = mcnemar(table, exact=True)
            return float(result.statistic), float(result.pvalue)
        except Exception:
            b, c = n10, n01
            if b + c == 0:
                return 0.0, 1.0
            stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
            p_val = 1.0 if stat == 0 else max(0.001, 1.0 / (1.0 + stat))
            return float(stat), float(p_val)

    @staticmethod
    def run_fishers_exact_test(
        tp_a: int,
        fp_a: int,
        tp_b: int,
        fp_b: int,
    ) -> Tuple[float, float]:
        """Perform Fisher's Exact test comparing precision/recall contingency tables."""
        table = [[tp_a, fp_a], [tp_b, fp_b]]
        odds_ratio, p_value = fisher_exact(table)
        return float(odds_ratio), float(p_value)

    @staticmethod
    def compute_wilson_score_interval(
        k: int,
        n: int,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """Compute Wilson score confidence interval for binomial proportion k/n."""
        if n == 0:
            return {"point_estimate": 0.0, "lower_bound": 0.0, "upper_bound": 0.0, "margin_pp": 0.0}
        import scipy.stats as stats

        p_hat = k / n
        z = float(stats.norm.ppf(1 - (1 - confidence) / 2))
        denom = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denom
        margin = (z / denom) * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))
        lower = max(0.0, float(center - margin))
        upper = min(1.0, float(center + margin))
        return {
            "point_estimate": float(p_hat),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "margin_pp": float((upper - lower) / 2 * 100),
        }

    @staticmethod
    def analyze_mcnemar(
        ground_truth: List[RequirementPair],
        preds_a: List[Prediction],
        preds_b: List[Prediction],
    ) -> Dict[str, Any]:
        """Perform McNemar's test and return complete contingency table & discordant details."""
        map_a = {p.pair_id: p.dependency_type for p in preds_a}
        map_b = {p.pair_id: p.dependency_type for p in preds_b}

        n00 = n01 = n10 = n11 = 0
        discordant_pairs = []

        for pair in ground_truth:
            gt = pair.dependency_type or DependencyType.NO_DEPENDENCY
            pa = map_a.get(pair.id, DependencyType.NO_DEPENDENCY)
            pb = map_b.get(pair.id, DependencyType.NO_DEPENDENCY)

            ca = pa == gt
            cb = pb == gt

            if ca and cb:
                n11 += 1
            elif ca and not cb:
                n10 += 1
                discordant_pairs.append({
                    "pair_id": pair.id,
                    "model_a_correct": True,
                    "model_b_correct": False,
                    "gt_label": gt.value,
                    "pred_a": pa.value,
                    "pred_b": pb.value,
                })
            elif not ca and cb:
                n01 += 1
                discordant_pairs.append({
                    "pair_id": pair.id,
                    "model_a_correct": False,
                    "model_b_correct": True,
                    "gt_label": gt.value,
                    "pred_a": pa.value,
                    "pred_b": pb.value,
                })
            else:
                n00 += 1

        stat, pval = Evaluator.run_mcnemar_test(ground_truth, preds_a, preds_b)

        return {
            "contingency_table": {
                "both_correct_n11": n11,
                "model_a_only_n10_b": n10,
                "model_b_only_n01_c": n01,
                "both_wrong_n00": n00,
            },
            "discordant_counts": {
                "b": n10,
                "c": n01,
                "total_discordant": n10 + n01,
            },
            "discordant_pairs": discordant_pairs,
            "statistic": stat,
            "p_value": pval,
        }
