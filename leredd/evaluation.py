"""Evaluation suite for computing metrics, confusion matrices, and statistical significance tests."""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Set
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from leredd.data_models import (
    AnnotationSource,
    RequirementPair,
    Prediction,
    DependencyType,
    EvaluationMetric,
    EvaluationReport,
    VerificationStatus,
)
from leredd.synthetic import is_synthetic


class EvaluationGuardrailError(ValueError):
    """Base exception for evaluation input-integrity failures."""

    def __init__(self, message: str, invalid_ids: List[str]):
        super().__init__(message)
        self.invalid_ids = sorted(set(invalid_ids))


class PredictionAlignmentError(EvaluationGuardrailError):
    """Raised when predictions do not align one-to-one with ground truth."""


class SplitOverlapError(EvaluationGuardrailError):
    """Raised when a declared system-disjoint split contains overlapping IDs."""


def validate_predictions(
    ground_truth: List[RequirementPair], predictions: List[Prediction]
) -> None:
    """Require unique predictions and complete ground-truth pair coverage."""
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for prediction in predictions:
        if prediction.pair_id in seen:
            duplicates.add(prediction.pair_id)
        seen.add(prediction.pair_id)

    if duplicates:
        ids = sorted(duplicates)
        raise PredictionAlignmentError(
            f"Predictions contain duplicate pair IDs: {', '.join(ids)}", ids
        )

    missing = sorted({pair.id for pair in ground_truth} - seen)
    if missing:
        raise PredictionAlignmentError(
            "Predictions are missing ground-truth pair IDs: " + ", ".join(missing),
            missing,
        )


def validate_system_disjoint_splits(
    train_pairs: List[RequirementPair],
    test_pairs: List[RequirementPair],
    dev_pairs: Optional[List[RequirementPair]] = None,
) -> Dict[str, Any]:
    """Check system, requirement, and pair IDs across train/test/dev partitions.

    Returns a compact audit when partitions are disjoint and raises
    ``SplitOverlapError`` with the exact overlapping IDs otherwise.
    """
    partitions = {"train": train_pairs, "test": test_pairs, "dev": dev_pairs or []}

    def identities(pairs: List[RequirementPair]) -> Dict[str, Set[str]]:
        return {
            "systems": {
                system
                for pair in pairs
                for system in (pair.req_a.system, pair.req_b.system)
                if system
            },
            "requirement_ids": {
                req_id
                for pair in pairs
                for req_id in (pair.req_a.id, pair.req_b.id)
                if req_id
            },
            "pair_ids": {pair.id for pair in pairs if pair.id},
        }

    ids_by_partition = {
        name: identities(pairs) for name, pairs in partitions.items()
    }
    overlap: Dict[str, Set[str]] = {
        category: set()
        for category in ("systems", "requirement_ids", "pair_ids")
    }
    names = list(partitions)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            for category in overlap:
                overlap[category].update(
                    ids_by_partition[left][category]
                    & ids_by_partition[right][category]
                )

    overlap = {key: values for key, values in overlap.items() if values}
    if overlap:
        invalid_ids = sorted(set().union(*overlap.values()))
        details = "; ".join(
            f"{category}: {sorted(values)}"
            for category, values in sorted(overlap.items())
        )
        raise SplitOverlapError(
            f"System-disjoint split contains overlapping identities ({details})",
            invalid_ids,
        )

    return {
        "is_disjoint": True,
        "overlapping_systems": [],
        "overlapping_requirement_ids": [],
        "overlapping_pair_ids": [],
    }


def ground_truth_provenance_problem(pair: RequirementPair) -> str | None:
    """Describe provenance that cannot substantiate a verified ground-truth label.

    Provenance-free legacy records remain a caller-level compatibility decision.
    """
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


class Evaluator:
    """Computes classification metrics and statistical tests matching paper methodology."""

    @staticmethod
    def evaluate(
        ground_truth: List[RequirementPair],
        predictions: List[Prediction],
        system_name: str = "EvaluationSystem",
        model_name: str = "LEREDD"
    ) -> EvaluationReport:
        """Compute Accuracy, Precision, Recall, F1 (per-class & macro/weighted) and confusion matrix."""
        Evaluator._validate_ground_truth(ground_truth)
        Evaluator._validate_prediction_coverage(ground_truth, predictions)
        # Align predictions by pair_id
        pred_map = {p.pair_id: p.dependency_type for p in predictions}

        y_true: List[str] = []
        y_pred: List[str] = []

        for pair in ground_truth:
            gt_type = pair.dependency_type.value
            pred_type = pred_map.get(pair.id, DependencyType.NO_DEPENDENCY).value
            y_true.append(gt_type)
            y_pred.append(pred_type)

        labels = [dt.value for dt in DependencyType]

        # Compute scikit-learn precision, recall, f1
        acc = float(accuracy_score(y_true, y_pred))
        p_per, r_per, f1_per, supp_per = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )

        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        per_class_metrics: Dict[str, EvaluationMetric] = {}
        for idx, label in enumerate(labels):
            per_class_metrics[label] = EvaluationMetric(
                precision=float(p_per[idx]),
                recall=float(r_per[idx]),
                f1_score=float(f1_per[idx]),
                support=int(supp_per[idx])
            )

        # Compute confusion matrix mapping
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
            per_class_metrics=per_class_metrics,
            confusion_matrix=cm_dict
        )

    @staticmethod
    def run_mcnemar_test(
        ground_truth: List[RequirementPair],
        preds_a: List[Prediction],
        preds_b: List[Prediction]
    ) -> Tuple[float, float]:
        """Perform McNemar's test comparing overall predictive accuracy of two models.

        Returns (statistic, p_value).
        """
        Evaluator._validate_ground_truth(ground_truth)
        Evaluator._validate_prediction_coverage(ground_truth, preds_a)
        Evaluator._validate_prediction_coverage(ground_truth, preds_b)
        map_a = {p.pair_id: p.dependency_type for p in preds_a}
        map_b = {p.pair_id: p.dependency_type for p in preds_b}

        # Contingency table:
        # both_correct, a_correct_b_wrong
        # a_wrong_b_correct, both_wrong
        n00 = n01 = n10 = n11 = 0

        for pair in ground_truth:
            gt = pair.dependency_type
            pa = map_a.get(pair.id, DependencyType.NO_DEPENDENCY)
            pb = map_b.get(pair.id, DependencyType.NO_DEPENDENCY)

            correct_a = (pa == gt)
            correct_b = (pb == gt)

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
            # Manual McNemar formula fallback
            b, c = n10, n01
            if b + c == 0:
                return 0.0, 1.0
            stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
            # Rough p-value approximation or default threshold
            p_val = 1.0 if stat == 0 else max(0.001, 1.0 / (1.0 + stat))
            return float(stat), float(p_val)

    @staticmethod
    def run_fishers_exact_test(
        tp_a: int, fp_a: int,
        tp_b: int, fp_b: int
    ) -> Tuple[float, float]:
        """Perform Fisher's Exact test comparing precision/recall contingency tables.

        Returns (odds_ratio, p_value).
        """
        table = [[tp_a, fp_a], [tp_b, fp_b]]
        odds_ratio, p_value = fisher_exact(table)
        return float(odds_ratio), float(p_value)

    @staticmethod
    def compute_wilson_score_interval(
        k: int, n: int, confidence: float = 0.95
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
        preds_b: List[Prediction]
    ) -> Dict:
        """Perform McNemar's test and return complete contingency table & discordant details."""
        Evaluator._validate_ground_truth(ground_truth)
        Evaluator._validate_prediction_coverage(ground_truth, preds_a)
        Evaluator._validate_prediction_coverage(ground_truth, preds_b)
        map_a = {p.pair_id: p.dependency_type for p in preds_a}
        map_b = {p.pair_id: p.dependency_type for p in preds_b}

        n00 = n01 = n10 = n11 = 0
        discordant_pairs = []

        for pair in ground_truth:
            gt = pair.dependency_type
            pa = map_a.get(pair.id, DependencyType.NO_DEPENDENCY)
            pb = map_b.get(pair.id, DependencyType.NO_DEPENDENCY)

            ca = (pa == gt)
            cb = (pb == gt)

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
    @staticmethod
    def _validate_ground_truth(ground_truth: List[RequirementPair]) -> None:
        """Refuse to score missing or explicitly unverified labels as negatives."""
        invalid = []
        for pair in ground_truth:
            if pair.dependency_type is None:
                invalid.append(f"{pair.id}: unlabeled")
            else:
                problem = ground_truth_provenance_problem(pair)
                if problem:
                    invalid.append(f"{pair.id}: {problem}")
        if invalid:
            raise ValueError(
                "Ground truth contains unlabeled or unverified pairs: "
                + ", ".join(invalid)
            )

    @staticmethod
    def _validate_prediction_coverage(
        ground_truth: List[RequirementPair], predictions: List[Prediction]
    ) -> None:
        validate_predictions(ground_truth, predictions)
