"""Evaluation suite for computing metrics, confusion matrices, and statistical significance tests."""

import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from leredd.data_models import (
    RequirementPair,
    Prediction,
    DependencyType,
    EvaluationMetric,
    EvaluationReport,
)


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
        # Align predictions by pair_id
        pred_map = {p.pair_id: p.dependency_type for p in predictions}

        y_true: List[str] = []
        y_pred: List[str] = []

        for pair in ground_truth:
            gt_type = pair.dependency_type.value if pair.dependency_type else DependencyType.NO_DEPENDENCY.value
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
        map_a = {p.pair_id: p.dependency_type for p in preds_a}
        map_b = {p.pair_id: p.dependency_type for p in preds_b}

        # Contingency table:
        # both_correct, a_correct_b_wrong
        # a_wrong_b_correct, both_wrong
        n00 = n01 = n10 = n11 = 0

        for pair in ground_truth:
            gt = pair.dependency_type or DependencyType.NO_DEPENDENCY
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
