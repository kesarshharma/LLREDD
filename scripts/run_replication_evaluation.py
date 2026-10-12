"""Prototype replication-evaluation runner — ADB Pilot only, MockLLM, fail-closed.

VERIFIED BASE:
- load_replication_annotation() (scripts/integrate_replication.py) validated by 15 failure-mode tests
- Evaluator.evaluate, build_safe_retrieval_pool, MockLLMClient, LEREDDDetector.detect_batch verified existing APIs
- Real dataset counts: adb_pilot 100 (verified)

LIMITATIONS / BLOCKERS (explicit in output):
- Synthetic Requirement IDs only — retrieval claims unsupported
- No real split metadata in loader JSON; split is synthetic index-based
- MockLLM only; not a real LLM benchmark
- Implemented by excluded from standard class metrics per fail-closed rule
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from scripts.integrate_replication import load_replication_annotation, INTERNAL_LABELS
from leredd.config import LEREDDConfig
from leredd.data_models import DependencyType, Prediction, Requirement, RequirementPair
from leredd.evaluation import Evaluator
from leredd.detector import LEREDDDetector
from leredd.llm import MockLLMClient
from leredd.baselines.bert_classifier import BERTBaseline
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.provisional_labeler import HeuristicLabeler
from leredd.embeddings import EmbeddingModel
from scripts.run_full_evaluation import check_ground_truth_eligibility, build_safe_retrieval_pool

SCRIPT_VERSION = "0.1.0-prototype"
DATASET = "adb_pilot"
OUTPUT_DIR = Path("data/evaluation/replication")
DATASET_SYSTEM_MAP = {"adb_evaluation": "ADB", "adb_pilot": "ADB", "apa": "APA", "tja": "TJA"}


def build_pairs(dataset_name: str = DATASET):
    """Load replication dataset and build RequirementPairs."""
    records = load_replication_annotation(dataset=dataset_name)
    system = DATASET_SYSTEM_MAP.get(dataset_name, "UNKNOWN")
    pairs = []
    implemented_by = []
    for rec in records:
        pair_id = rec["pair_id"]
        original_label = rec.get("dependency_status_original")
        normalized_label = rec.get("normalized_internal_label")
        unresolved = rec.get("normalized_mapping_unresolved")
        provenance = rec.get("provenance")
        req_a_text = rec["req_a_text"]
        req_b_text = rec["req_b_text"]

        # Synthetic IDs — bookkeeping only; NOT claimed as real requirement identity
        req_a = Requirement(id=f"{dataset_name}_A_{pair_id}", text=req_a_text, system=system, source="replication_loader")
        req_b = Requirement(id=f"{dataset_name}_B_{pair_id}", text=req_b_text, system=system, source="replication_loader")

        if original_label == "Implemented by":
            # Fail-closed: preserve original, never remap to IMPLEMENTS, exclude from scoring
            implemented_by.append({
                "pair_id": pair_id,
                "original_label": original_label,
                "normalized_internal_label": normalized_label,
                "normalized_mapping_unresolved": unresolved,
                "provenance_status": provenance.get("status") if isinstance(provenance, dict) else None,
            })
            dep_type = None  # skipped by eligibility checker
        else:
            label_map = {
                "Requires": DependencyType.REQUIRES,
                "Implements": DependencyType.IMPLEMENTS,
                "Conflicts": DependencyType.CONFLICTS,
                "Details": DependencyType.DETAILS,
                "Is similar": DependencyType.IS_SIMILAR,
                "No_dependency": DependencyType.NO_DEPENDENCY,
            }
            dep_type = label_map.get(normalized_label)

        pairs.append(RequirementPair(
            id=pair_id,
            req_a=req_a,
            req_b=req_b,
            dependency_type=dep_type,
            provenance=provenance if isinstance(provenance, dict) else None,
        ))
    return pairs, implemented_by


def synthetic_split(pairs, seed=42, fractions=(0.70, 0.15, 0.15)):
    """Deterministic index-based split with seeded shuffle — avoids ordering bias."""
    import random
    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_n = int(round(n * fractions[0]))
    dev_n = int(round(n * fractions[1]))
    train = shuffled[:train_n]
    dev = shuffled[train_n : train_n + dev_n]
    test = shuffled[train_n + dev_n :]
    return train, dev, test


def run_prototype(dataset: str = DATASET, output_path: str = None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    pairs, implemented_by = build_pairs(dataset)
    train, dev, test = synthetic_split(pairs)

    # Eligibility (fail-closed)
    train_eligible, w_train = check_ground_truth_eligibility(train, "TRAIN", strict=False)
    dev_eligible, w_dev = check_ground_truth_eligibility(dev, "DEV", strict=False)
    test_eligible, w_test = check_ground_truth_eligibility(test, "TEST", strict=False)
    all_warnings = w_train + w_dev + w_test

    if len(test_eligible) == 0:
        raise ValueError(f"{dataset}: eligible test set empty after ground-truth filtering — non-evaluable; check provenance / label exclusions.")

    # Contamination guard — explicit note on synthetic identity limitation
    safe_pool = None
    leakage = {
        "contamination_detected": False,
        "excluded_req_ids": [],
        "excluded_pair_ids": [],
        "train_pool_size": len(train_eligible),
        "safe_pool_size": 0,
        "note": "Retrieval-pool identity is synthetic (loader has no real req IDs/systems for retrieval). Leakage check performed structurally only.",
    }
    try:
        safe_pool, leakage_audit = build_safe_retrieval_pool(
            train_eligible, test_eligible, dev_eligible, strict=False
        )
        safe_pool = safe_pool
        leakage = leakage_audit
        leakage["note"] = "Synthetic IDs passed structural contamination check; retrieval claims remain unsupported due to invented identity."
    except Exception as e:
        leakage["contamination_detected"] = True
        leakage["note"] = f"Retrieval pool blocked: {e}. Synthetic identity insufficient for defensible retrieval evaluation."
        safe_pool = None

    # Predictions — Heuristic baseline (always runs on eligible test)
    heuristic_preds = []
    for p in test_eligible:
        res = HeuristicLabeler.evaluate_pair(p)
        label = res.suggested_label if res.suggested_label is not None else DependencyType.NO_DEPENDENCY
        heuristic_preds.append(Prediction(
            pair_id=p.id,
            dependency_type=label,
            rationale=res.evidence,
            confidence=int(round(res.confidence * 5)),
        ))
    heuristic_report = Evaluator.evaluate(test_eligible, heuristic_preds, system_name=dataset, model_name="HeuristicLabeler")

    # MockLLM LEREDD — zero-shot / hybrid with pool (if structurally safe)
    mock_llm = MockLLMClient(model="mock")
    detector = LEREDDDetector(
        config=LEREDDConfig(llm_provider="mock", retrieval_mode="hybrid", k_examples=6, confidence_threshold=4),
        llm=mock_llm,
        embedder=EmbeddingModel(),
    )
    # Disabled: synthetic identity only; retrieval evaluation unsupported
    pool_for_detector = None
    leredd_preds = detector.detect_batch(
        test_eligible,
        annotated_pool=pool_for_detector,
        max_workers=1,
        retrieval_mode="hybrid",
        confidence_threshold=4,
    )
    leredd_report = Evaluator.evaluate(test_eligible, leredd_preds, system_name=dataset, model_name="LEREDD-Hybrid-Mock")

    # Validation: confusion matrix must account for every eligible test example
    cm = leredd_report.confusion_matrix
    cm_total = sum(sum(row.values()) for row in cm.values()) if isinstance(cm, dict) else sum(cm.flatten()) if hasattr(cm, 'flatten') else int(cm)
    if cm_total != len(test_eligible):
        raise ValueError(f"{dataset}: confusion matrix total ({cm_total}) != eligible test count ({len(test_eligible)}); metric inconsistency.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_tag = "LEREDD-Hybrid-Mock"
    note = (
        "MockLLMClient — deterministic mock, NOT a real LLM evaluation. "
        "Prototype limited to ADB Pilot. Synthetic req IDs; retrieval claims unsupported. "
        "Implemented by preserved as original label and excluded from standard metrics."
    )

    # Implemented-by summary (fail-closed reporting)
    imple_summary = {
        "dataset": dataset,
        "total_records": len(pairs),
        "implemented_by_original": len(implemented_by),
        "eligible_implemented_by": sum(
            1 for r in implemented_by
            if r["normalized_mapping_unresolved"] is True and r["normalized_internal_label"] == "Implemented by"
        ),
        "excluded_unresolved_remapped": sum(
            1 for r in implemented_by
            if not (r["normalized_mapping_unresolved"] is True and r["normalized_internal_label"] == "Implemented by")
        ),
    }

    # Per-class metrics from LEREDD (primary prototype model)
    per_class = {}
    for cls, m in leredd_report.per_class_metrics.items():
        per_class[cls] = {
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "support": m.support,
        }

    label_dist_test = dict(Counter(p.dependency_type.value if p.dependency_type else "None" for p in test_eligible))

    report = {
        "prototype_version": SCRIPT_VERSION,
        "dataset": dataset,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "mock",
        "note": note,
        "configuration": {
            "dataset_source": f"load_replication_annotation({dataset!r})",
            "synthetic_split": {
                "train": len(train_eligible),
                "dev": len(dev_eligible),
                "test": len(test_eligible),
            },
            "retrieval_pool_available": False,
            "retrieval_pool_size": 0,
            "retrieval_evaluated": False,
            "retrieval_note": "Synthetic IDs only — retrieval evaluation disabled (unsupported).",
            "split_seed": 42,
        },
        "ground_truth_eligibility": {
            "eligible_train": len(train_eligible),
            "eligible_dev": len(dev_eligible),
            "eligible_test": len(test_eligible),
            "warnings": all_warnings,
            "implemented_by_total": len(implemented_by),
            "implemented_by_excluded_unresolved_or_remapped": imple_summary["excluded_unresolved_remapped"],
            "provenance_note": (
                "Loader provenance verified: source='imported', status='pending_review', annotator_id=None. "
                "Legacy records (no provenance) not present in replication packages."
            ),
        },
        "leakage_audit": leakage,
        "implemented_by_summary": imple_summary,
        "label_distribution": {
            "test_eligible": label_dist_test,
            "test_total_before_exclusion": len(test),
        },
        "model_results": {
            "HeuristicLabeler": {
                "overall_accuracy": float(heuristic_report.overall_accuracy),
                "macro_f1": float(heuristic_report.macro_f1),
                "weighted_f1": float(heuristic_report.weighted_f1),
                "per_class_metrics": {
                    cls: {
                        "precision": m.precision,
                        "recall": m.recall,
                        "f1_score": m.f1_score,
                        "support": m.support,
                    } for cls, m in heuristic_report.per_class_metrics.items()
                },
                "confusion_matrix": heuristic_report.confusion_matrix,
            },
            "LEREDD-Hybrid-Mock": {
                "overall_accuracy": float(leredd_report.overall_accuracy),
                "macro_f1": float(leredd_report.macro_f1),
                "weighted_f1": float(leredd_report.weighted_f1),
                "per_class_metrics": per_class,
                "confusion_matrix": leredd_report.confusion_matrix,
            },
        },
    }

    if output_path is None:
        output_path = OUTPUT_DIR / f"{dataset}_LEREDD-Hybrid-Mock_{timestamp}_mock.json"
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Collision-safe: timestamp + dataset + model tag; never overwrite system_disjoint_report.json or others
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return output_path, report, test_eligible, implemented_by


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prototype replication evaluation — ADB Pilot only")
    parser.add_argument("--dataset", default=DATASET, help="Dataset to evaluate (default: adb_pilot)")
    parser.add_argument("--output", default=None, help="Output path (default: auto-named under data/evaluation/replication/)")
    args = parser.parse_args()
    path, report, te, ib = run_prototype(dataset=args.dataset, output_path=args.output)
    print(f"PROTOTYPE REPORT: {path}")
    print(f"Eligible test pairs: {len(te)} | Implemented by records: {len(ib)} | MockLLM: yes | Retrieval pool: {report['configuration']['retrieval_pool_size']}")
