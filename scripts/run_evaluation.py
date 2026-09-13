"""Script to run end-to-end evaluation comparing LEREDD against baselines."""

import os
import json
from leredd.config import get_config
from leredd.utils import load_pairs_json
from leredd.detector import LEREDDDetector
from leredd.evaluation import Evaluator
from leredd.baselines.tfidf_lsa import TFIDF_LSABaseline
from leredd.baselines.bert_classifier import BERTBaseline

def main():
    config = get_config()
    annotated_file = "data/annotated/adb_pairs.json"
    if not os.path.exists(annotated_file):
        print(f"File {annotated_file} not found.")
        return

    pairs = load_pairs_json(annotated_file)
    print(f"Loaded {len(pairs)} ground truth requirement pairs for ADB system evaluation.")

    # 1. Run LEREDD
    print("\n--- Running LEREDD Engine ---")
    detector = LEREDDDetector(config=config)
    leredd_preds = detector.detect_batch(pairs, annotated_pool=pairs)
    leredd_report = Evaluator.evaluate(pairs, leredd_preds, system_name="ADB", model_name="LEREDD")

    # 2. Run TF-IDF & LSA
    print("--- Running TF-IDF & LSA Baseline ---")
    tfidf_baseline = TFIDF_LSABaseline()
    tfidf_preds = tfidf_baseline.predict_batch(pairs)
    tfidf_report = Evaluator.evaluate(pairs, tfidf_preds, system_name="ADB", model_name="TF-IDF & LSA")

    # 3. Run Fine-tuned BERT Baseline
    print("--- Running Fine-Tuned BERT Baseline ---")
    bert_baseline = BERTBaseline()
    bert_preds = bert_baseline.predict_batch(pairs)
    bert_report = Evaluator.evaluate(pairs, bert_preds, system_name="ADB", model_name="BERT")

    # Print Summary Results
    print("\n=================== BENCHMARK EVALUATION RESULTS ===================")
    print(f"{'Model':25s} | {'Accuracy':10s} | {'Macro F1':10s} | {'Weighted F1':10s}")
    print("-" * 65)
    print(f"{leredd_report.model_name:25s} | {leredd_report.overall_accuracy:10.4f} | {leredd_report.macro_f1:10.4f} | {leredd_report.weighted_f1:10.4f}")
    print(f"{bert_report.model_name:25s} | {bert_report.overall_accuracy:10.4f} | {bert_report.macro_f1:10.4f} | {bert_report.weighted_f1:10.4f}")
    print(f"{tfidf_report.model_name:25s} | {tfidf_report.overall_accuracy:10.4f} | {tfidf_report.macro_f1:10.4f} | {tfidf_report.weighted_f1:10.4f}")
    print("====================================================================")

    # Statistical Test
    stat, p_val = Evaluator.run_mcnemar_test(pairs, leredd_preds, bert_preds)
    print(f"\nMcNemar's Significance Test (LEREDD vs BERT): statistic={stat:.4f}, p_value={p_val:.4f}")

if __name__ == "__main__":
    main()
