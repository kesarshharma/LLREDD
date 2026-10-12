"""Script to demonstrate provisional requirement dependency labeling on candidate pairs."""

import json
import os
from leredd.data_models import RequirementPair
from leredd.provisional_labeler import ProvisionalLabeler, LabelingMode


def main():
    candidate_file = "data/candidates/adb_candidates_bm25.json"
    output_dir = "data/provisional"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(candidate_file):
        print(f"Candidate file '{candidate_file}' not found. Please run scripts/generate_candidates.py first.")
        return

    print(f"Loading candidate requirement pairs from '{candidate_file}'...")
    with open(candidate_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = [RequirementPair.model_validate(item) for item in data]
    print(f"Loaded {len(candidates)} candidate pairs.\n")

    # Initialize ProvisionalLabeler
    labeler = ProvisionalLabeler()

    # 1. Run Heuristic-only labeling
    print("Running Heuristic-only Provisional Labeling...")
    provisional_pairs_h, manifest_h = labeler.process_batch(
        candidates=candidates,
        mode=LabelingMode.HEURISTIC,
        source_identity="ADB_Candidates_BM25"
    )

    output_h_path = os.path.join(output_dir, "adb_provisional_heuristic.json")
    output_h_manifest = os.path.join(output_dir, "adb_provisional_heuristic_manifest.json")
    labeler.save_provisional_dataset(provisional_pairs_h, manifest_h, output_h_path, output_h_manifest)

    # 2. Run Hybrid labeling (Heuristic + Mock LLM / configured LLM)
    print("Running Hybrid Provisional Labeling...")
    provisional_pairs_hybrid, manifest_hybrid = labeler.process_batch(
        candidates=candidates,
        mode=LabelingMode.HYBRID,
        source_identity="ADB_Candidates_BM25"
    )

    output_hybrid_path = os.path.join(output_dir, "adb_provisional_hybrid.json")
    output_hybrid_manifest = os.path.join(output_dir, "adb_provisional_hybrid_manifest.json")
    labeler.save_provisional_dataset(provisional_pairs_hybrid, manifest_hybrid, output_hybrid_path, output_hybrid_manifest)

    print("\n=== PROVISIONAL LABELING COMPLETE ===")
    print(f"Heuristic Mode: {manifest_h.labeled_count}/{manifest_h.input_candidate_count} labeled, {manifest_h.unresolved_count} unresolved.")
    print(f"   Saved to: {output_h_path}")
    print(f"Hybrid Mode: {manifest_hybrid.labeled_count}/{manifest_hybrid.input_candidate_count} labeled, {manifest_hybrid.agreed_count} agreed, {manifest_hybrid.disagreed_count} disagreed, {manifest_hybrid.unresolved_count} unresolved.")
    print(f"   Saved to: {output_hybrid_path}")


if __name__ == "__main__":
    main()
