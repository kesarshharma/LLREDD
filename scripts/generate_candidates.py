"""Script to demonstrate candidate requirement pair generation on SRS documents."""

import os
import json
from leredd.utils import extract_requirements_from_file
from leredd.candidate_generator import CandidateGenerator, GenerationStrategy


def main():
    srs_file = "data/raw/adb_srs.txt"
    output_dir = "data/candidates"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Extracting requirements from '{srs_file}'...")
    reqs = extract_requirements_from_file(srs_file, system_name="ADB")
    print(f"Extracted {len(reqs)} unique requirements.\n")

    # 1. All-Pairs Strategy
    candidates_all, manifest_all = CandidateGenerator.generate_candidates(
        requirements=reqs,
        strategy=GenerationStrategy.ALL_PAIRS,
        source_identity="ADB_SRS",
        max_candidates=100
    )
    all_pairs_path = os.path.join(output_dir, "adb_candidates_all.json")
    all_manifest_path = os.path.join(output_dir, "adb_candidates_all_manifest.json")
    CandidateGenerator.save_candidates(candidates_all, manifest_all, all_pairs_path, all_manifest_path)

    # 2. BM25 Retrieval Strategy
    candidates_bm25, manifest_bm25 = CandidateGenerator.generate_candidates(
        requirements=reqs,
        strategy=GenerationStrategy.BM25,
        source_identity="ADB_SRS",
        top_k=3
    )
    bm25_pairs_path = os.path.join(output_dir, "adb_candidates_bm25.json")
    bm25_manifest_path = os.path.join(output_dir, "adb_candidates_bm25_manifest.json")
    CandidateGenerator.save_candidates(candidates_bm25, manifest_bm25, bm25_pairs_path, bm25_manifest_path)

    print("=== CANDIDATE GENERATION COMPLETE ===")
    print(f"1. ALL_PAIRS Strategy: {manifest_all.generated_count} candidates generated (exhaustive={manifest_all.is_exhaustive}, truncated={manifest_all.truncated_count})")
    print(f"   Saved to: {all_pairs_path}")
    print(f"2. BM25 Strategy: {manifest_bm25.generated_count} candidates generated (top_k={manifest_bm25.top_k})")
    print(f"   Saved to: {bm25_pairs_path}")


if __name__ == "__main__":
    main()
