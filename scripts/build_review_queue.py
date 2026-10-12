"""Script to demonstrate building and managing the human review queue from provisional labeled pairs."""

import json
import os
from leredd.data_models import RequirementPair
from leredd.review_queue import (
    QueueStatus,
    ReviewQueueConfig,
    ReviewQueueManager,
    ReviewReasonCode,
)


def main():
    provisional_file = "data/provisional/adb_provisional_hybrid.json"
    output_dir = "data/queue"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(provisional_file):
        print(f"Provisional file '{provisional_file}' not found. Run scripts/label_candidates.py first.")
        return

    print(f"Loading provisional labeled pairs from '{provisional_file}'...")
    with open(provisional_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = [RequirementPair.model_validate(item) for item in data]
    print(f"Loaded {len(pairs)} provisional requirement pairs.\n")

    # Reconcile outcome counts across the 31 candidates
    agreed = [p for p in pairs if p.provenance and "AGREEMENT" in (p.provenance.evidence or "")]
    disagreed = [p for p in pairs if p.provenance and "DISAGREEMENT" in (p.provenance.review_notes or "")]
    llm_only = [p for p in pairs if p.provenance and p.provenance.source.value == "llm"]
    heur_only = [p for p in pairs if p.provenance and p.provenance.source.value == "heuristic"]
    both_abstained = [p for p in pairs if p.provenance and "UNRESOLVED" in (p.provenance.review_notes or "")]

    print("=== PROVISIONAL CANDIDATE MUTUALLY EXCLUSIVE BREAKDOWN ===")
    print(f"Total Candidates Input : {len(pairs)}")
    print(f"1. Agreement (Ensemble): {len(agreed)}")
    print(f"2. Disagreement        : {len(disagreed)}")
    print(f"3. LLM-Only            : {len(llm_only)}")
    print(f"4. Heuristic-Only      : {len(heur_only)}")
    print(f"5. Both Abstained      : {len(both_abstained)}")
    print(f"Check Sum: {len(agreed) + len(disagreed) + len(llm_only) + len(heur_only) + len(both_abstained)} == {len(pairs)}\n")

    # Initialize ReviewQueueManager with default config
    manager = ReviewQueueManager(config=ReviewQueueConfig(include_single_method=True))

    print("Building Review Queue...")
    queue_items = manager.build_queue(pairs)
    manifest = manager.create_manifest(queue_items, source_identity="ADB_Provisional_Hybrid")

    queue_file = os.path.join(output_dir, "adb_review_queue.json")
    manifest_file = os.path.join(output_dir, "adb_review_queue_manifest.json")
    manager.save_queue(queue_items, manifest, queue_file, manifest_file)

    print("\n=== REVIEW QUEUE GENERATION SUMMARY ===")
    print(f"Total Items Queued : {manifest.total_items}")
    print(f"Pending Items      : {manifest.pending_count}")
    print(f"Reasons Breakdown  : {manifest.reasons_breakdown}")
    print(f"Saved queue to: '{queue_file}'\n")

    # Demonstrate recording a human decision on the highest priority pending item
    if queue_items:
        first_item = queue_items[0]
        print(f"Simulating Human Decision on Top Priority Queue Item [{first_item.queue_id}]...")
        print(f"   Pair: {first_item.pair_id} ({first_item.req_a_id} -> {first_item.req_b_id})")
        print(f"   Reasons: {[r.value for r in first_item.review_reasons]}")
        print(f"   Notes: {first_item.review_notes}")

        # Record decision: Accept if proposed label exists, else Correct to Requires
        decision_type = QueueStatus.ACCEPTED if first_item.proposed_label else QueueStatus.CORRECTED
        target_label = first_item.proposed_label or "Requires"

        updated_item, verified_pair = manager.record_decision(
            item=first_item,
            decision_type=decision_type,
            final_label=target_label,
            reviewer_id="lead_annotator_42",
            reviewer_notes="Verified direct requirement dependency after inspecting architectural specifications."
        )

        # Refresh queue idempotently
        refreshed_items = manager.refresh_queue(queue_items, pairs)
        refreshed_manifest = manager.create_manifest(refreshed_items, source_identity="ADB_Provisional_Hybrid")
        manager.save_queue(refreshed_items, refreshed_manifest, queue_file, manifest_file)

        print(f"\nRecorded Decision: Status={updated_item.status.value.upper()}, Final Label={verified_pair.dependency_type.value if verified_pair.dependency_type else 'None'}")
        print(f"Verified Pair Status: {verified_pair.provenance.status.value.upper()} (Source: {verified_pair.provenance.source.value.upper()})")
        print(f"Updated Queue Pending Count: {refreshed_manifest.pending_count} / {refreshed_manifest.total_items}")


if __name__ == "__main__":
    main()
