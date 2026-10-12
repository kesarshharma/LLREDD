=== LEREDD Replication Evaluation Protocol Design ===
Status: PROPOSED — not executed; current imported packages are not eligible for metrics until local verification is completed.
Paper: arXiv:2602.22456 / CC BY 4.0

1. Verified datasets (data/replication/*.json via integrate_replication.py loader):
   - adb_evaluation_annotation.json: 313 records (ADB Evaluation)
   - adb_pilot_annotation.json: 100 records (ADB Pilot)
   - apa_annotation.json: 200 records (APA)
   - tja_annotation.json: 200 records (TJA)

2. Label semantics (preserved, not remapped):
   - Original spreadsheet label: Implemented by
   - Internal schema: Requires / Implements / Conflicts / Details / Is similar / No dependency
   - Direction convention: Implemented by = higher-level fulfilled by lower-level (Eq 1/2) — unresolved without full paper text verification
   - Recommendation: Preserve original Implemented by; do not silently map to Implements until Eq direction confirmed.

   - Evaluation eligibility: records with provenance are ground truth only when local status is exactly `verified`. The loader's `pending_review` records remain excluded. The replication runner is expected to fail closed with an empty eligible test set and write no report in the current data state. Do not edit imported JSON provenance to bypass this gate; use the human review/export workflow to create verified annotations.

3. Proposed staged evaluation (pending approval):
   - Stage 1 (tuning/config): ADB Pilot (100) only — not primary evaluation ground truth.
   - Stage 2 (primary evaluation): ADB Evaluation (313) using existing system-disjoint split logic; do not include pilot in evaluation pool.
   - Stage 3 (cross-system): APA (200) + TJA (200) evaluated separately; maintain system-disjoint contamination guards.

4. Metrics (if executed): Per-class precision/recall/F1; macro-F1 (primary); accuracy / confusion matrix / dataset counts; report unresolved Implemented by counts explicitly.

5. Baselines / LLM: MockLLM (current) for reproducible baseline; real LLM not configured; do not claim live LLM performance.

6. Data split safeguards:
   - Use build_safe_retrieval_pool with strict=true to exclude test/dev IDs from train retrieval.
   - Shared EmbeddingModel is stateless; retrieval indexes per detector (not global).
   - No random 813 merge; keep datasets independent.

7. Unresolved blockers:
   - Implemented by Eq direction confirmation needed from paper definitions.
   - No openpyxl cell-level open duplicate audit completed (structurally 0 exact/reversed duplicates verified).
   - No independent replication URL / dataset-specific redistribution agreement found locally; paper CC BY 4.0 applies but dataset redistribution should include attribution.
   - Local imported annotations have status `pending_review`, no annotator IDs, and have not completed this repository's independent local review. Source spreadsheet labels are not locally verified ground truth.

8. Definition of done (before benchmark):
   - Protocol approved (stages, metrics, split design).
   - Label direction either confirmed or handled explicitly (flagged unresolved).
   - All JSON data loaded with provenance; loader tests pass; no existing fixtures/reports altered.
   - No claim of exact paper reproduction until full Eq verification and benchmark execution completed.
