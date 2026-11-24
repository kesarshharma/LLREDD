=== LEREDD Replication Evaluation Protocol Design ===
Status: PROPOSED — not executed; current imported packages are not eligible for metrics until local verification is completed.
Paper: arXiv:2602.22456 (paper record lists CC BY 4.0; dataset rights are separate)

1. Verified datasets (data/replication/*.json via integrate_replication.py loader):
   - adb_evaluation_annotation.json: 313 records (ADB Evaluation)
   - adb_pilot_annotation.json: 100 records (ADB Pilot)
   - apa_annotation.json: 200 records (APA)
   - tja_annotation.json: 200 records (TJA)

2. Label semantics (preserved, not remapped):
   - Original spreadsheet label: Implemented by
   - Internal schema: Requires / Implements / Conflicts / Details / Is similar / No dependency
   - Paper definition confirmed: Implements means the higher-level requirement is fulfilled by a lower-level requirement (Table I and Section III of arXiv v2).
   - Source-label crosswalk remains unresolved: the local source workbooks label the class Implemented by, while the paper scores Implements. A workbook example uses the same two requirements as the paper's example, but with the row order reversed; this is suggestive, not an explicit mapping rule.
   - Recommendation: Preserve the original Implemented by label and do not map it to Implements for metrics or release until the dataset source or authors provide an explicit crosswalk. The relation's semantic definition is known; the source-to-paper label mapping is not.

   - Evaluation eligibility: provenance-bearing records require local `verified` status, human source, a reviewer or annotator identity, and an `accepted_label` matching the scored `dependency_type`. The loader's `pending_review` records remain excluded. The replication runner is expected to fail closed with an empty eligible test set and write no report in the current data state. Do not edit imported JSON provenance to bypass this gate; use the human review/export workflow to create verified annotations.

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
   - Explicit source-to-paper mapping for Implemented by → Implements is still needed; paper definition alone does not specify how the imported workbook label should be normalized.
   - A read-only openpyxl audit found no exact or reversed duplicate requirement pairs in any of the four source workbooks after whitespace normalization and case-folding (ADB evaluation 313, ADB pilot 100, APA 200, TJA 200); all rows had both requirements and a label. This is a structural duplicate check, not human label verification.
   - The paper's arXiv record lists CC BY 4.0 for the paper. Its v2 text says the replication package will be made available upon publication, but the local imported package contains no dataset-specific license or redistribution agreement. Paper licensing is not evidence of separate dataset rights.
   - Local imported annotations have status `pending_review`, no annotator IDs, and have not completed this repository's independent local review. Source spreadsheet labels are not locally verified ground truth.

8. Definition of done (before benchmark):
   - Protocol approved (stages, metrics, split design).
   - Source-to-paper label mapping confirmed by an authoritative source or handled explicitly as unresolved and excluded from mapped metrics.
   - All JSON data loaded with provenance; loader tests pass; no existing fixtures/reports altered.
   - No claim of exact paper reproduction until source-label mapping, dataset rights, protocol approval, independent human verification, and benchmark execution are complete.
