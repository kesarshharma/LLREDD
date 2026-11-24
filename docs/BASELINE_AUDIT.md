# Part 1: Baseline and Repository Audit

**Date:** 2026-11-25  
**Repository:** `kesarshharma/LLREDD` (`main` @ `a7f3002`)  
**Environment:** Windows, CPython `3.11.15`, `pytest 9.2.0`, `ruff 0.17.6`, `HF_HUB_OFFLINE=1`

---

## 1. Baseline Test & Static Analysis Results

### 1.1 Automated Pytest Suite (Unmodified Baseline)
Command executed:
```powershell
$env:TEMP = (Resolve-Path .pytest-tmp).Path
$env:TMP = (Resolve-Path .pytest-tmp).Path
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m pytest tests/ --tb=short -q
```
- **Collected:** 456 items across 39 test modules
- **Passed:** 455
- **Skipped:** 1 (`tests/test_replication_dataset.py` — local-only private replication spreadsheet check skipped when untracked external files are absent)
- **Failed:** 0
- **Duration:** `51.26s`
- **Data Integrity:** `git status` clean across all tracked files under `data/`.

### 1.2 Static Analysis Baseline (`ruff`)
Command executed:
```powershell
.\.venv\Scripts\ruff.exe check leredd api ui scripts tests --select E9,F63,F7,F82,F811,PIE794
```
- **Total style/lint findings (all rules):** 803
- **Critical semantic/runtime-risk errors:** 10
  1. `PIE794` (`leredd/review_queue.py:127–128`): `ReviewItem.synthetic` and `ReviewItem.source_identity` are defined twice on the same Pydantic model due to a prior branch merge.
  2. `F821` (`leredd/synthetic.py:182`): Undefined type annotation `ReviewItem` in `load_synthetic_review_items` (masked only by `from __future__ import annotations`).
  3. `F821` (`scripts/run_full_evaluation.py:384, 415`): Undefined type annotation `Any` on `run_heuristic_model` and `run_leredd_variant`.
  4. `F811` (`scripts/run_full_evaluation.py:101, 105, 168, 172`): `GroundTruthViolationError`, `check_ground_truth_eligibility`, `ContaminationError`, and `build_safe_retrieval_pool` are imported from `leredd.evaluation` and then shadowed by duplicate local definitions, splitting exception class identity across modules.
  5. `F811` (`tests/test_safe_evaluation.py:888`): Duplicate `Evaluator` import.

---

## 2. Core Module Inspection & Proposal-vs-Implementation Mismatches

| Module | Current Behavior | Proposal / Specification Mismatch |
| :--- | :--- | :--- |
| [`leredd/detector.py`](../leredd/detector.py) | Preserves predicted `DependencyType` and sets `PredictionValidationStatus` (`requires_review` when `confidence <= threshold`). Supports 1–5 `stability_runs`. | **1. Silent malformed label fallback:** Lines 40–47 scan raw text for any `DependencyType` substring when `Dependency_Type:` is absent, turning conversational refusals (e.g., *"this requires human review"*) into valid `Requires` labels.<br>**2. Missing confidence defaults to maximum (`5`):** Lines 67–69 assign `confidence = 5` if `Confidence Score:` is omitted, allowing incomplete outputs to pass threshold screening.<br>**3. Compatibility policy undocumented in config:** No explicit deprecation/compatibility flag or validator preventing Likert scores from being misrepresented as calibrated probabilities. |
| [`leredd/hybrid_retrieval.py`](../leredd/hybrid_retrieval.py) | Fuses SBERT (`ExampleRetriever`), lexical (`BM25Retriever`), and exact-overlap metadata (`component`, `subsystem`, `system`) via RRF (`rrf_k=60`). | **1. Unverified & synthetic leakage via metadata ranker:** `_metadata_ranking` receives raw `annotated_pool` without filtering out `PROVISIONAL`, `REJECTED`, or `synthetic=True` pairs.<br>**2. Reversed self-pair leakage:** `filter_self` checks exact `(A, B)` text match only; reversed equivalent pairs `(B, A)` or swapped requirement IDs are not excluded.<br>**3. Dead config & non-configurable weights:** `rrf_k: int = 60` shadows `self.config.rrf_k`; `dense_weight`, `bm25_weight`, and `metadata_weight` are missing from `LEREDDConfig`; `Requirement` lacks `module` and structured `metadata` dict fields. |
| [`leredd/retrieval.py`](../leredd/retrieval.py) | Implements Eq. 1 (`average`) & Eq. 2 (`max`) pair similarity in `ExampleRetriever`, `RequirementRetriever`, and fixed-size `RAGRetriever`. | **1. Baseline RAG lacks tie determinism & diagnostics:** `RAGRetriever.retrieve` sorts by score without deterministic index tie-breaking and provides no measurable diagnostics for duplicate chunks, cited requirement IDs, or low-relevance chunks.<br>**2. Missing configurable context enhancements:** No requirement-boundary-aware chunking, duplicate/near-duplicate filtering, or cited-ID chunk boosting.<br>**3. Reversed equivalent pair exclusion:** `ExampleRetriever` and `BM25Retriever` (`bm25_retrieval.py` also lacks `not is_synthetic(item)`) only exclude forward `(A, B)` text matches. |
| [`leredd/prompts.py`](../leredd/prompts.py) | Implements Paper Table I six-category taxonomy definitions and Figure 2 prompt structure (`#Requirements`, `#Dependency Definitions`, `#Examples`, `#Context`, `#Instructions`). | Matches Paper Table I and Figure 2 cleanly, but lacks a deterministic `template_hash()` helper to bind evaluation reports to an exact prompt version (Part 8 reproducibility requirement). |
| [`leredd/provisional_labeler.py`](../leredd/provisional_labeler.py) | Coordinates `HeuristicLabeler`, `LLMLabeler`, and `MultiModelLLMLabeler` under `VerificationStatus.PROVISIONAL`. | **1. Unstructured string encoding:** Encodes disagreements, failures, and rationales into free-text `review_notes` and `evidence` strings rather than structured provenance fields.<br>**2. Swallowed LLM failures:** Malformed LLM outputs return `(None, None, "Malformed...")`, which `HYBRID` mode collapses into `"UNRESOLVED: Both heuristic and LLM abstained or failed"` (`source=HEURISTIC`), masking `LLM_FAILURE`.<br>**3. Dropped synthetic markers:** `process_pair` constructs a new `RequirementPair` without propagating `pair.synthetic` and `pair.source_identity`. |
| [`leredd/review_queue.py`](../leredd/review_queue.py) | Builds deterministic `ReviewItem` queues, supports atomic JSON persistence, idempotent `refresh_queue`, and human `record_decision`. | **1. Brittle phrase parsing:** `determine_eligibility` and `create_review_item` parse substrings (`"DISAGREEMENT"`, `"UNRESOLVED"`, `"FAILED"`, `split("'")`) out of `review_notes`.<br>**2. Broken routing for 3 synthetic fixture types:** `EXPLICIT_FLAG` is never appended in `determine_eligibility`; `MISSING_METADATA` ignores empty `system`/ID metadata; `LLM_FAILURE` from `parsing_failure_pairs.json` is misrouted as `LLM_ONLY`; `BOTH_ABSTAINED` also spuriously triggers `HEURISTIC_ONLY`.<br>**3. Duplicate Pydantic fields (`PIE794`):** `synthetic` and `source_identity` declared twice on `ReviewItem`. |
| [`leredd/evaluation.py`](../leredd/evaluation.py) | Enforces `validate_ground_truth`, `ground_truth_provenance_problem`, `validate_predictions`, `validate_system_disjoint_splits`, and `build_safe_retrieval_pool`. | **1. Missing synthetic filter in `leredd.evaluation.build_safe_retrieval_pool`:** Only the duplicate function in `scripts/run_full_evaluation.py` excluded synthetic training examples; the library function did not.<br>**2. Missing `HUMAN_REVIEWED` explicit regression & document-disjoint validator:** Needs `validate_document_disjoint_splits`, dev-only threshold tuning (`tune_confidence_threshold_on_dev`), and explicit separation helper for candidate recall vs. classifier accuracy. |
| [`leredd/dataset_splitter.py`](../leredd/dataset_splitter.py) | Supports `SYSTEM`, `DOCUMENT`, `REQUIREMENT`, and `PAIR` strategies with `LeakageReport`. | `DOCUMENT` split groups cross-document pairs into composite `"docA_docB"` buckets rather than enforcing or auditing document disjointness across `train`/`dev`/`test`; needs document-overlap auditing on `LeakageReport` and `validate_document_disjoint_splits`. |

---

## 3. Prioritized Implementation Backlog

1. **P0 — Data Integrity, Isolation & Evaluation Guardrails (Parts 2 & 3)**
   - Exclude synthetic and unverified records across `BM25Retriever`, `HybridRetriever._metadata_ranking`, `leredd.evaluation.build_safe_retrieval_pool`, `BERTBaseline.fit`, and `TFIDF_LSABaseline.fit`.
   - Unify duplicate guardrail functions/exceptions between `leredd/evaluation.py` and `scripts/run_full_evaluation.py`.
   - Preserve structured fixture metadata in `load_synthetic_pairs` and propagate `synthetic`/`source_identity` through `ProvisionalLabeler`.
   - Add `validate_document_disjoint_splits` and explicit tests for `HUMAN_REVIEWED`, `PROVISIONAL`, `PENDING_REVIEW`, `REJECTED`, unlabeled, and synthetic pairs.

2. **P0 — Confidence-Aware Validation & Malformed-Response Safety (Part 4)**
   - Remove unsafe substring label guessing in `ResponseParser.parse`; require explicit `Dependency_Type:` header or structured JSON `dependency_type`.
   - Route missing/unparseable `Confidence Score` to `confidence = 0` and `PredictionValidationStatus.REQUIRES_REVIEW` instead of silently defaulting to `5`.
   - Prevent Likert confidence from being stored or interpreted as `calibrated_probability`; document deprecation of legacy `<= threshold -> No_dependency` rewriting and add boundary tests (`<`, `==`, `>`, disabled `-1`).

3. **P1 — Structured Disagreement & Review Queue Routing (Part 5)**
   - Add structured fields (`heuristic_label`, `heuristic_rule_id`, `heuristic_evidence`, `llm_label`, `llm_rationale`, `llm_failed`, `llm_error`, `explicit_review_flag`, `structured_review_reasons`) to `AnnotationProvenance`.
   - Refactor `ProvisionalLabeler` and `ReviewQueueManager` to populate and route via structured fields first (keeping legacy string fallback for older JSON artifacts).
   - Remove duplicate `ReviewItem` fields (`PIE794`) and verify all 7 synthetic review fixture categories produce exact expected `ReviewReasonCode` values while preserving full machine + human audit trails.

4. **P1 — Metadata-Aware Hybrid Retrieval & Measurable RAG Context Selection (Parts 6 & 7)**
   - Add `module` and `metadata` fields to `Requirement` (preserving round-trip through `ReviewItem`) and configurable `dense_weight`, `bm25_weight`, `metadata_weight` to `LEREDDConfig`.
   - Enforce symmetric/equivalent self-pair exclusion (`(A, B)` and `(B, A)`) and verified-only example filtering in `ExampleRetriever`, `BM25Retriever`, and `HybridRetriever`.
   - Test baseline `RAGRetriever` (empty docs, overlap math, duplicates, cited IDs, query pairing/boundaries, tie determinism), then add `RAGRetrievalDiagnostics` (`retrieve_with_diagnostics`) plus opt-in requirement-boundary chunking, duplicate/near-duplicate chunk filtering, and cited-ID chunk boosting.

5. **P2 — Controlled Ablation Runner, Documentation & CI Static Checks (Parts 8 & 9)**
   - Upgrade `scripts/run_full_evaluation.py` to compare all 6 required configurations (`ZeroShot`, `Dense`, `BM25`, `Hybrid` without metadata, `Hybrid_Metadata`, `Context_RAG`), wire `seed` into baselines, record prompt/config hashes, perform development-set-only threshold tuning, validate system/document disjointness, and report candidate recall separately from classifier accuracy.
   - Clean up `ruff` lint errors across the codebase, add `ruff check` to `.github/workflows/ci.yml`, update `.env.example`, and document provenance/verification/synthetic boundaries and retrieval configuration in `README.md` and `docs/`.
