# Provenance, Data Tiers, and Retrieval Configuration Guide

This document defines the strict data-tier boundaries (**Verified**, **Provisional**, and **Synthetic**), confidence-screening rules, and retrieval/context configuration parameters enforced across LEREDD.

---

## 1. Data Tiers & Ground-Truth Guardrails

LEREDD enforces fail-closed separation across three distinct data categories:

| Data Tier | Identifying Fields | Allowed Uses | Blocked From |
| :--- | :--- | :--- | :--- |
| **Verified Ground Truth** | `synthetic=False`, non-null `dependency_type`, `provenance.status="verified"`, `provenance.source="human"`, non-placeholder `reviewer_id`/`annotator_id`, and `accepted_label == dependency_type`. *(Legacy no-provenance records are accepted only when `allow_legacy=True`, with explicit report warnings).* | Benchmark evaluation (`Evaluator.evaluate`), supervised baseline fitting (`BERTBaseline`, `TFIDF_LSABaseline`), dynamic ICL retrieval pools, and immutable dataset releases (`dataset-release`). | Overwriting by `ProvisionalLabeler.process_pair`. |
| **Provisional / Pending / Rejected** | `provenance.status` in `{"provisional", "pending_review", "human_reviewed", "rejected"}` or `dependency_type=None`. | Human review queue (`ReviewQueueManager`), blind agreement studies, and assisted effort studies. | Benchmark evaluation (`GroundTruthViolationError`), ICL example retrieval (`is_eligible_retrieval_example`), baseline training, and `dataset-release`. |
| **Synthetic / Demo Fixtures** | `synthetic=True` or `source_identity="synthetic-review-tests"` (or `synthetic:*` prefix) at pair, requirement, or provenance level. | Unit/integration tests (`tests/fixtures/synthetic/`), CLI `--synthetic` walkthroughs, and UI workflow demos. | **All research & verification outputs:** benchmark evaluation, ICL retrieval (`ExampleRetriever`, `BM25Retriever`, `HybridRetriever`), baseline training, `ReviewQueueManager.export_verified_pairs()`, and `dataset-release`. |

> **Synthetic-Data Warning:** Every fixture under `tests/fixtures/synthetic/` uses deterministic mock outputs and fixed IDs (`SYNTH-*`). Simulated human verification (`human_decision` or `status="verified"`) inside synthetic fixtures raises a `ValueError` at load time (`load_synthetic_fixture`). Never strip `synthetic: true` to pass synthetic records into evaluation metrics.

---

## 2. Confidence-Aware Validation Policy

1. **No Silent Relabeling:** `ResponseParser.parse()` preserves the model's parsed `DependencyType` across all Likert scores (`0`–`5`). Low confidence sets `Prediction.validation_status = "requires_review"` (`<= CONFIDENCE_THRESHOLD`) rather than rewriting the prediction to `No_dependency`.
2. **Legacy Compatibility Flag (Deprecated):** `LEGACY_REWRITE_LOW_CONFIDENCE_TO_NO_DEPENDENCY=false` (default) keeps non-destructive review routing active. Enabling it emits a `DeprecationWarning` and is retained solely for historical reproduction.
3. **Not Calibrated Probability:** Self-reported Likert confidence (`0`–`5`) is an uncalibrated ordinal screening cue. Supplying `calibrated_probability` to `Prediction` raises a `ValueError`.
4. **Malformed Output Fail-Closed:** Responses missing an explicit `Dependency_Type:` header (or valid JSON `dependency_type` key) raise `ValueError`—free-text refusals containing words like *"requires"* or *"conflicts"* are never converted into valid labels. Missing or non-numeric `Confidence Score:` fields default to `confidence=0` and `requires_review`.
5. **Development-Set-Only Tuning:** `Evaluator.tune_confidence_threshold_on_dev()` tunes review cutoffs strictly on `DEV` partitions and raises `EvaluationGuardrailError` if invoked on `TEST`.

---

## 3. Retrieval & Context Selection Configuration

### 3.1 Dynamic ICL Example Retrieval (`leredd/retrieval.py`, `leredd/bm25_retrieval.py`, `leredd/hybrid_retrieval.py`)
- **`RETRIEVAL_MODE=dense`**: SBERT (`all-MiniLM-L6-v2`) pair similarity using Paper Eq. 1 (`AGGREGATION=average`) or Eq. 2 (`AGGREGATION=max`).
- **`RETRIEVAL_MODE=bm25`**: Deterministic `BM25Plus` lexical pair retrieval (offline, no embedding model required).
- **`RETRIEVAL_MODE=hybrid_no_metadata`**: Reciprocal Rank Fusion (`RRF_K=60`) over SBERT and BM25 only (`METADATA_WEIGHT=0.0`).
- **`RETRIEVAL_MODE=hybrid` / `hybrid_metadata`**: RRF fusing SBERT (`DENSE_WEIGHT`), BM25 (`BM25_WEIGHT`), and normalized exact overlap on `Requirement.component`, `subsystem`, `module`, non-generic `system`, and scalar `metadata` entries (`METADATA_WEIGHT`).
- **Self-Pair & Verification Guards:** All retrievers exclude exact and reversed equivalent target pairs (`(A, B)` and `(B, A)` by ID or text) and retrieve verified, non-synthetic examples only.

### 3.2 Measurable SRS Context Selection (`RAGRetriever` in `leredd/retrieval.py`)
- **Optional by Default:** Controlled via `USE_RAG=false` globally or `use_rag=True/False` per CLI/API/detector call.
- **Chunking Modes (`RAG_CHUNKING_MODE`):**
  - `fixed` (default): Fixed character windows (`RAG_CHUNK_SIZE=1000`, `RAG_CHUNK_OVERLAP=200`).
  - `requirement_boundary`: Splits along `[REQ-...]` identifiers and paragraph boundaries so individual requirement statements remain intact up to `RAG_CHUNK_SIZE`.
- **Measurable Diagnostics (`retrieve_with_diagnostics`):** Reports `total_chunks`, `exact_duplicate_count`, `near_duplicate_count`, `duplicate_ratio`, `cited_ids`, `cited_id_chunks_available`, `cited_id_chunks_selected`, and `irrelevant_selected_count` (`< RAG_MIN_SIMILARITY`).
- **Optional Enhancements:** Set `RAG_DEDUPLICATE_CHUNKS=true` (filters exact and Jaccard `>= RAG_NEAR_DUPLICATE_THRESHOLD` near-duplicates) and `RAG_CITE_ID_BOOST=0.15` (boosts chunks defining requirement IDs explicitly cited in the target pair).
