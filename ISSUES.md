# ISSUES.md — Unresolved Bugs, Blockers, and Risks Log

This log tracks bugs, technical debts, risks, and blockers discovered during project execution.

---

## Active Issues & Blockers

### [ISSUE-009] CRITICAL: MockLLMClient Ignores Retrieved Examples — Ablation Invariance Under Mock Backend
- **Severity:** Critical / Architectural Caveat
- **Status:** 🟡 Open — Prompt Forwarding Verified, Mock Backend Limitation Documented (updated 2026-10-11)
- **Files:** `leredd/llm.py` (`MockLLMClient.generate()`, lines 179–210), `leredd/detector.py`, `tests/test_detector.py`
- **Verified System Behavior:**
  - `LEREDDDetector` correctly retrieves dynamic in-context examples via dense/BM25/hybrid strategies and passes them into `PromptBuilder.build()`.
  - The assembled prompt contains the full `#Examples:` section with the retrieved requirement pairs and dependency type headers.
  - Regression test `test_detector_passes_in_context_examples_to_llm_prompt` in `tests/test_detector.py` verifies using a controlled test double that:
    - With `k_examples=0`, the prompt contains no retrieved-example block (`"No in-context examples provided."`).
    - With `k_examples=1`, the prompt contains the retrieved example text and appropriate dependency type section (`"[Conflicts Examples]"`).
    - Verified with 1/1 passing in focused execution and 3/3 passing in `tests/test_detector.py`.
- **Unresolved Mock Backend Limitation:**
  - `MockLLMClient.generate()` receives the full assembled prompt but discards the `#Examples:` block. It only extracts `Requirement A:` and `Requirement B:` text lines via regex and applies static keyword heuristics.
  - Retrieved examples from Dense, BM25, or Hybrid retrieval are never parsed or incorporated by `MockLLMClient`.
- **Evaluation Caveat & Recommendation:**
  - The ablation study dimensions `retrieval_strategy` and `in_context_examples` in the offline evaluation report show no signal and are not meaningful when evaluated using `MockLLMClient`.
  - Evaluating the actual empirical effect of dynamic in-context examples requires running a real LLM (`OpenAILLMClient` / `OllamaLLMClient`, which forward the full prompt without modification) or conducting a separately designed evaluation using an appropriate controlled test double.

### [ISSUE-010] CRITICAL: MockLLMClient Always Outputs Confidence=5 — Threshold Ablation Invariant Under Mock Backend
- **Severity:** Critical / Architectural Caveat
- **Status:** 🟡 Open — Threshold Mechanism Verified, Mock Backend Invariance Documented (updated 2026-10-11)
- **Files:** `leredd/llm.py` (lines 189, 193, 197, 201, 205, 209), `leredd/detector.py`, `tests/test_detector.py`, `tests/test_ablation_benchmark.py`
- **Verified System Behavior:**
  - `ResponseParser.parse()` threshold re-annotation logic works correctly: predictions with `confidence <= confidence_threshold` (and `confidence_threshold >= 0`) are accurately reclassified to `DependencyType.NO_DEPENDENCY`.
  - The threshold mechanism is empirically tested and passing in existing unit and ablation tests:
    - `test_response_parser_threshold_reannotation` in `tests/test_detector.py:7–20` verifies that raw predictions with confidence 3 are re-annotated under threshold 4 and retained under threshold 2.
    - `test_threshold_behavior_reannotation_vs_disabled` in `tests/test_ablation_benchmark.py:91–112` uses a controlled test double (`LowConfMockLLM` returning confidence 3) to prove end-to-end re-annotation under threshold 4 versus passthrough under disabled threshold -1.
- **Unresolved Mock Backend Limitation:**
  - `MockLLMClient.generate()` hardcodes `confidence = 5` across all six heuristic classification branches.
  - In `scripts/run_full_evaluation.py`, the ablation compares threshold enabled (`confidence_threshold = 4`) against threshold disabled (`confidence_threshold = -1`).
  - Because `MockLLMClient` always outputs confidence 5, the re-annotation condition `5 <= 4` evaluates to `False` for every single test pair. Consequently, the re-annotation condition never triggers, leaving the predictions and metrics for `Threshold_On` and `Threshold_Off` 100% bit-for-bit identical.
- **Evaluation Caveat & Recommendation:**
  - Offline evaluation reports declare the backend as `MockLLMClient`, but do not explicitly state that its fixed confidence score of 5 causes threshold-ablation invariance.
  - The reported threshold-ablation results under `MockLLMClient` show no empirical signal and are not informative regarding the efficacy of real-LLM confidence thresholding.
  - Studying the empirical effect of confidence thresholding requires evaluating a real LLM (`OpenAILLMClient` / `OllamaLLMClient`, where confidence scores vary naturally between 0 and 5) or designing a separate controlled evaluation using an appropriate test double. `MockLLMClient` should not be modified merely to produce arbitrary scores, as doing so would break deterministic test baselines.

### [ISSUE-011] CRITICAL: LEREDD_MockLLM Entry in Report is a Copy-Paste Duplicate of LEREDD_Hybrid
- **Severity:** Critical
- **Status:** ✅ Resolved (2026-10-11)
- **File:** `scripts/run_full_evaluation.py`, lines 540–561
- **Description:** The `LEREDD_MockLLM` model entry was previously serialized using `leredd_hybrid_report` and `leredd_hybrid_preds` (the same Python objects returned by the Hybrid evaluation call) instead of an independent model run, producing a phantom duplicate.
- **Resolution:** Removed the redundant `LEREDD_MockLLM` entry from `scripts/run_full_evaluation.py`. The proposed system is represented cleanly by `LEREDD_Hybrid`, whose configuration and report notes accurately identify the MockLLM backend. Updated `tests/test_safe_evaluation.py` to assert that `LEREDD_MockLLM` is absent and `LEREDD_Hybrid` correctly retains backend identification. All 27 tests in `test_safe_evaluation.py` pass cleanly.

### [ISSUE-012] HIGH: Global Config Singleton Breaks Test Isolation
- **Severity:** High / Architectural Debt
- **Status:** 🟡 Partially Addressed — Explicit Reset Available; Automatic Isolation Pending (updated 2026-10-11)
- **Files:** `leredd/config.py` (lines 112–134), `tests/test_config.py`
- **Description:** `get_config()` uses a module-level `_config_instance` singleton that caches configuration in memory. Once initialized, subsequent calls return the cached instance, which previously prevented environment variable overrides or controlled reinitialization during tests.
- **Implemented Remediation:**
  - Added `reset_config()` in `leredd/config.py` to allow callers and test harnesses to explicitly clear `_config_instance = None`.
  - Added dedicated regression test suite in `tests/test_config.py` (4/4 passed cleanly):
    1. `test_get_config_returns_same_instance_when_not_reset`: Verifies identity caching (`cfg1 is cfg2`) when not reset.
    2. `test_reset_config_creates_new_instance`: Verifies `reset_config()` forces a new instance (`cfg1 is not cfg2`).
    3. `test_env_variable_reflected_after_reset`: Verifies environment variable changes (`LLM_PROVIDER`) take effect upon reinitialization following `reset_config()`.
    4. `test_reset_does_not_mutate_existing_reference`: Verifies resetting and reloading does not mutate an already-held reference to a prior configuration instance.
- **Remaining Limitation & Next Steps:**
  - Automatic isolation between tests (e.g., an `autouse=True` fixture in `conftest.py`) has not been implemented or evaluated across the broader suite. Tests that do not explicitly call `reset_config()` may still share the singleton instance across test boundaries.

### [ISSUE-013] MEDIUM: ResponseParser Rationale Regex Truncates Multi-Line LLM Output
- **Severity:** Medium
- **Status:** ✅ Resolved (2026-10-12)
- **File:** `leredd/detector.py`, lines 45–54
- **Description:** The previous rationale regex `r"Rationale:\s*\[?(.*?)\]?\*?$"` used `re.MULTILINE` without `re.DOTALL`, which caused text after the first newline to be silently dropped. Any multi-sentence or multiline LLM rationale was truncated.
- **Resolution:** Updated `ResponseParser.parse()` rationale extraction to use a non-greedy regex with `re.DOTALL` and a lookahead delimiter stopping at the next recognized field header (`Confidence Score` or `Dependency_Type`) or end of string (`\Z`). The parser strips markdown artifacts while preserving multiline formatting and handles header variants like `**Rationale:** text`. Added three regression tests in `tests/test_detector.py` (`test_response_parser_multiline_rationale_preserved`, `test_response_parser_rationale_after_confidence`, `test_response_parser_rationale_colon_inside_bold_variant`) alongside the existing `test_response_parser_threshold_reannotation`. All 4 parser tests (and all 6 tests in `tests/test_detector.py`) pass cleanly.

### [ISSUE-014] LOW: OllamaLLMClient Missing Retry Logic
- **Severity:** Low
- **Status:** ✅ Resolved (discovered 2026-10-11 code audit)
- **File:** `leredd/llm.py` (`_is_retryable_ollama_exception` ~line 119; decorated `_call_ollama()` ~line 140; `generate()` ~line 152)
- **Description:** `OllamaLLMClient.generate()` now retries transient network errors and HTTP 500/502/503/504 responses through a private `_call_ollama()` helper. Retry policy: 3 total attempts; bounded exponential backoff with a 2–4 second delay range; no retries for permanent HTTP errors (e.g., 400, 404) or malformed JSON responses.

### [ISSUE-015] LOW: DependencyType.normalize() Substring Ambiguity
- **Severity:** Low
- **Status:** 🔴 Open (discovered 2026-10-11 code audit)
- **File:** `leredd/data_models.py`, line 57
- **Description:** The condition `"No" in clean` uses Python substring matching (not word boundary). Any string containing "No" as a substring (e.g., "Notable", "Notification") would incorrectly normalize to `NO_DEPENDENCY`. Latent risk for unexpected LLM outputs.

---

### [ISSUE-001] Missing Local Dependencies in Python Environment
- **Severity:** High (Blocker for local test execution)
- **Status:** ✅ Resolved (2026-10-09)
- **Description:** System Python environment (`Python 3.13.3`) did not have the declared dependencies installed.
- **Resolution:** `.venv` virtual environment with all deps installed; `leredd` in editable mode; `pytest --collect-only` → 28 items, exit 0; `pytest tests/ -v` → 28 passed in 45.74s.

### [ISSUE-002] Potential Data Leakage in Intra-System Dataset Evaluation
- **Severity:** Medium
- **Status:** ✅ Mitigated (2026-10-10)
- **Description:** Existing dataset pairs (`adb_pairs.json`, `apa_pairs.json`, `tja_pairs.json`) use static JSON files without explicit train/validation/test split definitions, risking potential data leakage during baseline training and evaluation.
- **Resolution:** Implemented `DatasetSplitter` (`TASK-P2-02`) in `leredd/dataset_splitter.py` providing `REQUIREMENT` and `SYSTEM` disjoint splitting strategies. Verified 0 requirement leakage on generated split partitions in `data/splits/`.

### [ISSUE-003] Absence of BM25 Lexical Scorer
- **Severity:** Medium
- **Status:** ✅ Mitigated (2026-10-10)
- **Description:** Current retrieval in `leredd/retrieval.py` relied exclusively on SBERT dense embeddings (`all-MiniLM-L6-v2`). Lexical matching (BM25) was missing from example selection.
- **Resolution:** Implemented `BM25Retriever` (`TASK-P3-01`) in `leredd/bm25_retrieval.py` using `BM25Plus`. Provides standalone lexical retrieval and example pair retrieval. Full hybrid score fusion implemented in `TASK-P3-02`.

### [ISSUE-004] Directional Semantics and Retrieval Candidate Recall Trade-offs
- **Severity:** Low / Technical Debt
- **Status:** 🟡 Partially Resolved / Tracked for Phase 7
- **Description:** Canonical unordered pair deduplication (`(A,B)` vs `(B,A)`) prevents duplicate pairs during candidate generation, but certain asymmetric dependency types (e.g., `Requires`, `Implements`) have directional semantics. Furthermore, retrieval-based candidate generation (`BM25`/`DENSE`/`HYBRID`) reduces candidate count from $O(n^2)$ to $O(kn)$, but may omit true dependencies that lack keyword overlap or high embedding similarity.
- **Progress in TASK-P4-02:** Preserved requirement order and orientation in candidate evaluation. Heuristic rules explicitly check direct citation (`RULE-REF-ID-DIRECT`) vs reverse citation (`RULE-REF-ID-REVERSE`), ensuring swapping A and B yields directional accuracy while candidate set deduplication remains separate.
- **Target Resolution:** Phase 7 ablation experiments comparing exhaustive vs retrieval candidate pools.

### [ISSUE-005] Single-User Local Streamlit Concurrency and Optimistic File Locking
- **Severity:** Low / Operational Limitation
- **Status:** 🟡 Documented & Safeguarded (2026-10-10)
- **Description:** The Streamlit human review workspace (`ui/app.py`) and CLI commands operate on single local JSON manifest files (`ReviewQueueManifest`). Concurrent reviews by multiple annotators on the same JSON file could cause read-write race conditions or overwrite concurrent reviewer decisions.
- **Mitigation in TASK-P5-02:** Implemented explicit file write safety and state reload in `ReviewQueueManager`. In-memory decision objects are immediately updated and serialized. Documented multi-reviewer workflow recommendation: maintain separate system or document review manifests per reviewer prior to merging via `ReviewQueueManager.refresh_queue`.
- **Target Resolution:** Optional database backend or file-locking sidecar if multi-user distributed review is required in future releases.

### [BLOCKER-001] Retrieval Pool Not Isolated from Test Set — Contamination Risk Unguarded
- **Severity:** High (Blocks TASK-P7-01)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-02)
- **Description:** `LEREDDDetector` accepts a runtime `annotated_pool` argument for in-context example retrieval. If the full annotated dataset is passed as the retrieval pool when evaluating on test pairs from the same system, labeled test examples contaminate the model prompt.
- **Resolution:** Implemented `build_safe_retrieval_pool` in `leredd/evaluation.py` and `scripts/run_full_evaluation.py`. Enforces strict retrieval isolation: test/dev requirement IDs and test-pair identities are stripped from the candidate retrieval pool. Raises `ContaminationError` in strict mode if leakage occurs.

### [BLOCKER-002] Requirement-Disjoint Test Partition Is Empty
- **Severity:** High (Blocks requirement-disjoint evaluation)
- **Status:** 🔴 Active (identified 2026-10-10, TASK-P6-01 audit)
- **Description:** With only 20 total annotated pairs, the requirement-disjoint splitter drops 11 cross-partition pairs, leaving dev=0 and test=0. This split strategy requires substantially more annotated data.
- **Required Resolution:** Annotate additional requirement pairs from `data/raw/` SRS documents before using requirement-disjoint evaluation.

### [BLOCKER-003] Only 20 Original Annotated Pairs — Statistical Power Near-Zero
- **Severity:** Medium (Impacts evaluation credibility)
- **Status:** 🟡 Known limitation (2026-10-10, TASK-P6-01 audit & P6-02 evaluation)
- **Description:** 20 total pairs, no provenance metadata, annotator identity unknown. System-disjoint test set n=5 — 95% CI for accuracy is approximately ±40pp. Results are demonstrative only.
- **Required Resolution:** Annotate 50-100+ requirement pairs per system for publication-quality evaluation.

### [BLOCKER-004] CLI `evaluate` Lacks VerificationStatus Filtering
- **Severity:** Medium (Blocks safe automated evaluation)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-02)
- **Description:** `leredd-cli evaluate --ground-truth gt.json` accepted any JSON array as ground truth including PROVISIONAL records with no warning.
- **Resolution:** Implemented `check_ground_truth_eligibility` in `leredd/evaluation.py` and integrated into `scripts/run_full_evaluation.py`. Rejects `data/provisional/*` and review-queue records when `VerificationStatus.PROVISIONAL`, raising `ValueError` in strict mode.

### [BLOCKER-005] `adb_verified_ui_test.json` Contains an Explicit Test Fixture
- **Severity:** Low (Risk for automated dataset ingestion)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-02)
- **Description:** `data/verified/adb_verified_ui_test.json` (2 records) — 1 record has `review_notes` explicitly stating "UI persistence test... not an independently verified dependency annotation." Both carried `VerificationStatus.VERIFIED` or `PROVISIONAL`.
- **Resolution:** `check_ground_truth_eligibility` filters out provisional records and raises strict errors if provisional review queue / UI test exports are supplied as ground truth without human verification.

### [ISSUE-006] Evaluation CLI Argument Ambiguity & Supervised Cross-System Terminology Alignment
- **Severity:** Medium (Documentation & Metadata Integrity)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-03)
- **Description:** The `--model-type` parameter in `scripts/run_full_evaluation.py` selected the classification backend head for `BERTBaseline` (`rf` vs `lr`), but could be misconstrued as selecting which baseline model to run. Additionally, reports previously mislabeled supervised `BERTBaseline` fitting on ADB and testing on TJA as "zero-shot".
- **Resolution:** Added `--bert-classifier-type` parameter alias (with `--model-type` fallback) and updated help strings. Updated report configuration metadata (`bert_classifier_backend: "rf"`, `bert_classifier_description`). Re-labeled `BERTBaseline` and `TFIDF_LSABaseline` notes as **Supervised Cross-System Evaluation** (trained/fitted on ADB, tested on TJA) and added exploratory notices for McNemar's test on small samples ($n=5$). Added 4 regression tests (24 total in `test_safe_evaluation.py`).
