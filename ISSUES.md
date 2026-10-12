# ISSUES.md — Unresolved Bugs, Blockers, and Risks Log

This log tracks bugs, technical debts, risks, and blockers discovered during project execution.

---

## Active Issues & Blockers

### [ISSUE-018] MEDIUM: Generated Pair Labels Must Not Masquerade as Verified Ground Truth
- **Severity:** Medium / Data Semantics
- **Status:** Resolved (TASK-P9-13, 2026-10-12)
- **Files:** `leredd/utils.py`, `leredd/candidate_generator.py`, `leredd/data_models.py`, `leredd/cli.py`, `leredd/evaluation.py`, `scripts/run_full_evaluation.py`, downstream loaders/retrievers and regression tests
- **Description:** Generated candidates previously could receive `No_dependency` before annotation, making an unknown relation indistinguishable from an explicit negative label. Label handling also accepted ambiguous unknown text, and some consumers defaulted missing labels to `No_dependency`.
- **Resolution:** Both generation paths now preserve unknown labels as `None` and provisional provenance. Unknown explicit labels fail validation. Evaluation requires complete prediction coverage and rejects provenance-bearing ground truth unless status is exactly `VERIFIED`; retrieval and supervised fitting ignore unlabeled/unverified examples. The legacy pairs command's interactive annotation stores an explicit human VERIFIED decision. Regression coverage and the full suite pass (**236 passed**, 40 warnings).

### [ISSUE-019] HIGH: API Configuration Endpoint Could Return the OpenAI Secret
- **Severity:** High / Credential Exposure
- **Status:** Resolved (TASK-P9-14, verified by full suite 261 passed).
- **File:** `api/routes.py`
- **Description:** `GET /api/v1/config` serialized the complete settings model, including `openai_api_key`.
- **Resolution:** The route excludes the secret and reports only `openai_api_key_configured: bool`. `tests/test_api.py::test_api_config_never_returns_openai_secret` asserts the secret does not appear in the response.

### [ISSUE-020] HIGH: UI Evaluation Leaked Test Labels and Agreement Was Fabricated
- **Severity:** High / Evaluation Integrity
- **Status:** Resolved (TASK-P9-14, verified by full suite 261 passed).
- **File:** `ui/app.py`
- **Description:** The former evaluation page used the selected target dataset as its retrieval pool and let both baselines fit on that same target batch. Its kappa control compared a label vector with itself and displayed artificial perfect agreement. UI annotations were also only mutated in memory.
- **Resolution:** Page 4 now calls the system-disjoint runner with strict eligibility and contamination checks, writes each report into a unique run directory, and displays limitations. Page 2 atomically saves a separate verified output with reviewer identity and explicit replacement confirmation; Cohen's kappa is stated as unavailable without independent labels. AppTest covers the UI contract and safe output.

### [ISSUE-021] HIGH: Demonstration Script Invented a Human Review Decision
- **Severity:** High / Annotation Provenance
- **Status:** Resolved (TASK-P9-14, verified by full suite 261 passed).
- **File:** `scripts/build_review_queue.py`
- **Description:** The former script automatically recorded a made-up `lead_annotator_42` decision and promoted a provisional prediction to VERIFIED.
- **Resolution:** The script now creates only pending queue records, supports isolated output paths, and never records a decision. A synthetic script test verifies pending status and absent `human_decision`.

### [ISSUE-022] MEDIUM: Unfitted Baselines Could Fit on the Prediction Batch
- **Severity:** Medium / Data Leakage
- **Status:** Resolved (TASK-P9-14, verified by full suite 261 passed).
- **Files:** `leredd/baselines/bert_classifier.py`, `leredd/baselines/tfidf_lsa.py`
- **Description:** `predict_batch()` could fit an untrained baseline on the same pairs it was asked to predict; BERT also returned a hidden heuristic prediction when unfitted.
- **Resolution:** Prediction now requires an explicit prior `fit(training_pairs)` call and raises a clear `ValueError` otherwise. Regression tests cover single and batch prediction. The safe evaluator explicitly fits both baselines on train pairs.

### [ISSUE-023] LOW: Document Extraction Silently Returned No Requirements Without Optional Libraries
- **Severity:** Low / Optional Dependency Clarity
- **Status:** Resolved (TASK-P9-14, verified by full suite 261 passed).
- **Files:** `leredd/utils.py`, `pyproject.toml`
- **Description:** PDF/DOCX extraction swallowed import and parse errors and returned an empty list, while those libraries were not declared dependencies.
- **Resolution:** Added the `documents` optional extra (`pypdf`, `python-docx`), explicit missing-library errors, and surfaced parser errors. No packages were installed for this task. Tests simulate absent optional dependencies.

### [ISSUE-024] MEDIUM: `train-bert` Reported Saving a Model Without Writing an Artifact
- **Severity:** Medium / CLI Correctness
- **Status:** Resolved (TASK-P9-15, verified by 264-test full suite).
- **Files:** `leredd/baselines/bert_classifier.py`, `leredd/cli.py`, `pyproject.toml`, `requirements.txt`, `tests/test_baselines.py`, `tests/test_cli.py`
- **Description:** The command created the output directory and printed a successful save message but persisted no trained model, making subsequent prediction impossible.
- **Resolution:** Fitted BERT baselines now serialize atomically to versioned joblib artifacts and can be reloaded with `BERTBaseline.load()`. `train-bert` refuses to overwrite by default and its tests verify artifact creation, reload/prediction, and preservation of an existing artifact.
- **Caveat:** joblib/pickle artifacts are executable serialization; only load trusted files and preserve compatible Python/scikit-learn/joblib versions.

### [ISSUE-016] HIGH: Replication Loader Failure Tests Mutate Checked-In Dataset Files
- **Severity:** High / Test Safety
- **Status:** Resolved (TASK-P9-04; verified in TASK-P9-12)
- **File:** `tests/test_replication_loader_failure_modes.py`
- **Description:** The former `_temp_dataset()` implementation could overwrite checked-in files under `data/replication/` and leave them replaced if cleanup failed.
- **Resolution:** `load_replication_annotation()` accepts an optional keyword-only `data_root`; failure-mode cases now write synthetic records only under pytest temporary directories. The focused replication runner/loader/dataset coverage and full suite pass. The first 16 hexadecimal characters of each of the four imported JSON SHA-256 digests matched before and after the full-suite run, and the failure-mode tests write only to their `tmp_path` fixture.

### [ISSUE-017] HIGH: Full Test Suite Has No Current Clean Completion Evidence
- **Severity:** High / Verification Gap
- **Status:** Resolved (TASK-P9-12, 2026-10-12)
- **Description:** Earlier complete-suite attempts were interrupted. A later attempt in the restricted sandbox stalled because Windows asyncio's socket-pair fallback waited on local TCP `accept()` during Starlette TestClient portal startup.
- **Resolution:** Reconfirmed outside the restricted sandbox on 2026-10-13: latest `python -m pytest tests/` completed with **264 passed, 38 warnings in 170.21s**, exit code 0 (173.83s outer duration). All API tests, including `test_api_root`, passed. The Windows TCP socket-pair fallback stall recurred in the sandbox and did not recur outside it; no API code change was needed.

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
- **Status:** ✅ Resolved (TASK-P9-13, 2026-10-12)
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
  - Added an autouse fixture in `tests/conftest.py` that resets configuration before and after each test. Verified by the complete suite (**236 passed**, 40 warnings).

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
- **Status:** ✅ Resolved (TASK-P9-13, 2026-10-12)
- **File:** `leredd/data_models.py`, line 57
- **Description:** The former substring normalizer could map arbitrary text containing "No" to `NO_DEPENDENCY`.
- **Resolution:** Replaced substring matching with an explicit alias table and a `ValueError` for unknown labels. Regressions cover ambiguous text and the preserved `Implemented by` label. Full suite: **236 passed**, 40 warnings.

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
- **Resolution:** Evaluation guards reject provenance-bearing records unless status is exactly `VERIFIED`; the CLI evaluator also rejects unverified provenance through `Evaluator`. Focused regressions cover provisional, pending-review, and human-reviewed states. Legacy records without provenance retain the explicitly documented compatibility path with a warning. Full suite: **236 passed**, 40 warnings.

### [BLOCKER-005] `adb_verified_ui_test.json` Contains an Explicit Test Fixture
- **Severity:** Low (Risk for automated dataset ingestion)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-02)
- **Description:** `data/verified/adb_verified_ui_test.json` (2 records) — 1 record has `review_notes` explicitly stating "UI persistence test... not an independently verified dependency annotation." Both carried `VerificationStatus.VERIFIED` or `PROVISIONAL`.
- **Resolution:** `check_ground_truth_eligibility` filters out provisional records and raises strict errors if provisional review queue / UI test exports are supplied as ground truth without human verification.

### [BLOCKER-006] Replication Imports Have Not Been Locally Verified
- **Severity:** High (Blocks replication metrics)
- **Status:** 🔴 Active (confirmed TASK-P9-13, 2026-10-12)
- **Files:** `scripts/integrate_replication.py`, `scripts/run_replication_evaluation.py`, `docs/REPLICATION_NOTE.md`
- **Evidence:** Loader provenance for imported spreadsheet records is `source=imported`, `status=pending_review`, and `annotator_id=None`. The evaluation eligibility rule now requires provenance-bearing labels to be explicitly `VERIFIED`. The replication prototype therefore exits before initializing its model or writing a report for current imports.
- **Required resolution:** Complete local human review using auditable identity and decision records, export verified annotations separately, then evaluate those verified exports without altering the source imports.

### [BLOCKER-007] Replication Dataset Redistribution Rights Are Unverified
- **Severity:** Medium (Research/data governance)
- **Status:** 🔴 Active (confirmed TASK-P9-13, 2026-10-12)
- **Files:** `docs/REPLICATION_NOTE.md`, `docs/EVALUATION_PROTOCOL.md`
- **Evidence:** Local replication package files do not establish a separate dataset license or redistribution agreement. The paper's license is not proof that separately packaged annotation data may be redistributed.
- **Required resolution:** Obtain authoritative dataset-specific permission/license terms before redistribution or publication of the imported annotation files. Do not infer permission from the paper license alone.

### [ISSUE-006] Evaluation CLI Argument Ambiguity & Supervised Cross-System Terminology Alignment
- **Severity:** Medium (Documentation & Metadata Integrity)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-03)
- **Description:** The `--model-type` parameter in `scripts/run_full_evaluation.py` selected the classification backend head for `BERTBaseline` (`rf` vs `lr`), but could be misconstrued as selecting which baseline model to run. Additionally, reports previously mislabeled supervised `BERTBaseline` fitting on ADB and testing on TJA as "zero-shot".
- **Resolution:** Added `--bert-classifier-type` parameter alias (with `--model-type` fallback) and updated help strings. Updated report configuration metadata (`bert_classifier_backend: "rf"`, `bert_classifier_description`). Re-labeled `BERTBaseline` and `TFIDF_LSABaseline` notes as **Supervised Cross-System Evaluation** (trained/fitted on ADB, tested on TJA) and added exploratory notices for McNemar's test on small samples ($n=5$). Added 4 regression tests (24 total in `test_safe_evaluation.py`).
