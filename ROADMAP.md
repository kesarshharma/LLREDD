# ROADMAP.md — Dynamic Project Roadmap

This roadmap tracks the development phases for extending the LEREDD repository into the final Minor Project framework.

---

## Phase 0 — Repository Discovery and Safety
- [x] **TASK-P0-01: Initial Repository Reconnaissance & Inspection**
  - **Objective:** Deep inspection of Git history, remote URL, project structure, code files, and test status.
  - **Acceptance Criteria:** Comprehensive read-only handover report delivered and verified.
  - **Completion Evidence:** Inspection report produced; verified clean working tree and origin remote `https://github.com/kesarshharma/LLREDD.git`.
- [x] **TASK-P0-02: Proposal Alignment & Context Management Initialization**
  - **Objective:** Read `Minor_Project.pdf` and establish persistent context management files (`AGENTS.md`, `CONTEXT.md`, `ROADMAP.md`, `CHANGELOG.md`, `DECISIONS.md`, `ISSUES.md`).
  - **Acceptance Criteria:** All context management files created and synced with verified codebase state.
  - **Completion Evidence:** All 6 context files created in root directory.

---

## Phase 1 — Environment & Baseline Verification
- [x] **TASK-P1-01: Isolated Virtual Environment Setup & Dependency Resolution**
  - **Dependencies:** `TASK-P0-02`
  - **Objective:** Create python virtual environment, install requirements, and verify editable package installation.
  - **Acceptance Criteria:** `pytest --collect-only` runs without `ModuleNotFoundError`.
  - **Completion Evidence:** `.venv` (Python 3.13.3) with all deps installed; `leredd` editable; `pytest --collect-only` collected 28 items with exit code 0.
- [x] **TASK-P1-02: Test Suite Execution & Baseline Failure Resolution**
  - **Dependencies:** `TASK-P1-01`
  - **Objective:** Execute full unit test suite (`pytest tests/`) and record baseline pass/fail metrics.
  - **Acceptance Criteria:** All tests pass or genuine baseline issues are documented/fixed.
  - **Completion Evidence:** `pytest tests/ -v` → **28 passed, 0 failed** in 45.74s. 1 deprecation warning (httpx/starlette).
- [x] **TASK-P1-03: Offline Evaluation Benchmark Run (Mock LLM)**
  - **Dependencies:** `TASK-P1-02`
  - **Objective:** Run `scripts/run_evaluation.py` using `MockLLMClient` to record baseline evaluation metrics.
  - **Acceptance Criteria:** Baseline metrics recorded in `CHANGELOG.md` and `CONTEXT.md`.
  - **Completion Evidence:** Benchmark ran on ADB dataset (10 pairs). LEREDD(Mock): Acc=0.5000, MacroF1=0.4722. BERT: 1.0 (overfitting, no split). TF-IDF&LSA: Acc=0.3000. McNemar p=0.0625.

---

## Phase 2 — Dataset Integrity and Split Management
- [x] **TASK-P2-01: Dataset Schema & Provenance Extension**
  - **Dependencies:** `TASK-P1-03`
  - **Objective:** Extend `Requirement`, `RequirementPair`, and `Prediction` data models to support metadata, provenance (model version, annotator ID, timestamp), and verification status.
  - **Acceptance Criteria:** Pydantic models validate extended fields with backward compatibility.
  - **Completion Evidence:** Added `AnnotationSource` enum (5 values), `VerificationStatus` enum (5 values), `AnnotationProvenance` sub-model (16 fields), optional `provenance` field on `RequirementPair`, 4 provenance fields on `Prediction`. All new fields optional with None defaults. 35 new tests in `test_provenance.py` + 28 existing tests = **63/63 passed** in 131.57s. Legacy JSON datasets load without modification. No breaking changes.
- [x] **TASK-P2-02: Leakage-Resistant Dataset Splitter**
  - **Dependencies:** `TASK-P2-01`
  - **Objective:** Implement SRS/system-level dataset split generator ensuring disjoint train, retrieval, development, and test sets.
  - **Acceptance Criteria:** Splitter unit tests verify zero overlap between train and test pairs.
  - **Completion Evidence:** Implemented `leredd/dataset_splitter.py` with 4 splitting strategies (`SYSTEM`, `DOCUMENT`, `REQUIREMENT`, `PAIR`), `SplitConfig`, `DatasetSplit`, `LeakageReport`, and persistence functions. Integrated `leredd-cli split` command and `scripts/split_dataset.py`. Added 10 unit tests in `test_dataset_splitter.py`. Total test suite: **73/73 passed** (28 original + 35 provenance + 10 splitter) in 46.63s. Zero requirement leakage verified on disjoint splits.

---

## Phase 3 — Hybrid Retrieval
- [x] **TASK-P3-01: BM25 Lexical Retrieval Implementation**
  - **Dependencies:** `TASK-P2-02`
  - **Objective:** Implement BM25 lexical similarity scorer for requirement statements.
  - **Acceptance Criteria:** BM25 scoring unit tests pass.
  - **Completion Evidence:** Implemented `leredd/bm25_retrieval.py` using `BM25Plus` from `rank-bm25` (added dependency `rank-bm25>=0.2.2`). Features deterministic tokenization, inspectable corpus isolation, stable tie-breaking, requirement pair example retrieval, and pure offline execution. Added 10 unit tests in `test_bm25_retrieval.py`. Total test suite: **83/83 passed** (28 original + 35 provenance + 10 splitter + 10 BM25) in 47.61s. Zero regressions.
- [x] **TASK-P3-02: Hybrid Retrieval Engine (Semantic + Lexical + Metadata Fusion)**
  - **Dependencies:** `TASK-P3-01`
  - **Objective:** Implement Reciprocal Rank Fusion (RRF) / weighted score fusion combining SBERT embeddings, BM25, and metadata alignment.
  - **Acceptance Criteria:** `HybridRetriever` adapter compatible with existing `ExampleRetriever` API; unit tests pass.
  - **Completion Evidence:** Implemented `leredd/hybrid_retrieval.py` providing `HybridRetriever` and `reciprocal_rank_fusion` with configurable $rrf\_k=60$, stable tie-breaking, candidate deduplication, target self-exclusion, metadata/provenance preservation, and corpus isolation. Exported `HybridRetriever` from `leredd`. Added 9 unit tests in `test_hybrid_retrieval.py`. Total test suite: **92/92 passed** (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid) in 66.22s. Zero regressions.

---

## Phase 4 — Semi-Automated Dataset Generation
- [x] **TASK-P4-01: Candidate Requirement Pair Generation**
  - **Dependencies:** `TASK-P3-02`
  - **Objective:** Extract/accept requirements from SRS and generate reproducible candidate pairs via `ALL_PAIRS`, `BM25`, `DENSE`, or `HYBRID` strategies with canonical deduplication and audit manifests.
  - **Acceptance Criteria:** Candidate generator unit tests pass; audit manifest records generation configuration and truncation status.
  - **Completion Evidence:** Implemented `leredd/candidate_generator.py` providing `CandidateGenerator`, `CandidateGenerationManifest`, `GenerationStrategy` enum, and `get_canonical_pair_key`. Added `leredd-cli candidates` CLI command and `scripts/generate_candidates.py`. Added 9 unit tests in `test_candidate_generator.py`. Total test suite: **101/101 passed** (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid + 9 candidate generator) in 65.52s. Zero regressions.
- [x] **TASK-P4-02: Transparent Labeling Heuristics Engine & Pipeline**
  - **Dependencies:** `TASK-P4-01`
  - **Objective:** Implement provisional requirement dependency labeling combining deterministic rule-based heuristics (`HeuristicLabeler`) and LLM predictions (`LLMLabeler`) managed by a coordinator (`ProvisionalLabeler`) producing auditable provisional dataset records (`LabelingRunManifest`) with `VerificationStatus.PROVISIONAL`.
  - **Acceptance Criteria:** Provisional labeler unit tests pass; machine-generated labels explicitly preserved as provisional; agreement/disagreement recorded without overwriting verified ground truth.
  - **Completion Evidence:** Implemented `leredd/provisional_labeler.py` with `HeuristicLabeler` (deterministic rules `RULE-REF-ID-DIRECT`, `RULE-REF-ID-REVERSE`, `RULE-REF-IMPLEMENTS`, `RULE-REF-CONFLICTS`, `RULE-NEG-CONFLICT`, `RULE-SIM-JACCARD`, `RULE-ABSTAIN`), `LLMLabeler` (using existing LLM infrastructure and `MockLLMClient`), `ProvisionalLabeler` (coordinator handling agreement, disagreement, single-method, and unresolved states with `VerificationStatus.PROVISIONAL`), `LabelingRunManifest`, and dataset persistence. Added `leredd-cli label` CLI command and `scripts/label_candidates.py`. Added 12 unit tests in `tests/test_provisional_labeler.py`. Total test suite: **113/113 passed** in 10.97s. Zero regressions.

---

## Phase 5 — Human Review and Uncertainty Handling
- [x] **TASK-P5-01: Multi-Method Disagreement & Uncertainty Queue**
  - **Dependencies:** `TASK-P4-02`
  - **Objective:** Implement persistent, auditable human review queue (`ReviewQueueManager`, `ReviewItem`, `QueueStatus`, `ReviewReasonCode`, `HumanDecision`) with configurable eligibility rules, deterministic priority ranking, idempotent refresh, ground-truth protection (`VerificationStatus.VERIFIED` granted ONLY via explicit human decisions), and CLI integration (`queue-create`, `queue-list`, `queue-decide`).
  - **Acceptance Criteria:** Review queue unit tests pass; priority ordering enforced; human decision audit trail preserved without resetting reviewed items or overwriting raw predictions.
  - **Completion Evidence:** Implemented `leredd/review_queue.py` providing `ReviewQueueManager`, `ReviewItem`, `QueueStatus`, `ReviewReasonCode`, `ReviewPriority`, `HumanDecision`, `ReviewQueueConfig`, and `ReviewQueueManifest`. Added `leredd-cli queue-create`, `queue-list`, `queue-decide` CLI commands and `scripts/build_review_queue.py`. Added 12 unit tests in `tests/test_review_queue.py`. Total test suite: **125/125 passed** in 65.12s. Zero regressions.
- [x] **TASK-P5-02: Streamlit & CLI Review Interface Extension**
  - **Dependencies:** `TASK-P5-01`
  - **Objective:** Extend `ui/app.py` with Page 6 "6. Human Review Queue" (Operational Summary, Queue Browser & Filters, Side-by-Side Requirement Inspection, Method Prediction Comparison, Human Decision Controls, and Verified Annotation Export) and CLI (`queue-create`, `queue-list`, `queue-decide`).
  - **Acceptance Criteria:** Streamlit review workspace functional; decision controls validate input, update backend queue idempotently, and produce verified dataset exports without overwriting raw predictions.
  - **Completion Evidence:** Updated `ui/app.py` adding complete Streamlit Review Workspace. Added UI integration test suite `tests/test_ui_and_queue_integration.py`. Total test suite: **131/131 passed** in 65.23s. Zero regressions.

---

## Phase 6 — Confidence-Aware Classification
- [x] **TASK-P6-01: Evaluation Readiness and Dataset Integrity Audit**
  - **Dependencies:** `TASK-P5-02`
  - **Objective:** Read-only audit of all datasets, splits, evaluation capabilities, and evaluation blockers before running any model comparisons.
  - **Acceptance Criteria:** Audit report produced; dataset inventory complete; ground-truth eligibility determined; blockers documented; recommended next task specified.
  - **Completion Evidence:** Audit report produced at [`p6_01_audit_report.md`](file:///C:/Users/DN816/.gemini/antigravity-ide/brain/53a8af3f-ea50-421f-8b25-268eebad88da/p6_01_audit_report.md). Key findings: 20 defensible ground-truth pairs (no provenance metadata); system-disjoint split usable (10 train / 5 dev / 5 test); requirement-disjoint split unusable (empty dev/test); `adb_verified_ui_test.json` excluded (test fixture); all provisional/queue data excluded; PRIMARY BLOCKER is unguarded retrieval contamination risk. 5 blockers documented in ISSUES.md.
- [x] **TASK-P6-02: Safe Evaluation Script with Contamination Guard**
  - **Dependencies:** `TASK-P6-01`
  - **Objective:** Implement `scripts/run_full_evaluation.py` using system-disjoint split, isolated retrieval pool (train-only), and clean per-model evaluation pipeline (LEREDD, BERTBaseline, TFIDF_LSABaseline). Add contamination guard, VerificationStatus filtering, and JSON report output.
  - **Acceptance Criteria:** Script exits 0; 0 contaminated examples logged; evaluation report saved to `data/evaluation/system_disjoint_report.json`; unit test verifies contamination guard and report structure.
  - **Completion Evidence:** Implemented `scripts/run_full_evaluation.py`, `leredd/evaluation.py` extensions (`build_safe_retrieval_pool`, `check_ground_truth_eligibility`, `run_full_model_evaluation`), and test suite `tests/test_safe_evaluation.py` (20 tests). Retrieval contamination guard verifies zero test/dev requirement leakage into candidate pool (BLOCKER-001 resolved). Ground truth eligibility guard rejects provisional/UI-test records. System disjoint report written to `data/evaluation/system_disjoint_report.json`. Total test suite: **153/153 passed** in 156.37s.
- [x] **TASK-P6-03: Evaluation Results Audit and Methodology Consistency**
  - **Dependencies:** `TASK-P6-02`
  - **Objective:** Audit model names, CLI arguments, metrics, statistical test assumptions, and methodology descriptions across `scripts/run_full_evaluation.py` and generated reports.
  - **Acceptance Criteria:** CLI `--model-type` argument disambiguated via `--bert-classifier-type`; methodology terminology updated from "zero-shot" to "supervised cross-system evaluation" for fitted models (`BERTBaseline`, `TFIDF_LSABaseline`); exploratory notices added for McNemar test on small sample sizes ($n=5$); 4 new regression tests added to `tests/test_safe_evaluation.py`.
  - **Completion Evidence:** Updated `scripts/run_full_evaluation.py`, report metadata serialization, and report generator notes. Added 4 regression tests in `test_safe_evaluation.py`. Total test suite: **157/157 passed** in 167.09s. Zero regressions.
- [x] **TASK-P6-04: Reconcile Evaluation Metrics and Verify Statistical Claims**
  - **Dependencies:** `TASK-P6-03`
  - **Objective:** Investigate P6-02 vs P6-03 metric differences; replace hardcoded ±40% CI approximation with Wilson score binomial CI; implement full McNemar $2×2$ contingency table with exact binomial test; add unit tests verifying statistical functions.
  - **Acceptance Criteria:** Wilson score CI implemented; McNemar reports discordant counts $(b,c)$ and explicit exploratory status; all unit and regression tests pass (27/27 in `test_safe_evaluation.py`); no hardcoded margin-of-error approximations remain; `Prediction` construction in tests correctly includes required `rationale` field.
  - **Completion Evidence:** Added `compute_wilson_score_interval` and `analyze_mcnemar` to `leredd/evaluation.py`; updated `_serialize_report` and Step 7 in `scripts/run_full_evaluation.py`; fixed `Prediction(rationale=...)` in `tests/test_safe_evaluation.py`. **27/27 passed** in 122.80s. Zero regressions.

---

## Phase 7 — Integrated Evaluation and Ablation Studies
- [x] **TASK-P7-01: Benchmark Suite & Reproducible Comparison Script**
  - **Dependencies:** `TASK-P6-01`
  - **Objective:** Run full evaluation comparing semantic-only baseline, heuristic classifier, zero-shot LLM, and proposed hybrid system.
  - **Acceptance Criteria:** Comprehensive evaluation report generated with Macro-F1, per-class metrics, confusion matrices, and ablation analysis across retrieval strategies, in-context examples, and confidence thresholding.
- [x] **TASK-P7-02: Ablation Validity and Model Identity Audit**
  - **Dependencies:** `TASK-P7-01`
  - **Objective:** Audit model identities, fitting procedures, threshold mechanisms, and root cause of identical ablation metrics across variants.
  - **Acceptance Criteria:** Document actual implementation of all 5 primary models; capture per-example prediction matrix; prove configuration routing and threshold behavior via unit tests; verify ground-truth eligibility and retrieval isolation.
  - **Completion Evidence:** Verified that code routing correctly dispatches to dense, BM25, and hybrid retrievers and applies thresholding. Identified root cause of identical metrics: `MockLLMClient` evaluates regex patterns on Requirement A & B statement text only, ignoring prompt example blocks, and hardcodes confidence=5 (which is above threshold=4 in all branches). Created `scratch/audit_ablation_details.py` and added 2 regression tests to `tests/test_ablation_benchmark.py` (7 tests total). Total suite: **167/167 passed**.


## Phase 8 — Integration, Documentation, and Final Deliverables
- [x] **TASK-P8-01: System Verification & Multi-Interface Testing**
  - **Dependencies:** `TASK-P7-01`
  - **Objective:** Test end-to-end workflows across CLI, REST API, Streamlit UI, semi-automated dataset generation pipeline, and Docker setup.
  - **Acceptance Criteria:** All interfaces operational and verified with comprehensive integration test suite.
  - **Completion Evidence:** Verified end-to-end integration across CLI, REST API (`api/main.py`), Streamlit UI (`ui/app.py`), dataset pipeline, evaluation runner (`scripts/run_full_evaluation.py`), and Docker setup (`Dockerfile`, `docker-compose.yml`). Updated `tests/test_api.py` (7 tests) and created `tests/test_system_verification.py` (3 system tests). Comprehensive test suite of **174/174 tests** passed cleanly in 10.97s. Zero regressions.
- [x] **TASK-P8-02: Final Documentation & Artifact Alignment**
  - **Dependencies:** `TASK-P8-01`
  - **Objective:** Update README, API docs, architecture guides, and final context files aligned with academic submission requirements.
  - **Acceptance Criteria:** Repository documentation, API specifications, system architecture diagrams, and experiment summaries fully reflect empirical implementation.
  - **Completion Evidence:** Comprehensive rewrite of `README.md` with exact academic title, dual-workflow breakdown, architecture diagrams, taxonomy tables, setup/usage guide, benchmark summaries, and limitation statements. Created `docs/ARCHITECTURE.md` and `docs/EXPERIMENTS.md`. Updated `docs/API.md`. Updated all context tracking files (`ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `DECISIONS.md`, `ISSUES.md`). Full test suite: **174/174 passed** cleanly.

- [x] **TASK-P8-03: Final Academic Readiness and Evidence Audit**
  - **Dependencies:** `TASK-P8-02`
  - **Objective:** Audit codebase, datasets, evaluation reports, and documentation against submitted minor project proposal without adding features or inflating results.
  - **Acceptance Criteria:** Delivered evidence-based coverage matrix, empirical claims audit, deployment verification, academic honesty qualifications, and submission readiness declaration.
  - **Completion Evidence:** Audit report created at [`p8_03_academic_readiness_audit.md`](file:///C:/Users/DN816/.gemini/antigravity-ide/brain/53a8af3f-ea50-421f-8b25-268eebad88da/p8_03_academic_readiness_audit.md). Verified 174/174 unit and integration tests pass cleanly. Declared readiness status: **READY WITH CAVEATS**.
- [x] **TASK-P8-04: Final Submission Artifact Verification**
  - **Dependencies:** `TASK-P8-03`
  - **Objective:** Verify existence, completeness, and consistency of the P8-03 academic audit report and final project context files without adding features or inflating results.
  - **Acceptance Criteria:** Verified 10 submission criteria including explicit paper reference demarcation (0.880 F1), partially demonstrated annotation effort (no human time-study claimed), demonstrative $n=5$ sample size CIs, static Docker configuration checks, non-inflated 174 passing test reporting, and consistent readiness status across all documentation.
  - **Completion Evidence:** Verified artifacts [`p8_03_academic_readiness_audit.md`](file:///C:/Users/DN816/.gemini/antigravity-ide/brain/53a8af3f-ea50-421f-8b25-268eebad88da/p8_03_academic_readiness_audit.md), [`README.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/README.md), [`docs/EXPERIMENTS.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/EXPERIMENTS.md), [`docs/ARCHITECTURE.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/ARCHITECTURE.md), and [`docs/API.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/API.md). Final readiness status: **READY WITH CAVEATS**.

## Phase 9 — Project Health Audit
- [x] **TASK-P9-01: Whole-Project Analysis and Available Verification**
  - **Objective:** Inspect current project state, architecture, interfaces, documentation, recent replication additions, and run available verification.
  - **Evidence:** Current tree inspected (`git status`); `compileall` succeeded; CLI `--help` succeeded; FastAPI and Typer imports succeeded; focused replication dataset/loader/label-exclusion suite: **15 passed in 0.19s**; `tests/test_safe_evaluation.py`: **27 passed in 641.55s**.
  - **Limitations:** Full suite did not complete within this audit: pytest execution was interrupted after prolonged runs with no final summary. Default temp location caused a sandbox `PermissionError` in the first focused evaluation run; using a workspace temp directory allowed the suite to pass. Ruff executable is not installed in `.venv`. Replication-loader failure-mode tests overwrite checked-in data files while running and were not executed.
  - **Outcome:** Audit complete; project is not certified as “perfect.” See ISSUE-016/017 and active issues for test-safety, runtime, data, and evaluation caveats.
- [x] **TASK-P9-02: Functional Architecture Walkthrough**
  - **Objective:** Trace the full project purpose, data flow, model/retrieval stack, user interfaces, evaluation paths, and replication prototype from source and documentation.
  - **Evidence:** Inspected README/architecture/protocol/replication docs; core models, extraction and pair generation, candidate labeling/review, splitting, retrieval/detection, LLM clients, baselines/evaluation, CLI, API, Streamlit navigation, and replication loader/runner. No tests were requested or run for this read-only walkthrough.
  - **Outcome:** Functional map summarized in the task response and CONTEXT.md. Recorded legacy `pairs` command label semantics as ISSUE-018.
- [x] **TASK-P9-12: Windows TestClient Stall Diagnosis and Full-Suite Verification**
  - **Objective:** Diagnose the API root stall and establish a complete suite result without changing application or test behavior.
  - **Evidence:** Faulthandler located the stall in AnyIO portal startup while Windows asyncio's socket-pair fallback waited in TCP `accept()`. The sandboxed raw socket-pair probe also timed out; outside the sandbox, `test_api_root` passed in 0.75s, all 7 API tests passed in 50.70s, and the full suite passed **228/228 in 210.60s** (46 deprecation warnings). Four replication test files passed **38/38 in 40.11s**.
  - **Outcome:** No repository API/TestClient code change was warranted; the hang was specific to restricted sandbox loopback behavior. ISSUE-017 is resolved by the completed full-suite run; tests in this sandbox must use an execution context permitting local TCP loopback.
- [x] **TASK-P9-13: End-to-End Label Integrity and Stabilization Verification**
  - **Objective:** Resolve verified label-integrity defects, preserve provenance through generated-pair workflows, isolate evaluation from unverified annotations, strengthen regression coverage, and establish a current full-suite baseline.
  - **Implementation:** Generated pairs are unlabeled and carry candidate-generator provisional provenance; unknown relation strings now fail explicitly; annotation records a human VERIFIED decision; evaluators enforce verified provenance and complete prediction coverage; training/retrieval consumers exclude unlabeled or unverified records; test configuration resets global config state around each test. Replication source labels remain unchanged and evaluation blocks while imports are pending review.
  - **Focused evidence:** `tests/test_safe_evaluation.py::TestCheckGroundTruthEligibility` plus `tests/test_replication_evaluation_runner.py`: **15 passed in 2.09s**. `tests/test_cli.py::test_cli_evaluate`: **1 passed in 2.02s**.
  - **Full-suite evidence:** From PowerShell with `TEMP` and `TMP` set to `.pytest-tmp`, `python -m pytest tests/`: **236 passed, 40 warnings in 176.82s**, exit code 0; outer duration **179.59s**. The immediately preceding run had one incomplete CLI test fixture and was not counted as a pass; the fixture was corrected to cover all ground-truth IDs and the rerun passed.
  - **Data/status evidence:** The first 16 hexadecimal characters of the SHA-256 digest for each `data/replication/*.json` matched before and after the full-suite run. Failure-mode tests write only under `tmp_path`. No package install or Git state operation occurred. `git diff --check` reports no whitespace errors (only Git's LF-to-CRLF notices); remaining work is limited by research/data caveats.
  - **Remaining blockers:** Small legacy test set; mock-backend ablation limitations; source annotation `pending_review`; unresolved `Implemented by` mapping; dataset-specific redistribution rights; existing dependency deprecation warnings.

## Phase 10 — End-to-End Stabilization
- [x] **TASK-P10: Complete LEREDD End-to-End Functionality (local implementation and verification)**
  - **Objective:** Exercise the supported SRS-to-evaluation workflow and the CLI, FastAPI, and Streamlit surfaces; fix reproducible defects with controlled regressions.
  - **Implementation evidence:** Dense and hybrid candidate generation now use requirement-level rankings; retrieval candidate manifests report exhaustive output accurately; hash fallback embeddings use a stable feature space; split validation and leakage audits cover ratio bounds, cross-system pairs, and all partitions; CLI verified exports validate before mutation and publish atomically; malformed review queues are preserved. Synthetic end-to-end coverage exercises SRS extraction, provisional labeling, explicit synthetic review decisions, verified export, system-disjoint splitting, safe retrieval isolation, prediction, and report persistence. The fixture clearly identifies reviewer identity as synthetic test data.
  - **Verification:** `python -m pytest tests/` → **298 passed, 12 warnings in 69.21s**, exit 0 (PowerShell, `TEMP`/`TMP` set to `.pytest-tmp/task-p10/temp`, unrestricted Windows loopback execution). Focused workflow/retrieval subset: **120 passed in 8.02s**; embedding + ground-truth prompt regressions: **12 passed in 1.91s**; API module: **9 passed in 1.02s**. `git diff --check` returned no whitespace errors. Four imported replication JSON hashes and two historical evaluation report hashes matched before and after; `git diff -- data` was empty.
  - **Limits:** Hosted Python 3.10/3.11 GitHub Actions matrix remains unverified; real OpenAI/Ollama, Docker build, optional PDF/DOCX packages, clean-cache model download, replication annotation validity/rights, and human visual review were not exercised. Existing deprecation warnings remain.
  - **Next:** Obtain local human review and dataset-specific redistribution terms before replication metrics/publication; handle hosted CI verification as a separate user-authorized task.

- [x] **TASK-P9-14: Whole-Project End-to-End Stabilization and Feature Traceability**
  - **Objective:** Audit and exercise the supported workflows across CLI, scripts, API, Streamlit, data generation, review, splitting, detection, and evaluation; fix reproducible in-scope defects; publish evidence and limitations.
  - **Implementation:** Added `docs/FEATURE_TRACEABILITY.md`; protected source datasets and prior outputs in UI and script workflows; removed automatic synthetic human verification; fixed same-sample baseline fitting, UI evaluation leakage, fabricated kappa, API secret serialization, and silent optional document extraction failures.
  - **Verification:** Fresh unrestricted `python -m pytest tests/`: **261 passed, 40 warnings in 196.70s**, exit 0; outer duration **200.50s**. Focused suites also passed: scripts/UI/baselines 31 passed/70.70s, UI journeys 7 passed/10.08s, API 9 passed/28.66s. FastAPI `/api/v1/health` and Streamlit `/_stcore/health` each returned HTTP 200; both processes were stopped. A sandbox rerun stalled at `test_api_root` and was stopped; the unrestricted run passed that test.
  - **Integrity:** SHA-256 hashes for all four imported replication JSON files matched recorded pre-run hashes; `git diff -- data` was empty. `git diff --check` reported no whitespace errors (only LF-to-CRLF notices). Existing unrelated and untracked worktree changes were preserved. No package installation, external LLM call, or Git state operation occurred.
  - **Limits:** Real OpenAI/Ollama requests, cold-cache model download, Docker build, browser visual review, and BERT artifact persistence remain unverified/unsupported; replication labels remain pending local review and licensing/direction clarification. See `docs/FEATURE_TRACEABILITY.md`.
  - **Files:** See `docs/FEATURE_TRACEABILITY.md` for implementation, test, dependency, and status mapping.
- [x] **TASK-P9-15: Persist and Reload Trained BERT Baseline Artifacts**
  - **Finding:** Completion audit found `leredd-cli train-bert` claimed to save a model but only created an empty output directory.
  - **Implementation:** Added atomic, versioned joblib persistence and `BERTBaseline.load()`. CLI now writes `bert_baseline.joblib`, refuses replacement unless `--overwrite` is given, and checks collision before training. Added the direct joblib dependency and documented trusted-artifact handling.
  - **Verification:** Focused baseline/CLI suite: **14 passed, 1 deselected in 3.48s**. Fresh unrestricted `python -m pytest tests/`: **264 passed, 38 warnings in 170.21s**, exit 0; outer duration **173.83s**. Four replication dataset hashes matched before/after.
  - **Limits:** Artifact loading uses pickle serialization and therefore must only be used with trusted files; compatible Python/scikit-learn/joblib versions are required.
- [x] **TASK-P9-16: Use Module Invocation for CI Pytest Imports**
  - **Finding and change:** GitHub Actions collection failed to import top-level `api` and `scripts` when invoking the pytest console entry point. Changed the CI step to `python -m pytest` so the repository root is on `sys.path`.
  - **Verification:** Local full suite via `python -m pytest tests/ --cov=leredd --cov-report=xml:.pytest-tmp/coverage.xml --cov-report=term-missing` → **264 passed, 38 warnings**, exit 0. GitHub Actions Python 3.10/3.11 matrix rerun remains pending.

- [x] **TASK-P11: Research Workflow Validation and Submission Readiness**
  - **Objective:** Validate the user workflows across CLI, API, and Streamlit; audit provenance and leakage; inspect replication readiness; correct reproducible defects; document an isolated demonstration and remaining human/research blockers.
  - **Fixes:** System-disjoint splitting now reserves requested train/dev/test partitions when the available system count permits; evaluation requires coherent verified human provenance for provenance-bearing ground truth (human source, actor identity, and matching accepted/scored labels), rejecting known generic placeholder IDs. CLI and queue-manager verification require explicitly supplied nonblank actor identities; Streamlit no longer pre-populates a generic reviewer. Added split, evaluator, safe-runner, CLI, and queue safety regressions.
  - **Execution evidence:** Real Markdown CLI flow in `.pytest-tmp/task-p11/cli-demo` extracted stable explicit IDs, generated BM25 candidates, wrote provisional-only heuristic annotations, and persisted a pending review queue without fabricated reviewer action. The default three-system CLI split now writes non-empty partitions (5/10/5 fixture pairs; no leakage). The legacy ADB/APA/TJA split and MockLLM report were written under `.pytest-tmp/task-p11/legacy-flow`, with 10/5/5 examples and zero retrieval contamination; report flags the provenance-free legacy labels and MockLLM as demonstrative.
  - **Tests:** Focused CLI/queue/evaluation/provenance/UI suite: **116 passed, 12 warnings in 26.23s**. Full `python -m pytest tests/ -q` (outside restricted sandbox because its Windows loopback restriction stalls TestClient): **309 passed, 12 warnings in 23.02s**, exit 0. Warnings are the existing `tqdm`/`datetime.utcfromtimestamp()` deprecation.
  - **Integrity:** Four replication JSON and two historical report SHA-256 hashes unchanged; `git diff -- data` empty; `git diff --check` clean. No package installs, paid calls, CI edits, or Git state mutations.
  - **Limits / next:** Actual annotations remain pending until a human reviews them. Replication label direction and dataset redistribution permission are unresolved; no real OpenAI/Ollama request, Docker build, optional PDF/DOCX install, or browser visual review was performed.

- [x] **TASK-P12: End-to-End Product Stabilization and Honest Evidence Alignment (local scope)**
  - **Objective:** Exercise the supported document-to-evaluation path and independently verify CLI, FastAPI, and Streamlit workflows; fix reproducible defects without replacing real human, provider, or research evidence with fixtures.
  - **Implementation:** Evaluation report metadata now derives systems, partition sizes, provenance counts, label coverage, and sample-size caveats from the supplied split. Replaced unsupported paper/GPT-4 and per-class/significance claims in README and experiment documentation with the checked-in report's actual scope and limitations. Added a regression using synthetic DEMO systems.
  - **Execution evidence:** Under `.pytest-tmp/task-p12/e2e`, exercised SRS extraction, BM25 candidate generation, provisional labeling, review-queue creation, synthetic-only fixture decisions/export (explicitly not human evidence), system-disjoint splitting (3/3/3, no system overlap), BM25 retrieval, MockLLM prediction, and primary/ablation report persistence. Report showed 3 eligible pairs per partition and zero retrieval contamination; API and Streamlit journeys were independently covered by automated tests.
  - **Verification:** `tests/test_safe_evaluation.py`: **35 passed, 12 warnings in 21.69s**. API/UI suites: **32 passed in 6.43s**. Complete `python -m pytest tests/ -q`: **309 passed, 12 warnings in 23.06s**, exit 0, using Python 3.13.3, workspace-local TEMP/TMP, offline model mode, and unrestricted local loopback. `git diff --check` clean. Replication and historical report hashes were preserved; `git diff -- data` empty.
  - **Limits:** Synthetic decisions demonstrate software transitions only. Actual human annotation, identity authentication, dataset rights and label-direction resolution, real-provider requests, visual browser review, Docker build, optional PDF/DOCX installation, and hosted Python 3.10/3.11 CI remain unverified or externally blocked. No commit or push was made.

- [x] **TASK-P13: Launch and exercise the local product demonstration**
  - **Evidence:** Started Streamlit on 8501 and FastAPI on 8000; verified UI/API health and API extraction/detection. Browser-inspected the UI, generated synthetic provisional candidates, and ran the CLI review queue, system-disjoint synthetic split, and persisted mock evaluation reports under `.pytest-tmp`. Full suite: **310 passed, 12 warnings**.
  - **Fix:** Streamlit demo defaults use explicitly labeled synthetic SRS data and a disposable output path; empty legacy card shells are hidden.
  - **Limits:** No human review was fabricated; verified export requires genuine operator decisions. Evaluation output is synthetic software evidence, not research evidence.

- [x] **TASK-P14: Decouple CI replication tests from local-only data**
  - **Finding:** Hosted run `38442446610` failed 17 tests because the replication JSON files are intentionally absent from Git; the Docker step was skipped.
  - **Change:** Loader, dataset-contract, and replication-runner tests now use synthetic temporary records. Tests that inspect the actual imported source package run when it is supplied and skip with an explicit reason when absent.
  - **Verification:** Focused replication tests **35 passed**; full local suite **310 passed, 12 warnings in 131.43s**, exit 0. Exact CI coverage command in a complete tracked-file checkout without replication assets: **309 passed, 1 skipped, 12 warnings in 200.84s**, exit 0; focused run in that checkout: **34 passed, 1 skipped in 11.57s**.
  - **Next:** Review and publish the changes only with explicit authorization, then verify the hosted Python 3.10/3.11 matrix and Docker build. Do not include replication materials while redistribution rights remain unresolved.

- [x] **TASK-P15: Add blinded independent annotation and agreement tooling (local implementation)**
  - **Implementation:** Added a separate CLI workflow for two reviewers to annotate the same candidates without seeing generated labels, record optional rationale/abstention and elapsed time, resume from an in-progress manifest, and produce observed agreement, Cohen's kappa, confusion matrix, and timing summaries. Source/context hashes and pair-set identity are checked; outputs are blocked from imported replication and historical evaluation trees; results never auto-promote labels.
  - **Verification:** `python -m pytest tests/test_independent_annotation.py -q` → **15 passed in 1.01s**. Full `python -m pytest tests/` → **325 passed, 12 warnings in 69.69s**, exit 0. Warning source: `tqdm`'s `datetime.utcfromtimestamp()` deprecation.
  - **Limits / next:** Tooling has not been used by actual independent reviewers. Human review, disagreement adjudication, replication label direction and data rights, real-provider validation, Docker, browser visual review, and hosted CI remain open/unverified. Do not claim research results from synthetic tests.

- [x] **TASK-P16: Browser QA and graph readability correction**
  - **Finding / fix:** Direct Brave inspection found that visible node labels overlapped in the dependency graph. Requirement IDs and statements now appear in hover details rather than crowded permanent labels; added a focused regression.
  - **Execution evidence:** Streamlit navigation, synthetic SRS extraction, mock prediction, fixed legacy evaluation, graph and review queue were exercised in Brave. Extraction wrote 3 requirements and 3 provisional pairs beneath `.pytest-tmp/task-p16-ui`; mock detection returned `REQUIRES`, confidence 5/5. The evaluation page wrote a fresh unique report to `data/evaluation/ui_runs/20261123T094742Z-215fe692/system_disjoint_report.json`; it explicitly notes legacy provenance-free data, `n_test=5`, and MockLLM. No human decision was submitted.
  - **Verification:** Focused graph regression **1 passed in 12.09s**; UI journeys **8 passed in 20.45s**; full suite **326 passed, 12 warnings in 77.22s**, exit 0 (83.01s outer). Hashes of all `data/` files were identical before and after the full suite. Existing historical reports and source datasets were not changed.
  - **Limits:** The visual check used one local Brave desktop session; cross-browser/mobile behavior remains unverified. No actual human annotations or research-grade results were created.

- [x] **TASK-P17: Make MockLLM ablation limitations explicit in reports**
  - **Finding:** The offline evaluation runner compares retrieval strategy, ICL examples, and confidence threshold while using deterministic MockLLM that ignores retrieved examples and always returns confidence 5. Those comparisons are not empirical estimates of model effects.
  - **Change:** Added structured `ablation_validity` to primary and dedicated ablation reports and a matching evaluation limitation. Kept deterministic mock prediction behavior unchanged; did not claim the empirical issues are resolved.
  - **Verification:** Report-contract test **1 passed in 16.50s**; full safe-evaluation module **35 passed, 12 warnings in 20.06s**; final `python -m pytest tests/` **326 passed, 12 warnings in 29.50s**, exit 0 (33.97s outer). SHA-256 hashes of every `data/` file matched before and after the full run.
  - **Limits:** ISSUE-009 and ISSUE-010 remain open until an authorized real LLM or separately validated controlled test double is available.

- [x] **TASK-P18: Prevent generated requirement ID collisions**
  - **Finding / fix:** Requirement extraction could reuse an explicit SRS ID for an ID-less requirement when the explicit identifier matched the synthetic counter, including when it appeared later in the document. The allocator now reserves all explicit IDs in the input and skips occupied generated IDs while preserving explicit IDs and the existing schema.
  - **Verification:** `python -m pytest tests/test_utils.py -q`: **8 passed in 3.56s**. Full `python -m pytest tests/`: **327 passed, 12 warnings in 89.33s**, exit 0 (95.21s outer). Warnings are the existing `tqdm` `utcfromtimestamp()` deprecation. Before/after SHA-256 comparison of all existing `data/` files found **0 differences**.
  - **Limits:** This prevents generated-vs-explicit collisions; it does not rewrite or validate duplicate identifiers already repeated explicitly in a source document.

- [x] **TASK-P19: Preserve SRS source and extracted location**
  - **Finding / fix:** File extraction discarded source document identity and line location. Added an optional `Requirement.source_location` field; file extraction records a basename and extracted-text line, while API and UI accept/use a source label. The document name remains separate from line location so document-disjoint splitting can group records consistently. Explicit short and bracketed IDs used by the UI examples are preserved.
  - **Compatibility:** The new field is optional; old JSON records still load. Calls to text extraction without a source name keep `source=regex_extraction`. PDF/DOCX coordinates are extracted-text lines, not physical page/paragraph coordinates.
  - **Verification:** Focused utility, file-script, API, UI, and CLI tests: **43 passed in 24.11s**. Full suite: **329 passed, 12 warnings in 87.12s**, exit 0 (93.86s outer). The 12 warnings are existing tqdm datetime deprecations. All existing `data/` SHA-256 hashes matched before and after. A CLI smoke test preserved `REQ-DEMO-007` and `REQ-2`, source `demo-srs.md`, and extracted-text lines 1 and 2 in output under `.pytest-tmp`.

- [x] **TASK-P20: Add paired manual-versus-assisted effort-study workflow (tooling only)**
  - **Implementation:** Added a fixed-sample schedule bound to input/context hashes, counterbalanced assignment of each pair to two different reviewers and both conditions, CLI collection that hides provisional suggestions in manual mode and identifies them in assisted mode, and a descriptive report with per-condition time and label disagreement. Study observations are never verified ground truth. Added output guards that reject all paths under project `data/` and a regression test.
  - **Documentation:** README and annotation guide describe prerequisites, commands, timing limits, self-reported identities, rights-cleared input expectations, and non-research status of synthetic fixtures. Feature traceability lists scripts and evidence.
  - **Verification:** Focused annotation tests **27 passed in 1.98s**. Complete `python -m pytest tests/ -q`: **341 passed, 12 warnings in 92.81s**, exit 0 (PowerShell wall duration 98.50s, Python 3.13.3, `TEMP`/`TMP` set to `.pytest-tmp`, `HF_HUB_OFFLINE=1`). All existing `data/` file hashes matched; no files were added under `data/`.
  - **Limits:** No human reviewers or rights-cleared study sample were used; no effort estimate, annotation-quality result, or statistical inference exists. Research completion remains blocked on real annotation, adjudication, dataset permissions/label direction, and other external validations recorded below.

- [ ] **TASK-P21: Verify Docker image build and local Compose services**
  - **Progress:** Added `.dockerignore` because the Dockerfile's `COPY . .` otherwise included local `data/`, replication packages, reports, credentials, and temporary files in the image build context. Compose configuration parses successfully.
  - **Blocked verification:** Docker CLI is installed (29.1.5), but `docker info` cannot connect to `dockerDesktopLinuxEngine`; no image build or container launch was possible. Keep this task open until the daemon is available, then build and exercise API/UI health paths without publishing local data.
  - **Latest local check (2026-11-24):** `com.docker.service` is `Stopped` with `Manual` start type, and no Docker Desktop/backend process was found. A bounded `docker info` probe timed out. An elevated `Start-Service com.docker.service` attempt returned “Cannot open ... service”; no build or container state was changed. A local administrator must start Docker Desktop before this verification can continue.

- [x] **TASK-P22: Diagnose and verify live Streamlit rendering**
  - **Finding:** The blank-shell observation was time-bounded, not a persistent failure. On the documented port 8501, native Chrome first showed Streamlit's `Running...` shell; the dashboard rendered after approximately 16 seconds from navigation. A separate fresh instance on 8502 rendered after about six seconds. HTTP health/root checks succeeded on the diagnostic instance.
  - **Conclusion:** The UI does run in native Chrome. The earlier 7.5-second observation ended during cold startup; no application-code root cause was established, and no watcher workaround or dependency installation was justified. The isolated in-app browser websocket error was not reproduced in native Chrome.
  - **Limit:** Startup latency should be measured/optimized separately if it remains a demonstration concern. The initial browser state is not a hang once the UI proceeds to the dashboard.

- [x] **TASK-P23: Verify live Streamlit rendering and synthetic extraction in native Chrome**
  - **Evidence:** A diagnostic Streamlit instance on `127.0.0.1:8502` returned HTTP 200 for `/` and `/_stcore/health`. Native Chrome rendered that dashboard after about six seconds. The documented Streamlit command then launched on 8501 and native Chrome rendered its dashboard after about 16 seconds from navigation. The API process started on 8000; its logs record HTTP 200 for `/` and `/api/v1/health`.
  - **Synthetic UI flow:** Requirement Annotation rendered; its synthetic SRS action extracted 3 requirements and saved 3 candidate pairs to `.pytest-tmp/leredd-ui/candidates-demo-20261124.json`. Inspection confirmed all 3 remained unlabeled, `status=provisional`, `source=candidate_generator`. The UI refused an overwrite of a pre-existing file; that file was preserved.
  - **Limits / next:** No human label was submitted and no imported dataset was changed. The isolated in-app browser websocket failure did not reproduce in native Chrome. Remaining UI pages, real human annotation, eligible evaluation, and a complete real-data end-to-end run remain unverified.

- [x] **TASK-P24: Avoid unused embedding startup during zero-shot detection**
  - **Finding / fix:** `LEREDDDetector` eagerly created `EmbeddingModel` during construction, so a zero-shot API request could cold-load SentenceTransformer even when no examples or RAG context used embeddings. Detector embedding and retrieval helpers now initialize on first use; explicit injected embedders and the existing retrieval APIs remain supported.
  - **Verification:** Focused detector/API/embedding/dense/hybrid/BM25 tests: **46 passed in 9.24s**. Fresh local API process: `/extract` returned 2 identified requirements with source/line metadata; `/batch_detect` returned 2 predictions; malformed `/detect` returned 422. A zero-shot `/detect` returned HTTP 200 in **164 ms** without loading SentenceTransformer. Full `python -m pytest tests/`: **342 passed, 12 warnings in 128.88s**, exit 0. `git diff --check` exited 0. SHA-256 hashes for all 11 files under `data/replication/` and `data/evaluation/` matched before and after.
  - **Live UI:** Native Chrome rendered the dashboard; the synthetic extraction form saved 3 unlabeled provisional pairs to `.pytest-tmp/leredd-ui/task-p24-candidates-20261124.json`. The default MockLLM detection page produced `REQUIRES`, confidence 5/5, with rationale; the first dense-retrieval interaction took over one minute to return (not separately timed or attributed to a single operation). The annotation controls displayed a default output under `data/verified/`; no annotation was submitted.
  - **Limits / next:** Cold dense retrieval remains slow and needs isolated timing/diagnosis. Real-provider, human-review, replication rights/direction, hosted CI, Docker daemon/build, and optional PDF/DOCX validation remain outstanding.

- [x] **TASK-P25: Preserve alphanumeric requirement IDs during extraction**
  - **Finding / fix:** The extractor's explicit-ID regex truncated IDs such as `REQ-DEMO-A` to `REQ-DEMO`, causing collisions and an empty downstream candidate set. It now accepts complete hyphen/colon separated alphanumeric ID segments; a regression checks three distinct IDs and three generated pairs.
  - **Execution evidence:** Ran the README CLI sequence on synthetic Markdown under `.pytest-tmp/task-p24-cli-corrected`: extracted 3 requirements, generated 3 BM25 candidates, wrote 3 provisional records (1 heuristic suggestion, 2 abstentions), and built/listed 3 pending review items. Every pair remained `provisional`; no human decision or verified export was created.
  - **Verification:** `python -m pytest tests/test_utils.py tests/test_cli.py -q`: **21 passed in 3.29s**. Full suite: **343 passed, 12 warnings in 31.11s**, exit 0. The 12 warnings remain the `tqdm` UTC datetime deprecation. Existing replication/evaluation artifact hashes matched before and after the full suite.
  - **Limits / next:** The accepted ID grammar remains `REQ-` followed by alphanumeric segments separated by hyphen or colon; real SRS samples may require additional identifier formats.

- [x] **TASK-P26: Select retrieval mode in API and Streamlit detection**
  - **Implementation:** Added optional validated `retrieval_mode` fields (`dense`, `bm25`, `hybrid`) to single and batch API requests and a corresponding Streamlit detection selector. Omitted API fields continue to use the configured default. Added pass-through and invalid-mode regression coverage; updated the API reference.
  - **Verification:** Focused API/UI tests: **20 passed in 18.76s**. A fresh local API request using one synthetic example and `retrieval_mode=bm25` returned HTTP 200 and logged a BM25 index build without loading SentenceTransformer. Complete `python -m pytest tests/`: **346 passed, 12 warnings in 91.15s**, exit 0. `git diff --check` exited 0, and all 11 protected replication/evaluation hashes matched.
  - **Limits:** The already-running Streamlit process predates the selector change, so this selector was not inspected in a fresh live browser session; AppTest verified the page and mode pass-through. Dense cold-start latency remains unmeasured and potentially slow. No synthetic records were promoted to verified data.

- [x] **TASK-P27: Preserve predicted labels and expose confidence review status**
  - **Finding / fix:** The detector previously rewrote low-confidence relation labels to `No_dependency`. It now preserves the model's dependency class and adds `validation_status` plus the applied threshold. Streamlit displays the review cue; evaluation reports include screened counts and review rate. Updated architecture, API, replication, and feature-traceability descriptions.
  - **Verification:** Focused parser/API/UI/ablation/evaluation tests: **72 passed, 12 warnings in 105.79s**. Full `python -m pytest tests/`: **347 passed, 12 warnings in 102.35s**, exit 0. Warnings remain the existing tqdm `utcfromtimestamp()` deprecation.
  - **Limits:** The UI queue action is exercised with a controlled low-confidence fixture; MockLLM's fixed confidence 5 still provides no empirical review-rate or screening-utility evidence. Real-provider behavior, reviewer data, rights and label direction, hosted CI, and Docker remain open.

- [x] **TASK-P28: Persist low-confidence UI predictions into human review queue**
  - **Implementation:** The Streamlit detection page now offers an explicit action for a `requires_review` prediction. It adds the machine-labeled pair to the existing pending review queue as provisional, preserving its label and rationale; it does not mark the annotation verified. The UI atomically replaces its queue JSON. The shared `ReviewQueueManager.save_queue()` also stages queue and manifest JSON separately before atomic per-file replacement. Added write-failure regression coverage.
  - **Verification:** Focused queue/UI checks: **15 passed in 19.96s**. Full `python -m pytest tests/`: **348 passed, 12 warnings in 89.15s**, exit 0; PowerShell measured 97.71s. Warnings remain `tqdm`'s UTC datetime deprecation. `git diff --check` passed.
  - **Limits:** UI journey is verified with AppTest, not a fresh browser session. Two-file queue/manifest replacement cannot be a single filesystem transaction; if the second replacement fails after queue replacement, the queue remains authoritative and its manifest may lag. Real reviewers and real-provider confidence behavior remain unverified.

- [x] **TASK-P29: Verify and refresh the live detection-to-review UI path**
  - **Finding:** The pre-existing Streamlit process on port 8501 showed an import error for `PredictionValidationStatus`. A fresh Python import succeeded, and a fresh Streamlit instance on port 8503 rendered the dashboard. This points to stale in-memory runtime state, not a current source import failure.
  - **Live workflow:** In the fresh browser UI, ran MockLLM on two synthetic requirements with the confidence threshold set to 5. It displayed `No_dependency`, confidence 5/5, and the configured screening warning; saved a provisional queue item under `.pytest-tmp/task-goal-live-ui-20261124/review_queue.json`. The Human Review Queue page loaded that exact file and showed **1 total / 1 pending**, with `LLM_ONLY` and `LOW_CONFIDENCE`; no human decision or export was made. This is a software demonstration, not evidence of prediction quality.
  - **Runtime:** Replaced the stale 8501 Streamlit process with a fresh process on the documented default port. HTTP root and Streamlit health both returned 200; health body was `ok`. A separate 8503 test process was stopped; existing 8502/API processes were left alone.
  - **Limits / next:** Dense retrieval cold-start took roughly 40 seconds during this live interaction; zero-shot synthetic detection eventually completed. The previous stale process's in-memory import state was not independently recoverable. Actual human decisions, dataset rights and `Implemented by` direction, real-provider evidence, Docker daemon/build, and hosted Python 3.10/3.11 CI remain open.

- [x] **TASK-P30: Make confidence-cutoff review wording accurate at the inclusive boundary**
  - **Finding / fix:** A live `5/5` MockLLM result was routed for review at a configured cutoff of 5 by the existing inclusive rule (`score <= cutoff`), while the interface called the score “low confidence.” Updated detection, review-reason, and queue-note wording to state the cutoff rule explicitly; preserved the serialized `low_confidence` reason code and cutoff behavior.
  - **Verification:** Focused queue/UI checks: **16 passed in 26.32s**. Full `python -m pytest tests/`: **348 passed, 12 existing warnings in 114.93s**, exit 0; PowerShell duration **124.79s**. Warnings remain the `tqdm` UTC datetime deprecation.
  - **Limits:** AppTest verifies the copy change and queue behavior; the live browser observation preceded this copy-only source change. Real confidence calibration and review utility remain unmeasured.

- [x] **TASK-P31: Reuse dense embedding models across Streamlit reruns**
  - **Finding / fix:** The UI constructed a new `LEREDDDetector` for each prediction. With the local 10-item annotated pool and dense retrieval, each detector could initialize SentenceTransformer again. Added per-session embedder reuse keyed by model name/cache directory; the detector exposes a non-loading `loaded_embedder` accessor so zero-shot/BM25 runs do not force initialization.
  - **Verification:** Focused UI/detector/embedding tests: **15 passed in 23.16s**. The UI regression performs two predictions in one AppTest session and confirms the second detector receives the same initialized embedder. Full suite: **348 passed, 12 warnings in 114.94s**, exit 0; PowerShell duration **126.52s**. The first full run exposed one outdated OpenAI-path test double, which was updated to accept the optional embedder; the final run passed.
  - **Limits:** This proves model-instance reuse in a Streamlit session, not a measured wall-time improvement. A first dense use still pays model initialization; real provider and retrieval-quality behavior remain unverified.

- [x] **TASK-P32: Add metadata-aware ranking to hybrid example retrieval**
  - **Finding / fix:** Proposal objective 3.3.1 calls for component/subsystem metadata signals when available, while the example retriever fused only dense and BM25 rankings. Added optional `component` and `subsystem` fields to `Requirement` and a third, exact-overlap metadata ranking for component/subsystem/system values. The metadata ranking is omitted when the target has no meaningful values or no candidate matches; existing records with missing metadata retain dense/BM25 behavior.
  - **Verification:** Focused hybrid/detector/candidate/API tests: **62 passed in 11.36s**; final hybrid suite after the zero-weight guard: **11 passed in 0.27s**. Final full `python -m pytest tests/`: **350 passed, 12 existing tqdm warnings in 114.69s**, exit 0 (PowerShell duration 123.08s). Regression tests show matching component/subsystem values can change the selected example and generic/missing metadata creates no signal.
  - **Limits:** This establishes implemented capability only. Existing datasets do not provide component/subsystem labels at scale, and no retrieval-quality comparison or tuned metadata weight has been performed.

- [x] **TASK-P33: Add auditable multi-model provisional voting**
  - **Finding / fix:** The labeling workflow previously compared heuristics with a single LLM response. Added opt-in `multi_llm` CLI mode and model-list support for hybrid labeling. Only unanimous valid model votes produce a provisional suggestion; any disagreement or incomplete vote remains unlabeled and is routed for review, with model IDs, votes, confidence, and rationales recorded. LLM response cache keys now include provider/model identity to prevent cross-model response reuse.
  - **Verification:** Focused provisional-labeler/LLM/CLI suite: **35 passed in 11.46s**. Full `python -m pytest tests/`: **355 passed, 12 existing tqdm warnings in 115.92s**, exit 0 (PowerShell duration 124.57s). Controlled model clients verified unanimity, disagreement, unresolved output, CLI model validation, provisional provenance, and model-separated caching.
  - **Limits:** No real provider was called. Mock clients validate software plumbing only; model independence, output quality, costs, availability, and human review remain unverified. The cache-key version change intentionally does not reuse prior model-ambiguous cache entries.

- [x] **TASK-P34: Carry explicit component metadata through supported interfaces**
  - **Finding / fix:** The retrieval layer accepted component/subsystem values, but SRS extraction and single-pair API/UI detection offered no supported way to provide them. Added exact-ID sidecar mappings for CLI extraction/candidate generation, `metadata_by_id` for API extraction, a JSON mapping field in Streamlit extraction, and optional target metadata fields in API/UI detection. Candidate JSON and pair inputs preserve the values for retrieval.
  - **Verification:** Focused utility/CLI/API/UI tests: **50 passed in 29.85s**. They exercise exact-ID attachment, SRS-to-candidate serialization, API 200/422 behavior and target-pair metadata, and UI extraction/detection propagation. Full `python -m pytest tests/`: **364 passed, 12 warnings in 95.37s**, exit 0. The warnings are existing `tqdm` UTC datetime deprecations. Eleven replication/evaluation artifact SHA-256 hashes matched the pre-run baseline; `git diff --check` passed.
  - **Limits:** Values must come from the SRS or a human-supplied sidecar; the project does not infer components from prose. Existing datasets still lack established metadata coverage, so no retrieval-quality claim follows.

- [x] **TASK-P35: Preserve requirement metadata through human review and verified export**
  - **Finding / fix:** `ReviewItem` previously retained requirement IDs, text, and a shared system name only. The reviewer decision path rebuilt `Requirement` records from those fields, dropping per-requirement system, component, subsystem, source, and source location. Queue records now preserve those fields, the UI shows them beside each requirement, and verified decisions restore them in exported pairs. All added queue fields are optional so existing queue JSON remains loadable.
  - **Verification:** Focused queue/UI suites: **41 passed in 30.02s**. Full `python -m pytest tests/`: **366 passed, 12 warnings in 109.44s**, exit 0 (PowerShell duration **119.04s**). The warnings are the existing `tqdm` UTC datetime deprecation. Eleven protected replication/evaluation artifact hashes matched their pre-run values; `git diff --check` passed.
  - **Compatibility limit:** Old queue files load and remain decidable, but metadata omitted from a previously created queue cannot be reconstructed; regenerate the queue from the original candidate pairs to carry those values forward.
  - **Next:** obtain real reviewer decisions on a rights-cleared, metadata-bearing SRS before making research claims; Docker engine, hosted CI, provider-backed behavior, replication rights/direction, and human effort evidence remain open.

- [x] **TASK-P36: Publish immutable, verified dataset releases**
  - **Finding / fix:** Records carried a fixed `dataset_version`, but there was no release step to enforce human verification or bind a dataset snapshot to its source and output hashes. Added `leredd-cli dataset-release`, which accepts one or more verified JSON exports, rejects unlabeled, provenance-free, unverified, incoherent, or duplicate-ID records, stamps deep-copied records with the requested version, and publishes `pairs.json` plus an integrity/class-count manifest into a new directory via staged same-volume rename.
  - **Verification:** Focused release/CLI tests: **19 passed in 9.86s**. Full `python -m pytest tests/`: **374 passed, 12 warnings in 105.05s**, exit 0 (PowerShell duration **112.94s**). Warnings are the existing `tqdm` UTC datetime deprecation. All 11 protected replication/evaluation hashes matched their pre-run values; `git diff --check` passed.
  - **Limits:** Only synthetic verified test fixtures were released; this creates a reproducible software artifact format, not a real reviewed dataset. A release is never overwritten; choose a new destination for each version.

- [x] **TASK-P37: Measure candidate generation coverage against exhaustive pairs**
  - **Finding / fix:** Strategy generation reported only the number of emitted candidates and an exhaustive flag. Manifests and the `candidates` CLI now report the total unordered pair universe, pair-space recall, and uncovered pair count. Pair-space recall is explicitly candidate coverage, not recall of labeled dependencies.
  - **Verification:** Focused generator/CLI/script tests: **42 passed in 9.03s**. Full suite: **374 passed, 12 existing warnings in 101.49s**, exit 0 (PowerShell duration **110.51s**). `git diff --check` passed. All **311** files under `data/` had zero hash or file-set differences.
  - **Limits:** The metric measures coverage of possible pairs only. Estimate true dependency recall and strategy quality later with a rights-cleared sample and locally verified labels.

- [x] **TASK-P38: Screen repeated-inference consistency**
  - **Finding / fix:** The proposal calls for checking repeated-run prediction consistency, but the detector only recorded one response. Added an optional 1–5 same-prompt run check to Python, CLI, API, and Streamlit. Repeated calls bypass the built-in response cache; any label or self-reported confidence variation marks the prediction inconsistent and recommends human review while preserving the first run's predicted class. Default behavior remains one inference.
  - **Verification:** Focused detector/LLM/API/CLI/UI tests: **61 passed in 28.25s**. Full `python -m pytest tests/`: **383 passed, 12 existing `tqdm` deprecation warnings in 108.10s**, exit 0 (PowerShell duration **116.29s**). `git diff --check` passed. All **311** files under `data/` retained identical file paths and SHA-256 hashes.
  - **Limits:** No real-provider calls were made; the consistency check is not calibrated uncertainty or evidence of model quality. No human verification was fabricated. A sandboxed focused rerun stalled at the known Windows API loopback point; the unrestricted full suite passed `test_api_root` and all API tests.
  - **Next:** collect rights-cleared SRS material and actual independent human annotations, then evaluate repeated-run utility with authorized real-provider access. Hosted CI and Docker remain separate blockers.

- [x] **TASK-P39: Preserve tabular and textual SRS content during extraction**
  - **Finding / fix:** The proposal covers requirements in paragraphs and tables. DOCX extraction previously read only `document.paragraphs`, silently discarding requirements held in tables; the regex also removed the final period. Added source-ordered DOCX paragraph/table-row extraction, terminal-period preservation, and an actionable error for image-only PDFs that need unsupported OCR.
  - **Verification:** Synthetic DOCX table and PDF text regressions pass, including source IDs/locations and candidate generation; scanned-PDF failure has explicit OCR guidance. Focused `tests/test_utils.py tests/test_data_scripts.py -q`: **25 passed in 13.05s**. Full `.venv` suite: **386 passed, 1 existing Starlette deprecation warning in 109.83s**, exit 0 (PowerShell duration **115.85s**). All **311** data files retained identical paths and SHA-256 hashes; `git diff --check` passed.
  - **Limits:** Synthetic documents only. OCR is not implemented. Real SRS extraction recall and varied table layouts still need human review. The base system Python remains without the optional `documents` extra.
  - **Next:** use rights-cleared SRS documents to manually compare extracted requirements against the source; secure reviewer participation and source rights before generating research labels.

- [x] **TASK-P40: Route unusually short requirements for human inspection**
  - **Finding / fix:** The proposal calls for unusually short or incomplete requirement statements to be flagged for review. The queue previously handled empty text but not short non-empty text. Added a configurable minimum-character screening cue (default 20, `0` disables), exposed it through `queue-create`, and recorded the eligibility configuration in the queue manifest. The cue does not classify text as incomplete, rewrite labels, or verify annotations.
  - **Verification:** Review queue/CLI/UI integration tests: **45 passed in 18.06s**. Full `.venv` `python -m pytest tests/`: **387 passed, 1 existing Starlette deprecation warning in 101.78s**, exit 0 (PowerShell duration **108.13s**). All **311** files under `data/` retained identical paths and SHA-256 hashes. `git diff --check` passed.
  - **Limits:** The threshold is a configurable heuristic, not a validated linguistic rule. Semantic ambiguity and incompleteness still require a human reviewer. No human labels or real SRS content were used.

- [x] **TASK-P41: Reject unsafe SRS context retrieval parameters**
  - **Finding / fix:** `RAGRetriever.chunk_document()` could loop without advancing when overlap equaled or exceeded chunk size; zero chunk sizes and invalid counts were not validated. Added positive integer chunk-size checks, overlap bounds (`0 <= overlap < chunk_size`), non-negative integer retrieval-count validation, and an explicit empty result for `k=0` without embedding work.
  - **Verification:** Focused retrieval/detector tests: **31 passed in 1.09s**. Full `.venv` suite: **398 passed, 1 existing Starlette deprecation warning in 103.94s**, exit 0 (PowerShell duration **109.82s**). All **311** files under `data/` retained identical paths and SHA-256 hashes.
  - **Limits:** This validates parameter safety, not the semantic usefulness of retrieved context. Retrieval relevance still needs real SRS and evaluated labels.

- [x] **TASK-P42: Make optional RAG usable per API request**
  - **Finding / fix:** The API accepted `srs_text` but had no request field to enable RAG. With the default server configuration (`use_rag=False`), supplied text was silently unused. Added optional `use_rag` on single and batch requests and pass-through to detector; omission retains the server setting, while explicit `false` overrides an enabled setting.
  - **Verification:** Focused API/detector/retrieval tests: **49 passed, 1 existing Starlette deprecation warning in 4.53s**. Full `.venv` suite: **400 passed, 1 existing Starlette deprecation warning in 99.47s**, exit 0 (PowerShell duration **105.20s**). All **311** files under `data/` retained identical paths and SHA-256 hashes.
  - **Limits:** Controlled tests confirm context reaches the prompt. They do not establish improved predictions; real SRS, model, and human-verified evaluation remain necessary.

- [x] **TASK-P43: Verify replication label semantics and redistribution evidence**
  - **Finding:** The primary paper's v2 Table I and Section III confirm its Implements definition: a higher-level requirement is fulfilled by a lower-level requirement. The imported workbook uses the distinct source label Implemented by; one example matches the paper's example with reversed workbook row order, but no explicit source-to-paper crosswalk was found. The arXiv paper license is CC BY 4.0, while the paper states the replication package will be made available upon publication; the local package still has no dataset-specific license or redistribution terms.
  - **Evidence:** Read-only inspection of the official [arXiv v2 paper](https://arxiv.org/html/2602.22456v2), the arXiv record, the local ADB example, and all four source workbooks. An openpyxl audit found 0 exact/reversed duplicate requirement pairs, 0 unlabeled/incomplete rows, across 813 rows. No imported datasets were edited.
  - **Outcome:** Corrected docs/EVALUATION_PROTOCOL.md; retained active blockers for label mapping and dataset rights. No normalization, benchmark, release, or redistribution was performed. git diff --check passed.
  - **Next:** Obtain authoritative dataset-specific terms and an explicit source-label crosswalk; then conduct independent local human review before any mapped evaluation.

- [x] **TASK-P44: Revalidate live CLI, API, and synthetic Streamlit paths**
  - **Verification:** `python -m leredd.cli --help` exited 0 and listed the supported commands. Outside the restricted socket sandbox, the existing API returned HTTP 200 at `/` and `/api/v1/health` (`status=ok`). A fresh Streamlit instance on port 8504 rendered the detection page; synthetic SRS extraction persisted three candidate pairs under `.pytest-tmp/task-final-ui-demo/candidates.json`, all unlabeled and provisional. A BM25/MockLLM prediction rendered `NO DEPENDENCY`, 5/5, with the UI's warning that this is not a calibrated probability or human verification. `git diff --check` passed.
  - **Limits:** This is a synthetic interface smoke test, not accuracy evidence. No human review, verified export, or evaluation was performed. UI logs include a Windows `ConnectionResetError` during socket teardown; no route or visible UI failure reproduced. A sandboxed API probe was denied at the socket boundary; the same read-only probe succeeded outside that boundary. Existing CI, Docker, rights, label crosswalk, provider, and human-study blockers remain.
  - **Next:** Obtain authoritative replication data rights and label mapping, then collect independently reviewed real annotations before measuring end-to-end research outcomes.

- [x] **TASK-P45: Re-run the complete local suite and verify dataset integrity**
  - **Verification:** `.venv\Scripts\python.exe -m pytest tests/` completed outside the restricted socket sandbox: **400 passed, 1 Starlette deprecation warning in 176.51s**, exit 0; PowerShell duration **181.96s**. `test_api_root` passed. Before/after SHA-256 manifests matched for all **312** files under `data/`; `git diff --check` passed.
  - **Warning:** Installed Starlette warns that HTTPX-backed `TestClient` use is deprecated and recommends `httpx2`. Current PyPI metadata requires Python >=3.11 for `httpx2`, whereas the project supports Python >=3.10. No dependency change was attempted; investigate a cross-version-compatible test transport before changing requirements.
  - **External check:** The official arXiv v2 article still says the replication package will be made available upon publication; the arXiv record/page has no direct package link. Dataset rights and the source-label crosswalk remain unverified.
  - **Limits:** The suite does not establish real-provider quality or human-study results. Hosted CI and Docker remain separately unverified; replication licensing and label crosswalk remain open.

- [x] **TASK-P46: Replace deprecated API TestClient test transport**
  - **Finding / fix:** `tests/test_api.py` was the only repository user of Starlette's `TestClient`, which starts a worker-thread portal and emitted a deprecation warning under the current stack. Replaced it with a small synchronous adapter using the existing HTTPX `AsyncClient` and `ASGITransport`; strengthened the root test to assert the full response.
  - **Verification:** Root test **1 passed in 1.36s**; API module **18 passed in 1.67s**; full suite **400 passed with no warnings in 107.33s**, exit 0 (PowerShell duration **111.87s**). The full run occurred outside the restricted Windows socket sandbox. All **312** `data/` file hashes and paths matched before/after.
  - **Limits:** The restricted sandbox still stalls at Windows event-loop socket initialization, even without Starlette TestClient; do not use that restricted environment to judge this test. No new dependencies or application behavior were added. Research and data-rights blockers remain.

- [x] **TASK-P47: Build and smoke-test CPU-only Docker deployment**
  - **Finding / fix:** The default Linux PyPI Torch resolver selected CUDA/cuDNN packages for CPU-only API/UI services. Added the official CPU wheel index install before `requirements.txt` so Sentence Transformers reuses CPU Torch.
  - **Verification:** `docker compose build` succeeded for `minor-api` and `minor-ui`; `docker compose up -d api ui` started both. API root and health returned HTTP 200; Streamlit root returned HTTP 200 and `/_stcore/health` returned `ok`.
  - **Limits:** The subsequent Docker management API check returned 500/hung; a bounded `docker info` recheck timed out after 20 seconds, while API/UI health remained 200/`ok`. Final `compose ps` and image-content inspection remain incomplete. Browser visual inspection was blocked by the computer-use URL check. No full test suite was run because only the Docker build configuration changed. Human annotation, rights/crosswalk, real-provider, hosted-CI, and research evidence remain outstanding.

- [x] **TASK-P48: Exercise extraction and candidate generation on an external SRS**
  - **Fix:** Corrected `leredd-cli extract --help` and its command docstring to list Markdown support, which the implementation already provided; extended the existing Markdown CLI regression to assert the advertised `.md` and `.docx` formats.
  - **Execution:** Used NASA Trick's repository SRS as a temporary input. CLI extraction produced 99 unique requirements; all 99 retained source locations. Ninety-seven source lines contained modal requirement keywords, with two lines yielding two separate outputs each. All-pairs generation produced 4,851 pairs; heuristic labeling left 350 suggestions and 4,501 unresolved, all provisional. Queue creation produced 4,851 pending items and zero reviewer decisions. Artifacts stayed under workspace-local `.pytest-tmp` (untracked); no evaluation or verified release was created.
  - **Verification:** `tests/test_cli.py::test_cli_metadata_sidecar_flows_from_srs_to_candidate_pairs -q`: **1 passed**; `git diff --check` passed.
  - **Limits:** Keyword counts and source locations do not establish extraction precision/recall; this was a parser smoke trial, not a human-reviewed SRS. All-pairs coverage is pair-space exhaustiveness, not dependency recall. No paid model or human verification was used. Obtain source-specific permissions and reviewer labels before research evaluation or dataset release.

- [x] **TASK-P49: Measure retrieval candidate coverage on the same SRS**
  - **Execution:** On the 99 extracted requirements (4,851 possible unordered pairs), ran the existing BM25, dense, and hybrid strategies at `top_k=5`, plus the recorded all-pairs run. BM25 emitted 327 pairs (6.74% pair-space coverage), dense 335 (6.91%), hybrid 325 (6.70%), and all-pairs 4,851 (100%).
  - **Interpretation:** This measures emitted-pair coverage/review volume only, not recall of true dependency links. Without independently verified labels, the rate of missed positive dependencies is unknown and the top-k setting is not validated for research use.
  - **Side effects and verification:** CLI outputs are in `.pytest-tmp/task-p49-candidate-coverage`. Dense model encoding created 99 ignored embedding-cache files under the configured `data/embeddings` cache; the four imported replication JSONs and historical evaluation reports were not written by these commands. The run used `HF_HUB_OFFLINE=1`; the model loaded locally. No tests were run; `git diff --check` passed.
  - **Next:** Have two actual reviewers label a preregistered sample drawn from both generated and omitted pairs, then estimate positive dependency recall and review effort with the sampling design recorded.

- [x] **TASK-P50: Prepare a blinded candidate-coverage pilot packet**
  - **Prepared:** A fixed-seed sample of 50 hybrid top-5 selected pairs and 50 omitted pairs from the 4,851-pair universe. The reviewer file assigns opaque pair IDs, contains no strategy stratum or labels, and keeps all records provisional. A separate analysis map records each stratum's population/sample counts, inclusion probability, and weight.
  - **Integrity checks:** 100 input records, 50 per stratum, all unlabeled/provisional; reviewer-input SHA-256 `cbe751f617f258a3f1f58f48eb0683b2c13c24e1a99cb146`. `scripts/annotate_independent.py --help` confirms the blind collector accepts one fixed input and the same optional SRS context for both reviewers.
  - **Regression verification:** `.venv\Scripts\python.exe -m pytest tests/` completed with **400 passed, no warnings in 132.31s**, exit 0 (PowerShell duration 138.03s). `TEMP`, `TMP`, and `CACHE_DIR` pointed to `.pytest-tmp`; `HF_HUB_OFFLINE=1`.
  - **Limits:** This is an operational pilot sample, not a powered or preregistered research result. It has no annotations, agreement score, positive-recall estimate, or verified ground truth. The stratum map must remain hidden from both reviewers until their annotations are complete.
  - **Next:** Two actual reviewers independently annotate the supplied fixed file; then adjudicate disagreements and add a sampling-weighted positive-recall analysis before choosing candidate settings.

- [x] **TASK-P51: Add fail-closed weighted candidate-recall analysis**
  - **Implementation:** Added `leredd/candidate_coverage.py` and `scripts/analyze_candidate_coverage.py`. Analysis requires the pinned blinded sample, strata map, strategy output, full pair universe, source SRS, and an exact-size adjudicated pair file with coherent `VERIFIED` human provenance. It reports post-stratified positive totals, candidate positive-recall estimate, and a finite-population-corrected normal interval only when the stratum variance estimate is non-degenerate.
  - **Safety:** Provisional, missing, extra, duplicate, tampered, incorrectly stratified, or contradictory labels fail closed. It does not adjudicate independent reviewer vectors or create labels. `docs/ANNOTATION_GUIDE.md`, README, and feature traceability describe the command and limits.
  - **Verification:** Focused analysis/provenance/release suite after the final population-count regression: **36 passed in 16.77s**. The default restricted run stalled at the known Windows socket boundary after the first module; rerunning outside that restriction completed the full suite: **413 passed, no warnings in 38.38s**, exit 0 (PowerShell duration **42.26s**). `git diff --check` passed.
  - **Output correction:** Review found and fixed the report's `population_pair_count` field to return the pair-universe size, not that size plus estimated omitted positives. Regression asserts the field on a controlled eight-pair universe.
  - **Evidence boundary:** Synthetic fixtures verify estimator and guards only. The real pilot has no verified adjudications yet, so this estimator has not been run on human results and no positive-recall estimate exists.

- [x] **TASK-P52: Prevent synthetic/demo records from entering research ground truth**
  - Added synthetic markers/source identities and propagated them through extraction, pair generation, CSV persistence, and review decisions.
  - Excluded marked examples from retrieval and verified dataset export; evaluation rejects marked ground truth, including records carrying otherwise verified provenance. Added regression coverage across CLI, UI, retrieval, review, serialization, and evaluation.
  - Verification: focused suite **172 passed in 30.25s**; full suite **424 passed in 34.30s**, exit 0.
  - Evidence boundary: only controlled synthetic fixtures were used. Human review and research metrics remain unavailable until actual adjudications are collected.

- [x] **TASK-P53: Revalidate evaluation partitions before scoring**
  - Added a reusable validator for train/test/dev system, requirement, and pair-ID overlap. The full evaluation runner invokes it against actual loaded records for declared system-disjoint splits and records the audit in the report.
  - Duplicate prediction-ID errors now identify the invalid pair IDs. Updated contamination fixtures to declare pair-level strategy so they continue testing contamination filtering without claiming to be system-disjoint.
  - Verification: focused evaluation suite **45 passed in 35.98s**; full suite **428 passed in 35.96s**, exit 0. All **412** files under `data/` retained their hashes; `git diff --check` passed.
  - Remote review: Parts 2/3 artifacts are present at public `main` (`ec11351`), but local and remote histories diverge and the remote snapshot had two reproduced test failures. No Git state or remote branch was changed.

- [x] **TASK-P54: Reconcile and publish the approved software checkpoint**
  - Preserved both the three local commits and two newer `origin/main` commits with a merge; no reset, rebase, branch switch, or force push was used.
  - Published as `d419d998e8ed9c05098c4fb0bf4f6789df08c74f` on `origin/main`.
  - Verification: post-merge full suite **456 passed in 106.86s**; merge-specific evaluation/synthetic tests **42 passed**; `git diff --check` passed. Remote `refs/heads/main` exactly matched the pushed hash.
  - Kept local-only the PDF, imported replication JSON, external replication materials, generated reports, temporary logs/caches, and scratch files. Source rights and label-crosswalk blockers remain open.

- [x] **TASK-P55: Correct Linux CI test portability failures**
  - **Finding:** The supplied Python 3.11 CI log collected 456 tests and reported 449 passed, 4 skipped, and 3 failed. Two CLI tests searched ANSI-colored output where Click inserted escape codes between the option's leading hyphens. A protected-output test derived the repository root from `tmp_path`; on Linux that resolved to `/`, so it attempted `/data/...` and hit an OS permission error before the product's path guard.
  - **Fix:** Disabled color in the two CLI error assertions and derived the protected output from the test file's repository-relative location. Assertions still check required reviewer rejection and protected-path rejection.
  - **Verification:** Focused regressions: **3 passed in 12.45s**. Full local Windows suite: **456 passed in 171.76s**, exit 0 (PowerShell duration **178.50s**). `git diff --check` passed before tracking updates.
  - **CI-equivalent verification:** `python -m pytest tests/ --cov=leredd --cov-report=xml --cov-report=term-missing` with coverage outputs directed to `.pytest-tmp/task-p57-coverage` completed **456 passed in 197.92s**, **89% coverage**, exit 0 (PowerShell duration **206.61s**). The pre-existing root `.coverage` hash was unchanged.
  - **Boundary:** This does not prove the hosted Linux job now passes; it must be rerun after the reviewed changes are published. Docker verification is still blocked by an unresponsive daemon.

- [x] **TASK-P58: Normalize Rich CLI output in Linux regression assertions**
  - **Finding:** The hosted run for commit `1643a7840bd475f5da8bd1b00b361b48ec11ec0a` reached pytest and reported **450 passed, 2 failed, 4 skipped**. Rich emitted ANSI escapes inside the option names even with `CliRunner(color=False)`; the independent-annotation protected-path regression passed.
  - **Fix:** Strip ANSI styling with Click's `unstyle()` before asserting option spellings. Keep command-failure and queue-nonmutation assertions.
  - **Verification:** Both regressions passed locally. Full local coverage run: **453 passed, 3 skipped, 12 warnings in 225.32s**, exit 0; coverage **88%**. Logs and coverage were isolated under `.pytest-tmp/task-p58-coverage`.
  - **Next:** Publish the reviewed test-only correction and inspect Python 3.10/3.11 Actions results. Hosted CI is not yet verified green; Docker remains separately blocked by local daemon availability.
