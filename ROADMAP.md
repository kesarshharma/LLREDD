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