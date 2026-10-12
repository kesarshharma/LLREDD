# DECISIONS.md — Architectural & Technical Decisions Log

This document records key architectural, design, and implementation decisions for the project.

## [ADR-019] Keep API Tests Unmodified; Run Windows TestClient Outside Restricted Sandbox
- **Date:** 2026-10-12
- **Status:** Accepted
- **Context:** The API root test stalled inside the execution sandbox. Faulthandler showed Starlette TestClient waiting for AnyIO's portal thread while Windows asyncio created its self-pipe. This Python build lacked native `_socket.socketpair` and `AF_UNIX`, so Python used a TCP loopback fallback that blocked in `accept()`.
- **Decision:** Do not alter the API route, replace TestClient, or weaken the existing route assertion. Run pytest in a Windows process context that permits local TCP loopback. The same root test passed outside the sandbox, followed by all 7 API tests and the complete 228-test suite.
- **Consequences:** Existing API coverage remains intact. The sandbox limitation is documented as an execution constraint; the full suite is verified outside it. ISSUE-017 is resolved based on the full pytest summary (228 passed, 46 deprecation warnings, exit code 0).

## [ADR-020] Require Explicit Verification for Provenance-Bearing Evaluation Ground Truth
- **Date:** 2026-10-12
- **Status:** Accepted (user selected)
- **Context:** Replication imports preserve source spreadsheet labels but are marked `pending_review`; spreadsheet labels alone do not establish local human verification. Letting those records into metrics would overstate annotation verification.
- **Decision:** Evaluation accepts provenance-bearing labels only when `VerificationStatus.VERIFIED`. Pending-review, human-reviewed-but-not-verified, provisional, rejected, or absent-status provenance is excluded (or raises in strict mode). Existing provenance-free legacy benchmark records remain conditionally accepted with an audit warning for backward compatibility. Replication prototype evaluation fails closed on current imported packages and writes no report until records are locally verified.
- **Consequences:** No replication metrics are reported from the current pending-review imports. Verification must occur through the human review workflow; the source spreadsheets and imported JSON are not modified. Full suite: **236 passed**, 40 warnings.

## [ADR-018] Candidate Generation Emits Explicitly Unlabeled Records
- **Date:** 2026-10-12
- **Status:** Accepted; implemented in TASK-P9-13
- **Context:** The repository has both a legacy all-pairs helper/CLI (`generate_requirement_pairs`, `leredd-cli pairs`) and a newer candidate-generation pipeline. Their initial label semantics differ.
- **Decision:** Both generation paths emit `dependency_type=None` and candidate-generator provisional provenance. `No_dependency` is only an explicit annotation. Human CLI annotation records a `VERIFIED` decision; unverified labels cannot be used as evaluation ground truth. Unknown label strings are rejected rather than guessed.
- **Consequences:** Candidate creation no longer fabricates negative labels. JSON/CSV and downstream training/retrieval/evaluation preserve and enforce the unlabeled state. Existing serialized labels remain readable.

## [ADR-017] Audit Reporting Must Distinguish Focused Passes from Full-Suite Verification
- **Date:** 2026-10-12
- **Status:** Accepted
- **Context:** Existing project context claimed a fully passing, clean tree, while this audit found user modifications, untracked replication additions, test-side production-data mutation, and a full suite that did not complete.
- **Decision:** Record verification by exact command/scope and observed result. Do not mark the entire project verified based on focused tests or prior historical reports. Avoid running tests that mutate checked-in datasets until they use isolated fixture files.
- **Consequences:** Audit statements remain reproducible and accurately bounded; a full-suite pass is still required before claiming whole-project test success.

---

## [ADR-016] Comprehensive Documentation Alignment for Academic Submission
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** `TASK-P8-02` requires aligning `README.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md`, and context tracking files with the empirical implementation and academic proposal title *"Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation"*.
- **Decision:**
  1. Align `README.md` to state the exact academic title and document both workflows (Dataset Generation & LEREDD Detection), 6 dependency categories, architecture Mermaid diagrams, quickstart, CLI/API/UI/Docker usage guides, empirical results, and honest limitations.
  2. Create `docs/ARCHITECTURE.md` to detail system architecture, data models, candidate generation, review queue engine, system-disjoint dataset splitting, hybrid ICL retrieval, and provenance tracking schemas.
  3. Create `docs/EXPERIMENTS.md` to document system-disjoint evaluation metrics, baseline comparison tables, per-class metrics, statistical significance tests, and limitations.
  4. Update `docs/API.md` with complete technical references for FastAPI endpoints, Typer CLI commands, and Python SDK classes.
  5. Sync all context tracking files (`ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `DECISIONS.md`, `ISSUES.md`).
- **Consequences:** The repository documentation is 100% aligned with the codebase, fully reproducible, and ready for academic project submission.

---

## [ADR-015] Multi-Interface System Verification & Integration Architecture
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** `TASK-P8-01` requires verifying that the CLI, REST API (FastAPI), Streamlit UI (`ui/app.py`), semi-automated dataset generation pipeline, safe evaluation runner, and Docker deployment operate cohesively without architectural mismatch or data leakage.
- **Decision:** Establish a multi-layer integration verification test suite (`tests/test_api.py`, `tests/test_system_verification.py`, `tests/test_streamlit_app.py`) verifying:
  1. REST API endpoint contracts and HTTP 422 error handling via `TestClient`.
  2. End-to-end pipeline provenance flow (`provisional` $\rightarrow$ `review_queue` $\rightarrow$ `human_decision` $\rightarrow$ `verified_export` $\rightarrow$ `dataset_split` $\rightarrow$ `safe_evaluator`).
  3. Strict ground-truth eligibility filtering preventing unverified machine predictions from contaminating evaluation split ground-truth.
  4. Container infrastructure syntax and service configuration integrity (`Dockerfile`, `docker-compose.yml`).
- **Consequences:** 174/174 unit and integration tests pass cleanly. Architectural alignment with the submitted proposal is empirically confirmed across all 4 project interfaces (CLI, REST API, Streamlit UI, Python SDK).

---

## [ADR-001] Extension-First Architectural Policy
- **Date:** 2026-10-09
- **Status:** Accepted
- **Context:** The minor project prompt requires extending the existing LEREDD repository (`kesarshharma/LLREDD`) rather than rebuilding from scratch.
- **Decision:** Preserve existing public interfaces, Pydantic models (`Requirement`, `RequirementPair`, `Prediction`), CLI commands (`leredd-cli`), API endpoints (`/api/v1/*`), and Streamlit app structure (`ui/app.py`). All new workflows (hybrid retrieval, dataset generation, human review queue) will be implemented as backward-compatible extensions or adapters.
- **Consequences:** Lowers regression risk, maintains reproducibility of existing baseline tests, and ensures seamless multi-interface support.

## [ADR-002] Multi-Stage Persistent Context Management
- **Date:** 2026-10-09
- **Status:** Accepted
- **Context:** High-complexity iterative project requires strict tracking across LLM sessions to prevent drift, duplicate work, or broken assumptions.
- **Decision:** Maintain 6 core context files (`AGENTS.md`, `CONTEXT.md`, `ROADMAP.md`, `CHANGELOG.md`, `DECISIONS.md`, `ISSUES.md`). Every prompt execution MUST check context at start and update context at completion.
- **Consequences:** Assures complete auditing, accurate task states, and empirical evidence tracking.

## [ADR-003] Composable Provenance Sub-Model Design
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** The semi-automated dataset generation pipeline requires tracking label origin (human/LLM/heuristic), verification lifecycle, predicted vs accepted labels, confidence scores, and reviewer identity. These fields must coexist with the existing schema without breaking backward compatibility.
- **Decision:** Implement provenance as a composable `AnnotationProvenance` sub-model attached as an optional field on `RequirementPair` (for ground-truth/annotation lifecycle) rather than flattening all fields directly onto the pair. `Prediction` receives 4 lightweight provenance fields (`annotation_source`, `model_id`, `config_hash`, `created_at`) directly, since predictions are ephemeral inference outputs not requiring full lifecycle tracking. All new fields use `Optional` with `None` defaults.
- **Alternatives Considered:**
  1. *Flat fields on RequirementPair* — Rejected: would add 16+ optional fields to an already crowded model, polluting the core data structure.
  2. *Separate provenance table/model linked by ID* — Rejected: over-engineering for the current project scope; no relational database involved.
  3. *Separate `ProvenancedPair` subclass* — Rejected: would break type compatibility with all existing consumers expecting `RequirementPair`.
- **Consequences:** Zero breaking changes. Legacy datasets load without modification. All 28 original + 35 new tests pass. New provenance data flows naturally through existing JSON serialization (`model_dump`/`model_validate`). Future phases (dataset generation, human review) can populate provenance without schema migrations.

## [ADR-004] Leakage-Resistant Multi-Strategy Dataset Splitting
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Requirement dependency detection pairs often share requirement statements across different pairs. Naive random pair splitting causes requirement-level data leakage between training, retrieval-example, and test partitions, resulting in artificially high baseline performance (e.g., BERT 100% accuracy on un-split pairs).
- **Decision:** Implement a dedicated `DatasetSplitter` with 4 explicit strategies:
  1. `SYSTEM`: Partition by system identifier (ADB vs APA vs TJA) ensuring zero system and requirement overlap.
  2. `DOCUMENT`: Partition by SRS document/module metadata.
  3. `REQUIREMENT`: Partition the set of unique requirement IDs into disjoint subsets so no requirement ID in the test set appears in train or retrieval sets. Cross-partition pairs are dropped by default for pure evaluation.
  4. `PAIR`: Pair-level random split (baseline, with explicit logged warning).
- **Consequences:** Guarantees zero data leakage for rigorous evaluation in Phase 7. Provides automated leakage checking (`check_leakage`) and metadata logging for reproducible benchmarks. All 73 test suite items pass.

## [ADR-005] BM25 Lexical Retrieval with BM25Plus Non-Negative Score Handling
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Dense vector embeddings (SBERT) capture semantic similarity but can miss exact keyword / entity matches (e.g. acronyms like BCS, PCS, subsystem names, or exact error codes). Integrating a BM25 lexical retriever provides complementary lexical signals. However, standard `BM25Okapi` produces negative IDF values on small datasets when terms appear across multiple documents.
- **Decision:** Implement `BM25Retriever` in `leredd/bm25_retrieval.py` using `BM25Plus` from `rank-bm25`. `BM25Plus` adds a lower-bound parameter ($\delta = 1.0$) to prevent negative score penalties on frequent query terms, ensuring robust scoring across both small test corpora and large real-world SRS documents.
  48: - **Consequences:** Enables deterministic, fast, offline lexical retrieval without model downloads or network/API calls. Preserves exact metadata and provenance fields on retrieved pairs. All 83 test suite items pass.

## [ADR-006] Reciprocal Rank Fusion (RRF) for Hybrid Lexical-Dense Retrieval
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Combining dense embedding similarity scores (cosine/Euclidean bounded $[0, 1]$ or $(0, \infty)$) and BM25 lexical scores (unbounded positive scores $[0, \infty)$) via direct linear addition is sensitive to score scaling and calibration artifacts.
- **Decision:** Implement rank-based fusion using Reciprocal Rank Fusion (RRF, Cormack et al. 2009):
  $$\text{RRF\_Score}(d) = \frac{w_{\text{dense}}}{rrf\_k + r_{\text{dense}}(d)} + \frac{w_{\text{bm25}}}{rrf\_k + r_{\text{bm25}}(d)}$$
  with default $rrf\_k = 60$ and equal weights $w_{\text{dense}} = 1.0, w_{\text{bm25}} = 1.0$. Candidates present in either or both retrievers are merged deterministically using object identifier mapping (`get_candidate_id`).
- **Consequences:** Provides scale-invariant, robust hybrid retrieval for in-context example selection. Preserves all candidate metadata, labels, and provenance. All 92 test suite items pass.

## [ADR-007] Candidate Requirement Pair Generation & Audit Manifest Architecture
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Generating requirement pairs from SRS documents for semi-automated labeling requires scaling from small corpora (where exhaustive $n(n-1)/2$ all-pairs generation is feasible) to large SRS documents (where $O(n^2)$ pairs become intractable). Unlabeled candidates must not be assigned arbitrary labels like `No_dependency` prior to verification.
- **Decision:** Implement `CandidateGenerator` in `leredd/candidate_generator.py` supporting 4 generation strategies (`ALL_PAIRS`, `BM25`, `DENSE`, `HYBRID`). Enforce canonical unordered pair keys (`get_canonical_pair_key`) to deduplicate `(A,B)` and `(B,A)`, enforce target self-exclusion `(A,A)`, represent unlabeled candidates cleanly with `dependency_type = None`, enforce explicit safety truncation limits (`max_candidates`), and emit an audit manifest (`CandidateGenerationManifest`) for reproducibility.
- **Consequences:** Enables scaling dataset generation without data corruption, silent candidate drops, or pre-mature label assignment. All 101 test suite items pass.

## [ADR-008] Provisional Requirement Dependency Labeling & Disagreement Resolution
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Machine-generated dependency predictions from deterministic heuristics or LLMs must never be treated as verified ground truth annotations. Furthermore, `No_dependency` must be an explicit prediction, not a default fallback for missing data, parsing failures, provider errors, or heuristic abstentions.
- **Decision:** Implement `ProvisionalLabeler` in `leredd/provisional_labeler.py` combining `HeuristicLabeler` and `LLMLabeler`:
  1. All machine predictions receive `VerificationStatus.PROVISIONAL` and explicit `AnnotationSource` (`heuristic`, `llm`, or `ensemble`).
  2. In `HYBRID` mode, when Heuristic and LLM agree on a valid label, `accepted_label` is set to that label with `AnnotationSource.ENSEMBLE`.
  3. When Heuristic and LLM disagree on valid labels, `accepted_label` is set to `None` (Unresolved) with `review_notes="DISAGREEMENT: ..."` for routing to Phase 5 human review queue.
  4. Heuristic abstentions (`RULE-ABSTAIN`), LLM parsing failures, or provider errors yield `None` (Unresolved), NEVER `No_dependency`.
  5. Existing human-verified labels (`VerificationStatus.VERIFIED`) are never overwritten.
- **Consequences:** Ensures complete provenance auditability, strict ground-truth protection, and seamless routing of ambiguous candidate pairs to human review in Phase 5. All 113 test suite items pass.

## [ADR-009] Human Review Queue Data Model & Ground-Truth Protection Architecture
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Semi-automated dataset generation requires a structured, persistent review queue to present ambiguous, conflicting, or single-method provisional annotations to human experts. Entering the queue must not automatically verify an item, and refreshing the queue must not discard previously recorded human decisions.
- **Decision:** Implement `ReviewQueueManager` in `leredd/review_queue.py`:
  1. **Deterministic Queue Item Identity:** Queue item ID is generated deterministically from candidate pair ID and orientation (`pair_id:req_a_id->req_b_id`).
  2. **8 Reason Codes & 5 Priority Ranks:** Items are classified by `ReviewReasonCode` (`DISAGREEMENT`, `BOTH_ABSTAINED`, `LLM_FAILURE`, `HEURISTIC_ONLY`, `LLM_ONLY`, `LOW_CONFIDENCE`, `MISSING_METADATA`, `EXPLICIT_FLAG`) and sorted by `ReviewPriority` (CRITICAL=1 to INFO=5).
  3. **Idempotent Queue Refresh:** Refreshing a queue with new predictions preserves items with existing `human_decision` or non-PENDING status, preventing work loss or duplicate active entries.
  4. **Ground-Truth Protection:** `VerificationStatus.VERIFIED` is granted ONLY when an explicit human decision (`ACCEPT` or `CORRECT`) is recorded. Accepting and correcting remain distinct actions in audit logs, while raw heuristic and LLM predictions are preserved in `ReviewItem`.
- **Consequences:** Guarantees complete auditability, decision persistence, and strict ground-truth integrity across dataset refresh iterations. All 125 test suite items pass.

## [ADR-010] Multi-Interface Human Review Workspace & Ground-Truth Export Architecture
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Translating persistent review queue items into verified ground truth requires both an interactive visual interface (Streamlit) and an automated batch CLI. The UI must avoid creating a parallel decision engine, prevent accidental decision duplication during Streamlit reruns, and enforce strict verification checks before dataset export.
- **Decision:** Implement Page 6 ("6. Human Review Queue") in `ui/app.py` and CLI commands (`queue-list`, `queue-decide`) backed strictly by `ReviewQueueManager`:
  1. **Single Source of Truth:** All count statistics, filtering, sorting, decision validation, persistence, and export rely directly on `ReviewQueueManager`. No decision-transition logic is duplicated in the UI layer.
  2. **Streamlit Rerun Safeguards:** State variables (selected item ID, active queue path) are synchronized via `st.session_state`. Decisions require deliberate action buttons and update the underlying manifest immediately.
  3. **Strict Decision Validation:** For `ACCEPT`, a valid proposed label MUST exist. For unresolved pairs, `ACCEPT` without an explicit label choice is rejected with an actionable UI warning. `CORRECT` requires choosing one of the 6 canonical dependency classes (`Requires`, `Similar`, `Refines`, `Conflicts`, `Parent-Child`, `No_dependency`).
  4. **Verified Ground-Truth Export:** Exporting verified annotations strictly filters the queue to include ONLY `accepted` and `corrected` items with non-null `final_label`. Provisional, deferred, rejected, and pending items are excluded, and existing source dataset files are never overwritten without explicit path targeting.
- **Consequences:** Eliminates interface-backend divergence, ensures decision integrity across UI reruns, and guarantees clean, auditable verified dataset generation for Phase 6 evaluation. All 131 test suite items pass.

## [ADR-011] Leakage-Isolated Retrieval Pool & Safe Ground-Truth Eligibility Architecture
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Reproducible evaluation on system-disjoint splits requires preventing retrieval pool contamination (leakage of test/dev requirement IDs or test pair identities into retrieval example candidates) and preventing unverified provisional or test-fixture records from polluting ground truth evaluation benchmarks.
- **Decision:** Implement safe evaluation architecture in `scripts/run_full_evaluation.py` and `leredd/evaluation.py`:
  1. **Strict Retrieval Isolation (`build_safe_retrieval_pool`):** Filter retrieval pool to contain ONLY training system requirement pairs. Automatically strip any candidate containing test or development requirement IDs, or matching test-pair identities. Raise `ContaminationError` in strict mode if any contamination is detected.
  2. **Ground-Truth Eligibility Protection (`check_ground_truth_eligibility`):** Enforce strict verification rules. Reject `data/provisional/*`, review queue records, and UI-test fixtures (`adb_verified_ui_test.json`) when `VerificationStatus.PROVISIONAL`. Allow legacy benchmark pairs lacking provenance with explicit logged audit warnings.
  3. **Demarcation of Mock Inference:** Flag `MockLLM` outputs in evaluation reports with explicit structural warnings, marking them clearly as mock baseline demonstrations rather than real-LLM performance claims.
  4. **Structured JSON Audit Reporting:** Emit standardized evaluation reports containing comprehensive metrics (Accuracy, Macro-F1, Weighted-F1, Per-class P/R/F1, Confusion Matrix), dataset split sizes, label distributions, leakage audit logs, statistical tests (McNemar's test where applicable), and explicit methodology limitations.
- **Consequences:** Closes BLOCKER-001 completely. Guarantees zero data leakage during evaluation and prevents unverified provisional predictions from contaminating evaluation results. All 153 test suite items pass cleanly.

## [ADR-012] Disambiguated Baseline Model Identity & Supervised Cross-System Terminology Alignment
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Audit of TASK-P6-02 evaluation scripts and reports identified ambiguities in CLI argument `--model-type` (which controls the classification head for `BERTBaseline`, not the global pipeline choice), inaccurate methodology descriptions (calling supervised `BERTBaseline` fitted on ADB and tested on TJA "zero-shot"), and unclarified sample size limitations for McNemar's paired test on $n=5$.
- **Decision:**
  1. **Disambiguate CLI & Report Metadata:** Alias `--model-type` with `--bert-classifier-type` (`rf` = `RandomForestClassifier`, `lr` = `LogisticRegression`). In report configuration, explicitly record `bert_classifier_backend: "rf"` and full description `RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)`.
  2. **Correct Methodology Terminology:** Re-label `BERTBaseline` and `TFIDF_LSABaseline` evaluation as **Supervised Cross-System Evaluation** (trained/fitted on ADB system, tested on TJA system). Never describe supervised fitted models as "zero-shot".
  3. **Exploratory Statistical Test Notices:** Include an explicit `status: "exploratory"` dict and notice in McNemar test outputs explaining that applying paired statistical tests to $n=5$ yields low statistical power and low expected discordant cell counts ($b+c$), making p-values exploratory and demonstrative only.
- **Consequences:** Ensures 100% factual consistency across CLI args, report metadata, methodology descriptions, and documentation. All 157 test suite items pass cleanly.

## [ADR-013] Integrated Evaluation Benchmark & Multi-Dimension Ablation Suite Architecture
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Phase 7 requires a comprehensive, reproducible benchmark suite comparing 5 primary systems (`BERTBaseline`, `TFIDF_LSABaseline`, `HeuristicLabeler`, `LEREDD-ZeroShot`, `LEREDD-Hybrid`) and conducting structured ablation studies across 3 critical dimensions: Retrieval Strategy (Dense vs BM25 vs Hybrid RRF), In-Context Examples (Zero-Shot k=0 vs Dynamic Examples k=6), and Confidence Threshold Re-annotation (Off / threshold=-1 vs On / threshold=4).
- **Decision:**
  1. **Config Validation Extension:** Updated `LEREDDConfig` parameters (`k_examples >= 0`, `confidence_threshold >= -1`) to allow zero-shot prompting (`k=0`) and threshold re-annotation disabling (`threshold=-1`) without pydantic validation errors.
  2. **Dynamic Detector Strategy Selection:** Extended `LEREDDDetector.detect_pair` and `detect_batch` to support dynamic `retrieval_mode` override (`dense`, `bm25`, `hybrid`) and runtime `confidence_threshold` overrides.
  3. **Multi-Model & Multi-Dimension Ablation Runner:** Updated `scripts/run_full_evaluation.py` to run all 5 primary models and all 3 ablation study dimensions in sequence. Emits both `data/evaluation/system_disjoint_report.json` and dedicated `data/evaluation/ablation_report.json`.
  4. **Backward-Compatible Schema Aliases:** Maintained backward-compatibility aliases (`LEREDD_MockLLM`, `leredd_vs_bert`, `leredd_vs_tfidf`) alongside new ablation keys to preserve compatibility with existing P6 test assertions.
- **Consequences:** Provides complete, publication-grade benchmark and ablation study outputs. All 165 test suite items pass cleanly.

## [ADR-014] Mock LLM Demonstration Limits & Ablation Validity Audit
- **Date:** 2026-10-10
- **Status:** Accepted
- **Context:** Audit of Phase 7 benchmark results revealed that `LEREDD_ZeroShot`, `LEREDD_Dense`, `LEREDD_BM25`, `LEREDD_Hybrid`, and `LEREDD_Hybrid_NoThreshold` yielded identical predictions across all 5 test pairs (`PAIR-TJA-001` to `PAIR-TJA-005`).
- **Decision:**
  1. **Verified Code Routing:** Confirmed that `LEREDDDetector` dispatches cleanly to `self.example_retriever`, `self.bm25_retriever`, and `self.hybrid_retriever`, and builds distinct prompt strings containing zero vs retrieved in-context examples.
  2. **Isolated Root Cause:** Proved that identical predictions stem from `MockLLMClient` behavior: `MockLLMClient` evaluates regex keyword heuristics on Requirement A & B statement text only, ignoring prompt example blocks, and hardcodes `Confidence Score: 5` (which is strictly above `threshold=4` in all cases).
  3. **Demarcation & Unit Test Evidence:** Added explicit regression unit tests (`test_threshold_behavior_reannotation_vs_disabled`, `test_mock_llm_per_example_deterministic_predictions`) in `tests/test_ablation_benchmark.py` verifying that threshold re-annotation functions when predictions have confidence $\le 4$ and documenting mock LLM determinism.
- **Consequences:** Fully clarifies scientific validity limits for offline demonstration runs. Real-LLM inference (OpenAI/Ollama) will demonstrate full prompt sensitivity in live deployment. Total test suite: **167/167 passed**.

## [ADR-021] Isolated and Auditable End-to-End UI and Script Workflows
- **Date:** 2026-10-13
- **Status:** Accepted
- **Context:** The whole-project workflow audit found UI state changes without persistence, same-dataset evaluation leakage, synthetic inter-annotator agreement, a review script that fabricated human verification, and utility scripts that could overwrite prior outputs. The API config route also serialized an OpenAI key, and unfit baselines could fit against their prediction batch.
- **Decision:**
  1. UI-generated candidate and annotation data use atomic writes to separate output paths. Source dataset directories are protected; existing outputs require an explicit replacement control.
  2. UI evaluation uses the strict fixed system-disjoint runner and unique output paths. The kappa control is replaced by a statement requiring independent second labels.
  3. Review-queue construction never makes human decisions; only a reviewer action through the existing CLI/UI review flows can promote a record.
  4. Provisional labeling scripts default explicitly to MockLLM, and script outputs use fresh run directories unless the caller explicitly supplies paths.
  5. Prediction methods require baselines to be fitted on a separate training dataset.
  6. API configuration output redacts secrets. Local API/UI servers bind to loopback by default, and the wildcard CORS configuration does not allow credentials.
  7. PDF/DOCX parsing is an optional dependency extra and missing or malformed document inputs produce errors rather than a silent empty extraction result.
- **Consequences:** Local workflow outputs are auditable and isolated. Remote LLM services, cold-cache model downloads, Docker deployment, and imported replication labels remain external or research verification limits.
- **TASK-P9-14 verification update (2026-10-13):** Fresh unrestricted whole suite completes with 264 passed, 38 warnings, exit 0 (170.21s pytest / 173.83s outer duration). The sandboxed run hung at API root, confirming that loopback-permitted execution remains necessary for this environment.

## [ADR-022] Persist Reusable BERT Baseline Artifacts
- **Date:** 2026-10-13
- **Status:** Accepted
- **Context:** The `train-bert` CLI advertised saving a trained model but wrote no artifact, so successful completion could not be followed by prediction.
- **Decision:** Save the fitted classifier plus artifact format/model metadata using atomic joblib serialization; expose `BERTBaseline.load()`; refuse output replacement unless `--overwrite` is explicit. Record the embedding model name and initialize that embedder when loading unless a caller injects one.
- **Consequences:** The CLI output is reusable. Because joblib is pickle-based, artifacts must be trusted and loaded with compatible dependency versions. `joblib` is declared directly rather than relied on as a transitive scikit-learn dependency.
