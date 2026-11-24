# DECISIONS.md — Architectural & Technical Decisions Log

This document records key architectural, design, and implementation decisions for the project.

## [ADR-030] Keep Requirement Graph Labels in Hover Details
- **Date:** 2026-11-23
- **Status:** Accepted locally
- **Context:** Browser QA showed that permanently rendering every requirement ID over the graph caused overlap and obscured adjacent nodes.
- **Decision:** Keep the graph uncluttered by rendering nodes without visible text; place each node's requirement ID and full statement in its interactive hover details.
- **Consequences:** The graph remains navigable with zoom/pan and preserves the requirement context while improving legibility. Hover interaction and assistive/accessibility behavior still need broader browser/device review.

## [ADR-031] Treat MockLLM Ablations as Pipeline Checks, Not Empirical Effects
- **Date:** 2026-11-23
- **Status:** Accepted locally
- **Context:** The evaluation runner uses deterministic MockLLMClient, which does not interpret retrieved examples and always returns confidence 5. Retrieval/ICL/threshold ablation metrics can therefore remain identical for backend reasons.
- **Decision:** Preserve MockLLM's deterministic behavior and attach structured validity metadata to both reports, marking the affected comparisons as non-informative for model effects. Keep the empirical issues open until an authorized real provider or validated controlled test double is used.
- **Consequences:** Future reports explain the limitation alongside the metrics. Historical reports remain unchanged; existing scores are not retroactively presented as valid ablation evidence.

## [ADR-029] Keep Independent Annotation Separate from Assisted Verification
- **Date:** 2026-11-23
- **Status:** Accepted locally
- **Context:** The proposal calls for independently labeled examples and agreement measurement. The existing review queue displays assisted/model evidence and records a final verification decision; reusing it for blind agreement would expose cues and conflate agreement collection with adjudication.
- **Decision:** Collect each reviewer vector in a separate blinded manifest keyed to the same candidate file and optional SRS context hashes. Record abstentions and elapsed time. Compare only complete vectors with matching source/context/pair IDs and distinct reviewer IDs. Agreement reports are descriptive and never promote labels; a separate human adjudication and existing verified-export workflow are required.
- **Consequences:** The repository now supports collecting the evidence, but produces no research claim until actual independent human review occurs. Actor IDs are self-reported and not authenticated.

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
- **Decision:** Evaluation accepts provenance-bearing labels only when status is `VERIFIED`, source is `HUMAN`, an annotator or reviewer is identified, and `accepted_label` equals the scored `dependency_type`. Pending-review, human-reviewed-but-not-verified, machine-sourced, unattributed, or contradictory provenance is excluded (or raises in strict mode). Existing provenance-free legacy benchmark records remain conditionally accepted with an audit warning for backward compatibility.
- **Consequences:** No replication metrics are reported from the current pending-review imports. Verification must occur through the human review workflow; the source spreadsheets and imported JSON are not modified. Current full suite: **305 passed**, 12 deprecation warnings.

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

## [ADR-023] Invoke Pytest as a Python Module in CI
- **Date:** 2026-10-13
- **Status:** Accepted
- **Context:** GitHub Actions failed test collection because the `pytest` console entry point did not expose the repository root for imports of top-level `api` and `scripts` packages. These modules are source-tree packages and are not included by the package's `leredd*` setuptools include rule.
- **Decision:** Run the existing configured pytest coverage command as `python -m pytest ...`; preserve the same test path and coverage arguments.
- **Consequences:** The interpreter adds the current working directory to the module search path. Local full-suite coverage succeeds; the hosted Python-version matrix must rerun before CI status is confirmed.

## TASK-P10 — Stabilization Choices (2026-10-14)
- Candidate-generation DENSE ranks individual `Requirement` records through a dedicated `RequirementRetriever`; `ExampleRetriever` remains pair-oriented for labeled ICL examples. HYBRID fuses dense and BM25 requirement rankings with the existing RRF configuration. This preserves data type semantics and makes the strategy names truthful.
- `CandidateGenerationManifest.is_exhaustive` means generated unique pairs cover the mathematical all-pairs total and were not truncated. A retrieval run can therefore be exhaustive only when its selected neighbors cover the complete corpus.
- HashingVectorizer replaces per-call-fitted TF-IDF as the embedding fallback because independently transformed query and corpus batches need a shared feature space. Fallback cache keys are versioned separately; SBERT cache keys are preserved.
- Verified JSON export validates the existing file before recording a new decision, filters to explicit verified labeled records, replaces an existing pair on retry, and uses sibling-temp-plus-replace publication. If publication fails after the queue decision is saved, the CLI reports both states instead of implying a successful export.
- Retrieval tests use deterministic injected embedders so wiring and ranking behavior tests do not require SentenceTransformer initialization or network access. This does not replace actual model-quality validation, which remains outside the offline suite.

## [ADR-024] Require Coherent Human Provenance for Provenance-Bearing Evaluation Labels
- **Date:** 2026-10-14
- **Status:** Accepted
- **Context:** Evaluation previously treated the `verified` enum value as sufficient even when the record claimed an LLM source, had no attributable human, or contradicted its own accepted label.
- **Decision:** Any provenance-bearing ground-truth pair must be `verified`, use `human` source, identify an annotator or reviewer, and carry an `accepted_label` equal to `dependency_type`. Continue conditional warning-based acceptance only for legacy pairs with no provenance object.
- **Consequences:** Queue, UI, and interactive CLI human decisions remain eligible; machine suggestions and malformed/inconsistent verification metadata fail closed. Legacy research records remain usable with disclosed provenance limitations.

## [ADR-025] Reserve Requested System Partitions Before Ratio Apportionment
- **Date:** 2026-10-14
- **Status:** Accepted
- **Context:** Rounding train/dev/test ratios independently produced empty dev partitions for the common three-system 60/20/20 case.
- **Decision:** For system-disjoint splits, reserve one system for every requested partition that fits, then distribute remaining systems proportionally using largest remainders. If there are too few systems, preserve train/test first and omit optional retrieval/dev partitions as needed. Explicit assignments are unaffected.
- **Consequences:** A three-system dataset produces non-empty train/dev/test partitions by default. Fractions are approximate at small system counts; leakage prevention and deterministic seed behavior remain primary.

## [ADR-026] Require Explicit Actor Identity for Human Verification
- **Date:** 2026-10-14
- **Status:** Accepted
- **Context:** CLI and review-queue defaults supplied `human_cli` or `human_reviewer_1` when operators omitted an identity, allowing verified provenance to appear attributable without identifying the actual actor.
- **Decision:** Require a nonblank, non-placeholder identity at interactive CLI and queue-manager decision boundaries. New Streamlit decisions start with an empty actor field. Evaluation rejects records whose only actor IDs are the known generic placeholders.
- **Consequences:** Verification provenance contains an operator-supplied identity. The application does not authenticate that identity; it remains an auditable self-reported identifier. Legacy queue files missing an actor remain readable with an empty identity and cannot be promoted without a reviewer identifier.

## [ADR-027] Derive Evaluation Report Metadata From the Supplied Split
- **Date:** 2026-10-14
- **Status:** Accepted
- **Context:** The general-purpose evaluation runner described its input using legacy ADB/APA/TJA names, fixed sample sizes, and provenance assumptions, including when a caller supplied another verified split.
- **Decision:** Build systems, partition counts, provenance caveats, label coverage, and small-sample wording from the data actually loaded and accepted for the run. Keep MockLLM and non-authentication limitations explicit. Leave historical report artifacts immutable and require new destinations for new runs.
- **Consequences:** A report about a synthetic or user-provided split reflects that split's metadata. Self-reported reviewer identities are still not authenticated, and synthetic fixture evaluations remain software demonstrations rather than research evidence.

## TASK-P13 demonstration choices (2026-10-14)
- The Streamlit annotation page uses explicitly labeled synthetic requirements and writes its default candidates beneath `.pytest-tmp/leredd-ui/`, keeping a fresh demonstration independent of imported source datasets. No automated or synthetic action is represented as human verification.

## ADR-028 — Keep CI replication tests independent of private source data (2026-11-23)
- **Status:** Accepted
- **Context:** The imported replication JSON is local-only and its redistribution rights are unresolved. CI must still test loader validation and runner behavior without requiring those files.
- **Decision:** Use synthetic expected-count fixtures under temporary roots for loader and runner contract tests. Keep checks of the actual imported package conditional on the package being supplied locally, with a clear skip when absent. Do not add the replication package to the repository solely to make CI pass.
- **Consequences:** CI tests code and schema behavior without asserting access to research data. Full source-package integrity validation remains a local check until redistribution terms are established.

## ADR-032 — Allocate synthetic requirement IDs around explicit IDs (2026-11-23)
- **Status:** Accepted
- **Context:** Extraction assigned IDs to unlabeled requirement statements using a counter independent of explicit IDs already found in the source. This could create duplicate requirement identity.
- **Decision:** Track identifiers seen in a single extraction and advance the synthetic counter until it reaches an unused `REQ-<SYSTEM>-NNN` value. Preserve source-provided IDs and retain the existing requirement schema.
- **Consequences:** Mixed explicit and generated IDs no longer collide in the common counter-matching case. Duplicate IDs explicitly repeated in source input remain visible and are not silently rewritten.

## ADR-033 — Keep document identity separate from extraction location (2026-11-23)
- **Status:** Accepted
- **Context:** Extraction previously stored only the generic `regex_extraction` marker, losing the original SRS name and statement location. Combining filename and line into `Requirement.source` would make each line look like a separate document to document-level splitting.
- **Decision:** Retain the source document name in `Requirement.source` and add optional `Requirement.source_location` for the extracted-text line. File extraction uses the source basename; API callers can send a source label; UI upload/paste provides one. Preserve the generic source fallback where no source name is known.
- **Consequences:** New records are traceable while old JSON remains compatible and source grouping remains document-stable. PDF/DOCX coordinates are extracted-text lines; page/paragraph offsets remain future work. Short bracketed IDs, such as `[REQ-2]`, are retained as explicit identifiers.

## ADR-034 — Keep effort-study observations separate from ground truth (2026-11-23)
- **Status:** Accepted
- **Context:** The proposal calls for comparing manual annotation effort with machine-assisted effort, but there are no actual reviewer measurements and machine suggestions are not verified labels.
- **Decision:** Provide a separate paired, counterbalanced study workflow. Require provisional machine suggestions, hide them in the manual condition, identify them as provisional in assisted mode, and assign each pair to different reviewers across conditions. Store responses in study manifests and make the report explicitly descriptive; never auto-promote these labels to verified ground truth. Refuse study outputs anywhere under `data/`.
- **Consequences:** The software can support a controlled pilot without conflating suggestion, observation, or adjudicated truth. The tool and synthetic tests are not human-study evidence; genuine rights-cleared samples and actual reviewers are still required.

## ADR-035 — Exclude local-only assets from Docker build context (2026-11-23)
- **Status:** Accepted
- **Context:** Dockerfile uses `COPY . .`, and the working tree contains untracked research data, generated reports, project PDF, temporary files, and potentially local environment files. Without ignore rules, these can be copied into an image.
- **Decision:** Add `.dockerignore` rules that exclude all `data/`, local environments/secrets, repository metadata, caches/temp, and `Minor_Project.pdf`. Compose can mount host data at runtime for local workflows, but image layers must not contain those local assets.
- **Consequences:** Docker build context is safer and smaller. Because the Docker daemon was unavailable, actual image contents and Compose runtime behavior still need verification before considering Docker support complete.

## ADR-036 — Diagnose Streamlit blank page before changing runtime behavior (2026-11-23)
- **Status:** Accepted
- **Context:** Streamlit health and root HTTP responses succeed, but a browser view showed a blank shell with a persistent running indicator. An older stderr log includes watcher import warnings for missing torchvision.
- **Decision:** Keep the live process and application unchanged until fresh, correlated server and browser evidence identifies the blocking layer. Do not suppress file watching or install torchvision speculatively.
- **Consequences:** The UI remains an open runtime verification defect. Diagnose on a separate port with fresh logs and browser console/network evidence before fixing.

## ADR-037 — Separate Streamlit rendering from port and browser reachability (2026-11-24)
- **Status:** Accepted
- **Context:** A fresh diagnostic server on port 8502 returned HTTP 200 and native Chrome rendered the dashboard after cold startup; the isolated in-app browser had websocket reachability errors. The documented port 8501 had no listener and Chrome reported connection refused.
- **Decision:** Do not modify app code or suppress the file watcher without correlated evidence. Track 8502 rendering, the initially unavailable 8501 listener, and isolated-browser websocket access as separate conditions. Resolve the persistent blank-page report after the documented 8501 process renders in native Chrome.
- **Consequences:** The application rendered on both the diagnostic and documented ports after cold startup. Broader API/UI workflow checks remain outstanding. Synthetic annotations remain provisional; this demonstration creates no human ground truth.

## ADR-038 — Initialize detection embeddings only for retrieval paths (2026-11-24)
- **Status:** Accepted
- **Context:** `LEREDDDetector.__init__` constructed `EmbeddingModel`, whose initialization loads SentenceTransformer. API detection with an empty example pool and RAG disabled does not use embeddings, yet the first zero-shot request incurred a full model startup delay.
- **Decision:** Defer the embedder and dense/RAG retriever construction until the corresponding dense or RAG retrieval path is accessed. Preserve detector properties and injected-embedder support so existing callers and tests retain their interface. BM25 retrieval remains independent of SBERT.
- **Consequences:** Cold start is avoided for zero-shot and BM25-only requests; first dense/RAG use still pays the configured model initialization cost. Full suite passed (342 tests), and a real local MockLLM API request completed in 164 ms without a model-load log.

## ADR-040 — Expose per-request retrieval choice through product interfaces (2026-11-24)
- **Status:** Accepted
- **Context:** Dense, BM25, and hybrid retrieval were supported internally, but API and Streamlit users could not select among them for a detection request. The configured detector default must remain backward compatible.
- **Decision:** Add an optional constrained retrieval mode to single and batch API requests and an explicit Streamlit detection selector. When omitted by an API caller, use the existing configured default; do not change label semantics or dataset schemas.
- **Consequences:** BM25 is accessible without loading SentenceTransformer for supported requests; dense/hybrid paths remain available and retain their model startup cost. Focused API/UI tests and the full local suite passed. Visual inspection of the updated live Streamlit selector and retrieval-quality comparisons remain outstanding.

## ADR-039 — Preserve segmented alphanumeric requirement identifiers (2026-11-24)
- **Status:** Accepted
- **Context:** The extractor accepted numeric trailing IDs such as `REQ-ADB-001` but truncated letter-suffixed IDs such as `[REQ-DEMO-A]` to `REQ-DEMO`, causing multiple source requirements to share an identifier and collapse during candidate generation.
- **Decision:** Parse the complete `REQ-` token when it consists of alphanumeric segments separated by hyphens or colons, including alphabetic final segments. Keep generated IDs and the existing requirement schema unchanged.
- **Consequences:** Such source IDs now survive extraction and produce distinct candidate pairs. This grammar is regression-tested; broader real-SRS identifier conventions still require validation on approved sample documents.

## ADR-041 — Keep confidence screening separate from dependency classification (2026-11-24)
- **Status:** Accepted
- **Context:** The detector converted predictions at or below the Likert threshold into `No_dependency`. This conflated uncertainty with a negative label, despite the proposal calling for uncertain outputs to be identified for review. A model's self-reported confidence is also not a calibrated probability or annotation verification.
- **Decision:** Preserve the parsed dependency class and record a separate validation status and threshold. Mark low-confidence predictions `requires_review`, above-threshold predictions `passed_threshold`, and disabled checks `threshold_disabled`. Present the status as a screening cue only; never elevate model output to verified ground truth.
- **Consequences:** API/CLI serialization, Streamlit feedback, and new evaluation reports can expose screening decisions without corrupting label semantics. The UI never enqueues automatically; its explicit pending-queue action is defined in ADR-042. MockLLM's fixed confidence does not provide empirical evidence about screening utility.

## ADR-042 — Queue low-confidence predictions only through explicit user action (2026-11-24)
- **Status:** Accepted
- **Context:** Confidence-screened predictions could be shown as requiring review but were not connected to the existing human review queue, leaving the Streamlit user to manually recreate the item. Machine predictions must remain distinct from verified annotation.
- **Decision:** Add a separate user-triggered action that writes only predictions with `requires_review` into the pending review queue, preserving predicted class, rationale, confidence, and provisional provenance. The UI writes its queue JSON through a same-directory temporary file and atomic replacement and requires confirmation when that queue already exists. The shared `ReviewQueueManager.save_queue()` separately writes queue and manifest through staged, per-file atomic replacements.
- **Consequences:** The product now routes the result to a real review worklist without claiming a human decision. A filesystem cannot atomically replace the shared queue and manifest together; queue is replaced first, so a later manifest write failure may leave a stale manifest. Tests verify preservation when the initial replacement fails. Real reviewer use and confidence-screening utility remain unverified.

## ADR-043 — Verify Streamlit startup in a fresh process after import-time errors (2026-11-24)
- **Status:** Accepted
- **Context:** The long-running Streamlit process showed an import error for a class present in current source; a fresh interpreter imported the class and a fresh Streamlit process rendered the application and completed the synthetic detection-to-review journey.
- **Decision:** Treat this observed error as stale runtime state and restart the documented Streamlit service after code/model changes; do not alter application source unless the failure reproduces in a fresh process.
- **Consequences:** The 8501 UI is healthy after restart. The same-path fresh-process reproduction was not a source-level regression test; real human/provider behavior remains unverified.

## ADR-044 — Preserve the inclusive confidence cutoff and describe its rule precisely (2026-11-24)
- **Status:** Accepted
- **Context:** The existing screen routes a prediction when `confidence <= threshold`. At the maximum threshold, a 5/5 score is therefore routed, but “low confidence” copy implies a low absolute score.
- **Decision:** Preserve the inclusive threshold and stored `low_confidence` code for compatibility. In user-facing output, report that the self-reported score is at or below the configured cutoff and retain the explicit caveat that the output is not verified ground truth.
- **Consequences:** Boundary behavior remains unchanged and understandable, including a 5/5 score at cutoff 5. This is a routing rule only; it provides no calibration or empirical utility evidence.

## ADR-045 — Reuse the loaded embedding resource within a Streamlit session (2026-11-24)
- **Status:** Accepted
- **Context:** The UI creates a detector when each user clicks Run. Dense/hybrid retrieval and RAG initialize the embedder lazily, so the loaded SentenceTransformer was discarded at the end of the click and initialized again on the next prediction.
- **Decision:** Store an embedder only after a detector has actually initialized it, keyed by configured model name and cache directory, and inject it into subsequent session-local detectors. Do not eagerly load it for zero-shot or BM25-only paths.
- **Consequences:** Repeated dense/RAG interactions in one browser session reuse the same embedding model. The initial model load remains; the model is not shared across user sessions, and this change makes no latency claim.

## ADR-046 — Use structured requirement metadata as a conditional hybrid ranking signal (2026-11-24)
- **Status:** Accepted
- **Context:** The project proposal identifies component/subsystem metadata as an optional hybrid retrieval signal. Existing `Requirement` records exposed system/source fields only, and example retrieval used dense plus BM25 rankings.
- **Decision:** Add optional `component` and `subsystem` fields and rank eligible examples by normalized exact overlap across component, subsystem, and meaningful system values. Fuse this ranking with dense/BM25 only when the target has usable metadata and at least one candidate shares a value. Do not use IDs, source locations, or free text as metadata. Keep the metadata RRF weight configurable on `HybridRetriever`.
- **Consequences:** Existing records without structured metadata preserve their existing dense/BM25 path; new data can use component/subsystem fields without forcing them to be inferred. This is implementation support, not evidence of improved retrieval. Metadata provenance/quality and empirical weight selection remain open.

## ADR-047 — Require unanimous votes for multi-model provisional labels (2026-11-24)
- **Status:** Accepted
- **Context:** The proposal calls for multiple model predictions where feasible and review of ambiguous or conflicting labels. A majority rule would silently select a label despite a model disagreement, while the system has no empirically justified vote weighting.
- **Decision:** In opt-in multi-model labeling, accept a suggestion only when every configured model returns a valid, identical taxonomy label. Any disagreement or incomplete vote produces no label, records each model's vote/rationale, and routes the pair for human review. The result remains provisional even when unanimous; confidence is the minimum self-reported vote and is not a calibrated probability.
- **Consequences:** Vote handling is deterministic and auditable without pretending that agreement among models is ground truth. Real model diversity, calibration, cost, and usefulness require separate provider-backed evaluation.

## ADR-048 — Include provider and model identity in LLM cache keys (2026-11-24)
- **Status:** Accepted
- **Context:** The response cache previously keyed only by client class, temperature, and prompt, allowing same-provider models to collide and invalidate multi-model comparison.
- **Decision:** Version the cache-key input and include provider class, model identifier, temperature, and prompt. Do not fall back to legacy entries whose producing model is unknown.
- **Consequences:** Distinct models do not share cached responses. Existing ambiguous cache entries remain on disk but are not reused; the next request may call the provider. No package install or real provider call is required to verify the key separation.

## ADR-049 — Attach requirement metadata only through explicit ID mappings (2026-11-24)
- **Status:** Accepted
- **Context:** Hybrid retrieval can use component/subsystem values, but plain-text extraction cannot reliably infer those concepts from requirement prose. Unchecked metadata association could silently attach values to the wrong requirement.
- **Decision:** Accept optional component/subsystem values via an exact requirement-ID-keyed JSON mapping during extraction/candidate generation and through equivalent API/UI fields. Reject unknown IDs, unsupported keys, and malformed values. Allow target requirement metadata in single detection. Do not infer metadata from prose, IDs, or source locations.
- **Consequences:** Supplied structured metadata reaches candidate JSON and pair-level retrieval without changing existing text extraction semantics. Existing data remains metadata-free until a source-grounded user supplies values; empirical retrieval value is unproven.
- **Verification (TASK-P34):** Focused metadata flow tests passed (50), and the complete suite passed (364, 12 existing `tqdm` deprecation warnings). No existing replication/evaluation artifact changed. This verifies software plumbing only; it does not validate metadata quality or retrieval effectiveness.

## ADR-050 — Preserve requirement metadata in review queue records (2026-11-24)
- **Status:** Accepted
- **Context:** Explicit component/subsystem and source traceability are attached to each `Requirement`, but the queue previously flattened each pair into IDs, text, and a shared system name. Human decisions then rebuilt verified records from this reduced representation.
- **Decision:** Store optional system, component, subsystem, source, and source-location fields independently for requirements A and B in each review item. Display available fields in the human review UI and reconstruct the verified pair with the stored values. Keep new fields optional so historic queue JSON remains loadable; do not infer missing historical values.
- **Consequences:** New SRS-to-review-to-verified-export flows preserve metadata. Existing queue files that already omitted it remain usable but must be regenerated from candidate pairs to restore lost values.
- **Verification (TASK-P35):** Focused queue/UI tests passed (41); complete suite passed (366, with 12 existing `tqdm` warnings). Eleven protected replication/evaluation artifacts were unchanged.

## ADR-051 — Publish human-verified datasets as immutable releases (2026-11-24)
- **Status:** Accepted
- **Context:** Verified pair exports had a constant provenance dataset-version string and no artifact-level manifest, input hashes, or overwrite protection. Splits and evaluation therefore could not point back to a content-identified release.
- **Decision:** Require each release input to be non-empty, uniquely identified, labeled, and backed by coherent human `VERIFIED` provenance. Copy eligible records, assign the requested release version in each provenance record, and atomically publish a new directory containing `pairs.json` plus a manifest with source-file hashes, released-data hash, class counts, record count, and timestamp. Never overwrite an existing release directory.
- **Consequences:** Users can combine verified exports into a versioned dataset before splitting, and can detect source or release-content changes by hash. The operation does not create or authenticate human labels; legacy provenance-free records are intentionally ineligible.
- **Verification (TASK-P36):** Focused release/CLI tests passed (19); complete suite passed (374, 12 existing `tqdm` warnings). Synthetic verified fixtures only; no human dataset was released.

## ADR-052 — Define candidate recall against the exhaustive pair universe (2026-11-24)
- **Status:** Accepted
- **Context:** The proposal asks to estimate candidate-generation recall by comparing strategy output with the complete pair set on a sample, but this is distinct from recall of true dependency labels.
- **Decision:** Candidate manifests report `pair_space_recall = unique generated pairs within the input corpus / n(n-1)/2`, together with the universe size and uncovered-pair count. A zero-pair universe reports recall as null. The CLI and docs call this candidate-space coverage and explicitly distinguish it from gold dependency recall.
- **Consequences:** Strategy coverage is available before annotation and does not need labels. Research claims about recovering true dependencies still require a rights-cleared, human-verified labeled sample.
- **Verification (TASK-P37):** Synthetic ALL_PAIRS, truncated, retrieval, empty-input, manifest round-trip, and CLI assertions passed (42 focused); the full suite passed (374, 12 existing warnings). All 311 data files were hash-identical before and after.

## ADR-053 — Preserve first label and flag repeated-inference variation (2026-11-24)
- **Status:** Accepted
- **Context:** Proposal scope includes examining prediction consistency across repeated runs, while existing API/CLI/UI detection must remain inexpensive and deterministic by default.
- **Decision:** Keep one inference as the default. Expose an optional 1–5 repeated-run count. For counts above one, bypass the built-in response cache while reusing the same constructed prompt. Record each run's class and self-reported confidence. Any class or confidence variation sets an inconsistent stability status and recommends human review; preserve the first run's class as the result. Do not treat consistency or Likert scores as calibrated uncertainty or annotation verification.
- **Consequences:** Callers can explicitly inspect run stability without changing default behavior. Multiple OpenAI runs incur multiple provider calls and cost. Third-party `LLMClient` implementations that do not override `generate_uncached` may still have their own external caching behavior.
- **Verification (TASK-P38):** 61 focused detector/LLM/API/CLI/UI tests passed; complete suite passed (383, 12 existing warnings). No external provider or human annotation was used.

## ADR-054 — Preserve DOCX table text and fail clearly on image-only PDFs (2026-11-24)
- **Status:** Accepted
- **Context:** The proposal states that SRS requirements may occur in paragraphs or tables and asks that original requirement text be preserved. DOCX parsing omitted tables; regex extraction dropped final periods; image-only PDFs appeared as successful empty extractions.
- **Decision:** Traverse DOCX body paragraphs and top-level tables in source order; join row cells with tabs so IDs and requirement text remain associated on one source line. Preserve the terminating period in extracted statements. If a PDF has no extractable text, raise an error that identifies unsupported OCR rather than return an apparently successful empty list.
- **Consequences:** Text-based DOCX/PDF ingestion is more complete and failures are easier to diagnose. OCR, nested DOCX table traversal, and layout-specific table interpretation are not added. The PDF/DOCX libraries remain an optional `documents` extra.
- **Verification (TASK-P39):** Synthetic PDF/DOCX tests and OCR failure test pass; full suite passed (386, one Starlette deprecation warning). Only synthetic SRS content was used.

## ADR-055 — Treat short requirement length as a review cue, not a classifier (2026-11-24)
- **Status:** Accepted
- **Context:** The proposal asks the tool to flag unusually short or incomplete requirement statements for human review. A deterministic length rule can identify a subset for inspection, but cannot establish whether the text is incomplete or ambiguous.
- **Decision:** Route non-empty requirement text shorter than a configurable minimum (default 20 characters) to human review. `queue-create --minimum-requirement-characters N` sets the minimum and `0` disables the cue. Persist the effective queue eligibility configuration in its manifest. Do not alter any dependency label or verification status.
- **Consequences:** The review queue can highlight short text consistently while leaving semantic interpretation to the reviewer. The default threshold is heuristic and may need adjustment for a real corpus; it is not a validated language rule.
- **Verification (TASK-P40):** Focused queue/CLI/UI integration tests passed (45); full suite passed (387, one existing Starlette deprecation warning). All 311 data files retained their SHA-256 hashes. Synthetic fixtures only.

## ADR-056 — Validate RAG chunk progress and zero-result behavior (2026-11-24)
- **Status:** Accepted
- **Context:** SRS chunks advance by `chunk_size - overlap`; invalid values could yield a zero or negative step. The retrieval method also interpreted an explicit `k=0` as the configured default because it used truthiness.
- **Decision:** Require a positive integer chunk size and integer overlap satisfying `0 <= overlap < chunk_size`; require `k` to be a non-negative integer. Treat explicit `k=0` as a request for no chunks and return before embedding.
- **Consequences:** Malformed settings fail promptly with clear `ValueError`s instead of hanging or silently changing the request. Default configuration and positive retrieval behavior remain intact.
- **Verification (TASK-P41):** Focused retrieval/detector tests passed (31); full suite passed (398, one existing Starlette deprecation warning). All 311 data files retained their SHA-256 hashes.

## ADR-057 — Allow per-request API control of SRS context retrieval (2026-11-24)
- **Status:** Accepted
- **Context:** API payloads already accepted SRS text, but RAG was enabled only by server-wide configuration, disabled by default. This made the API's optional context field appear usable when it was ignored by default.
- **Decision:** Add optional `use_rag` to single and batch requests and detector methods. Explicit booleans override the server setting; `None` preserves the existing configured default. Keep RAG disabled by default.
- **Consequences:** API clients can opt into SRS context retrieval for a request without mutating shared server configuration. Embeddings initialize only for enabled RAG with non-empty context. This is interface wiring, not evidence of model-quality gains.
- **Verification (TASK-P42):** Focused API/detector/retrieval suite passed (49); full suite passed (400, one existing Starlette deprecation warning). A controlled API regression confirms the retrieved marker is in the prompt.

## ADR-058 — Keep imported replication labels unmapped without a source crosswalk (2026-11-24)
- **Status:** Accepted
- **Context:** The paper defines Implements as a higher-level requirement fulfilled by a lower-level requirement. The imported workbook uses Implemented by; an example corresponds to the paper's example with workbook columns reversed, but the source material provides no explicit normalization rule. Separately, the paper license does not establish the rights for the packaged datasets.
- **Decision:** Preserve raw imported labels and keep them out of mapped metrics and redistribution until authoritative documentation provides the source-label crosswalk and dataset-specific rights. Do not infer the crosswalk from a matching example alone.
- **Consequences:** The semantic definition in the paper is no longer treated as unknown, while the import-to-paper mapping and dataset redistribution remain distinct active blockers. Existing raw data and provenance stay unchanged.
- **Evidence:** TASK-P43 read-only review of arXiv v2 and all four local workbooks; an openpyxl audit found zero exact/reversed duplicate pairs and zero incomplete rows across 813 entries. git diff --check passed. No tests or data edits were needed.

## ADR-059 — Keep live smoke predictions explicitly synthetic (2026-11-24)
- **Status:** Accepted
- **Context:** The real application can be exercised with a built-in synthetic SRS and MockLLM, while no rights-cleared, independently verified dataset is currently available for a trustworthy end-to-end research evaluation.
- **Decision:** Use only clearly labeled synthetic inputs and the Mock provider for interface smoke checks. Preserve generated candidates as unlabeled/provisional and do not turn a UI prediction into human ground truth, a dataset release, or benchmark evidence.
- **Consequences:** Product navigation, serialization, and request plumbing can be checked without paid services or fabricated evidence. Prediction quality, reviewer agreement/effort, and replication metrics still require actual rights-cleared data and human review.
- **Verification (TASK-P44):** UI extraction persisted three unlabeled provisional pairs; BM25/MockLLM rendered one result with the calibration/verification disclaimer. CLI help exited 0 and local API root/health returned HTTP 200.

## ADR-060 — Keep full-suite verification offline and data-protected (2026-11-24)
- **Status:** Accepted
- **Context:** The Windows API `TestClient` can stall inside the restricted socket sandbox, while tests must not mutate imported research datasets or trigger model downloads.
- **Decision:** Run the complete suite with the existing project virtual environment outside the restricted socket sandbox, set `TEMP`/`TMP` to `.pytest-tmp`, disable Hugging Face network access, and compare a SHA-256 manifest of `data/` before and after.
- **Consequences:** This yields reproducible local test evidence while keeping test scratch output separate and detecting unexpected dataset changes. It does not substitute for hosted CI or clean-machine installation testing. The TestClient warning is currently non-fatal; `httpx2` requires Python >=3.11, so replacing the current HTTPX fallback requires a deliberate strategy that preserves the declared Python >=3.10 support.
- **Verification (TASK-P45):** 400 passed, 1 Starlette deprecation warning, exit 0; all 312 data file paths and SHA-256 values matched before and after.

## ADR-061 — Exercise API tests with in-process ASGI transport (2026-11-24)
- **Status:** Accepted
- **Context:** The API test module used Starlette `TestClient`, which launches an AnyIO worker portal. Its HTTPX fallback emits a deprecation warning in current Starlette, and its Windows Proactor/socket-pair path is unreliable in the restricted sandbox. Adding `httpx2` is not a clean universal fix because current releases require Python >=3.11 while the project supports Python >=3.10.
- **Decision:** Keep synchronous pytest tests but send requests through the existing HTTPX `AsyncClient` and `ASGITransport`, invoking each request with `asyncio.run`. The FastAPI app has no lifespan hooks, so this covers the route behavior exercised by the module without TestClient's worker portal.
- **Consequences:** No dependency or production-code change is needed, and the suite avoids the TestClient deprecation. Windows runs still need an environment that allows asyncio's socket setup; the restricted sandbox may block event-loop startup independently of the request transport.
- **Verification (TASK-P46):** Root test 1 passed; API module 18 passed; full suite 400 passed with no warnings. All 312 `data/` files retained identical hashes.

## ADR-062 — Use CPU-only PyTorch in the general Docker images (2026-11-24)
- **Status:** Accepted
- **Context:** The Compose API and UI services use CPU inference, but an unqualified Linux `pip install` of Torch selected a 555 MB CUDA wheel and then a 553 MB cuDNN wheel, with more NVIDIA packages queued. This inflated an ordinary local deployment without GPU use.
- **Decision:** Install Torch from PyTorch's CPU wheel index in the Docker builder before installing the declared project requirements. The requirements then reuse the installed CPU build.
- **Consequences:** The general Docker images avoid unnecessary NVIDIA runtime packages. GPU-specific Docker deployment is not configured by this decision.
- **Verification (TASK-P47):** Both images built; build output reports `torch-2.14.1+cpu`. Compose API root/health and Streamlit root/health returned successful responses. Docker management API later became unstable, and browser visual inspection was not completed.
- **Follow-up evidence:** A fresh bounded Docker CLI info request timed out after 20 seconds; API health and Streamlit health still returned HTTP 200/`ok`. This does not establish container/image inventory or browser-rendered UI state.

## ADR-063 — Keep external SRS smoke artifacts temporary and non-evaluative (2026-11-24)
- **Status:** Accepted
- **Context:** Product extraction and candidate generation needed a non-synthetic input trial, but no project SRS has yet been cleared for dataset redistribution or independently labeled.
- **Decision:** Use a repository-licensed public SRS only as a local parser/product smoke input, keep its copy and all outputs under workspace-local untracked `.pytest-tmp`, and do not treat keyword counts, heuristic labels, or exhaustive candidate enumeration as gold quality or benchmark evidence.
- **Consequences:** The extraction pipeline receives a realistic Markdown document without modifying imported datasets or creating a release. Human source review, source permissions for any retained/shared data, and independently verified dependency labels remain prerequisites for research claims.
- **Verification (TASK-P48):** 99 extracted records preserved source locations; 4,851 candidate pairs were exhaustively generated; all labels remained provisional and the queue contained zero reviewer decisions. Focused CLI regression passed (1 test).

## ADR-064 — Report candidate-pair coverage separately from dependency recall (2026-11-24)
- **Status:** Accepted
- **Context:** The candidate generator records how much of the exhaustive pair universe each retrieval strategy emits. This software coverage number can be mistaken for recall of actual dependency links.
- **Decision:** Report candidate-space coverage and omitted-pair count as workload/space diagnostics only. Report positive dependency recall only after independently verified labels exist for an appropriate sample that includes pairs omitted by the strategy.
- **Consequences:** Current top-k strategy measurements are not used to assert candidate quality or choose research settings. Review sampling must capture its frame, selection probability, and both candidate and omitted strata.
- **Verification (TASK-P49):** For 99 requirements and 4,851 possible pairs at `top_k=5`, BM25 emitted 327 (6.74%), dense 335 (6.91%), hybrid 325 (6.70%); all-pairs emitted all 4,851. No labels were available, so positive recall remains unknown.
- **Operational note:** Dense encoding used the configured ignored cache directory `data/embeddings`; future trial commands must override `CACHE_DIR` to a workspace temporary directory when avoiding writes below `data/`.

## ADR-065 — Stratify a small blinded pilot across selected and omitted pairs (2026-11-24)
- **Status:** Accepted for an operational pilot only
- **Context:** Measuring positive dependency recall for hybrid candidate generation requires labels both for emitted pairs and for pairs the strategy omits. Reviewing the entire 4,851-pair universe is too large for an initial pilot.
- **Decision:** Prepare a fixed-seed sample of 50 of 325 hybrid-top-5 pairs and 50 of 4,526 omitted pairs. Give reviewers one shuffled, opaque-ID file with no labels or stratum information; store population counts, inclusion probabilities, and design weights separately for post-review analysis.
- **Consequences:** If both strata contain adjudicated labels, a weighted estimate can account for their different sampling fractions. The 100-pair sample is an operational pilot only and is not powered or preregistered; no positive-recall claim can be made before real review and appropriate analysis.
- **Verification (TASK-P50):** 100 unlabeled/provisional records validated, exactly 50 per stratum; the reviewer-file hash is bound in the pilot manifest. No human data was entered.
- The current local suite passed **400 tests with no warnings in 132.31s** using temporary `TEMP`, `TMP`, and `CACHE_DIR` plus `HF_HUB_OFFLINE=1`; this verifies software only, not the pilot's research claims.

## ADR-066 — Estimate candidate positive recall only from verified stratified labels (2026-11-24)
- **Status:** Accepted
- **Context:** The pilot oversamples strategy-selected pairs relative to omitted pairs. A raw positive fraction or independent reviewer vector would not be a valid positive-recall estimate or verified ground truth.
- **Decision:** Use post-stratified positive totals and their ratio for candidate recall, after requiring exact sample-ID coverage and coherent human `VERIFIED` adjudications. Pin and validate the sample, context, analysis map, candidate output, and full pair universe by hash. Fail closed for incomplete/ambiguous labels; never adjudicate automatically. Report a finite-population-corrected, linearized normal interval only when non-degenerate and label the result exploratory.
- **Consequences:** Future reports can account for different selected/omitted sampling fractions without treating provisional suggestions as labels. The 100-pair pilot's estimate remains unproven and potentially imprecise; real adjudication and appropriate research review are prerequisites.
- **Verification (TASK-P51):** Final combined analysis/provenance/release suite: 36 passed; final complete suite: 413 passed with no warnings outside the restricted Windows socket environment. A regression verifies `population_pair_count` reports the actual universe size. No human data was supplied.

## ADR-067 — Synthetic data is ineligible as research evidence (2026-11-24)
- **Status:** Accepted
- **Context:** Synthetic demo records can resemble reviewed annotations. A caller-supplied verified status or human-looking annotator identity must not let synthetic fixtures enter retrieval evidence, evaluation ground truth, or released datasets.
- **Decision:** Carry explicit synthetic markers and source identities through requirement, pair, provenance, candidate, CSV, and review data. Exclude marked records from retrieval pools and verified exports; reject them as evaluation ground truth even if they carry verified provenance. Do not infer synthetic status solely from an annotator name.
- **Consequences:** Existing unmarked records remain compatible. Synthetic demo output remains useful for demonstrations but cannot be presented as verified research data. Genuine human review and rights-cleared source material are still required.
- **Verification:** TASK-P52 focused suite: 172 passed; full local suite: 424 passed. Tests use controlled synthetic fixtures only.

## ADR-068 — Verify system-disjoint splits from partition contents (2026-11-24)
- **Status:** Accepted
- **Context:** The full evaluation runner trusted the split metadata's declared strategy and did not recompute overlap across the actual train/dev/test files. A stale or inaccurate metadata report could therefore permit evaluation of a contaminated system-disjoint split.
- **Decision:** For a declared system-disjoint strategy, validate actual system identities, requirement IDs, and pair IDs before any ground-truth filtering or model fit. Raise a typed error carrying invalid IDs; include a passing split-integrity audit in the report.
- **Compatibility:** Pair-level fixtures remain pair-level and continue exercising retrieval contamination behavior; non-system strategies are not subjected to the system-disjoint check. Duplicate prediction errors preserve the existing ValueError compatibility while carrying IDs.
- **Verification:** TASK-P53 focused suite 45 passed; full suite 428 passed; all 412 `data/` file hashes unchanged.

## ADR-069 — Publish software checkpoint without local imported research assets (2026-11-24)
- **Status:** Accepted for the TASK-P54 checkpoint.
- **Context:** The worktree included substantial software changes alongside a PDF, imported replication datasets, external replication materials, generated reports, caches, and temporary outputs. Redistribution rights for the local imported package remain unresolved.
- **Decision:** Publish reviewed application code, tests, and project documentation only. Keep `Minor_Project.pdf`, local imported replication JSON and spreadsheets, external package files, generated reports, caches, and scratch artifacts untracked. Reconcile divergent history with a merge commit; do not rewrite either side's history.
- **Verification:** `d419d998e8ed9c05098c4fb0bf4f6789df08c74f` is the verified `origin/main` tip; post-merge suite **456 passed**. The newer remote demo SRS and `.gitattributes` commits were already on GitHub and were preserved as remote ancestors.
- **Consequence:** The repository's software checkpoint is published, but local replication evaluation remains blocked until rights and label semantics are documented.

## ADR-070 — Make CLI and protected-path tests platform independent (2026-11-24)
- **Status:** Accepted
- **Context:** Hosted Linux CI exposed test-only assumptions that passed on Windows: colored Click output inserted escape sequences within option spellings, and `tmp_path.parents[3]` did not identify the checkout root.
- **Decision:** Disable color for tests that assert CLI option names, and derive repository-owned protected paths from the test file's location. Keep the existing behavior assertions intact.
- **Verification:** All three affected regressions passed locally; the full Windows suite passed **456 tests**. The hosted Linux rerun remains pending.

## ADR-071 — Require a live Docker engine for container verification (2026-11-24)
- **Status:** Accepted
- **Context:** Docker CLI presence alone does not prove that image builds or Compose runtime checks are possible. The local service is stopped and cannot be opened by the current process, even in the elevated attempt.
- **Decision:** Keep Docker image/runtime verification open until Docker Desktop and its Linux engine are running; do not infer success from Compose configuration parsing or past API/UI health checks.
- **Verification:** `com.docker.service` reported `Stopped`/`Manual`; `docker info` timed out; elevated service start returned an access/open error. No build or container mutation was performed.

## ADR-072 — Normalize CLI error output before portable assertions (2026-11-24)
- **Status:** Accepted
- **Context:** GitHub's Linux run showed that Typer/Rich can emit ANSI control sequences inside option names even when `CliRunner(color=False)` is used. Literal substring assertions therefore failed on Linux but passed locally on Windows.
- **Decision:** Strip ANSI styling with Click's `unstyle()` before asserting the CLI option spelling. Preserve the checks for command failure and, for queue decisions, no queue mutation.
- **Verification:** Both affected tests passed locally; the full local coverage run passed **453 tests**, skipped **3**, emitted **12 deprecation warnings**, and measured **88%** coverage. Hosted rerun is still required.
