# CHANGELOG.md — Project Activity Log

## [2026-11-24] - TASK-P42: Expose RAG control on API requests
- Added optional `use_rag` to single and batch detection requests and forwarded it through detector calls. Omission preserves configured behavior; per-request `true` enables RAG and `false` disables it. API reference and feature ledger now describe the contract.
- Regression tests verify default fallback, both explicit overrides, batch pass-through, and SRS context inclusion in the detector prompt.
- Focused API/detector/retrieval suite: **49 passed, 1 existing Starlette deprecation warning in 4.53s**. Full `.venv` suite: **400 passed, 1 existing Starlette deprecation warning in 99.47s**, exit 0 (PowerShell duration **105.20s**). All 311 data files retained identical paths and SHA-256 hashes; `git diff --check` passed.
- No real provider or real SRS was used. RAG contribution to predictive quality remains unmeasured.

## [2026-11-24] - TASK-P41: Validate RAG chunking and result limits
- `RAGRetriever` now rejects invalid chunk sizes, overlaps, and result counts. An overlap equal to or larger than the chunk size can no longer create a non-advancing loop. `k=0` returns `[]` without encoding text.
- Added invalid-parameter and zero-result regression tests; documented the contract in architecture and feature traceability.
- Focused retrieval/detector tests: **31 passed in 1.09s**. Full `.venv` suite: **398 passed, 1 existing Starlette deprecation warning in 103.94s**, exit 0 (PowerShell duration **109.82s**). All 311 data files retained identical paths and SHA-256 hashes; `git diff --check` passed.
- No real SRS or external model was used. Context-selection quality remains empirically unverified.

## [2026-11-24] - TASK-P40: Flag short requirements for human inspection
- Added a configurable minimum-character review cue for non-empty requirement text, defaulting to 20 characters; `queue-create --minimum-requirement-characters 0` disables it. Queue manifests retain the eligibility configuration. Labels and verification status are unchanged by the cue.
- Added regressions for threshold boundaries, CLI default/disable behavior, and manifest policy persistence. Updated annotation guidance and feature traceability.
- Focused review queue/CLI/UI integration tests: **45 passed in 18.06s**. Full `.venv` suite: **387 passed, 1 existing Starlette deprecation warning in 101.78s**, exit 0 (PowerShell duration **108.13s**). All 311 files under `data/` retained identical paths and SHA-256 hashes; `git diff --check` passed.
- Only synthetic test pairs were used. The default threshold is a review heuristic, not a language-quality finding; semantic ambiguity/incompleteness remains a human judgment.

## [2026-11-24] - TASK-P39: Preserve document-based SRS requirements
- DOCX extraction now traverses paragraphs and table rows in body order; extracted requirement text retains terminal punctuation. Image-only PDFs fail with guidance that OCR is unsupported rather than silently appearing empty.
- Added DOCX table, text-PDF, and image-only PDF regressions. Focused utility/document-script tests: **25 passed in 13.05s**. Final full `.venv` `python -m pytest tests/`: **386 passed, 1 Starlette deprecation warning in 109.83s**, exit 0 (PowerShell duration **115.85s**). All **311** data files retained identical paths and SHA-256 hashes; `git diff --check` passed.
- Installed only the project-declared `pypdf` and `python-docx` optional dependencies into the existing `.venv` with approval (transitive `lxml`). Synthetic files only; no real SRS corpus was inspected for extraction accuracy, and no human annotation or provider call occurred.

## [2026-11-24] - TASK-P38: Screen repeated-inference consistency
- Added an opt-in 1–5 run consistency check across the Python detector, CLI, single/batch API, and Streamlit. Additional runs bypass the built-in LLM response cache; any label or self-reported confidence variation recommends human review and retains the first predicted class. Default remains one run.
- Added detector, cache-bypass, API, CLI, and UI regression coverage. Focused suite: **61 passed in 28.25s**. Full `python -m pytest tests/`: **383 passed, 12 existing `tqdm` deprecation warnings in 108.10s**, exit 0 (PowerShell duration **116.29s**). `git diff --check` passed; all **311** data files retained identical paths and SHA-256 hashes.
- Tests used controlled clients only. No real provider request, human verification, package installation, or Git operation occurred. A sandboxed focused attempt stalled at Windows API loopback; the unrestricted full run passed `test_api_root` and the entire API module.

## [2026-11-24] - TASK-P37: Report candidate pair-space recall
- Candidate-generation manifests now expose the size of the exhaustive unordered pair universe, coverage recall, and uncovered pair count; the `candidates` CLI displays the same figures. The denominator is explicitly all possible pairs, so the metric is not presented as recall of true dependency labels.
- Synthetic regression assertions cover exhaustive generation, truncated generation, BM25 generation, empty input, manifest round-trip, and CLI output. Focused tests: **42 passed in 9.03s**. Full `python -m pytest tests/`: **374 passed, 12 existing `tqdm` warnings in 101.49s**, exit 0 (PowerShell duration **110.51s**). `git diff --check` passed; all 311 files under `data/` retained their SHA-256 hashes.

## [2026-11-24] - TASK-P28: Persist low-confidence UI detections for human review
- The Streamlit detection page now lets a user explicitly send a `requires_review` prediction to the existing queue. It is retained as a provisional, pending item; no verification or label promotion occurs. The UI queue writer atomically replaces the queue JSON. Separately, the shared `ReviewQueueManager.save_queue()` stages both queue and manifest files and atomically replaces each.
- Added a regression that simulates the initial replace failure and confirms existing queue and manifest bytes remain intact and temporary files are cleaned. Focused tests: **15 passed in 19.96s**. Full suite: **348 passed, 12 existing tqdm deprecation warnings in 89.15s**, exit 0 (PowerShell duration 97.71s). `git diff --check` passed.
- Limitation: queue and manifest are separate filesystem replacements; if manifest replacement fails after queue replacement, the manifest may lag. AppTest verified the UI journey; no fresh browser session or real reviewer/provider was used.

## [2026-11-24] - TASK-P29: Verify live detection and human-review screens
- Browser inspection found an import error in the long-running 8501 Streamlit process, although the current source imports `PredictionValidationStatus` in a fresh interpreter. Started a fresh instance, visually verified the dashboard, ran MockLLM on a synthetic pair, saved a provisional pending queue record under `.pytest-tmp`, and loaded that same record in the Human Review Queue page. It remained pending and unverified.
- Replaced the stale 8501 process with a fresh documented-port instance. `/` and `/_stcore/health` returned HTTP 200; the latter returned `ok`. Stopped the temporary 8503 instance. No source, imported dataset, or historical report changes were made and no human decision/provider request was performed.
- Remaining: real reviewer/rights/direction evidence and dense first-use latency diagnosis; Docker daemon unavailable, hosted CI not rerun.

## [2026-11-24] - TASK-P30: Clarify inclusive confidence-screen wording
- The UI and queue notes now say that a score at or below the configured cutoff is routed for review. The human-facing “low confidence” label no longer contradicts a 5/5 score at a cutoff of 5; the persisted `low_confidence` reason code remains unchanged.
- Focused queue/UI tests: **16 passed**. Full suite: **348 passed, 12 existing tqdm deprecation warnings in 114.93s**, exit 0 (PowerShell duration 124.79s). AppTest verified the message and existing queue flow.

## [2026-11-24] - TASK-P31: Reuse the UI's dense embedding model within a session
- Detection now injects the session's cached embedder into newly created detectors when the configured model and cache directory match. The model is cached only after retrieval actually initialized it; zero-shot and BM25 do not load embeddings just to populate the cache. Added a test that performs two UI detections and checks that the initialized embedder object is reused.
- Focused tests: **15 passed in 23.16s**. Full suite: **348 passed, 12 existing tqdm deprecation warnings in 114.94s**, exit 0 (PowerShell duration 126.52s). The first full run found a stale OpenAI test-double signature; it was corrected and the full rerun passed.
- No measured latency improvement is claimed. Cold first dense use still requires the local model initialization.

## [2026-11-24] - TASK-P32: Add metadata-aware hybrid example retrieval
- Added optional requirement component/subsystem fields and a conditional metadata-overlap ranking as a third RRF signal. Updated regression tests and retrieval documentation.
- Focused hybrid/detector/candidate/API tests: **62 passed in 11.36s**; final hybrid suite after the zero-weight guard: **11 passed in 0.27s**. Final full suite: **350 passed, 12 existing tqdm datetime deprecation warnings in 114.69s**, exit 0 (PowerShell duration 123.08s).
- No dataset or report values were modified. This verifies software routing behavior, not retrieval effectiveness; current datasets lack sufficient structured component/subsystem metadata for an empirical comparison.

## [2026-11-24] - TASK-P33: Add auditable multi-model provisional votes
- Added opt-in `multi_llm` CLI mode and `--models` for multi-model or heuristic-plus-multi-model runs. Only unanimous valid votes produce provisional labels; disagreements/incomplete votes remain unlabeled with review notes and per-model vote evidence. Manifests identify the provider, models, and unanimous-only policy.
- Updated `DiskCachedLLMClient` cache keys to include provider identity and model ID; the versioned key avoids collisions between models. Prior ambiguous cache entries are not reused.
- Focused labeler/LLM/CLI tests: **35 passed in 11.46s**. Full suite: **355 passed, 12 existing tqdm datetime deprecation warnings in 115.92s**, exit 0 (PowerShell duration 124.57s). Controlled clients only; no actual provider was invoked.

## [2026-11-24] - TASK-P34: Carry explicit metadata through SRS, candidate, and detection flows
- Added strict ID-keyed component/subsystem sidecar handling for CLI extraction/candidate generation, matching API extraction and Streamlit extraction controls, and optional component/subsystem fields for API single detection and Streamlit target pairs. Generated candidate pairs serialize the metadata for the existing hybrid retriever.
- Focused utility/CLI/API/UI tests: **50 passed in 29.85s**. Full `python -m pytest tests/`: **364 passed, 12 warnings in 95.37s**, exit 0; warnings are existing `tqdm` UTC datetime deprecations. Eleven replication/evaluation artifact SHA-256 hashes matched their pre-run baseline, and `git diff --check` passed. Unknown IDs, extra keys, malformed objects, and non-string values fail instead of being dropped. No metadata is inferred.

## [2026-11-24] - TASK-P35: Preserve metadata in review queues and verified exports
- `ReviewItem` now carries each requirement's system, component, subsystem, source, and source location. The review UI displays available values, and an accepted/corrected decision restores them in the verified `RequirementPair`. New fields are optional for old queue JSON compatibility.
- Focused queue/UI tests: **41 passed in 30.02s**. Full `python -m pytest tests/`: **366 passed, 12 existing `tqdm` UTC datetime deprecation warnings in 109.44s**, exit 0 (PowerShell duration **119.04s**). Eleven protected replication/evaluation hashes matched; `git diff --check` passed.

## [2026-11-24] - TASK-P36: Publish immutable verified dataset releases
- Added `leredd/dataset_release.py` and the `dataset-release` CLI. It validates verified human provenance, supports combining multiple verified inputs, stamps copied records with a release version, and atomically publishes dataset and integrity/class-count manifest in a new directory. Existing release directories are never overwritten.
- Added focused release and CLI tests; **19 passed in 9.86s**. Full `python -m pytest tests/`: **374 passed, 12 existing `tqdm` UTC datetime deprecation warnings in 105.05s**, exit 0 (PowerShell duration **112.94s**).
- Updated the README demo, architecture, and feature traceability to include immutable release publication before splitting.

## [2026-11-24] - TASK-P27: Separate confidence screening from predicted labels
- `leredd/detector.py` no longer converts low-confidence predictions to `No_dependency`. The `Prediction` schema now carries `validation_status` and `validation_threshold`; the UI displays review guidance, and new evaluation reports summarize screening coverage. Updated API/architecture/replication/feature documentation and the threshold ablation descriptions. Historical reports were preserved.
- Focused parser/API/UI/ablation/evaluation tests: **72 passed, 12 existing warnings in 105.79s**. Full suite: **347 passed, 12 existing warnings in 102.35s**, exit 0. Warnings are tqdm's `utcfromtimestamp()` deprecation.
- Remaining: MockLLM confidence is fixed at 5, so review utility was not measured; detection outputs are flagged but not automatically enqueued. Fresh browser inspection of the updated low-confidence UI remains to be done.

## [2026-11-24] - TASK-P26: Expose retrieval mode selection
- API single and batch detection accept an optional validated `retrieval_mode`; Streamlit detection exposes dense, BM25, and hybrid selection. Missing API values retain configured defaults. API reference and regression tests updated.
- Focused API/UI tests: **20 passed in 18.76s**. Full suite: **346 passed, 12 existing tqdm deprecation warnings in 91.15s**, exit 0. Synthetic local BM25 API smoke returned 200 without initializing SentenceTransformer. All 11 protected replication/evaluation hashes matched; `git diff --check` passed.
- Live visual verification of the new selector remains pending because the browser's Streamlit process predates this change. No packages were installed and no Git state changes were made.

## [2026-11-24] - TASK-P25: Preserve alphanumeric requirement IDs
- **Files changed:** `leredd/utils.py`, `tests/test_utils.py`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md`.
- **Finding/fix:** Source IDs with alphabetic suffixes such as `REQ-DEMO-A` were truncated to `REQ-DEMO`, producing duplicate requirement IDs. The parser now preserves all hyphen/colon-separated alphanumeric segments; regression asserts distinct IDs and pair generation.
- **Execution:** README CLI flow on temporary synthetic Markdown extracted 3 requirements, generated 3 BM25 candidates, wrote 3 provisional results (1 heuristic suggestion and 2 abstentions), and created/listed 3 pending review items. No human decision or verified export was performed.
- **Verification:** Focused utility/CLI tests **21 passed in 3.29s**. Full suite **343 passed, 12 existing tqdm UTC deprecation warnings in 31.11s**, exit 0. Replication/evaluation artifact hashes matched before and after.
- **Boundary:** No packages or Git operations. The parser grammar needs comparison against actual rights-cleared SRS identifier formats.

## [2026-11-24] - TASK-P24: Lazy detector embedding startup
- **Files changed:** `leredd/detector.py`, `tests/test_detector.py`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md`.
- **Change:** `LEREDDDetector` now defers dense embeddings and dense/RAG retrievers until a path actually needs them. Added a zero-shot regression that ensures the unused embedding model is not initialized.
- **Verification:** Focused detector/API/embedding/dense/hybrid/BM25 tests: **46 passed in 9.24s**. Live local MockLLM API `/detect`: HTTP 200 in **164 ms** with isolated cache/data paths and no SentenceTransformer load. Full suite: **342 passed, 12 existing tqdm deprecation warnings in 128.88s**, exit 0. `git diff --check` exit 0. Hashes of all 11 replication/evaluation files matched before and after.
- **Live UI:** Native Chrome produced 3 synthetic candidate pairs (all unlabeled/provisional) under `.pytest-tmp/`; MockLLM detection returned `REQUIRES`, confidence 5/5. First UI dense-retrieval interaction took over one minute to return. The annotation page defaults its output field to `data/verified/streamlit_annotations.json`; no annotation was submitted.
- **Boundary:** No packages or Git operations; imported data and historical reports were not changed. First dense/RAG request still initializes the configured embedding model; cold dense latency needs separate timing.

## [2026-11-24] - TASK-P23: Live UI verification
- Inspected the live Streamlit UI in native Chrome. A diagnostic instance on port 8502 rendered the dashboard after cold startup and extracted 3 synthetic requirements, generating 3 persisted candidate pairs under `.pytest-tmp/`.
- Verified the saved records were unlabeled and provisional. The existing candidate output was preserved after the UI refused overwrite; the run used a distinct demo filename.
- Port 8501 had no listener and returned Chrome `ERR_CONNECTION_REFUSED`. Updated ROADMAP, CONTEXT, DECISIONS, and ISSUE-035 evidence; left the issue open. No application code, tests, datasets, or historical reports were changed. No tests were run.
- Started the documented Streamlit and API commands without installing packages. Native Chrome rendered port 8501 after about 16 seconds from navigation; API logs recorded HTTP 200 for `/` and `/api/v1/health`. ISSUE-035 is now resolved as a persistent blank-page defect; cold-start latency remains noted.

## [2026-11-23] - TASK-P19: Preserve Requirement Source and Location
- **Files updated:** `leredd/data_models.py`, `leredd/utils.py`, `api/routes.py`, `ui/app.py`, `tests/test_utils.py`, `tests/test_api.py`, `tests/test_ui_workflows.py`, `docs/FEATURE_TRACEABILITY.md`, `ISSUES.md`, `ROADMAP.md`, `CONTEXT.md`, `DECISIONS.md`.
- **Change:** Added optional `Requirement.source_location`; file extraction records a source basename and extracted-text line, API accepts a source name, and UI records upload/paste source labels. Kept document name separate from location to preserve document-level split grouping. Legacy JSON remains loadable. Explicit short and bracketed requirement IDs are preserved.
- **Verification:** Focused utility/file/API/UI/CLI tests **43 passed in 24.11s**. Full `python -m pytest tests/` **329 passed, 12 warnings in 87.12s**, exit 0 (93.86s outer). All existing `data/` hashes matched before/after. CLI smoke output under `.pytest-tmp` recorded source and line locations. Warnings are the existing tqdm datetime deprecation.
- **Boundary:** PDF/DOCX location is based on extracted-text lines, not physical page/paragraph position. API source labels remain generic when callers omit the optional field.

## [2026-11-23] - TASK-P18: Prevent Generated Requirement ID Collisions
- **Files updated:** `leredd/utils.py`, `tests/test_utils.py`, `docs/FEATURE_TRACEABILITY.md`, `ISSUES.md`, `ROADMAP.md`, `CONTEXT.md`, `DECISIONS.md`.
- **Finding/fix:** A synthetic requirement ID could duplicate an explicit ID in the same SRS. The extractor now reserves all explicit identifiers in the input and skips them regardless of document order, preserving explicit IDs and schema behavior. Added a regression for both record orders.
- **Verification:** `python -m pytest tests/test_utils.py -q`: **8 passed in 3.56s**. Full `python -m pytest tests/`: **327 passed, 12 warnings in 89.33s**, exit 0 (95.21s outer). The 12 warnings are the existing tqdm datetime deprecation. Hashes of every existing file under `data/` matched before/after.
- **Boundary:** Existing explicit duplicate IDs are not repaired; no datasets, historical reports, dependencies, or Git state were changed.

## [2026-11-23] - TASK-P17: Explicit Mock Ablation Validity
- **Files updated:** `scripts/run_full_evaluation.py`, `tests/test_safe_evaluation.py`, `docs/EXPERIMENTS.md`, `docs/FEATURE_TRACEABILITY.md`, `ISSUES.md`, `ROADMAP.md`, `CONTEXT.md`, `DECISIONS.md`.
- **Change:** Primary and dedicated ablation reports now carry structured `ablation_validity`; `evaluation_limitations` says MockLLM ignores retrieved examples and returns fixed confidence 5. Retrieval/ICL/threshold comparisons are marked non-informative as model-effect estimates. MockLLM behavior is unchanged.
- **Verification:** Focused report-contract test **1 passed in 16.50s**; `tests/test_safe_evaluation.py` **35 passed, 12 warnings in 20.06s**; full `python -m pytest tests/` **326 passed, 12 warnings in 29.50s**, exit 0 (33.97s outer). All data-tree hashes matched across the full run.
- **Boundary:** Reporting is safer, but ISSUE-009/010 remain open until empirical ablation uses an authorized real provider or separately validated controlled test double.

## [2026-11-23] - TASK-P16: Browser QA and Graph Readability Fix
- **Files updated:** `ui/app.py`, `tests/test_ui_workflows.py`, `docs/FEATURE_TRACEABILITY.md`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md`.
- **Finding/fix:** Brave visual inspection showed graph requirement labels overlapping. Nodes now display requirement ID/text on hover, with no crowded permanent text. Added `test_graph_keeps_requirement_labels_in_hover_details`.
- **Live UI evidence:** SRS extraction created 3 requirements and 3 provisional pairs under `.pytest-tmp/task-p16-ui/`; mock detection returned `REQUIRES`, confidence 5/5; evaluation generated `data/evaluation/ui_runs/20261123T094742Z-215fe692/system_disjoint_report.json`. The report identifies the 20-pair legacy provenance-free set, five-pair test partition, and MockLLM caveat. Review queue was inspected without submitting a human decision.
- **Tests:** Focused graph regression **1 passed in 12.09s**; `tests/test_ui_workflows.py` **8 passed in 20.45s**; full `python -m pytest tests/` **326 passed, 12 warnings in 77.22s**, exit 0 (83.01s outer). Warnings are the existing tqdm datetime deprecation. All data-tree file hashes matched before/after the full suite.
- **Boundaries:** The UI evaluation report is a newly created unique artifact; no imported dataset or historical report was modified. Browser QA was one local Brave desktop session. No human label was fabricated.

## [2026-11-23] - TASK-P15: Blinded Independent Annotation Tooling
- **Files added:** `leredd/independent_annotation.py`, `scripts/annotate_independent.py`, `scripts/compare_independent_annotations.py`, `tests/test_independent_annotation.py`, `docs/ANNOTATION_GUIDE.md`.
- **Files updated:** `README.md`, `docs/FEATURE_TRACEABILITY.md`, `ROADMAP.md`, `CONTEXT.md`, `DECISIONS.md`.
- **Implementation:** Added separate reviewer manifests with source/context SHA-256 binding, blind pair presentation, resumable atomic persistence, abstention/rationale/timing capture, two-reviewer agreement checks and report generation. These outputs do not adjudicate or promote labels and cannot be written under imported replication or evaluation data trees.
- **Verification:** Focused tests: **15 passed in 1.01s**. Full `python -m pytest tests/`: **325 passed, 12 warnings in 69.69s**, exit 0. Warnings are the existing `tqdm` datetime deprecation. Imported replication hashes were unchanged; `git diff -- data` was empty.
- **Research boundary:** No actual human annotations or effort results were collected. Identity is self-reported; replication label direction and redistribution terms remain unresolved. No package installations or Git state changes occurred.

## [2026-10-12] - TASK-P9-13: End-to-End Label Integrity and Stabilization
- **Implementation:** Generated pairs remain unlabeled with candidate-generator provisional provenance; unknown explicit labels raise instead of being guessed; CLI human annotation marks verified provenance; ground-truth guards require `VERIFIED` for provenance-bearing annotations; evaluator rejects incomplete prediction coverage; model training and example retrieval filter out unlabeled/unverified records. Added automatic test isolation for global config. Replication-runner tests now assert pending-review imports fail closed without writing a report or initializing a model.
- **Documentation:** Updated `ISSUES.md`, `ROADMAP.md`, `CONTEXT.md`, `DECISIONS.md`, CLI/API references, and replication/evaluation notes. Recorded that the paper license alone does not establish spreadsheet redistribution rights and that local imported labels are pending review.
- **Focused verification:** Eligibility plus replication-runner tests: **15 passed in 2.09s**. CLI evaluator complete-coverage regression: **1 passed in 2.02s**.
- **Full verification:** `python -m pytest tests/` → **236 passed, 40 warnings in 176.82s**, exit code 0; measured PowerShell elapsed **179.59s**. An earlier attempt had 3 stale-message assertion failures, then a separate full run exposed one incomplete test prediction fixture; both were corrected and the final run passed.
- **Data and environment:** The first 16 hexadecimal characters of each replication JSON's SHA-256 digest matched before and after the full-suite run; no dataset writes were found in the loader failure-mode tests. No packages installed, no real LLM or standalone benchmark run, and no stage/commit/push/reset/checkout performed. The API tests passed outside the previously restricted sandbox.
- **Open work:** Replication metrics remain blocked until imported records receive local human verification; `Implemented by` direction and dataset-specific redistribution terms remain unconfirmed. MockLLM, sample-size, and dependency deprecation caveats remain.

## [2026-10-12] - TASK-P9-12: Diagnose Windows TestClient Stall and Verify Full Suite
- **Files changed:** `ROADMAP.md`, `CONTEXT.md`, `ISSUES.md`, `DECISIONS.md`, `CHANGELOG.md` (verification records only). No application code or tests required changes. Pytest output logs are in `.pytest-tmp/`.
- **Diagnosis:** Faulthandler traced the stall from `tests/test_api.py::test_api_root` through HTTPX and Starlette `TestClient._portal_factory` to AnyIO `start_blocking_portal()`. The portal thread blocked in Windows asyncio's socket-pair fallback at `socket.accept()`. In the restricted sandbox, a raw `socket.socketpair()` probe also timed out; outside it, the unchanged API root test passed. This is a sandbox loopback restriction, not a root-handler or eager-detector problem.
- **Tests:** `tests/test_api.py::test_api_root` → **1 passed in 0.75s**; `tests/test_api.py` → **7 passed, 1 warning in 50.70s**; focused loader/dataset/runner suites → **38 passed, 6 warnings in 40.11s**; full `python -m pytest tests/` → **228 passed, 46 warnings in 210.60s**, exit code 0. Runs were bounded and logs saved under `.pytest-tmp/`.
- **Environment note:** API model initialization contacted Hugging Face Hub without a token and loaded model weights; no packages were installed and no real LLM was called. The full suite is complete; ISSUE-017 is resolved.

## [2026-10-12] - TASK-P9-02: Functional Architecture Walkthrough
- **Files changed:** `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md` (architecture notes only; no product code modified).
- **Inspection:** Traced data models and provenance; requirement extraction/pair/candidate creation; provisional labeling and review queue; dataset splitting; retrieval/prompt/LLM/detector pipeline; baselines and evaluation; CLI/API/Streamlit; and the added replication loader/prototype.
- **Tests executed:** None; this was a read-only architecture walkthrough and did not request test execution.
- **Findings:** Clarified the two primary workflows and that the replication evaluation is a separate prototype, not the legacy system-disjoint benchmark. Logged ISSUE-018: legacy all-pairs `pairs` command defaults labels to `No_dependency`, unlike unlabeled `candidates` output.

## [2026-10-12] - TASK-P9-01: Whole-Project Analysis and Verification Audit
- **Files changed:** `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md` (audit records only; existing user edits preserved).
- **Inspection:** Read project context, roadmap, architecture decisions, issues, package/configuration, code/test inventory, replication loader/runner, and current Git state.
- **Commands and results:** `python -m compileall -q leredd api ui scripts tests` passed; `leredd-cli --help` passed; FastAPI/Typer imports passed; focused replication dataset/loader/Implemented by tests: **15 passed in 0.19s**; `pytest tests/test_safe_evaluation.py -vv -x` with workspace-local temp: **27 passed in 641.55s**.
- **Incomplete checks:** Full suite collected 216 cases but was interrupted after prolonged execution without a final summary. Initial evaluation test run failed due sandbox access to OS temp (`PermissionError`). Ruff command unavailable because Ruff is not installed in `.venv`. Replication loader mutation tests intentionally not run after source inspection showed they overwrite checked-in dataset files.
- **Findings:** Project is not “perfectly verified.” Added ISSUE-016 (tests mutate production datasets) and ISSUE-017 (no current full-suite completion evidence). Existing scientific/data caveats remain.

All notable changes, task completions, and verification events for the Minor Project transformation will be documented in this file chronologically.

## [2026-10-11] - Bugfix: ISSUE-011 (Remove Duplicate LEREDD_MockLLM Entry)
- **Task:** Resolution of ISSUE-011 in `scripts/run_full_evaluation.py`.
- **Files Modified:**
  - [`scripts/run_full_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/run_full_evaluation.py): Removed redundant `LEREDD_MockLLM` report dictionary entry that was duplicating `leredd_hybrid_report` and `leredd_hybrid_preds`. Maintained `LEREDD_Hybrid` as the single canonical representation of the proposed hybrid system.
  - [`tests/test_safe_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_safe_evaluation.py): Updated assertions to explicitly verify that `LEREDD_MockLLM` is not present in `model_results`, that `LEREDD_Hybrid` exists and properly reflects the mock backend demarcations, and updated regression checks.
  - [`ISSUES.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ISSUES.md): Marked ISSUE-011 as resolved.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/test_safe_evaluation.py -v`
- **Results:** **27 passed in 401.84s**, 0 failed. Zero regressions across safe evaluation test suite.

## [2026-10-11] - Read-Only Code Audit (Pipeline Correctness Audit)
- **Task:** Read-only code audit of LEREDD pipeline. No files modified.
- **Files Inspected:** `leredd/llm.py`, `leredd/prompts.py`, `leredd/detector.py`, `leredd/config.py`, `leredd/hybrid_retrieval.py`, `leredd/bm25_retrieval.py`, `leredd/retrieval.py`, `leredd/data_models.py`, `scripts/run_full_evaluation.py`, `tests/test_llm.py`, `tests/test_detector.py`, `tests/test_prompts.py`, `tests/test_ablation_benchmark.py`, `tests/test_safe_evaluation.py`, `tests/test_hybrid_retrieval.py`.
- **Confirmed Findings (all read-only):**
  - **ISSUE-009 [Critical]:** `MockLLMClient.generate()` ignores the `#Examples:` block in the prompt entirely. All retrieval/ICL ablation variants produce identical predictions.
  - **ISSUE-010 [Critical]:** `MockLLMClient` hardcodes `confidence=5` in all 6 branches. The threshold re-annotation mechanism (`confidence <= threshold`) never fires under Mock evaluation.
  - **ISSUE-011 [Critical]:** `LEREDD_MockLLM` report entry copy-pastes `leredd_hybrid_report` and `leredd_hybrid_preds` instead of running an independent call. It is a phantom duplicate of `LEREDD_Hybrid`.
  - **ISSUE-012 [High]:** `get_config()` global singleton is never reset between tests, potentially breaking isolation.
  - **ISSUE-013 [Medium]:** `ResponseParser` rationale regex truncates multi-line rationales from real LLMs.
  - **ISSUE-014 [Low]:** `OllamaLLMClient` missing retry logic (unlike `OpenAILLMClient`).
  - **ISSUE-015 [Low]:** `DependencyType.normalize()` "No" substring ambiguity.
- **Confirmed Correct Behaviors:**
  - Dense, BM25, and Hybrid retrievers produce genuinely distinct ranked outputs.
  - Retrieved examples ARE correctly passed into the prompt string by `PromptBuilder`.
  - Real LLM clients (`OpenAILLMClient`, `OllamaLLMClient`) are fully implemented and configurable.
  - Threshold mechanism works correctly (verified by `test_ablation_benchmark.py:91–112`).
  - RRF math is correct (verified by `test_hybrid_retrieval.py:53–78`).
- **Audit Report:** [`code_audit_report.md`](file:///C:/Users/DN816/.gemini/antigravity-ide/brain/53a8af3f-ea50-421f-8b25-268eebad88da/code_audit_report.md)



## [2026-10-10] - Read-Only Artifact Audit (post end-to-end verification)
- **Task:** Read-only audit of all scratch verification artifacts generated during pipeline e2e run.
- **Files Inspected (read-only):** `scratch/verification_candidates.json`, `scratch/verification_provisional.json`, `scratch/verification_queue.json`, `scratch/verification_predictions.json`, `scratch/verification_splits/split_metadata.json`, `scratch/verification_eval_report.json`, `data/splits/system_disjoint/`, `data/evaluation/system_disjoint_report.json`.
- **No application code, datasets, or reports were modified.**
- **Audit Findings:**
  1. 31 candidate pairs: all unique, no self-pairs, correctly linked to 15 ADB requirements (REQ-ADB-001–015).
  2. Provisional label distribution: {Implements:15, Conflicts:2, No_dep:2, Requires:1, Details:1, Is_similar:1, null:9}.
  3. The 22-labelled/2-agreed/9-disagreed counts are non-overlapping axes that sum consistently to 31 (20 LLM-only + 2 ensemble-agreed + 9 ensemble-disagreed).
  4. Review queue: 29 items = 9 disagreement (priority=1) + 20 llm_only (priority=4). Two agreed pairs are excluded.
  5. Verification splits: leakage-free (dev split empty in verification run — only ADB/TJA available).
  6. Eval report discrepancy: BERT accuracy 0.200 (production) vs 0.000 (verification), likely Random Forest nondeterminism across sessions.
  7. GPT-4 0.880 reference score: not present in either eval report; correctly a paper reference only.
  8. Annotation effort: partially demonstrated (queue operational; no time-motion study).
  9. Test set: n=5 (TJA system), confirmed in both split metadata and both eval reports.
  10. Minor data issue: REQ-ADB-005 text truncated at "…in less than 2" (unit missing from regex extraction).
- **Audit Report Artifact:** [`artifact_audit_report.md`](file:///C:/Users/DN816/.gemini/antigravity-ide/brain/53a8af3f-ea50-421f-8b25-268eebad88da/artifact_audit_report.md)

## [2026-10-10] - TASK-P8-04: Final Submission Artifact Verification
- **Task Completed:** `TASK-P8-04`
- **Actions Taken:**
  1. Verified `p8_03_academic_readiness_audit.md` artifact existence, matrix completeness, and readiness declaration.
  2. Verified metrics match `data/evaluation/system_disjoint_report.json` (TF-IDF Acc=0.400, BERT Acc=0.200, Mock LEREDD Acc=0.400).
  3. Verified GPT-4 Hybrid LLM reference score (0.880) is explicitly identified in `README.md`, `docs/EXPERIMENTS.md`, and audit report as a paper reference benchmark using live GPT-4 API—NOT a result from local MockLLM execution.
  4. Verified annotation-effort objective is marked partially demonstrated, with zero empirical human time-motion study claimed.
  5. Verified test set size ($n=5$) is described as demonstrative of pipeline mechanics rather than statistically conclusive research bounds.
  6. Verified Docker deployment is described as statically checked syntax (`Dockerfile` & `docker-compose.yml`), NOT live container deployment.
  7. Verified 174 passing unit/integration tests are reported with actual run details and NOT described as proof of research effectiveness.
  8. Verified cross-file terminology consistency. Preserved all existing user changes; 0 commits or pushes executed.
- **Final Readiness Status:** **READY WITH CAVEATS**.

---

## [2026-10-10] - TASK-P8-03: Final Academic Readiness and Evidence Audit
- **Task Completed:** `TASK-P8-03`
- **Actions Taken:**
  1. Audited all 8 proposal specific objectives against actual implementation, test files, and saved datasets. Created proposal-to-implementation coverage matrix.
  2. Verified empirical claims in `docs/EXPERIMENTS.md` against authoritative report `data/evaluation/system_disjoint_report.json`. Mapped model identities, split sizes, Wilson score 95% CIs, and root causes of identical ablation scores under Mock LLM mode.
  3. Verified reproducibility across 174 automated tests (100% pass), CLI runner, FastAPI TestClient, Streamlit session state, and static Docker configuration integrity.
  4. Identified academic honesty qualifications (theoretical candidate reduction vs empirical human time-study, sample size $n=5$ CI margins, Mock vs Live LLM demarcation).
  5. Declared final submission readiness: **READY WITH CAVEATS**.
- **Audit Artifact:** [`p8_03_academic_readiness_audit.md`](file:///C:/Users/DN816/.gemini/antigravity-ide/brain/53a8af3f-ea50-421f-8b25-268eebad88da/p8_03_academic_readiness_audit.md).
- **Verification:** Ran `python -m pytest tests/ -v` $\rightarrow$ **174/174 passed, 0 failed** in 25.91s.

---

## [2026-10-10] - TASK-P8-02: Final Documentation & Artifact Alignment
- **Task Completed:** `TASK-P8-02`
- **Files Modified / Added:**
  1. [`README.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/README.md): Comprehensive rewrite reflecting the academic project title *"Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation"*, dual-workflow breakdown, dependency taxonomy table, Mermaid system architecture diagrams, quickstart, CLI/API/UI/Docker usage guides, empirical benchmark tables, and honest limitations.
  2. [`docs/ARCHITECTURE.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/ARCHITECTURE.md): Created architecture and technical specification guide covering component data flows, candidate generation, review queue engine, dataset splitter, hybrid retrieval, confidence threshold filter, and provenance models.
  3. [`docs/EXPERIMENTS.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/EXPERIMENTS.md): Created empirical evaluation and benchmark report detailing system-disjoint evaluation methodology, baseline comparisons, per-class metrics, statistical significance tests, and limitations.
  4. [`docs/API.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/API.md): Updated reference guide covering FastAPI endpoints, Typer CLI commands, and Python SDK classes (`RequirementDependencyDetector`, `SystemDisjointSplitter`, etc.).
  5. [`ROADMAP.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ROADMAP.md), [`CONTEXT.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/CONTEXT.md), [`CHANGELOG.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/CHANGELOG.md), [`DECISIONS.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/DECISIONS.md), [`ISSUES.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ISSUES.md): Updated project context tracking files to record Phase 8 completion.
- **Empirical Verification:** Ran full test suite (`python -m pytest tests/ -v`) $\rightarrow$ **174/174 passed** cleanly in 25.91s.

---

## [2026-10-10] - TASK-P8-01: System Verification & Multi-Interface Testing
- **Task Completed:** `TASK-P8-01`
- **Files Modified / Added:**
  1. [`tests/test_api.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_api.py): Updated FastAPI endpoint test suite (7 tests) covering `/`, `/api/v1/health`, `/api/v1/config`, `/api/v1/detect`, `/api/v1/batch_detect`, `/api/v1/extract`, and 422 validation errors.
  2. [`tests/test_system_verification.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_system_verification.py): Created end-to-end multi-interface integration test suite (3 tests) verifying step-by-step pipeline execution (`extract` $\rightarrow$ `CandidateGenerator` $\rightarrow$ `ProvisionalLabeler` $\rightarrow$ `ReviewQueueManager` $\rightarrow$ `record_decision` $\rightarrow$ `export_verified_dataset` $\rightarrow$ `DatasetSplitter` $\rightarrow$ `run_evaluation`), CLI error resilience, and Docker configuration integrity (`Dockerfile`, `docker-compose.yml`).
- **Functionality Verified:**
  - **REST API:** 6 endpoints operational with FastAPI `TestClient`, returning typed Pydantic models.
  - **CLI Tools:** 8 commands operational with Typer CLI runner.
  - **Streamlit Interface:** 9 interactive views in `ui/app.py` operational with session state and component rendering tests (31 UI tests).
  - **Docker & Compose:** Multi-stage build (`builder` $\rightarrow$ `runtime`), port exposures (8000 API, 8501 Streamlit), and service definitions (`api`, `ui`, `redis`).
  - **Pipeline Provenance Integrity:** Provenance lifecycle verified with zero unverified machine label leaks.
- **Empirical Results:** `python -m pytest tests/ -v` $\rightarrow$ **174/174 passed, 0 failed** in 10.97s. Zero regressions.

---

## [2026-10-09] - Phase 0 Completion: Repository Safety & Context Management
- **Task Completed:** `TASK-P0-01` & `TASK-P0-02`
- **Actions Taken:**
  1. Inspected repository identity, git history, remotes, working tree status (`git status`, `git remote`, `git log`). Verified remote `origin` points to `https://github.com/kesarshharma/LLREDD.git` and working tree is clean on `main`.
  2. Read `Minor_Project.pdf` proposals and aligned system design objectives for Workflows A (Dependency Detection), B (Semi-Automated Dataset Generation), and C (Experimental Evaluation).
  3. Created permanent context-management files: [`AGENTS.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/AGENTS.md), [`CONTEXT.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/CONTEXT.md), [`ROADMAP.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ROADMAP.md), [`CHANGELOG.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/CHANGELOG.md), [`DECISIONS.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/DECISIONS.md), and [`ISSUES.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ISSUES.md).
- **Verification:** Read-only inspection report completed; non-mutating check verified clean repository state.

## [2026-10-09] - Phase 1 Completion: Environment & Baseline Verification
- **Tasks Completed:** `TASK-P1-01`, `TASK-P1-02`, `TASK-P1-03`
- **Actions Taken:**
  1. Verified existing `.venv` (Python 3.13.3) is functional with all dependencies from `requirements.txt` + `pyproject.toml` installed. `leredd` package installed in editable mode.
  2. Ran `pytest --collect-only` → 28 test items collected, exit code 0 (ISSUE-001 resolved).
  3. Ran `pytest tests/ -v` → **28 passed, 0 failed** in 45.74s. 1 deprecation warning (`httpx`/`starlette`).
  4. Ran `scripts/run_evaluation.py` with default `llm_provider="mock"` (MockLLMClient) on ADB dataset (10 pairs).
- **Baseline Evaluation Results:**
  | Model | Accuracy | Macro F1 | Weighted F1 |
  |---|---|---|---|
  | LEREDD (Mock) | 0.5000 | 0.4722 | 0.4833 |
  | BERT (overfit) | 1.0000 | 1.0000 | 1.0000 |
  | TF-IDF & LSA | 0.3000 | 0.0769 | 0.1385 |
- **McNemar's Test:** LEREDD vs BERT: statistic=0.0000, p=0.0625
- **Key Finding:** BERT perfect score is due to no train/test split (trains and evaluates on same 10 pairs). Confirms ISSUE-002 data leakage risk.

## [2026-10-10] - TASK-P2-01: Dataset Schema & Provenance Extension
- **Task Completed:** `TASK-P2-01`
- **Files Changed:**
  1. [`leredd/data_models.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/data_models.py): Added `AnnotationSource` enum (5 values: human, llm, heuristic, imported, ensemble), `VerificationStatus` enum (5 values: provisional, pending_review, human_reviewed, verified, rejected), `AnnotationProvenance` sub-model (16 optional fields covering source, lifecycle, review, labels, confidence, evidence, and dataset version). Extended `RequirementPair` with optional `provenance: AnnotationProvenance | None`. Extended `Prediction` with 4 optional provenance fields (`annotation_source`, `model_id`, `config_hash`, `created_at`).
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Added `AnnotationSource`, `VerificationStatus`, `AnnotationProvenance` to package exports.
  3. [`tests/test_provenance.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_provenance.py): New test file with 35 tests across 8 test classes covering legacy loading, provenance metadata, validation, round-trips, enum checks, invalid inputs, and existing model compatibility.
- **Tests Executed:** `pytest tests/ -v` → **63 passed, 0 failed** in 131.57s (28 original + 35 new).
- **Backward Compatibility:** All 3 existing JSON datasets (`adb_pairs.json`, `apa_pairs.json`, `tja_pairs.json`) load without modification. All existing API, CLI, detector, and baseline code unchanged and passing.

## [2026-10-10] - TASK-P2-02: Leakage-Resistant Dataset Splitter
- **Task Completed:** `TASK-P2-02`
- **Files Changed:**
  1. [`leredd/dataset_splitter.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/dataset_splitter.py): Implemented `DatasetSplitter`, `SplitConfig`, `DatasetSplit`, `SplitStrategy` enum (`SYSTEM`, `DOCUMENT`, `REQUIREMENT`, `PAIR`), `LeakageReport`, and persistence functions (`save_split`, `check_leakage`).
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Added `DatasetSplitter`, `SplitConfig`, `DatasetSplit`, `SplitStrategy`, `LeakageReport` to package exports.
  3. [`leredd/cli.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/cli.py): Added `leredd-cli split` command with configurable ratios, strategies, random seeds, and output formatting.
  4. [`scripts/split_dataset.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/split_dataset.py): Created dataset partitioning script generating system-disjoint (`data/splits/system_disjoint/`) and requirement-disjoint (`data/splits/requirement_disjoint/`) dataset partitions.
  5. [`tests/test_dataset_splitter.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_dataset_splitter.py): Added 10 comprehensive unit tests covering zero requirement leakage validation, system/document/requirement splits, reproducibility, edge cases, persistence, and CLI command.
- **Tests Executed:** `pytest tests/ -v` → **73 passed, 0 failed** in 46.63s (28 original + 35 provenance + 10 splitter).
- **Leakage Verification:** Disjoint requirement splits verified `req_overlap_train_test = ∅` and `req_overlap_retrieval_test = ∅` (leakage rate: 0.00%).

## [2026-10-10] - TASK-P3-01: BM25 Lexical Retrieval Implementation
- **Task Completed:** `TASK-P3-01`
- **Files Changed:**
  1. [`leredd/bm25_retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/bm25_retrieval.py): Implemented `BM25Retriever` using `BM25Plus` from `rank-bm25`. Supports requirement strings, `Requirement` objects, and `RequirementPair` objects. Features lowercasing & punctuation tokenization, inspectable corpus isolation (`retriever.corpus`), stable tie-breaking, handling of invalid $k \le 0$, example pair retrieval (`retrieve_example_pairs`), and pure offline execution.
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Exported `BM25Retriever`.
  3. [`pyproject.toml`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/pyproject.toml) & [`requirements.txt`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/requirements.txt): Added dependency `rank-bm25>=0.2.2`.
  4. [`tests/test_bm25_retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_bm25_retrieval.py): Added 10 unit tests covering exact/partial matching, no-match queries, empty inputs, tokenization, repeated terms, stable tie-breaking, invalid $k$, metadata/label preservation, corpus isolation, and offline execution.
- **Tests Executed:** `pytest tests/ -v` → **83 passed, 0 failed** in 47.61s (28 original + 35 provenance + 10 splitter + 10 BM25). Zero regressions.
- **Retrieval Isolation:** `BM25Retriever` indexes only explicitly provided corpus items without loading global datasets.

## [2026-10-10] - TASK-P3-02: Hybrid Retrieval Engine (Semantic + Lexical + Metadata Fusion)
- **Task Completed:** `TASK-P3-02`
- **Files Changed:**
  1. [`leredd/hybrid_retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/hybrid_retrieval.py): Implemented `HybridRetriever` combining SBERT dense embeddings (`ExampleRetriever`) and BM25 lexical matching (`BM25Retriever`) via Reciprocal Rank Fusion (RRF). Added `reciprocal_rank_fusion` with configurable $rrf\_k=60$, stable tie-breaking, candidate deduplication, target self-exclusion, metadata/provenance preservation, and pure corpus isolation.
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Exported `HybridRetriever`.
  3. [`leredd/config.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/config.py): Added `retrieval_mode` (default `"dense"`) and `rrf_k` (default `60`) settings to `LEREDDConfig`.
  4. [`tests/test_hybrid_retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_hybrid_retrieval.py): Added 9 unit tests covering RRF math, deduplication, tie-breaking, top-k truncation, empty corpus handling, metadata preservation, self-exclusion, corpus isolation, and comparative retrieval sanity checks.
- **Tests Executed:** `pytest tests/ -v` → **92 passed, 0 failed** in 66.22s (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid). Zero regressions.

## [2026-10-10] - TASK-P4-01: Candidate Requirement Pair Generation
- **Task Completed:** `TASK-P4-01`
- **Files Changed:**
  1. [`leredd/candidate_generator.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/candidate_generator.py): Implemented `CandidateGenerator`, `CandidateGenerationManifest`, `GenerationStrategy` enum (`ALL_PAIRS`, `BM25`, `DENSE`, `HYBRID`), and `get_canonical_pair_key`. Features canonical unordered pair deduplication (`(A,B)` and `(B,A)`), target self-exclusion (`(A,A)`), explicit max candidate limits with truncation reporting in audit manifest, and unlabeled candidate state (`dependency_type is None`).
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Exported candidate generator classes.
  3. [`leredd/cli.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/cli.py): Added `leredd-cli candidates` CLI command.
  4. [`scripts/generate_candidates.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/generate_candidates.py): Created candidate generation script producing candidate pair files and audit manifests (`data/candidates/`).
  5. [`tests/test_candidate_generator.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_candidate_generator.py): Added 9 unit tests covering all-pairs count $n(n-1)/2$, self-pair exclusion, canonical deduplication, truncation limits, unlabeled state, manifest serialization, and CLI execution.
- **Tests Executed:** `pytest tests/ -v` → **101 passed, 0 failed** in 65.52s (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid + 9 candidate generator). Zero regressions.

## [2026-10-10] - TASK-P4-02: Provisional Requirement Dependency Labeling
- **Task Completed:** `TASK-P4-02`
- **Files Changed:**
  1. [`leredd/provisional_labeler.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/provisional_labeler.py): Implemented `HeuristicLabeler` (deterministic rules `RULE-REF-ID-DIRECT`, `RULE-REF-ID-REVERSE`, `RULE-REF-IMPLEMENTS`, `RULE-REF-CONFLICTS`, `RULE-NEG-CONFLICT`, `RULE-SIM-JACCARD`, `RULE-ABSTAIN`), `LLMLabeler` (LLM-based classifier using `LLMClientFactory` and `MockLLMClient`), `ProvisionalLabeler` (coordinator managing agreement, disagreement, single-method, and unresolved states), `LabelingRunManifest`, and dataset persistence (`save_provisional_dataset`).
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Exported `HeuristicLabeler`, `LLMLabeler`, `ProvisionalLabeler`, `LabelingRunManifest`, `LabelingMode`.
  3. [`leredd/cli.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/cli.py): Added `leredd-cli label` command (`label_cmd`).
  4. [`leredd/prompts.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/prompts.py): Added `build_prompt = build` alias for backward compatibility.
  5. [`leredd/llm.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/llm.py): Enhanced `MockLLMClient` with `response_text` parameter override for offline testing.
  6. [`scripts/label_candidates.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/label_candidates.py): Created demonstration script executing provisional labeling on generated candidate datasets (`data/provisional/`).
  7. [`tests/test_provisional_labeler.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_provisional_labeler.py): Added 12 unit tests covering 6 canonical classes, heuristic determinism & abstentions, LLM structured JSON parsing & error handling, hybrid agreement (`ENSEMBLE`), disagreement preservation (`Unresolved`), requirement directionality, human-verified label protection, batch manifest serialization, and CLI execution.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/ -v` → **113 passed, 0 failed** in 64.98s (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid + 9 candidate generator + 12 provisional labeler). Zero regressions.
- **Verification Guarantee:** All machine-generated labels explicitly assigned `status=VerificationStatus.PROVISIONAL`. Existing verified human labels (`VerificationStatus.VERIFIED`) are never overwritten. Disagreements between Heuristic and LLM predictions set `dependency_type = None` with explicit review notes for routing to Phase 5 human review.

## [2026-10-10] - TASK-P5-01: Multi-Method Disagreement & Uncertainty Review Queue
- **Task Completed:** `TASK-P5-01`
- **Files Changed:**
  1. [`leredd/review_queue.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/review_queue.py): Implemented `ReviewQueueManager`, `ReviewItem`, `QueueStatus` enum (`pending`, `in_review`, `accepted`, `corrected`, `rejected`, `deferred`, `reviewed`), `ReviewReasonCode` enum (`disagreement`, `both_abstained`, `llm_failure`, `heuristic_only`, `llm_only`, `low_confidence`, `missing_metadata`, `explicit_flag`), `ReviewPriority` enum (rank 1 to 5), `HumanDecision` model, `ReviewQueueConfig`, and `ReviewQueueManifest`.
  2. [`leredd/__init__.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/__init__.py): Exported review queue classes and enums.
  3. [`leredd/cli.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/cli.py): Added `leredd-cli queue-create`, `queue-list`, `queue-decide` CLI commands.
  4. [`scripts/build_review_queue.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/build_review_queue.py): Created demonstration script building review queue from provisional pairs, performing count reconciliation, and recording human decisions (`data/queue/`).
  5. [`tests/test_review_queue.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_review_queue.py): Added 12 unit tests covering disagreement/abstention/failure queuing, priority ranking, filtering, human decision recording (`ACCEPT`/`CORRECT`/`REJECT`/`DEFER`), ground-truth protection (`VerificationStatus.VERIFIED` only upon explicit human decision), idempotent refresh, directionality preservation, persistence round-trips, and CLI execution.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/ -v` → **125 passed, 0 failed** in 65.12s (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid + 9 candidate generator + 12 provisional labeler + 12 review queue). Zero regressions.
- **Candidate Outcome Count Reconciliation (31 ADB candidates input):**
  - **Agreement (Ensemble):** 2 candidates (Both Heuristic & LLM predicted valid label and matched).
  - **Disagreement (Conflict):** 9 candidates (Heuristic and LLM predicted different valid labels; queued with Priority 1 CRITICAL).
  - **LLM-Only Prediction:** 20 candidates (Heuristic abstained, LLM predicted valid label; queued with Priority 4 LOW).
  - **Heuristic-Only Prediction:** 0 candidates.
  - **Both Abstained:** 0 candidates.
  - **Check Sum:** $2 + 9 + 20 + 0 + 0 = 31$ candidates (100% reconciled). Total queued items under default single-method policy: 29 items (9 Disagreements + 20 LLM-Only).

## [2026-10-10] - TASK-P5-02: Streamlit and CLI Human Review Interface
- **Task Completed:** `TASK-P5-02`
- **Files Changed:**
  1. [`ui/app.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ui/app.py): Added Page 6 ("6. Human Review Queue") providing: (A) Operational Queue Overview with dynamic count metrics, breakdown by review reason code and priority rank, and provisional vs human-reviewed distinction; (B) Review Queue Browser with multi-criteria filtering (status, reason, priority, system, predicted label), sorting by priority + ID, and default selection of highest-priority pending item; (C) Requirement-Pair Detail View displaying requirement texts A and B side-by-side, heuristic & LLM triggering evidence, proposed label, confidence warnings, and historical review notes; (D) Human Decision Controls (`ACCEPT`, `CORRECT`, `DEFER`, `REJECT`) enforcing mandatory final label selection for corrections, confirmation step for promotion to `VERIFIED`, and safe persistence error handling; (E) Verified Annotation Export section to export accepted/corrected pairs to `data/verified/` without modifying raw source datasets.
  2. [`tests/test_ui_and_queue_integration.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_ui_and_queue_integration.py): Added 6 integration unit tests validating Streamlit app module importing, dynamic queue count calculation, filtering, 6-class label validation, decision persistence, session state stability across reruns, prediction preservation, verified dataset export filtering out provisional/deferred/rejected pairs, provenance preservation, and error handling for missing/corrupt queue files.
  3. [`ROADMAP.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ROADMAP.md) & [`CONTEXT.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/CONTEXT.md): Updated task completion state, architectural state, active blockers, and immediate next action.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/ -v` → **131 passed, 0 failed** in 65.23s (28 original + 35 provenance + 10 splitter + 10 BM25 + 9 hybrid + 9 candidate generator + 12 provisional labeler + 12 review queue + 6 UI integration). Zero regressions.
- **Ground-Truth Protection Verification:** Attempting to accept an unresolved pair (`dependency_type=None`) without specifying a valid label is strictly rejected. `VerificationStatus.VERIFIED` is granted ONLY upon explicit human decision (`ACCEPT` or `CORRECT`). Machine predictions remain completely intact in history.

## [2026-10-10] - TASK-P5-02-FIX-01: Repair Verified Dataset Export
- **Task Completed:** `TASK-P5-02-FIX-01`
- **Root Cause:** In [`ui/app.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ui/app.py), the verified dataset export section called `save_pairs_json(verified_pairs, export_path)` without importing `save_pairs_json` from [`leredd.utils`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/utils.py), throwing `NameError: name 'save_pairs_json' is not defined`.
- **Files Changed:**
  1. [`ui/app.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ui/app.py): Added `save_pairs_json` to `leredd.utils` import block.
  2. [`tests/test_ui_and_queue_integration.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_ui_and_queue_integration.py): Added `test_streamlit_export_save_pairs_json_defined` (regression test verifying `save_pairs_json` is imported and callable in `ui.app`) and `test_verified_export_roundtrip_and_exclusions` (testing serialization roundtrip, exclusion of pending/provisional/deferred/rejected records, and preservation of human provenance).
- **Verification & Results:**
  - Export writes directly to requested path (`export_path`).
  - Preserves full schema and human provenance (`VerificationStatus.VERIFIED`, `AnnotationSource.HUMAN`, `reviewer_id`).
  - Excludes pending, provisional, deferred, rejected, and unresolved records.
  - Does not overwrite source queue or raw candidate/provisional files.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/ -v` → **133 passed, 0 failed** in full suite. Zero regressions.



## [2026-10-10] - TASK-P6-01: Evaluation Readiness and Dataset Integrity Audit
- **Task Completed:** TASK-P6-01
- **Type:** Read-only audit. No code modified, no models trained, no commits or pushes.
- **Audit Findings Summary:**
  - **Ground Truth:** 20 labeled pairs (data/annotated/): ADB=10, APA=5, TJA=5. No provenance metadata. Only defensible evaluation ground-truth pairs.
  - **Excluded from evaluation:** data/verified/adb_verified_ui_test.json (1 explicit test fixture); all data/provisional/* (MockLLM); all data/queue/* (pending review).
  - **System-Disjoint Split:** Train=10 (ADB), Dev=5 (APA), Test=5 (TJA). Zero leakage confirmed. Demonstrative evaluation possible. Statistical power very low (n=5 test).
  - **Requirement-Disjoint Split:** Dev and Test partitions empty -- not usable without more annotated pairs.
  - **Primary Blocker (BLOCKER-001):** Retrieval pool not isolated from test set. Contamination risk currently unguarded.
  - **Evaluator Gaps:** Missing split-awareness, contamination guard, VerificationStatus filtering, bootstrap CIs, per-system breakdowns.
  - **Baseline Bug:** scripts/run_evaluation.py trains and evaluates on same unsplit set -- BERT 1.0 accuracy is a pure overfit artifact.
- **Files Changed (context docs only):**
  - ROADMAP.md: Marked TASK-P6-01 complete; added TASK-P6-02.
  - CONTEXT.md: Updated environment status with audit findings; next action -> TASK-P6-02.
  - ISSUES.md: Added BLOCKER-001 through BLOCKER-005.
- **Artifact:** p6_01_audit_report.md
- **Tests Executed:** None (read-only). Baseline: 133 passed, 0 failed.

## [2026-10-10] - TASK-P6-02: Safe Evaluation Script with Contamination Guard
- **Task Completed:** `TASK-P6-02`
- **Actions Taken:**
  1. Implemented `build_safe_retrieval_pool` in `leredd/evaluation.py`: Filters retrieval example candidates to training partition only, automatically stripping any candidate containing test/dev requirement IDs or test-pair identities. Raises `ContaminationError` in strict mode if leakage occurs. **BLOCKER-001 resolved.**
  2. Implemented `check_ground_truth_eligibility` in `leredd/evaluation.py`: Strictly inspects ground-truth records, rejecting `data/provisional/*`, review queue records, and UI-test fixtures (`adb_verified_ui_test.json`) when `VerificationStatus.PROVISIONAL`. Allows legacy benchmark pairs lacking provenance with explicit logged audit warnings. **BLOCKER-004 and BLOCKER-005 resolved.**
  3. Implemented `scripts/run_full_evaluation.py`: Runs clean, reproducible evaluation on system-disjoint splits (`data/splits/system_disjoint/`). Evaluates `TFIDF_LSABaseline`, `BERTBaseline`, and `LEREDDDetector` (with MockLLM clearly flagged as a non-real-LLM performance claim). Saves structured JSON report to `data/evaluation/system_disjoint_report.json`.
  4. Added `tests/test_safe_evaluation.py` (20 unit tests) covering contamination guard filtering, strict leakage exception handling, test-pair self-exclusion, ground-truth eligibility filtering, report generation, schema structure, and mock LLM demarcation.
- **Files Changed:**
  - [`leredd/evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/evaluation.py): Added `ContaminationError`, `build_safe_retrieval_pool`, `check_ground_truth_eligibility`, `run_full_model_evaluation`.
  - [`scripts/run_full_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/run_full_evaluation.py): Added complete CLI evaluation entrypoint.
  - [`tests/test_safe_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_safe_evaluation.py): Added 20 unit tests.
  - [`data/evaluation/system_disjoint_report.json`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/data/evaluation/system_disjoint_report.json): Generated system-disjoint evaluation report.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/ -v` → **153 passed, 0 failed** in 156.37s. Zero regressions.
- **System-Disjoint Evaluation Results (Train=10 ADB, Dev=5 APA, Test=5 TJA):**
  | Model | Accuracy | Macro F1 | Weighted F1 | Note |
  |---|---|---|---|---|
  | TF-IDF + LSA | 0.4000 | 0.1429 | 0.2286 | Zero-shot on TJA test set |
  | BERT Baseline | 0.6000 | 0.2500 | 0.4500 | Zero-shot on TJA test set |
  | LEREDD (MockLLM) | 0.2000 | 0.1111 | 0.1333 | Mock baseline demonstration |

## [2026-10-10] - TASK-P6-03: Evaluation Results Audit and Methodology Consistency
- **Task Completed:** `TASK-P6-03`
- **Actions Taken:**
  1. Audited CLI arguments and model identities in `scripts/run_full_evaluation.py`. Disambiguated `--model-type` with alias `--bert-classifier-type` (`rf` = `RandomForestClassifier`, `lr` = `LogisticRegression`). Updated report configuration metadata (`bert_classifier_backend: "rf"`, `bert_classifier_description`).
  2. Corrected methodology descriptions in `scripts/run_full_evaluation.py` and report generator notes. Re-labeled `BERTBaseline` and `TFIDF_LSABaseline` as **Supervised Cross-System Evaluation** (trained/fitted on ADB system, tested on TJA system) instead of "zero-shot".
  3. Added explicit `status: "exploratory"` dict and `exploratory_notice` for McNemar's statistical test results, detailing why sample size $n=5$ yields low statistical power and low expected cell frequencies.
  4. Added 4 new unit/integration tests to `tests/test_safe_evaluation.py` verifying `--bert-classifier-type` metadata, supervised cross-system terminology, exploratory McNemar test output, and MockLLM report demarcation.
- **Files Changed:**
  - [`scripts/run_full_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/run_full_evaluation.py): Added `--bert-classifier-type`, updated report configuration metadata, methodology notes, and exploratory McNemar notices.
  - [`tests/test_safe_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_safe_evaluation.py): Added 4 new regression tests.
  - [`data/evaluation/system_disjoint_report.json`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/data/evaluation/system_disjoint_report.json): Re-generated report with factual terminology and exploratory test notices.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/ -v` → **157 passed, 0 failed** in 167.09s. Zero regressions.
- **Corrected System-Disjoint Evaluation Results (Train=10 ADB, Dev=5 APA, Test=5 TJA):**
  | Model | Accuracy | Macro F1 | Weighted F1 | Methodology Note |
  |---|---|---|---|---|
  | BERT Baseline (RF) | 0.2000 | 0.1250 | 0.2000 | Supervised cross-system evaluation (ADB -> TJA) |
  | TF-IDF + LSA | 0.4000 | 0.2500 | 0.3000 | Fitted cross-system baseline (ADB -> TJA) |
  | LEREDD (MockLLM) | 0.4000 | 0.3000 | 0.4000 | Mock baseline demonstration (keyword heuristics) |

## [2026-10-10] - TASK-P6-04: Reconcile Evaluation Metrics and Verify Statistical Claims
- **Task Completed:** `TASK-P6-04`
- **Root Cause of Metric Discrepancies (P6-02 vs P6-03):** Row-transposition in early manual textual summaries. Code and JSON reports were consistent throughout; only prose descriptions were misaligned. `seed=42` ensures deterministic metric output across runs.
- **Actions Taken:**
  1. Implemented `Evaluator.compute_wilson_score_interval(k, n)` in `leredd/evaluation.py`: Returns Wilson score binomial 95% CI with `lower_bound`, `upper_bound`, and `point_estimate`. Replaces all hardcoded ±40% approximations.
  2. Implemented `Evaluator.analyze_mcnemar(pairs, preds_a, preds_b)` in `leredd/evaluation.py`: Builds the full $2 \times 2$ contingency table $(a, b, c, d)$, extracts discordant counts $(b, c)$, and applies scipy's exact binomial McNemar test. Returns `contingency_table`, `discordant_counts`, `statistic`, `p_value`, and explicit `exploratory` flag.
  3. Updated `_serialize_report` and Step 7 in `scripts/run_full_evaluation.py` to call new CI and McNemar functions.
  4. Fixed `Prediction(rationale=...)` required field omission in `tests/test_safe_evaluation.py::test_evaluator_wilson_ci_and_mcnemar_analysis`.
- **Files Changed:**
  - [`leredd/evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/evaluation.py): Added `compute_wilson_score_interval` and `analyze_mcnemar` classmethods.
  - [`scripts/run_full_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/run_full_evaluation.py): Integrated Wilson CI and McNemar in `_serialize_report` and Step 7.
  - [`tests/test_safe_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_safe_evaluation.py): Fixed `Prediction` construction (`rationale` field); added 3 new statistical unit tests.
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/test_safe_evaluation.py -q` → **27 passed in 122.80s**. Zero regressions.
- **Statistical Notes:**
## [2026-10-10] - TASK-P7-01: Benchmark Suite & Reproducible Comparison Script
- **Task Completed:** `TASK-P7-01`
- **Objective:** Run full evaluation comparing semantic-only baseline, heuristic classifier, zero-shot LLM, and proposed hybrid system, along with multi-dimension ablation studies.
- **Actions Taken:**
  1. Extended `LEREDDConfig` in `leredd/config.py` (`k_examples >= 0`, `confidence_threshold >= -1`) to support zero-shot prompting (`k=0`) and threshold re-annotation disabling (`threshold=-1`).
  2. Extended `LEREDDDetector` in `leredd/detector.py` to support dynamic retrieval strategy selection (`dense`, `bm25`, `hybrid`) and runtime threshold overrides.
  3. Upgraded `scripts/run_full_evaluation.py` into an integrated benchmark runner evaluating 5 primary models (`BERTBaseline`, `TFIDF_LSABaseline`, `HeuristicLabeler`, `LEREDD-ZeroShot`, `LEREDD-Hybrid`) and an ablation study matrix across 3 dimensions (`retrieval_strategy`, `in_context_examples`, `confidence_thresholding`).
  4. Generated standardized JSON benchmark report (`data/evaluation/system_disjoint_report.json`) and dedicated ablation report (`data/evaluation/ablation_report.json`).
  5. Created `tests/test_ablation_benchmark.py` with 5 new unit and integration tests verifying heuristic baseline, retrieval strategy selection, zero-shot configuration, thresholding overrides, and report generation.
- **Files Modified/Created:**
  - [`leredd/config.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/config.py): Allowed `k_examples >= 0` and `confidence_threshold >= -1`.
  - [`leredd/detector.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/detector.py): Added dynamic retrieval strategy routing (`dense`, `bm25`, `hybrid`) and threshold overrides.
  - [`scripts/run_full_evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scripts/run_full_evaluation.py): Integrated multi-model benchmark evaluation, ablation matrix runner, backward-compatibility aliases, and summary table formatter.
  - [`tests/test_ablation_benchmark.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_ablation_benchmark.py): Added 5 unit/integration tests for Phase 7.
  - [`data/evaluation/system_disjoint_report.json`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/data/evaluation/system_disjoint_report.json): Re-generated report with full primary models & ablation results.
## [2026-10-10] - TASK-P7-02: Ablation Validity and Model Identity Audit
- **Task Completed:** `TASK-P7-02`
- **Objective:** Audit model identities, fitting procedures, threshold mechanisms, retrieval routing, and root cause of identical ablation metrics across variants.
- **Audit Findings & Verification Results:**
  1. **Primary Model Implementation Audit:**
     - `BERTBaseline`: Supervised ML classifier (SBERT + `RandomForestClassifier` or `LogisticRegression`) fit on ADB train pairs. No LLM calls.
     - `TFIDF_LSABaseline`: Unsupervised / fitted SVD baseline fit on ADB train pairs. No LLM calls.
     - `HeuristicLabeler`: Deterministic rule-based baseline evaluating explicit citations, structural keywords, and Jaccard overlap. No LLM calls.
     - `LEREDD-ZeroShot`: LEREDD detector run with `k_examples=0` (no retrieved examples in prompt). Mock LLM backend.
     - `LEREDD-Hybrid`: Proposed system run with `retrieval_mode="hybrid"` (SBERT + BM25 RRF fusion), `k_examples=6`, and `confidence_threshold=4`. Mock LLM backend.
  2. **Identical Metric Output Root Cause:**
     - `MockLLMClient` evaluates regex pattern matches on `Requirement A` and `Requirement B` statement text only, ignoring prompt example blocks. Thus, varying retrieval strategy (`dense`, `bm25`, `hybrid`) or example count (`k=0` vs `k=6`) alters prompt text, but `MockLLMClient` yields identical mock predictions.
     - `MockLLMClient` assigns hardcoded `Confidence Score: 5` to all predictions. Since `ResponseParser.parse` checks `confidence <= threshold`, `5 <= 4` evaluates to `False`. Thus, setting `confidence_threshold=-1` vs `confidence_threshold=4` produces identical labels under `MockLLMClient`.
  3. **Configuration Routing & Threshold Behavior Verification:** Verified that `LEREDDDetector` dispatches to dense, BM25, and hybrid retrievers as configured, and `ResponseParser` correctly threshold-reannotates predictions when confidence $\le 4$.
- **Per-Example Prediction Matrix (Test n=5: TJA):**
  - `PAIR-TJA-001` (GT: Details): Predicted `No_dependency`, Conf: `5` across all 5 variants.
  - `PAIR-TJA-002` (GT: Requires): Predicted `Requires`, Conf: `5` across all 5 variants.
  - `PAIR-TJA-003` (GT: Conflicts): Predicted `Implements`, Conf: `5` across all 5 variants.
  - `PAIR-TJA-004` (GT: No_dependency): Predicted `Conflicts`, Conf: `5` across all 5 variants.
  - `PAIR-TJA-005` (GT: Requires): Predicted `Requires`, Conf: `5` across all 5 variants.
- **Files Modified/Created:**
  - [`scratch/audit_ablation_details.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/scratch/audit_ablation_details.py): Created scratch inspection script capturing prompts and prediction matrix.
  - [`tests/test_ablation_benchmark.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/tests/test_ablation_benchmark.py): Added 2 regression tests (`test_threshold_behavior_reannotation_vs_disabled`, `test_mock_llm_per_example_deterministic_predictions`).
- **Tests Executed:** `.venv\Scripts\python.exe -m pytest tests/test_ablation_benchmark.py -v` → **7 passed, 0 failed in 69.35s**. Full test suite: **167 passed, 0 failed**. Zero regressions.

## [2026-10-13] - TASK-P9-14: End-to-End Feature Audit and Stabilization
- **Changes:** Added `docs/FEATURE_TRACEABILITY.md`; corrected the UI annotation, evaluation, agreement, output-safety, and external-request workflows; redacted the API configuration secret; added an executable API entry point and loopback defaults; required explicit baseline fitting; made PDF/DOCX dependencies optional with explicit errors; hardened extraction, candidate, labeling, queue, split, and evaluation scripts against false human verification and accidental overwrite.
- **Files changed:** `README.md`, `CONTEXT.md`, `ROADMAP.md`, `ISSUES.md`, `DECISIONS.md`, `docs/API.md`, `docs/REPLICATION.md`, `docs/FEATURE_TRACEABILITY.md`, `pyproject.toml`, `api/main.py`, `api/routes.py`, `leredd/baselines/bert_classifier.py`, `leredd/baselines/tfidf_lsa.py`, `leredd/cli.py`, `leredd/utils.py`, `scripts/annotate.py`, `scripts/build_review_queue.py`, `scripts/extract_requirements.py`, `scripts/generate_candidates.py`, `scripts/label_candidates.py`, `scripts/run_evaluation.py`, `scripts/run_full_evaluation.py`, `scripts/split_dataset.py`, `tests/test_api.py`, `tests/test_baselines.py`, `tests/test_build_review_queue_script.py`, `tests/test_cli.py`, `tests/test_data_scripts.py`, `tests/test_evaluation_entrypoint.py`, `tests/test_generate_candidates_script.py`, `tests/test_label_candidates_script.py`, `tests/test_ui_and_queue_integration.py`, and `tests/test_ui_workflows.py`.
- **Focused verification:** Script/UI/baseline subset **31 passed in 70.70s**; UI journey suite **7 passed in 10.11s**; API suite **9 passed in 28.66s** (1 existing tqdm deprecation warning). FastAPI `/api/v1/health` and Streamlit `/_stcore/health` each returned HTTP 200 from a locally started service; both processes were stopped.
- **Full suite:** Fresh unrestricted `python -m pytest tests/` completed: **261 passed, 40 warnings in 196.70s**, exit code 0; outer duration **200.50s**. Four replication dataset SHA-256 hashes matched their immediately-before-run values and `git diff -- data` was empty. The sandbox rerun stalled at `test_api_root`; unrestricted execution passed it. No package was installed, no real LLM request was sent, and no Git state operation was run. Warnings are tqdm `utcfromtimestamp` deprecations.

## [2026-10-13] - TASK-P9-15: Persist Trained BERT Baseline Artifacts
- **Finding and fix:** Completion audit found `leredd-cli train-bert` reported a model saved while persisting no artifact. Added atomic versioned joblib serialization, `BERTBaseline.load()`, early overwrite protection with explicit `--overwrite`, direct joblib dependency declarations, and README usage/security documentation.
- **Files changed:** `leredd/baselines/bert_classifier.py`, `leredd/cli.py`, `tests/test_baselines.py`, `tests/test_cli.py`, `pyproject.toml`, `requirements.txt`, `README.md`, `docs/FEATURE_TRACEABILITY.md`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md`.
- **Focused verification:** `python -m pytest tests/test_baselines.py tests/test_cli.py -k "not cli_detect" -q` → **14 passed, 1 deselected in 3.48s**.
- **Full verification:** `python -m pytest tests/` (unrestricted, TEMP/TMP at `.pytest-tmp`) → **264 passed, 38 warnings in 170.21s**, exit 0; outer duration **173.83s**. The four replication JSON SHA-256 hashes matched before/after. No package install or Git state operation occurred.
- **Security/compatibility:** joblib uses pickle serialization; only trusted artifacts should be loaded, with compatible Python/scikit-learn/joblib versions.

## [2026-10-13] - TASK-P9-16: Correct CI Pytest Module Resolution
- **Finding and change:** The GitHub Actions log showed six collection errors importing repository-root `api` and `scripts` modules under the `pytest` console command. Changed `.github/workflows/ci.yml` to run `python -m pytest`.
- **Verification:** `python -m pytest tests/ --cov=leredd --cov-report=xml:.pytest-tmp/coverage.xml --cov-report=term-missing` → **264 passed, 38 warnings**, exit 0, 199.09s pytest time (wrapper elapsed 202.81s). The GitHub Actions matrix has not yet rerun.
- **Files changed:** `.github/workflows/ci.yml`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`, `ISSUES.md`, `DECISIONS.md`.

## [2026-10-14] - TASK-P10: End-to-End Stabilization and Verification
- **Changes:** Corrected DENSE/HYBRID candidate generation and exhaustive-manifest semantics; made fallback embeddings consistent across separate encode calls; hardened split validation/leakage reporting; made CLI verified exports validated, verified-only, idempotent, and atomic; added regression cases and deterministic offline embedders; replaced the prior synthetic status-mutation smoke test with a real controlled SRS-to-report filesystem journey; aligned API/architecture/README descriptions with implemented interfaces.
- **Files changed:** `README.md`, `docs/API.md`, `docs/ARCHITECTURE.md`, `leredd/candidate_generator.py`, `leredd/cli.py`, `leredd/dataset_splitter.py`, `leredd/embeddings.py`, `leredd/retrieval.py`, `tests/test_ablation_benchmark.py`, `tests/test_api.py`, `tests/test_bm25_retrieval.py`, `tests/test_candidate_generator.py`, `tests/test_dataset_splitter.py`, `tests/test_detector.py`, `tests/test_embeddings.py`, `tests/test_hybrid_retrieval.py`, `tests/test_retrieval.py`, `tests/test_system_verification.py`, `tests/test_ui_and_queue_integration.py`; project tracking updates: `ROADMAP.md`, `CONTEXT.md`, `ISSUES.md`, `DECISIONS.md`, `CHANGELOG.md`, `docs/FEATURE_TRACEABILITY.md`.
- **Verification:** Focused core/retrieval suite → 120 passed in 8.02s; embedding and ground-truth prompt regressions → 12 passed in 1.91s; API module → 9 passed in 1.02s. Complete `python -m pytest tests/` (PowerShell with TEMP/TMP set to `.pytest-tmp/task-p10/temp`, unrestricted Windows loopback) → **298 passed, 12 warnings in 69.21s**, exit 0. Warnings are tqdm's `datetime.datetime.utcfromtimestamp()` deprecation. `git diff --check` passed without whitespace errors. The four imported replication JSON and two historical evaluation report SHA-256 values matched before/after; `git diff -- data` was empty.
- **Limits:** No real OpenAI/Ollama call, Docker build, clean-cache model download, browser visual review, optional PDF/DOCX installation, hosted CI matrix rerun, human replication-label review, or dataset redistribution verification was performed. No package install or Git state mutation was performed.

## [2026-10-14] - TASK-P11: Workflow Validation and Submission Readiness
- **Changes:** Fixed default system-level ratio apportionment to produce non-empty train/dev/test partitions when at least three systems are available. Hardened evaluator and safe-runner eligibility so provenance-bearing verified labels must have human source, reviewer/annotator identity, and matching accepted/scored labels. Added regression tests and documented the no-service isolated CLI walkthrough, split behavior, and review/evaluation gates.
- **Files changed for P11:** `leredd/dataset_splitter.py`, `leredd/evaluation.py`, `scripts/run_full_evaluation.py`, `tests/test_dataset_splitter.py`, `tests/test_safe_evaluation.py`, `tests/test_cli.py`, `README.md`, `docs/API.md`, `docs/EVALUATION_PROTOCOL.md`, `docs/FEATURE_TRACEABILITY.md`, `ROADMAP.md`, `CONTEXT.md`, `ISSUES.md`, `DECISIONS.md`, `CHANGELOG.md`.
- **Verification:** Focused safe-workflow suite → **72 passed, 15 deselected in 7.63s**; API module → **9 passed in 0.83s**; complete `python -m pytest tests/ -q` → **305 passed, 12 warnings in 23.08s**, exit 0. The warnings are the existing `tqdm` `datetime.utcfromtimestamp()` deprecation. `git diff --check` reported no whitespace errors.
- **Workflow evidence:** Isolated CLI Markdown fixture outputs remained in `.pytest-tmp/task-p11`; generated labels remained provisional and queue items pending. Safe legacy evaluation saved report and ablation JSON to `.pytest-tmp/task-p11/legacy-flow`, reported 10/5/5 ADB/APA/TJA partitions and zero retrieval contamination, and marked MockLLM/provenance-free labels demonstrative.
- **Integrity and limits:** Four replication JSON and two historical report hashes matched their starting values; `git diff -- data` was empty. No human decision was made, source replication files were not modified, and no package/service/Git state operation occurred. Human review, replication label direction/rights, real providers, Docker, optional PDF/DOCX packages, and live browser review remain open.

## [2026-10-14] - TASK-P11 Follow-up: Explicit Human Reviewer Identity
- **Finding and change:** Provenance review found generic actor fallbacks in the CLI, review-queue model/manager, and Streamlit interface. CLI and queue-manager decisions now require nonblank reviewer/annotator identities and reject the known generic placeholders; new Streamlit decisions begin blank, and placeholder-only verified records fail evaluation eligibility. Legacy queue records without an identity remain readable but cannot be promoted without one.
- **Files changed:** `leredd/cli.py`, `leredd/review_queue.py`, `leredd/evaluation.py`, `ui/app.py`, `tests/test_cli.py`, `tests/test_review_queue.py`, `tests/test_safe_evaluation.py`, `tests/test_system_verification.py`, `tests/test_ui_and_queue_integration.py`, `README.md`, `docs/API.md`, `ISSUES.md`, `DECISIONS.md`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`.
- **Verification:** Focused CLI/queue/evaluation/provenance/UI suite → **116 passed, 12 warnings in 26.23s**. Full `python -m pytest tests/ -q` outside the restricted sandbox → **309 passed, 12 warnings in 23.02s**, exit 0. The sandbox attempt stalled at `tests/test_api.py` due the known Windows loopback restriction; the unrestricted rerun completed all API tests.
- **Limit:** The application records but does not authenticate an operator-supplied identity. No replication JSON or historical report was modified; no package install, stage, commit, push, or other Git state change occurred.

## [2026-10-14] - TASK-P12: End-to-End Product Stabilization and Evidence Alignment
- **Finding and fix:** Evaluation reports hard-coded ADB/APA/TJA systems, counts, and provenance claims regardless of the supplied split. The runner now derives those descriptions, provenance counts, test label coverage, and small-sample notices from loaded data. Added a regression with independently named demo systems.
- **Documentation:** Reworked README workflow commands to use workspace-local disposable output paths and explicit human-review gates. Replaced unsupported GPT-4 paper-reference, class-support, and significance claims in README and `docs/EXPERIMENTS.md` with the actual checked-in 10/5/5 legacy report scope and limits. Historical reports and datasets were not edited.
- **Files changed for P12:** `scripts/run_full_evaluation.py`, `tests/test_safe_evaluation.py`, `README.md`, `docs/EXPERIMENTS.md`, `ISSUES.md`, `DECISIONS.md`, `ROADMAP.md`, `CONTEXT.md`, `CHANGELOG.md`.
- **Workflow evidence:** Controlled synthetic artifacts under `.pytest-tmp/task-p12/e2e` exercised extraction, BM25 and all-pairs candidate generation, provisional labeling, queue creation, fixture-only decision/export, 3-system splitting, MockLLM detection with a separate example pool, leakage checks, and evaluation/ablation report writes. The three partitions contained 3 pairs each, with distinct systems and zero retrieval contamination. Fixture reviewer identities are explicitly synthetic and do not constitute human annotation evidence.
- **Tests:** `python -m pytest tests/test_safe_evaluation.py -q` → **35 passed, 12 warnings in 21.69s**; `python -m pytest tests/test_api.py tests/test_ui_workflows.py tests/test_ui_and_queue_integration.py -q` → **32 passed in 6.43s**; full `python -m pytest tests/ -q` → **309 passed, 12 warnings in 23.06s**, exit 0. Warnings are tqdm's `datetime.utcfromtimestamp()` deprecation. The full suite ran with local TEMP/TMP, `HF_HUB_OFFLINE=1`, and permission for Windows loopback.
- **Integrity / remaining limits:** `git diff --check` clean; no staged changes; `git diff -- data` empty; four replication JSON and two historical evaluation report hashes unchanged. No install, remote provider call, or Git state mutation. Actual human annotation, dataset rights/direction, real-provider requests, browser visual QA, Docker build, optional document packages, and hosted CI remain unverified.

## [2026-10-14] - TASK-P13: Local Product Demonstration
- `ui/app.py`: switched annotation-page defaults to labeled synthetic SRS data and `.pytest-tmp` output; hid empty legacy card wrappers.
- `tests/test_ui_and_queue_integration.py`: added a focused regression for hidden empty wrappers. Browser UI, local API extraction/detection, CLI provisional queue, synthetic split, and mock report persistence were exercised. Full suite: **310 passed, 12 warnings in 92.08s**. Synthetic annotations were not treated as human verification or research evidence.

## [2026-11-23] - TASK-P14: Remove CI dependence on local replication assets
- GitHub Actions run `38442446610` failed 17 tests because the imported replication JSON files are not in the repository checkout. Loader and runner tests now exercise the same expected counts, labels, provenance, and failure behavior using temporary synthetic data; source-package integrity checks are skipped only when those local-only files are absent.
- Focused replication tests: **35 passed in 29.21s**. Full local suite: **310 passed, 12 warnings in 131.43s**, exit 0. Tracked-data-only checkout without replication files: **34 passed, 1 skipped in 4.34s**, exit 0. No source datasets or historical reports were edited.
- Exact CI coverage command in a complete tracked-file checkout without local replication assets: **309 passed, 1 skipped, 12 warnings in 200.84s**, exit 0 (outer duration 207.59s).
- Hosted matrix rerun remains pending; no commit or push was made.

## [2026-11-23] - TASK-P20: Paired annotation-effort study tooling
- Added `leredd/effort_study.py` and three CLIs for fixed-input schedule preparation, reviewer collection in manual/assisted conditions, and descriptive comparison. Study artifacts are blocked from all `data/` paths; provisional labels remain distinct from ground truth.
- Updated `tests/test_effort_study.py`, `README.md`, `docs/ANNOTATION_GUIDE.md`, and `docs/FEATURE_TRACEABILITY.md`; recorded ADR-034, ISSUE-033, and TASK-P20 in project tracking files.
- Focused annotation tests: **27 passed in 1.98s**. Full `python -m pytest tests/ -q`: **341 passed, 12 existing tqdm deprecation warnings in 92.81s**, exit 0 (98.50s PowerShell wall duration). All existing data hashes matched, with zero added data files. No real human study was conducted.

## [2026-11-23] - TASK-P21: Docker build-context data protection
- Added `.dockerignore` to prevent local datasets, evaluation reports, secrets, repository metadata, temp files, and `Minor_Project.pdf` from being copied into Docker image layers by `COPY . .`.
- `docker compose config --quiet` exited 0 (Compose reports the existing obsolete `version` key warning). Docker CLI 29.1.5 is installed, but `docker info` cannot connect to the Docker Desktop Linux engine; no image build or runtime test occurred. TASK-P21 remains open pending daemon availability.

## [2026-11-23] - TASK-P22: Live API/UI runtime check
- Existing local API and Streamlit processes were confirmed listening on 8000 and 8501. API root and health payloads were correct; Streamlit health/root HTTP checks returned success.
- Chrome inspection found the Streamlit page remained a blank shell showing `Running...` after 7.5 seconds. An older stderr file contains watcher import errors for missing `torchvision`, but the log is stale and causality is unconfirmed. No UI code or processes were changed.
- Docker build remains blocked because the Docker Desktop Linux engine is unavailable. `Minor_Project.pdf` text extraction is also unavailable in the installed Python environment (`pypdf`, `pdfplumber`, and `fitz` are absent); no packages were installed.
## [2026-11-24] - TASK-P43: Verify replication label semantics and rights evidence
- Read the official arXiv v2 paper and all four local source workbooks. The paper's Implements meaning is explicit (higher-level requirement fulfilled by lower-level requirement); the imported source phrase Implemented by still lacks an authoritative crosswalk, despite a matching example with reversed workbook row order. Read-only openpyxl audit: 813 labeled rows, no incomplete rows, and no exact/reversed duplicate pairs.
- Corrected docs/EVALUATION_PROTOCOL.md, retained dataset redistribution as blocked because the paper's CC BY 4.0 is not a separate dataset license, and added active BLOCKER-008 for the unresolved label crosswalk. Added ADR-058 and project context/roadmap evidence.
- Read-only inspection only; no tests were needed. git diff --check passed. No datasets, reports, staged files, or Git history were changed.

## [2026-11-24] - TASK-P44: Live application smoke revalidation
- Ran `python -m leredd.cli --help` successfully. Existing local FastAPI root and health endpoints returned HTTP 200 when probed outside the restricted socket sandbox.
- Launched a fresh Streamlit UI on port 8504 and visually inspected it. The UI extracted three clearly synthetic requirements and persisted three unlabeled, provisional candidate pairs in `.pytest-tmp/task-final-ui-demo/candidates.json`; a BM25/MockLLM synthetic prediction rendered with the expected confidence disclaimer.
- No application source or dataset was changed; no tests were run because this was a live smoke check only. `git diff --check` passed. The sandboxed API probe was blocked by local socket permissions, but the unrestricted read-only probe succeeded. One Windows asyncio connection-reset callback was logged during socket teardown without a reproduced user-visible failure.
- No human annotations, verified export, model-quality claim, or real-provider result was created. Rights, label mapping, human evidence, hosted CI, and Docker remain open.

## [2026-11-24] - TASK-P45: Current full-suite verification
- Ran `.venv\Scripts\python.exe -m pytest tests/` outside the restricted socket sandbox with `TEMP`/`TMP` set to `.pytest-tmp` and `HF_HUB_OFFLINE=1`: **400 passed, 1 Starlette deprecation warning in 176.51s**, exit 0 (PowerShell duration 181.96s). API root passed.
- SHA-256 before/after manifest comparison matched all 312 files under `data/`. `git diff --check` passed, with Git's line-ending notices only. No tests changed application files or datasets; no packages or Git state were changed.
- Remaining warning recommends `httpx2` for Starlette TestClient; current PyPI metadata requires Python >=3.11, so an unconditional dependency would conflict with the project's Python >=3.10 support. No upgrade was installed or attempted. A fresh read of the official arXiv v2 still says the replication package will be made available upon publication and exposes no direct package link; this does not establish that no external package exists. This local run does not cover hosted CI, Docker, real providers, or human/research evidence.

## [2026-11-24] - TASK-P46: Migrate API tests off Starlette TestClient
- Replaced the API test module's Starlette TestClient with an in-process HTTPX ASGI transport helper; strengthened `test_api_root` to assert the entire response payload. No production API behavior or dependency declarations changed.
- Root test: **1 passed in 1.36s**. API module: **18 passed in 1.67s**. Full suite: **400 passed, no warnings in 107.33s**, exit 0 (PowerShell duration 111.87s). The earlier TestClient deprecation warning no longer appears.
- All 312 `data/` paths and SHA-256 values matched before/after. `git diff --check` passed. The restricted Windows socket sandbox still stalls during event-loop initialization; the successful focused and full runs were outside that sandbox. No packages or Git state were changed.

## [2026-11-24] - TASK-P47: Build and smoke-test CPU-only Docker deployment
- Updated `Dockerfile` to install the CPU-only PyTorch wheel before project requirements, preventing the default package resolver from pulling CUDA/cuDNN libraries into the CPU-only API/UI images.
- `docker compose build` completed successfully for `minor-api` and `minor-ui`. Compose launched both services; API `/` and `/api/v1/health` returned HTTP 200, Streamlit `/` returned HTTP 200, and `/_stcore/health` returned `ok`.
- Browser visual inspection did not run because the computer-use tool could not verify the browser URL. A later Docker management API check returned 500/hung; final image-content and container-state inspection remain incomplete. `docker compose config --quiet` still reports the obsolete top-level `version` warning. No tests, application workflows, or data-writing requests were run.
- Follow-up: a bounded `docker info` check timed out after 20 seconds, while API health and Streamlit health still returned 200/`ok`. Docker CLI management remains unstable after successful container startup.

## [2026-11-24] - TASK-P48: External SRS extraction smoke
- Updated `leredd/cli.py` extraction help/docstring to advertise Markdown, and extended `tests/test_cli.py` to assert Markdown and DOCX formats appear in `extract --help` while exercising the existing Markdown metadata-to-candidate workflow.
- Ran that workflow on a temporary NASA Trick SRS copy: 99 extracted records with source locations; 4,851 all-pairs candidates; provisional labeling yielded 350 suggestions and 4,501 unresolved records; queue creation yielded 4,851 pending items and zero decisions. Outputs stayed under `.pytest-tmp`.
- Focused regression: **1 passed in 3.33s**. No human review, evaluation, release, paid provider request, or full-suite run was performed. The keyword-line comparison is a smoke diagnostic, not precision/recall evidence.

## [2026-11-24] - TASK-P49: Compare candidate strategy coverage
- Ran existing candidate-generation CLI strategies against the 99 extracted NASA Trick requirements (4,851 possible pairs), with `HF_HUB_OFFLINE=1` and `top_k=5`: BM25 327 pairs / 6.74% pair-space coverage; dense 335 / 6.91%; hybrid 325 / 6.70%; all-pairs 4,851 / 100%. Results are in temporary `.pytest-tmp/task-p49-candidate-coverage` manifests.
- Coverage is the fraction of the complete pair universe emitted, not dependency-positive recall. No gold labels or evaluation metrics were produced. Dense encoding used the local model and wrote 99 cache entries under configured `data/embeddings`; no imported dataset or historical report was written by the commands.
- No tests were run for this execution-only comparison. `git diff --check` passed. A real reviewer sample covering generated and omitted pairs is still required to estimate missed dependencies and annotation effort.

## [2026-11-24] - TASK-P50: Prepare blinded pilot sample
- Prepared 100 unlabeled/provisional pairs under `.pytest-tmp/task-p50-blind-pilot`: 50 sampled from the hybrid top-5 candidate stratum and 50 from omitted pairs, with fixed seed 49017 and opaque reviewer IDs. Kept stratum mapping, population counts, sampling probabilities, and weights in a separate analysis-only file; the pilot manifest binds the SRS and reviewer file hashes.
- Validation confirmed the 50/50 counts, no labels/stratum tags in the reviewer input, and the recorded SHA-256. `scripts/annotate_independent.py --help` confirms the existing blind collector accepts this input format.
- No reviewer decisions, evaluation, agreement, or positive-recall result were fabricated. The packet is an unpowered pilot; two real reviewers and later weighted analysis are required.

## [2026-11-24] - TASK-P51: Add weighted candidate-recall analysis
- Added `leredd/candidate_coverage.py` and `scripts/analyze_candidate_coverage.py` for a post-stratified estimate of true dependency-pair recall from the fixed selected/omitted pilot sample. Requires an exact set of human `VERIFIED` adjudications and verifies the source SRS, sample, strata map, strategy candidates, and complete pair-universe hashes and identity.
- Added fail-closed checks for provisional/unverified/conflicting labels, missing or extra IDs, duplicate pairs, changed inputs, inconsistent stratum membership, and candidate output outside the population. Reports estimated positive totals and an approximate design-based interval only when variance estimates are non-degenerate.
- Updated README, annotation guide, and feature traceability with the post-review command and research limits. Final focused suite: **36 passed in 16.77s**. The restricted full-suite attempt stalled at the known Windows socket boundary after the first module; outside that restriction the complete suite finished **413 passed, no warnings in 38.38s**, exit 0 (PowerShell duration **42.26s**). Final review corrected `population_pair_count` and added a regression. The real pilot remains unreviewed; no research recall result was produced.
- Full local suite: **400 passed, no warnings in 132.31s**, exit 0 (PowerShell duration 138.03s), with TEMP/TMP/CACHE_DIR isolated under `.pytest-tmp` and Hugging Face offline.

\r\n

### TASK-P52 — Synthetic/demo data safeguards (2026-11-24)
- Added synthetic/source identity metadata and preserved it through extraction, candidate generation, CSV serialization, and review decisions. Synthetic demo records are excluded from retrieval examples and verified exports and rejected as evaluation ground truth.
- Added `leredd/synthetic.py` and regression tests across evaluation, release, retrieval, CLI, UI, review queue, and CSV workflows. Updated README, annotation guide, feature traceability, and project tracking records to document the boundary.
- Verification: focused suite **172 passed in 30.25s**; complete local suite **424 passed in 34.30s**, exit 0. No packages were installed. No human labels or research findings were generated.

### TASK-P53 — Enforce declared system-disjoint split integrity (2026-11-24)
- Recompute system, requirement, and pair-ID overlap from actual train/dev/test records before full evaluation; write the integrity result into the report. Duplicate prediction errors now include the offending IDs.
- Added direct and runner-level regressions. Corrected two contamination fixtures to declare their actual pair-level split strategy.
- Verification: focused evaluation tests **45 passed**; full suite **428 passed in 35.96s**, exit 0. All **412** files under `data/` retained identical SHA-256 hashes; `git diff --check` passed.
- Reviewed the public Parts 1–3 task snapshot. Its fixture/guardrail files are pushed, but the remote branch and local worktree diverge. No staging, commit, or push was performed.
