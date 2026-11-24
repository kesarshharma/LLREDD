# ISSUES.md — Unresolved Bugs, Blockers, and Risks Log

This log tracks bugs, technical debts, risks, and blockers discovered during project execution.

---

## Active Issues & Blockers

### [ISSUE-039] Low confidence was silently converted into `No_dependency`
- **Severity:** High / Label integrity
- **Status:** ✅ Resolved locally (TASK-P27, 2026-11-24)
- **Description:** `ResponseParser` changed any non-negative prediction at or below the confidence cutoff into `No_dependency`. This conflated uncertainty with an explicit negative prediction and conflicted with the proposal's requirement to identify uncertain outputs for review.
- **Resolution:** Predictions now preserve the parsed dependency class and separately record `validation_status` and `validation_threshold`. Low confidence is marked `requires_review`; high confidence is marked `passed_threshold`; a negative threshold is recorded as `threshold_disabled`. Streamlit displays the screening result and explains that it is not ground-truth verification or calibrated probability. Evaluation reports summarize screening coverage and review rate.
- **Verification:** Focused parser/API/UI/ablation/evaluation tests: **72 passed, 12 warnings in 105.79s**. Full suite: **347 passed, 12 warnings in 102.35s**, exit 0. All 12 warnings are the existing tqdm UTC datetime deprecation.
- **Boundary:** The UI offers an explicit action to add low-confidence predictions to the pending review queue (TASK-P28); it does not verify them. MockLLM always reports confidence 5, so review-rate behavior and utility require a variable-confidence controlled fixture or an authorized real provider. UI verification used AppTest rather than a fresh browser session.

#### ISSUE-039 follow-up (TASK-P28, 2026-11-24)
- A controlled Streamlit AppTest produced a low-confidence `Requires` prediction and explicitly queued it as `pending` with provisional machine provenance and a `low_confidence` reason. The label was retained and no human decision/export occurred.
- The UI queue JSON is atomically replaced. Separately, shared queue persistence stages queue and manifest JSON in sibling temporary files and atomically replaces each. Regression: a simulated failure of the first replacement leaves both pre-existing files unchanged and removes temporary files. The pair of replacements is not a cross-file transaction; failure of the second replacement can leave the manifest stale while the queue is authoritative. The UI derives counts directly from queue items and does not depend on a sidecar manifest.
- Full suite: **348 passed, 12 warnings in 89.15s**, exit 0 (PowerShell duration 97.71s). The remaining confidence-screening utility limitation is empirical, not a queue-routing defect.

### [ISSUE-040] Confidence-cutoff copy mislabeled a maximum score as “low confidence”
- **Severity:** Low / UI clarity
- **Status:** ✅ Resolved locally (TASK-P30, 2026-11-24)
- **Description:** The inclusive review rule routes scores at or below the configured cutoff. At cutoff 5, a 5/5 score was correctly routed but the UI and queue rationale described it as “low confidence,” which contradicted the displayed score.
- **Resolution:** Detection and review pages now explain that the score is at or below the selected cutoff; queue rationale uses the same precise wording. The stored `low_confidence` enum string and inclusive behavior remain compatible.
- **Verification:** Focused UI/queue tests **16 passed**; full suite **348 passed, 12 warnings**, exit 0.
- **Boundary:** This only clarifies the rule. It does not establish that self-reported confidence is calibrated or that the cutoff improves review quality.

### [ISSUE-041] Streamlit reloaded the dense embedding model for each detection
- **Severity:** Medium / Repeated interaction latency
- **Status:** ✅ Resolved locally for repeated runs within one Streamlit session (TASK-P31, 2026-11-24)
- **Evidence:** The detection click constructed a fresh `LEREDDDetector`; with the local 10-pair annotated pool and dense retrieval, that detector lazily constructed `EmbeddingModel` each time. The session did not retain that resource.
- **Resolution:** Keep the loaded embedder in session state keyed by embedding model and cache directory, and pass it into subsequent detectors. A regression performs two controlled UI detections and asserts the second uses the same embedder object. Full suite: **348 passed, 12 warnings**, exit 0.
- **Boundary:** First dense use still incurs model initialization. This change establishes object reuse, not an end-to-end latency measurement; concurrency and clean-machine model download were not tested.

### [ISSUE-042] Hybrid example retrieval did not use available requirement metadata
- **Severity:** Medium (proposal alignment and retrieval experiment capability)
- **Status:** 🟡 Software capability added locally (TASK-P32); empirical contribution remains unverified.
- **Description:** The detection-time hybrid retriever fused dense and BM25 rankings but did not use structured component/subsystem metadata, although the proposal identifies metadata alignment as an intended hybrid signal when such values are available.
- **Implementation:** Added optional `Requirement.component` and `Requirement.subsystem` fields and a conditional exact-overlap RRF ranking, including existing `system` metadata. It is omitted when no target metadata or candidate overlap exists. Regression tests verify a controlled ranking change and no signal for generic/missing metadata.
- **Evidence gap:** Current legacy/imported datasets do not establish reliable component/subsystem coverage. No retrieval effectiveness, weight tuning, or ablation result is claimed. Use a rights-cleared dataset with dependable structured metadata and verified labels before making a research claim.
- **TASK-P34 update:** The values can now be supplied explicitly through CLI sidecar JSON, API extraction, and Streamlit extraction; API/UI detection inputs also accept target metadata. Focused tests passed (50); full suite passed (364, 12 existing warnings), and 11 replication/evaluation artifact hashes were unchanged. This closes the interface-wiring gap but does not add metadata to historical datasets or establish retrieval effectiveness.

### [ISSUE-043] Provisional labeling could not collect multiple LLM votes
- **Severity:** Medium (proposal's semi-automated labeling and uncertainty review)
- **Status:** 🟡 Software capability added locally (TASK-P33); independent model behavior and empirical value unverified.
- **Description:** The CLI offered heuristic, single-LLM, or heuristic-plus-single-LLM labeling, so it could not collect multiple model predictions as the proposal calls for where feasible.
- **Resolution:** Added opt-in `multi_llm` mode and `--models` support for multi-model or hybrid runs. Only unanimous valid votes yield a provisional label; disagreements and incomplete votes are unresolved and record each model vote for review. CLI and controlled-client tests pass.
- **Remaining:** No real provider/model was invoked. The default MockLLM path is a software test, not independent model evidence. Human review and label-quality evaluation remain required.

### [ISSUE-044] LLM disk cache omitted provider/model identity
- **Severity:** High for multi-model comparison validity
- **Status:** ✅ Resolved locally (TASK-P33).
- **Evidence:** Cache keys previously used client class, temperature, and prompt but not the configured model ID. Same-provider models given the same prompt could therefore receive a cached response produced by another model.
- **Resolution:** Versioned cache keys now include provider class and model ID in addition to temperature and prompt. A regression confirms separate model outputs and same-model cache reuse.
- **Compatibility note:** Legacy keys are deliberately not read because they cannot prove which model produced a stored response; a first request may call the provider again. No real provider was called during this fix.

### [ISSUE-038] Retrieval strategy cannot be selected per detection request
- **Severity:** Medium (supported workflow access)
- **Status:** ✅ Resolved locally (TASK-P26, 2026-11-24); fresh-browser inspection of the updated Streamlit selector remains pending.
- **Description:** The detector supported dense, BM25, and hybrid retrieval, but API requests and the Streamlit detection form did not expose a per-request choice. Users could not select the fast sparse path through those interfaces.
- **Resolution:** Added validated optional API request fields for single and batch detection and a Streamlit selector; omission retains the configured default. API/UI regression tests cover pass-through and invalid API values. Focused tests: **20 passed**; full suite: **346 passed, 12 existing warnings**, exit 0. A local synthetic BM25 API smoke returned 200 without SentenceTransformer initialization.
- **Remaining verification:** Restart Streamlit and visually inspect the selector; measure dense cold-start latency separately. No retrieval-quality benchmark was run and no result is claimed.

### [ISSUE-036] Zero-shot API prediction unnecessarily initialized SentenceTransformer
- **Severity:** Medium / First-request latency
- **Status:** ✅ Resolved locally (TASK-P24, 2026-11-24)
- **Files:** `leredd/detector.py`, `tests/test_detector.py`
- **Description:** Detector construction eagerly initialized the embedding model even when detection had no in-context examples and RAG was disabled. On a cold API process, this made a MockLLM zero-shot request pay model startup cost despite embeddings being unused.
- **Resolution:** Deferred embedding and dense/RAG retriever construction until retrieval accesses them. Added a regression that fails if zero-shot detection constructs `EmbeddingModel`. Focused tests: **46 passed**; full suite: **342 passed, 12 warnings**, exit 0. Live local MockLLM `/detect` returned HTTP 200 in **164 ms**, with no model-load log. All 11 replication/evaluation artifact hashes were unchanged.
- **Boundary:** Dense retrieval still loads the model on first use. Hosted CI and clean-cache model initialization remain separate verification work.

### [ISSUE-037] Extractor truncated alphabetic requirement-ID suffixes
- **Severity:** Medium / Requirement identity integrity
- **Status:** ✅ Resolved locally (TASK-P25, 2026-11-24)
- **Files:** `leredd/utils.py`, `tests/test_utils.py`
- **Description:** The explicit identifier pattern accepted `REQ-ADB-001` but stopped at the first alphabetic segment in IDs such as `[REQ-DEMO-A]`. Multiple requirements could therefore receive the same truncated ID and candidate generation could collapse them.
- **Resolution:** Accept complete `REQ-` identifiers containing alphanumeric segments separated by hyphens or colons. Regression checks preserve three alphabetic suffixes and produce three distinct pairs. The README CLI sequence then completed with 3 extracted requirements, 3 BM25 pairs, 3 provisional records, and 3 pending queue items. Focused tests: **21 passed**; full suite: **343 passed, 12 warnings**, exit 0.
- **Boundary:** The pattern remains limited to `REQ-` identifiers; validate other identifier conventions against rights-cleared SRS examples before broadening further.

### [ISSUE-032] Requirement Extraction Discarded Source Document and Location
- **Severity:** Medium / Dataset traceability
- **Status:** ✅ Resolved locally (TASK-P19, 2026-11-23)
- **Files:** `leredd/data_models.py`, `leredd/utils.py`, `api/routes.py`, `ui/app.py`, extraction tests.
- **Description:** Requirements extracted from files were all labeled `source=regex_extraction`; the source filename and statement line were not retained. This weakened auditability and prevented reliable document-level grouping from extracted file data.
- **Resolution:** Added optional `Requirement.source_location`; file extraction records a basename in `source` and the extracted-text line in `source_location`. API accepts an optional `source_name`; UI uses the uploaded filename or `pasted SRS`. Legacy records without the field still load. Focused extraction/API/UI/CLI tests: **43 passed in 24.11s**; full suite: **329 passed, 12 warnings in 87.12s**, exit 0. No `data/` hashes changed.
- **Boundary:** PDF and DOCX locations refer to line positions in extracted text, not guaranteed physical page or paragraph coordinates. API callers that omit `source_name` retain the generic `regex_extraction` source label.

### [ISSUE-031] Generated Requirement IDs Could Collide With Explicit SRS IDs
- **Severity:** Medium / Requirement identity integrity
- **Status:** ✅ Resolved locally (TASK-P18, 2026-11-23)
- **Files:** `leredd/utils.py`, `tests/test_utils.py`
- **Description:** The text extractor incremented its synthetic ID counter only for requirements without explicit IDs. An explicit `REQ-<SYSTEM>-001` followed by an ID-less requirement could therefore assign the same ID to both records.
- **Resolution:** Reserve IDs explicitly present in the input and advance the synthetic counter past occupied IDs, regardless of where the explicit ID appears. Preserve short IDs such as `[REQ-1]` and system-prefixed bracketed IDs. Full suite after related extraction changes: **329 passed, 12 warnings in 87.12s**, exit 0. Hashes of existing `data/` files were unchanged.

### [ISSUE-030] Dependency Graph Node Labels Overlap in the Dashboard
- **Severity:** Low / UI readability
- **Status:** ✅ Resolved locally (TASK-P16, 2026-11-23)
- **Files:** `ui/app.py`, `tests/test_ui_workflows.py`
- **Description:** The legacy dependency graph drew all requirement IDs permanently; nearby nodes caused overlapping labels in the running dashboard.
- **Resolution:** Removed always-visible node text and moved requirement IDs and statements into interactive hover details. Brave visual inspection confirmed the graph is legible; a regression checks the node trace exposes hover details without persistent text. UI workflow suite: **8 passed**; full suite: **326 passed**.

### [ISSUE-022] Hosted CI Depends on Local-Only Replication Datasets
- **Severity:** Medium / CI Reliability
- **Status:** Local tests pass; latest hosted run failed on `ec11351` (run `45328417724`). Keep open.
- **Files:** `tests/test_replication_loader.py`, `tests/test_replication_dataset.py`, `tests/test_replication_evaluation_runner.py`.
- **Description:** The pushed `python -m pytest` workflow collected 264 tests, but 17 failed because `data/replication/*.json` are local untracked assets absent from GitHub checkout. The Python 3.10 job failed and the Python 3.11 job was canceled; Docker was skipped.
- **Resolution in local worktree:** Loader and runner behavior tests now use synthetic expected-count records; source-data-specific count checks run only when the private files are present. Full local suite: **310 passed, 12 warnings**, exit 0. The exact coverage command in a complete tracked-file checkout without replication assets passed **309 tests with 1 intentional skip and 12 warnings**, exit 0. Keep open until the updated hosted matrix and Docker step complete successfully.
- **Latest hosted evidence (2026-11-24):** Run `45328417724` on `ec11351c2ff027352aaa2060a1c84ebe6a65fe39` completed with failure. Python 3.11 failed at “Run Pytest with Coverage”; Python 3.10 was canceled during dependency installation; Docker was skipped. See [run 45328417724](https://github.com/kesarshharma/LLREDD/actions/runs/45328417724). The local 428-pass suite does not resolve hosted CI.

### [ISSUE-029] Evaluation Reports Hard-Coded Legacy Split Metadata
- **Severity:** Medium / Evaluation Integrity
- **Status:** Resolved locally (TASK-P12; regression and full suite passed).
- **Files:** `scripts/run_full_evaluation.py`, `tests/test_safe_evaluation.py`.
- **Description:** Per-model notes and limitations stated ADB/APA/TJA systems, fixed sample counts, and provenance-free labels even when the runner was given a different split. This could misdescribe a report and its evidence.
- **Resolution:** Derive system names, partition counts, provenance-free counts, label coverage, and small-sample wording from the supplied split and eligibility results. Regression uses differently named synthetic systems and checks the generated report. No checked-in evaluation report was changed.

### [ISSUE-025] Candidate Retrieval Strategy and Exhaustiveness Were Misreported
- **Severity:** Medium / Candidate Integrity
- **Status:** Resolved (TASK-P10, 2026-10-14; full suite 298 passed).
- **Files:** `leredd/candidate_generator.py`, `leredd/retrieval.py`, `leredd/embeddings.py`, candidate/retrieval tests.
- **Description:** DENSE candidate generation used pair-oriented example retrieval rather than ranking Requirements, HYBRID generation did not fuse the intended dense and lexical requirement rankings, and retrieval output could be marked exhaustive merely because it was not truncated. The fitted TF-IDF fallback also learned incompatible feature spaces when query and corpus were encoded in separate calls.
- **Resolution:** Added `RequirementRetriever`, fused DENSE and BM25 rankings with RRF for HYBRID, checked generated unique pairs against n(n-1)/2, and used stable hashing fallback features. Added controlled regression coverage for strategy ranking, feature-space consistency, and manifest semantics.

### [ISSUE-026] Queue CLI Could Corrupt or Misstate Verified Exports
- **Severity:** High / Annotation Integrity
- **Status:** Resolved (TASK-P10, 2026-10-14; full suite 298 passed).
- **Files:** `leredd/cli.py`, `tests/test_ui_and_queue_integration.py`.
- **Description:** `queue-create` could ignore an unreadable existing queue and overwrite it. `queue-decide` could ignore malformed verified output and replace it, export rejected/non-verified records, duplicate a record on retry, or leave a partially written JSON file.
- **Resolution:** Validate existing JSON before mutation, export only explicitly verified labeled decisions, replace matching pair IDs idempotently, and publish through a sibling temporary file followed by atomic replace. Failure regressions verify the previous export and queue are preserved where applicable.

### [ISSUE-027] Verified Status Could Contradict Annotation Provenance
- **Severity:** High / Evaluation Integrity
- **Status:** Resolved (TASK-P11; full suite 309 passed, 12 warnings, exit 0).
- **Files:** `leredd/evaluation.py`, `scripts/run_full_evaluation.py`, `leredd/cli.py`, `leredd/review_queue.py`, `ui/app.py`, `tests/test_safe_evaluation.py`, `tests/test_cli.py`, `tests/test_review_queue.py`, `tests/test_ui_and_queue_integration.py`, `tests/test_system_verification.py`.
- **Description:** Evaluation previously trusted `VerificationStatus.VERIFIED` alone, accepting provenance that named an LLM as the source, lacked a human actor or accepted label, or recorded a final label different from the scored dependency label.
- **Resolution:** Shared eligibility validation now requires a matching verified human annotation with an annotator/reviewer identity and matching `accepted_label`, and rejects generic placeholder identities. CLI and queue-manager decisions require an explicit nonblank reviewer/annotator; Streamlit starts new decisions with a blank identity and clears historical placeholders for re-review. Provenance-free legacy records retain the warning-based compatibility path.

### [ISSUE-028] Default System Split Could Leave Development Empty With Three Systems
- **Severity:** Medium / Evaluation Readiness
- **Status:** Resolved (TASK-P11; full suite 309 passed, 12 warnings, exit 0).
- **Files:** `leredd/dataset_splitter.py`, `tests/test_dataset_splitter.py`.
- **Description:** Independent rounding allocated two of three systems to training and the remaining one to test under default 60/20/20 ratios, producing an empty development partition despite three available systems.
- **Resolution:** System allocation reserves one system for each requested partition that fits, then assigns remaining systems by largest remainder. Explicit mappings and single-system behavior remain unchanged.

### [ISSUE-018] MEDIUM: Generated Pair Labels Must Not Masquerade as Verified Ground Truth
- **Severity:** Medium / Data Semantics
- **Status:** Resolved (TASK-P9-13, 2026-10-12)
- **Files:** `leredd/utils.py`, `leredd/candidate_generator.py`, `leredd/data_models.py`, `leredd/cli.py`, `leredd/evaluation.py`, `scripts/run_full_evaluation.py`, downstream loaders/retrievers and regression tests
- **Description:** Generated candidates previously could receive `No_dependency` before annotation, making an unknown relation indistinguishable from an explicit negative label. Label handling also accepted ambiguous unknown text, and some consumers defaulted missing labels to `No_dependency`.
- **Resolution:** Both generation paths now preserve unknown labels as `None` and provisional provenance. Unknown explicit labels fail validation. Evaluation requires complete prediction coverage and rejects provenance-bearing ground truth unless status is verified with coherent human source/identity/final-label provenance; retrieval and supervised fitting ignore unlabeled/unverified examples. The legacy pairs command's interactive annotation stores an explicit human VERIFIED decision. Regression coverage and full suite pass.

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
- **Resolution:** Reconfirmed outside the restricted sandbox on 2026-11-24: current `.venv\Scripts\python.exe -m pytest tests/` completed with **400 passed, no warnings in 107.33s**, exit 0 (PowerShell duration 111.87s). `test_api_root` passed. `tests/test_api.py` now uses HTTPX `ASGITransport` instead of Starlette TestClient; the full run exercised the replacement and removed the deprecation warning. A SHA-256 manifest confirmed all 312 `data/` files were unchanged.
- **Execution boundary:** A focused run with the replacement transport still did not finish in the restricted Windows sandbox; the prior stack evidence attributes this to asyncio Proactor socket-pair startup being denied there. The focused test and full suite passed outside that boundary. This is not evidence of an application route defect.
- **TASK-P45 follow-up:** Current `httpx2` PyPI metadata requires Python >=3.11 while `pyproject.toml` declares Python >=3.10. The in-process ASGI transport avoids adding that dependency.

### [ISSUE-009] CRITICAL: MockLLMClient Ignores Retrieved Examples — Ablation Invariance Under Mock Backend
- **Severity:** Critical / Architectural Caveat
- **Status:** 🟡 Open — Reports now mark retrieval/ICL ablations non-informative under MockLLM (TASK-P17); empirical effect remains unverified.
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
  - **TASK-P17 reporting correction:** New primary and ablation reports carry `ablation_validity` metadata explicitly marking both dimensions `not_informative_with_mock_backend`. This does not resolve the need for an empirical-capable backend.

### [ISSUE-010] CRITICAL: MockLLMClient Always Outputs Confidence=5 — Threshold Ablation Invariant Under Mock Backend
- **Severity:** Critical / Architectural Caveat
- **Status:** 🟡 Open — Confidence screening is implemented locally (TASK-P27), but its empirical utility remains unverified because MockLLM emits fixed confidence.
- **Files:** `leredd/llm.py` (lines 189, 193, 197, 201, 205, 209), `leredd/detector.py`, `tests/test_detector.py`, `tests/test_ablation_benchmark.py`
- **Verified System Behavior (TASK-P27):**
  - Low-confidence outputs retain their parsed label and carry `validation_status=requires_review`; outputs above the threshold carry `passed_threshold`; `-1` records `threshold_disabled`.
  - `Prediction.validation_threshold` and evaluation report `confidence_screening` counts preserve the configured screening decision. The Streamlit page displays a human-review recommendation without presenting the prediction as verified truth.
  - Regression tests confirm confidence changes status, never `dependency_type`, and exercise serialized screening coverage.
- **Unresolved Mock Backend Limitation:**
  - `MockLLMClient.generate()` hardcodes `confidence = 5` across all six heuristic classification branches.
  - In `scripts/run_full_evaluation.py`, the ablation compares threshold enabled (`confidence_threshold = 4`) against threshold disabled (`confidence_threshold = -1`).
  - Because `MockLLMClient` always outputs confidence 5, no item is flagged for review at threshold 4. Class metrics are identical by design because screening does not rewrite the predicted class; mock runs cannot estimate review utility or risk-coverage behavior.
- **Evaluation Caveat & Recommendation:**
  - New primary and ablation reports identify the fixed-confidence limitation. Studying screening utility requires an authorized real LLM with recorded version/configuration or a separately validated variable-confidence test double. Do not modify MockLLM merely to fabricate confidence variation.
  - The historical report artifacts still contain the previous `<=4 -> No_dependency` method description; they are preserved as historical outputs and have not been rewritten. New reports use the confidence-screening status contract.

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
- **TASK-P15 update (2026-11-23):** Added a blinded two-reviewer CLI and agreement report (`scripts/annotate_independent.py`, `scripts/compare_independent_annotations.py`). Tool tests pass, but no real human vectors have been collected; sample-size and research-validity blocker remains active.

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
- **Evidence:** The arXiv record lists CC BY 4.0 for the paper, while the paper's v2 text says the replication package will be made available upon publication. The local imported package has no dataset-specific license or redistribution agreement. The paper license does not establish rights for separately packaged annotations or SRS PDFs.
- **Required resolution:** Obtain authoritative dataset-specific permission/license terms before redistribution or publication of the imported annotation files. Do not infer permission from the paper license alone.

### [BLOCKER-008] Replication Implemented by source-label mapping is undocumented
- **Severity:** High (Blocks mapped replication metrics)
- **Status:** 🔴 Active
- **Evidence:** The paper defines Implements as a higher-level requirement fulfilled by a lower-level requirement (Table I / Section III). The local workbook's source label is Implemented by; the corresponding example pair appears in reverse row order from the paper's example. This suggests a relationship but does not document a normalization rule from the workbook label to the paper's class.
- **Required resolution:** Obtain an explicit source-to-paper label crosswalk from the replication package documentation or authors. Preserve imported labels unchanged and keep mapped evaluation blocked until then.

### [ISSUE-006] Evaluation CLI Argument Ambiguity & Supervised Cross-System Terminology Alignment
- **Severity:** Medium (Documentation & Metadata Integrity)
- **Status:** ✅ Resolved (2026-10-10, TASK-P6-03)
- **Description:** The `--model-type` parameter in `scripts/run_full_evaluation.py` selected the classification backend head for `BERTBaseline` (`rf` vs `lr`), but could be misconstrued as selecting which baseline model to run. Additionally, reports previously mislabeled supervised `BERTBaseline` fitting on ADB and testing on TJA as "zero-shot".
- **Resolution:** Added `--bert-classifier-type` parameter alias (with `--model-type` fallback) and updated help strings. Updated report configuration metadata (`bert_classifier_backend: "rf"`, `bert_classifier_description`). Re-labeled `BERTBaseline` and `TFIDF_LSABaseline` notes as **Supervised Cross-System Evaluation** (trained/fitted on ADB, tested on TJA) and added exploratory notices for McNemar's test on small samples ($n=5$). Added 4 regression tests (24 total in `test_safe_evaluation.py`).

### TASK-P13 demonstration status (2026-10-14)
- Local Streamlit and FastAPI surfaces ran; synthetic extraction, provisional queue, system-disjoint split, and mock report persistence were exercised. No issue status is changed by this demonstration. Actual human annotation, replication rights/direction, and real-provider behavior remain open or unverified as recorded above.

### [ISSUE-033] Manual-versus-assisted annotation effort tooling unavailable
- **Severity:** Medium (proposal workflow support)
- **Status:** ✅ Resolved as software capability (TASK-P20, 2026-11-23); empirical study remains unperformed.
- **Description:** The project had a blind independent-annotation and agreement workflow, but no controlled way to compare time with versus without provisional machine suggestions.
- **Resolution:** Added a paired counterbalanced study schedule, separate manual/assisted collection, hash-bound inputs, descriptive timing/disagreement report, and output protections against writing into project data. Focused suite: **27 passed**; full suite: **341 passed, 12 warnings**, exit 0. See ADR-034 and `docs/ANNOTATION_GUIDE.md`.
- **Remaining evidence gap:** No real participant annotations, effort estimate, correctness comparison, or significance test has been produced. A rights-cleared real sample and two actual reviewers are required; synthetic test records do not satisfy that requirement.

### [ISSUE-034] Docker image build context can include local research data
- **Severity:** Medium (data handling / packaging)
- **Status:** 🟡 Build and API/UI HTTP runtime verified locally (TASK-P47); post-start Docker management API inspection was unstable.
- **Evidence:** Dockerfile copies the repository with `COPY . .`; there was no `.dockerignore`. The local working tree contains untracked replication JSON and evaluation artifacts.
- **Mitigation:** `.dockerignore` excludes local `data/`, the project PDF, environment/secret files, repository metadata, and caches/temp. Both Docker images built using CPU-only Torch. Compose launched API/UI; API root and health returned 200, and Streamlit root and health returned 200/`ok`. No dataset-writing operation was requested.
- **Remaining verification:** A later Docker management API probe returned 500/hung; a bounded `docker info` recheck timed out after 20 seconds. In that same check, API health and Streamlit health still returned 200/`ok`, so app responsiveness is confirmed while Docker CLI management remains unstable. Image-content inspection and a final `docker compose ps` could not be completed. Browser visual inspection was also blocked because the computer-use tool could not verify the browser URL. `docker compose config` still warns that the top-level `version` key is obsolete.
- **Next:** Recheck Docker Desktop management commands and visually inspect `http://127.0.0.1:8501`; retain the `.dockerignore` data exclusions and do not publish mounted local research data.

### [ISSUE-035] Live Streamlit page remains blank after health success
- **Severity:** High (blocks a clean interactive demonstration)
- **Status:** ✅ Resolved as a persistent blank-page defect (2026-11-24); cold-start latency remains a usability consideration.
- **Evidence:** Local `/` and `/_stcore/health` returned HTTP 200, but Chrome at `http://127.0.0.1:8501` displayed only the shell and persistent `Running...` indicator, without dashboard content after 7.5 seconds. The serving process remained alive.
- **Hypothesis only:** An older server stderr log contains repeated `ModuleNotFoundError: No module named 'torchvision'` during Streamlit's Transformers module-path inspection. Its timestamp does not correlate it to the observed browser session and it may be a nonfatal file-watcher warning.
- **Next:** Reproduce on a separate port with fresh logs and browser console/network evidence. Fix only the identified failing layer; then rerun browser QA and UI tests.

#### ISSUE-035 update (2026-11-24)
The fresh diagnostic instance on `127.0.0.1:8502` returned HTTP 200 for `/` and `/_stcore/health`; native Chrome displayed the dashboard after six seconds. Port 8501 initially had no listener, but after launching the documented UI command, native Chrome rendered its dashboard after about 16 seconds from navigation. The synthetic annotation page generated and persisted three provisional pairs under `.pytest-tmp/`. The in-app browser websocket errors were environment-specific and did not reproduce in native Chrome. No app-code change was required. Keep only the cold-start latency as a separate usability observation, not a blank-page failure.

#### TASK-P29 runtime follow-up (2026-11-24)
The long-running 8501 process displayed an import error for `PredictionValidationStatus`, while a fresh Python import from the same checkout succeeded. A new Streamlit process rendered normally and the documented 8501 process was replaced; `/` and `/_stcore/health` returned 200 (`ok`). This is consistent with stale process module state; no source fix was warranted. A live synthetic MockLLM result was persisted as provisional and loaded by the review page as pending. It was not human-reviewed or exported. This follow-up does not establish prediction quality or resolve the external research blockers.

### [ISSUE-045] Human review discarded per-requirement metadata
- **Severity:** Medium (breaks metadata traceability and metadata-aware evaluation after review)
- **Status:** ✅ Resolved locally (TASK-P35, 2026-11-24)
- **Description:** Candidate pairs could contain distinct per-requirement system, component, subsystem, source-document, and source-location values, but `ReviewItem` did not store them. `record_decision()` reconstructed reviewed pairs from IDs/text and one shared system value, so verified exports silently lost the structured metadata.
- **Resolution:** Queue items now persist the optional per-requirement values, the Streamlit review workspace displays available metadata, and verified records restore it. Regressions cover queue JSON round-trip, human decision/export reconstruction, legacy queue compatibility, and the review UI.
- **Evidence:** Focused queue/UI suites: **41 passed**; full suite: **366 passed, 12 existing warnings**, exit 0. Eleven protected dataset/report hashes were unchanged.
- **Boundary:** Historical queue files remain readable, but metadata absent from those files cannot be recovered except by rebuilding the queue from its original candidate pairs.

### [ISSUE-046] Verified dataset snapshots had no immutable release operation
- **Severity:** Medium (dataset provenance and reproducibility)
- **Status:** ✅ Resolved locally (TASK-P36, 2026-11-24).
- **Description:** Human-verified exports were plain JSON files and each review decision used a fixed `dataset_version`; there was no release operation to validate all records, assign a chosen version, hash source and released content, or preserve a non-overwritable snapshot.
- **Resolution:** Added `leredd-cli dataset-release`. It accepts one or more verified exports, rejects empty/duplicate/unlabeled/provenance-free/unverified/incoherent data, stamps copied records with the requested version, and atomically publishes `pairs.json` and a manifest in a new release directory. Existing destinations are rejected.
- **Focused evidence:** `tests/test_dataset_release.py tests/test_cli.py -q`: **19 passed in 9.86s**. Synthetic records only; no real reviewer data was released.
- **Evidence:** Full suite: **374 passed, 12 existing warnings**, exit 0. Synthetic records only; no real reviewer dataset was released.
- **Remaining evidence:** Produce releases from real reviewer-verified, rights-cleared annotations before making dataset or research claims.

### [ISSUE-047] Candidate-generation coverage was not reported
- **Severity:** Medium (candidate-pipeline auditability)
- **Status:** ✅ Resolved locally (TASK-P37, 2026-11-24).
- **Description:** The proposal requires estimating candidate-generation recall against the complete pair universe on a sample, but candidate manifests exposed only the generated count and an exhaustive boolean.
- **Resolution:** Manifests and CLI report the exhaustive pair-space size, candidate-space recall, and uncovered-pair count. The metric is explicitly not gold dependency recall.
- **Verification:** Focused generator/CLI/script tests: **42 passed**. Full suite: **374 passed, 12 existing warnings**, exit 0. All 311 files under `data/` retained their hashes; `git diff --check` passed.
- **Evidence boundary:** This resolves the missing software coverage metric. A real strategy-coverage/positive-recall study still needs a rights-cleared SRS sample; gold dependency recall additionally requires verified labels.

### [ISSUE-048] Repeated prediction consistency was unavailable
- **Severity:** Medium (proposal's confidence-aware assessment)
- **Status:** ✅ Resolved locally (TASK-P38, 2026-11-24).
- **Description:** Detection returned one model response and had no supported way to identify whether repeated inferences on the same prompt vary.
- **Resolution:** Added optional 1–5 fresh-run consistency screening to the detector, CLI, single/batch API, and Streamlit. Calls above one bypass the built-in response cache. Any label or self-reported confidence variation is recorded and routed as a human-review recommendation while retaining the first predicted class.
- **Verification:** Focused detector/LLM/API/CLI/UI tests: **61 passed**. Full suite: **383 passed, 12 existing warnings**, exit 0. All 311 data files retained their hashes.
- **Evidence boundary:** Controlled tests verify software behavior only. Real-provider variability, calibration, prediction quality, and review benefit remain unmeasured; no labels are verified by this screen.

### [ISSUE-049] DOCX table requirements were silently omitted
- **Severity:** Medium (SRS extraction completeness)
- **Status:** ✅ Resolved locally (TASK-P39, 2026-11-24).
- **Description:** The PDF/DOCX extractor read only top-level DOCX paragraphs, so requirements stored in tables were silently lost. The shared text parser also removed terminal periods, and image-only PDFs could silently produce an empty extraction result.
- **Resolution:** DOCX extraction now traverses top-level paragraphs and table rows in body order, preserving separate cells with tabs. Requirement extraction retains terminal punctuation. PDFs with no extractable text raise an actionable message explaining that OCR is unsupported.
- **Verification:** Synthetic DOCX table/PDF success and image-only PDF failure regressions pass. Focused tests: **25 passed**; complete suite: **386 passed, 1 existing Starlette deprecation warning**, exit 0. All 311 data files retained their hashes.
- **Evidence boundary:** Only controlled synthetic documents were tested. This does not establish recall on real SRS layouts; OCR is unsupported, and the active base Python needs the optional `documents` extra.

### [ISSUE-050] Short non-empty requirements were not routed for review
- **Severity:** Medium (proposal uncertainty-review coverage)
- **Status:** ✅ Resolved locally as a screening capability (TASK-P40, 2026-11-24).
- **Description:** The proposal calls for flagging unusually short or incomplete requirement statements for human review. The queue identified empty text but did not route short non-empty statements based on length.
- **Resolution:** Added a configurable minimum-character cue (default 20; CLI `queue-create --minimum-requirement-characters N`; `0` disables it). The queue manifest records the eligibility configuration, and the cue only recommends inspection; it does not infer incompleteness, change labels, or verify annotations.
- **Verification:** Focused review queue/CLI/UI integration tests: **45 passed**. Full suite: **387 passed, 1 existing Starlette deprecation warning**, exit 0. All 311 `data/` files remained hash-identical; `git diff --check` passed.
- **Boundary:** This threshold is not linguistically validated. Semantic ambiguity/incompleteness still requires human judgment; no real SRS or human review was performed.

### [ISSUE-051] Invalid RAG overlap could prevent chunking from terminating
- **Severity:** High (optional SRS-context retrieval could hang on malformed configuration)
- **Status:** ✅ Resolved locally (TASK-P41, 2026-11-24).
- **Description:** `chunk_document()` advanced by `chunk_size - overlap`; an overlap equal to or greater than the chunk size allowed a zero or negative step. Zero chunk sizes and malformed `k` values were also not handled explicitly.
- **Resolution:** Validate positive integer chunk size, overlap in `[0, chunk_size)`, and non-negative integer `k`. `k=0` returns immediately without embedding.
- **Verification:** Focused retrieval/detector tests: **31 passed**. Full suite: **398 passed, 1 existing Starlette deprecation warning**, exit 0. All 311 `data/` files remained hash-identical; `git diff --check` passed.
- **Boundary:** The fix ensures termination and parameter behavior; retrieval relevance and classification benefit remain unmeasured.

### [ISSUE-052] API accepted SRS context without a request-level RAG switch
- **Severity:** Medium (API context retrieval path inaccessible under default configuration)
- **Status:** ✅ Resolved locally (TASK-P42, 2026-11-24).
- **Description:** Single and batch request schemas accepted `srs_text`, but the detector only retrieved it when global `USE_RAG` was enabled. API callers could not enable RAG per request, so the default configuration silently ignored supplied context.
- **Resolution:** Added optional `use_rag` to both schemas and detector APIs. Explicit true/false override server config; omitted values preserve it. The API reference explains this behavior.
- **Verification:** Focused API/detector/retrieval tests: **49 passed**; full suite: **400 passed**, each with one existing Starlette deprecation warning. Context inclusion in the actual constructed prompt is asserted; 311 `data/` files retained their hashes.
- **Boundary:** Controlled prompt routing only; no claim of retrieval or prediction quality. Real source text and verified test labels are still needed.

### [ISSUE-053] Extract CLI help omitted supported Markdown inputs
- **Severity:** Low (CLI usability / command discoverability)
- **Status:** ✅ Resolved locally (TASK-P48, 2026-11-24).
- **Description:** The extractor accepted Markdown files, but `leredd-cli extract --help` listed only `.txt`, `.pdf`, and `.docx`.
- **Resolution:** Updated the option help and command docstring to include `.md`; extended the Markdown CLI integration regression to check `.md` and `.docx` help text.
- **Verification:** `tests/test_cli.py::test_cli_metadata_sidecar_flows_from_srs_to_candidate_pairs -q`: **1 passed in 3.33s**. The test uses temporary fixtures.
- **Boundary:** This fixes the advertised file formats only; it does not establish extraction quality on arbitrary SRS documents.

### TASK-P48 extraction-trial evidence boundary (2026-11-24)
- NASA Trick SRS parser smoke output contained 99 unique records with source locations. A keyword scan found 97 source lines with `shall`/`should`/`must`, including two lines that emitted two records each. This does not establish precision/recall or correctness; human comparison remains open.
- Pair generation was exhaustive over 99 extracted requirements (4,851 unordered pairs). This is pair-space exhaustiveness only, not gold dependency recall. Provisional output contained 350 heuristic suggestions and 4,501 unresolved pairs; the review queue had 4,851 pending items and no human decisions. No evaluation or release was performed.
- The temporary source and outputs are in workspace-local untracked `.pytest-tmp`; no dataset permission, label mapping, reviewer evidence, or research result is inferred from this software smoke test.

#### TASK-P49 update to ISSUE-047 evidence boundary (2026-11-24)
- On 99 extracted requirements / 4,851 possible pairs, `top_k=5` generated 327 BM25 pairs (6.74% of pair space), 335 dense pairs (6.91%), and 325 hybrid pairs (6.70%); all-pairs generated 4,851 (100%). This is workload/space coverage, not positive dependency recall.
- No gold labels were used. The rate of true links missed by retrieval remains unknown; do not select a strategy or claim recall based on these percentages. The next empirical step requires two independent reviewers to label a documented sample from both emitted and omitted pairs.
- Dense encoding wrote 99 ignored cache entries to the configured `data/embeddings` location. Imported replication JSON and historical reports were not written by these runs; set `CACHE_DIR` to `.pytest-tmp` for future local embedding trials.

#### TASK-P50 pilot preparation status (2026-11-24)
- Created a 100-pair fixed-seed pilot sample: 50 hybrid top-5 candidates and 50 omitted pairs, drawn from the same 4,851-pair universe. A separate analysis map records the stratified sampling design; the reviewer file contains opaque IDs and no strata or labels.
- Status is `prepared_not_reviewed`. The sample has no human annotations and cannot yet produce positive dependency recall, agreement, or effort results. Reviewer actions must be independent and use the same pinned input/context; the stratum map must remain blinded until both complete.
- Full suite after the CLI correction and packet preparation: **400 passed, no warnings in 132.31s**, exit 0. This is software regression evidence only and does not resolve the gold-label or sampling-quality gap.

### [ISSUE-054] Stratified candidate pilot had no weighted positive-recall analysis
- **Severity:** Medium (proposal evaluation workflow incomplete)
- **Status:** ✅ Resolved as software capability (TASK-P51, 2026-11-24); real estimate pending human adjudication.
- **Description:** The project could measure the fraction of pair space emitted but had no way to estimate the proportion of verified dependency links recovered from a stratified sample of selected and omitted pairs.
- **Resolution:** Added a hash-pinned post-stratified ratio estimator and CLI. It accepts only an exact set of coherent human `VERIFIED` adjudications, verifies sample and population identity plus stratum weights, and reports an explicitly approximate pilot interval with boundary limitations.
- **Verification:** Final focused tests: **36 passed in 16.77s**; complete suite: **413 passed, no warnings in 38.38s**, exit 0 outside the restricted Windows socket environment (PowerShell duration 42.26s). Final review corrected the reported universe-size field and added a regression. Real pilot labels were not supplied; no positive-recall metric has been generated.
- **Remaining evidence:** Two reviewers must label independently; disagreements must be adjudicated and exported through the verified workflow. The sample is an operational pilot, not powered or preregistered.

### TASK-P44 — Live application smoke revalidation (2026-11-24)
- `python -m leredd.cli --help` exited 0; the existing API returned HTTP 200 for `/` and `/api/v1/health` when probed outside the restricted socket sandbox. A fresh Streamlit process on port 8504 rendered and executed synthetic extraction plus BM25/MockLLM detection.
- Three records were persisted only under `.pytest-tmp/task-final-ui-demo/candidates.json`; all remained unlabeled/provisional. The UI displayed the expected notice that Mock confidence is neither calibrated probability nor human verification.
- A Windows asyncio `ConnectionResetError` appeared in the Streamlit log during a connection teardown; the smoke actions completed and no route failure reproduced. Track only if it recurs as a user-visible failure. No open issue is resolved by synthetic smoke evidence.

### [ISSUE-055] Synthetic/demo records could be mistaken for research evidence
- **Severity:** High (evaluation and dataset integrity)
- **Status:** Resolved locally (TASK-P52, 2026-11-24); not committed or pushed.
- **Description:** Synthetic demo records could be assigned human-looking provenance and pass through retrieval/evaluation/release paths without an explicit synthetic-data boundary.
- **Resolution:** Added and propagated synthetic markers/source identities. Retrieval and verified export exclude marked records; evaluation rejects marked ground truth even when provenance claims verified status. Added regression tests across interfaces and persistence.
- **Verification:** Focused suite **172 passed**; full local suite **424 passed**, exit 0. This verifies software guards only, not real annotation quality.

### [ISSUE-056] Replication source rights and label crosswalk remain undocumented
- **Severity:** High (dataset legality and ground-truth validity)
- **Status:** Open; external evidence required.
- **Description:** The imported replication data and source spreadsheet labels lack verified redistribution rights and an authoritative crosswalk to the project's six dependency labels. `Implemented by` records also require explicit mapping or exclusion.
- **Next action:** Obtain dataset license/permission and the source codebook/crosswalk from an authoritative source; document the mapping before releasing or reporting replication metrics. Do not infer permission or semantics from the files themselves.

### [ISSUE-057] Full evaluation trusted declared split metadata without checking actual partition overlap
- **Severity:** High (evaluation leakage risk)
- **Status:** ✅ Resolved locally (TASK-P53, 2026-11-24); not committed or pushed.
- **Description:** A split could be labeled `system` in metadata while actual train/dev/test files shared systems, requirement IDs, or pair IDs; the runner previously proceeded to model fitting.
- **Resolution:** Revalidate actual loaded partitions before scoring and report the integrity audit. Duplicate prediction coverage errors now name invalid IDs.
- **Verification:** Focused evaluation suite **45 passed**; full suite **428 passed**, exit 0; all 412 files under `data/` hash-identical before and after.
- **Boundary:** The check follows the declared split strategy. It does not establish that source labels or provenance are truthful, and it does not validate research quality.

### TASK-P54 checkpoint status (2026-11-24)
- Local synthetic-data and split-integrity safeguards are now included in public `origin/main` at `d419d998e8ed9c05098c4fb0bf4f6789df08c74f` after a non-destructive merge with the latest remote commits.
- The post-merge local suite passed **456 tests**. This does not resolve ISSUE-056: imported replication rights and the `Implemented by` crosswalk still require authoritative evidence. Hosted CI has not yet been verified for this commit.
