# LEREDD Feature Traceability and Verification Ledger

Audit date: 2026-10-14. This ledger distinguishes code-path verification from real-service, research, and data-governance evidence. “Verified working” means the documented local contract has passing unit/integration or controlled end-to-end evidence; it does not imply paper reproduction or validity of unverified labels.

## End-to-End workflows

| Workflow / feature | Implementation | Dependencies / inputs | Test and evidence | Status |
|---|---|---|---|---|
| Fresh environment setup | `README.md`, `pyproject.toml`, `requirements.txt` | Python `>=3.10`; core dependencies; optional `.[dev]` and `.[documents]` extras | Install/start instructions inspected; no clean virtual environment was created and no packages were installed during this audit | Unverified on a fresh environment; setup commands are documented |
| Text/Markdown ingestion and requirement extraction | `leredd/utils.py`; `scripts/extract_requirements.py`; CLI `extract`; UI page 2; API `/extract` | UTF-8 text, `.txt` or `.md`; regex rules; optional explicit ID-keyed component/subsystem sidecar; CLI `--synthetic` and UI demo marker | `tests/test_utils.py`, `tests/test_cli.py`, `tests/test_api.py`, `tests/test_ui_workflows.py`, `tests/test_synthetic_data.py` | Verified: metadata flows by exact ID; extraction can explicitly mark demo input, and the prefilled UI sample is marked by default while custom text is unmarked; markers survive pair generation, review, and CSV/JSON export. Marked records are rejected from evaluation/release and excluded from retrieval |
| PDF and DOCX extraction | `leredd/utils.py` | Optional `pypdf` and `python-docx`, declared as `leredd[documents]` | Synthetic PDF text, DOCX paragraphs/table rows, and missing-package behavior; `tests/test_utils.py`, `tests/test_data_scripts.py` | Verified in `.venv` with the documents extra: extractable text and DOCX body tables preserve IDs, source, source lines, punctuation, and candidate generation. Image-only PDFs now report that OCR is unsupported. The base system Python still lacks the optional extra. |
| Exhaustive pair generation | `leredd/utils.py`; CLI `pairs`; UI page 2 | Extracted `Requirement` records | `tests/test_utils.py`, `tests/test_cli.py`, `tests/test_ui_workflows.py` | Verified working; output stays unlabeled/provisional |
| Candidate generation (ALL_PAIRS/BM25/DENSE/HYBRID) | `leredd/candidate_generator.py`; CLI `candidates`; `scripts/generate_candidates.py` | Requirement corpus; dense strategies use embeddings | `tests/test_candidate_generator.py`, `tests/test_generate_candidates_script.py`, BM25/hybrid suites | Verified: DENSE ranks requirements, HYBRID fuses dense + BM25 requirement rankings, and each manifest reports pair-space recall and uncovered pairs against the complete unordered pair universe. This metric is not gold dependency recall. |
| Provisional heuristic/LLM/multi-model labeling | `leredd/provisional_labeler.py`; CLI `label`; `scripts/label_candidates.py` | Explicit provider; default is MockLLM; optional distinct model IDs | `tests/test_provisional_labeler.py`, `tests/test_label_candidates_script.py`, `tests/test_llm.py` | Verified with controlled clients: unanimous model votes produce provisional suggestions; disagreement/incomplete votes remain unresolved with vote evidence. No real OpenAI/Ollama calls made |
| Human review and verification | `leredd/review_queue.py`; CLI queue commands; UI page 6 | Human reviewer, explicit decision, review queue JSON | `tests/test_review_queue.py`, `tests/test_cli.py`, UI queue AppTest journey | Verified with synthetic fixtures; the queue routes non-empty requirements shorter than a configurable threshold (default 20 characters) for reviewer inspection, without changing labels. CLI and manager require a nonblank non-placeholder identity; new UI decisions start blank. Per-requirement system/component/subsystem/source/location survive queue persistence and verified export and are shown to reviewers. Identity is self-reported, not authenticated. Older queue files load, but cannot restore metadata that was never stored |
| Immutable verified dataset release | `leredd/dataset_release.py`; CLI `dataset-release` | One or more verified pair JSON exports; unique version; new output directory | `tests/test_dataset_release.py` covers single/multiple sources, integrity hashes, eligibility rejection, no-overwrite, and publish failure cleanup | Verified as software behavior: release requires coherent human `VERIFIED` provenance, stamps copied records with the requested version, and writes dataset plus source/hash/class-count manifest atomically. Synthetic records only; no real reviewer dataset was released |
| Pair JSON export / validation | `leredd/utils.py`; CLI `annotate` (copy/validation utility); UI pages 2 and 6 | JSON pair schema | `tests/test_data_scripts.py`, `tests/test_ui_and_queue_integration.py`, `tests/test_ui_workflows.py` | Verified; UI writes are atomic and cannot target source dataset folders |
| System/document/requirement/pair dataset splitting | `leredd/dataset_splitter.py`; CLI `split`; `scripts/split_dataset.py` | Pair JSON inputs; output directory | `tests/test_dataset_splitter.py`, `tests/test_cli.py`, `tests/test_data_scripts.py` | Verified on synthetic and legacy inputs; default 3-system split reserves train/dev/test; requirement-disjoint legacy split remains empty |
| Dense, BM25, and hybrid example retrieval; optional RAG | `leredd/retrieval.py`, `leredd/bm25_retrieval.py`, `leredd/hybrid_retrieval.py` | SBERT for dense; `rank-bm25` for lexical; optional component/subsystem/system metadata; optional SRS text | `tests/test_retrieval.py`, `tests/test_bm25_retrieval.py`, `tests/test_hybrid_retrieval.py`, detector tests | Verified locally: hybrid conditionally fuses dense, BM25, and exact metadata-overlap rankings. RAG rejects non-positive chunk sizes, invalid overlap (including overlap >= chunk size), and invalid chunk counts; requesting zero chunks returns without embedding. Current data lacks demonstrated metadata coverage; no retrieval-quality claim |
| Dependency detection and response parsing | `leredd/detector.py`, `leredd/prompts.py`, `leredd/llm.py` | MockLLM, OpenAI, or local Ollama; config and optional example pool | `tests/test_detector.py`, `tests/test_prompts.py`, `tests/test_llm.py`, `tests/test_api.py`, controlled UI journey | Verified locally with MockLLM and parser fixtures. Prediction class is preserved; confidence screening emits `passed_threshold`, `requires_review`, or `threshold_disabled`, with the threshold included. Optional 1–5 fresh repeated runs flag label/confidence variation for review and retain the first class. Review status is not ground-truth verification; remote/provider service behavior is unverified |
| Baselines | `leredd/baselines/bert_classifier.py`, `leredd/baselines/tfidf_lsa.py` | Training pairs; BERTBaseline additionally uses embeddings | `tests/test_baselines.py`, safe evaluation tests, BERT joblib save/load roundtrip | Verified when explicitly fit before prediction; BERT artifacts are persisted and reloadable; predictions reject unfit models |
| Leakage-checked evaluation/report | `leredd/evaluation.py`; `scripts/run_full_evaluation.py`; safe `scripts/run_evaluation.py`; UI page 4 | System-disjoint split; provenance and contamination guards; MockLLM | `tests/test_safe_evaluation.py`, `tests/test_evaluation_entrypoint.py`, `tests/test_system_verification.py`, controlled UI report-contract test; TASK-P11 temp report run | Verified as demonstrative offline evaluation; provenance-bearing labels require matching human verification metadata; legacy benchmark n=5 has no provenance and remains conditionally accepted; report explicitly marks retrieval/ICL/threshold ablations invalid as empirical effects under MockLLM |
| Replication annotations | `scripts/integrate_replication.py`, `scripts/run_replication_evaluation.py` | Four imported JSON files, local verification status | `tests/test_replication_loader.py`, failure-mode, dataset, and evaluation-runner tests | Loader verified; metrics blocked until `pending_review` records are locally verified |

## Interfaces and user-facing controls

| Surface | Features / controls | Evidence | Status |
|---|---|---|---|
| Typer CLI (`leredd/cli.py`) | `extract`, `pairs`, `annotate`, `detect`, `evaluate`, `train-bert`, `split`, `candidates`, `label`, `queue-create`, `queue-list`, `queue-decide`, `dataset-release`, `serve` | `tests/test_cli.py`, `tests/test_dataset_release.py`, `tests/test_system_verification.py`, focused script tests; `leredd-cli --help` | Commands covered; `annotate` requires `--annotator`, `queue-decide` requires `--reviewer`; `train-bert` writes a versioned, reloadable classifier artifact atomically; `dataset-release` creates a hashed immutable release and refuses unverified records or an existing destination. |
| FastAPI (`api/main.py`, `api/routes.py`) | `GET /`, `/api/v1/health`, `/api/v1/config`; `POST /api/v1/detect`, `/batch_detect`, `/extract`; request validation and startup CLI | `tests/test_api.py`; actual local Uvicorn health request returned HTTP 200 and process exited after stop | Verified locally; config endpoint omits API secret and returns only a configured boolean. Single/batch requests can explicitly enable or disable RAG per request; omission follows server configuration |
| Streamlit page 1 | Home overview and research caveats | `AppTest` all-page render | Verified render; browser visual polish was not separately assessed |
| Streamlit page 2 | System ID, `.txt/.md` uploader, paste area, candidate output path, overwrite confirmation, extraction/pair generation, pair index, label, annotator ID, verified output path, overwrite confirmation; independent second-annotator agreement is collected by the separate blinded CLI workflow | `tests/test_ui_workflows.py` triggers upload, extraction, provisional serialization, annotation and verified serialization; `tests/test_ui_and_queue_integration.py` checks persistence safety; independent manifests/agreement are covered by `tests/test_independent_annotation.py` | Verified locally; only actual independent human vectors can produce meaningful agreement, and results do not auto-verify records |
| Streamlit page 3 | Provider/model, retrieval mode, example count, RAG, confidence threshold, and repeated-inference controls; OpenAI send confirmation; pair text; detection action, confidence/stability disposition and raw response | Controlled detector AppTest; detector/API suites | Verified with controlled clients, including low-confidence and repeated-run instability review warnings. Predicted class is preserved; real OpenAI and Ollama not exercised |
| Streamlit page 4 | Fixed system-disjoint evaluation, report metrics, limitations, confusion matrix; unique output directory | Controlled AppTest verifies split/strict flags and result display; safe evaluation tests exercise actual runner | Verified wiring; UI test substitutes the runner and does not itself retrain real models |
| Streamlit page 5 | Legacy ADB dependency graph, zoom/pan controls, and source-data caveat | AppTest plus Brave visual inspection; graph uses read-only legacy JSON and exposes requirement ID/text in hover detail to avoid overlapping permanent labels | Visually verified after graph-label readability fix; graph remains illustrative because source records have no verification provenance |
| Streamlit page 6 | Queue path/reload; status/reason/priority filters; search and item selection; decision/rationale/reviewer fields; explicit confirmation; persisted decision; verified export path and replace confirmation | UI AppTest creates a temporary queue, records a correction, reloads JSON, and exports a verified pair | Verified on synthetic temporary queue; new decisions have no default reviewer identity; multi-reviewer concurrent writes remain unsupported |

## TASK-P10 execution evidence (2026-10-14)

`python -m pytest tests/` completed with **298 passed, 12 warnings in 69.21s**, exit code 0, using the existing Windows Python 3.13.3 environment and workspace-local `TEMP`/`TMP`. The API TestClient test requires an unrestricted local loopback context on this host; it passed in the full unrestricted run. The focused candidate/retrieval/CLI/split/UI/end-to-end subset passed **120 tests in 8.02s**, embedding and ground-truth prompt regressions passed **12 tests in 1.91s**, and the separately isolated API module passed **9 tests in 1.02s**. The synthetic end-to-end test inspected serialized provisional annotations, persisted queue/verified export/split/report files under `tmp_path`, confirmed verified status is created only through the queue decision method, and checked strict retrieval exclusion for dev/test requirement IDs. A detector regression confirms changing the target pair's held-out label does not change the generated prompt. No `data/` diffs occurred; four imported replication JSON hashes and two historical evaluation report hashes were unchanged.

Remaining external or research verification: hosted Python 3.10/3.11 Actions, real OpenAI/Ollama behavior, Docker build, optional PDF/DOCX packages, cold-cache model download, browser visual review, annotation correctness, replication “Implemented by” semantics, and dataset redistribution rights.

## Python module inventory

## TASK-P11 workflow validation (2026-10-14)

- Real CLI walkthrough used only `.pytest-tmp/task-p11/cli-demo`: Markdown ingestion preserved three explicit IDs; BM25 generated two provisional candidates; heuristic labeling kept provisional status; queue creation/listing left both items pending without recording a human decision.
- `leredd-cli split` on the three legacy systems originally produced train=15/dev=0/test=5 with default 60/20/20 ratios. The split apportionment fix now gives each of three requested system partitions one system; the real fixture split wrote train=5/dev=10/test=5 with no requirement or system overlap.
- `scripts/split_dataset.py` plus `scripts/run_full_evaluation.py` ran in `.pytest-tmp/task-p11/legacy-flow`; train/dev/test were ADB/APA/TJA with counts 10/5/5, primary and ablation reports were written, and the contamination audit found zero contaminated retrieval pairs. Legacy ground truth had no provenance, was explicitly warned/conditionally accepted, and the report identifies MockLLM. Results are illustrative, not research-grade.
- Integrity regressions now reject provenance-bearing ground truth unless it is verified, human-sourced, attributable to an annotator/reviewer, and its accepted label matches the scored label. Legacy provenance-free compatibility remains.
- Final verification: focused provenance/split/CLI/API/UI path coverage **72 passed, 15 deselected in 7.63s**; `python -m pytest tests/test_api.py -q` **9 passed in 0.83s**; complete `python -m pytest tests/ -q` **305 passed, 12 warnings in 23.08s**, exit 0, using workspace-local `TEMP`/`TMP` and `HF_HUB_OFFLINE=1`. The warnings are `tqdm`'s deprecated `datetime.utcfromtimestamp()` call. `git diff --check` found no whitespace errors.
- Integrity: SHA-256 hashes for all four imported replication JSON files and both historical reports matched the pre-task values; `git diff -- data` was empty. The four imported files contain 313/100/200/200 unique IDs with no cross-file pair-ID overlap, all records remain `pending_review`, and unresolved `Implemented by` counts are 18/6/3/10. The copied package code describes `Implements`, but does not establish the direction mapping for the source label `Implemented by`; source direction and separate-dataset redistribution rights remain unresolved.
- Boundaries: no new human annotation or browser-based review was performed. UI AppTest and API tests passed, but a project reviewer must make real decisions before fresh candidate annotations can be exported as verified ground truth. OpenAI/Ollama and Docker remain unverified; optional PDF/DOCX packages were not installed. No CI workflow edits, package installations, or Git state mutations occurred.

### P11 provenance follow-up
- Removed generic reviewer identity defaults from CLI, queue manager/model, and Streamlit. Explicit identities are required for human decisions; known `human_cli` / `human_reviewer_1` placeholders are rejected by decision and evaluation paths. Legacy queue entries missing reviewer identity remain readable with an empty value and cannot be promoted until one is supplied.
- Focused CLI/queue/evaluation/provenance/UI suite: **116 passed, 12 warnings in 26.23s**. Full `python -m pytest tests/ -q` outside the restricted Windows loopback sandbox: **309 passed, 12 warnings in 23.02s**, exit 0. Warnings are the existing `tqdm.datetime.utcfromtimestamp()` deprecation.
- Actor identifiers are self-reported and not authenticated; reviewer identity governance remains an operational responsibility.

| Module(s) | Public responsibility | Main test coverage |
|---|---|---|
| `leredd/data_models.py`, `leredd/__init__.py` | Dependency labels, requirement/pair/prediction/report schemas, provenance and verification enums | `test_provenance.py`, all workflow suites |
| `leredd/config.py` | Pydantic settings, `.env`/environment configuration and singleton reset | `test_config.py`, `tests/conftest.py` |
| `leredd/utils.py` | Extraction, pair generation, JSON/CSV I/O, Cohen’s kappa primitive | `test_utils.py`, `test_data_scripts.py` |
| `leredd/candidate_generator.py` | Canonical pair identity, strategies and generation manifests | `test_candidate_generator.py`, script journey |
| `leredd/provisional_labeler.py` | Heuristic/LLM labeling and provisional manifests | `test_provisional_labeler.py`, script journey |
| `leredd/review_queue.py` | Eligibility, prioritization, state, human decisions, queue manifests and persistence | `test_review_queue.py`, UI/CLI integration |
| `leredd/dataset_splitter.py` | Four split strategies, persistence and leakage reports | `test_dataset_splitter.py`, script journey |
| `leredd/embeddings.py` | SentenceTransformer embedding and disk cache with TF-IDF fallback | `test_embeddings.py`, retrieval/baseline suites |
| `leredd/retrieval.py`, `bm25_retrieval.py`, `hybrid_retrieval.py` | Dense, lexical and reciprocal-rank-fusion retrieval | `test_retrieval.py`, `test_bm25_retrieval.py`, `test_hybrid_retrieval.py` |
| `leredd/prompts.py`, `leredd/llm.py`, `leredd/detector.py` | Prompt construction, Mock/OpenAI/Ollama clients, parsing and detection | `test_prompts.py`, `test_llm.py`, `test_detector.py`, API/UI tests |
| `leredd/baselines/bert_classifier.py`, `tfidf_lsa.py` | Supervised SBERT-feature classifier and TF-IDF/LSA baseline | `test_baselines.py`, `test_safe_evaluation.py` |
| `leredd/evaluation.py` | Eligibility/contamination enforcement, metrics, Wilson interval, McNemar | `test_evaluation.py`, `test_safe_evaluation.py`, `test_implemented_by_exclusion.py` |
| `leredd/cli.py` | All Typer commands listed above | `test_cli.py`, `test_system_verification.py`, workflow-specific tests |
| `api/main.py`, `api/routes.py` | FastAPI app, request models, health/config/extract/detection routes | `test_api.py` and actual local health request |
| `ui/app.py` | Six-page Streamlit dashboard and safe local output helpers | `test_ui_and_queue_integration.py`, `test_ui_workflows.py` |

## Scripts and support files

| File | Purpose and safety status |
|---|---|
| `scripts/extract_requirements.py` | Parameterized extraction with unique default output; tested on synthetic text |
| `scripts/generate_candidates.py` | Parameterized ALL_PAIRS/BM25 generation; run-specific default output; tested synthetically |
| `scripts/label_candidates.py` | Parameterized provisional labeling; defaults to MockLLM even if environment selects remote provider; tested synthetically |
| `scripts/build_review_queue.py` | Creates pending queue only; no simulated human decision; run-specific output and source path guard; tested synthetically |
| `scripts/split_dataset.py` | Parameterized system/requirement split outputs; refuses existing output by default; tested synthetically |
| `scripts/annotate.py` | Validates/copies pair JSON; does not annotate; distinct staging output and source path guard |
| `scripts/train_baselines.py` | Legacy in-memory demonstration; explicitly unsupported as an artifact/export workflow and not covered by end-to-end verification; use `leredd-cli train-bert` for persisted BERT classifier artifacts |
| `scripts/run_evaluation.py` | Backward-compatible, leakage-checked runner with unique run directory; tested for safe delegation and no overwrite |
| `scripts/run_full_evaluation.py` | Primary system-disjoint benchmark; CLI checks target output and source folders before writing |
| `scripts/run_replication_evaluation.py` | Replication prototype; blocked by local pending-review provenance; covered by runner tests |
| `scripts/integrate_replication.py` | Strict source loader, keyword-only temporary-root override used by failure-mode tests |
| `scripts/annotate_independent.py`, `scripts/compare_independent_annotations.py` | Blinded separate reviewer manifests, timing, abstentions, agreement report; outputs protected from imported replication and evaluation trees; no adjudication or automatic verification |
| `scripts/analyze_candidate_coverage.py`; `leredd/candidate_coverage.py` | Hash-pinned, stratified selected/omitted pair analysis; requires exact sample coverage and coherent VERIFIED human adjudications; estimates positive dependency recall with explicit exploratory uncertainty limits |
| `scripts/prepare_annotation_effort_study.py`, `scripts/run_annotation_effort_study.py`, `scripts/compare_annotation_effort.py` | Paired counterbalanced manual/assisted review, per-condition timing, descriptive comparison; machine suggestions required and explicitly hidden in the manual condition; outputs blocked from all project `data/` trees; no ground-truth promotion |
| `pyproject.toml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml` | Package dependencies, pytest config, container build and API/UI/Redis services; CPU-only API/UI images built and HTTP health-smoked locally, Docker management inspection remains incomplete |

Pytest uses `tests/`, files named `test_*.py`, and `--strict-markers -v` from `pyproject.toml`; no custom marker declarations or timeout plugin are configured. The suite includes: `test_ablation_benchmark`, `test_api`, `test_baselines`, `test_bm25_retrieval`, `test_build_review_queue_script`, `test_candidate_coverage`, `test_candidate_generator`, `test_cli`, `test_config`, `test_data_scripts`, `test_dataset_splitter`, `test_detector`, `test_embeddings`, `test_evaluation`, `test_evaluation_entrypoint`, `test_generate_candidates_script`, `test_hybrid_retrieval`, `test_implemented_by_exclusion`, `test_independent_annotation`, `test_label_candidates_script`, `test_llm`, `test_prompts`, `test_provenance`, `test_provisional_labeler`, four replication tests, `test_retrieval`, `test_review_queue`, `test_safe_evaluation`, `test_system_verification`, `test_ui_and_queue_integration`, `test_ui_workflows`, and `test_utils`.

## Configuration and unverified integrations

## TASK-P15 independent annotation capability (2026-11-23)

- Added a separate blinded CLI path for two independent reviewers to label the same candidate pairs, record abstentions/rationales and per-pair time, resume safely, and calculate observed agreement, Cohen's kappa, confusion matrix, and time summaries.
- Manifests bind to the candidate file and optional SRS context by SHA-256. Agreement comparison checks reviewer/source/context/pair-set consistency. Annotation outputs cannot target `data/replication/` or `data/evaluation/`, and writes use sibling temporary files with atomic replacement.
- Focused `tests/test_independent_annotation.py`: **15 passed in 1.01s**. Full `python -m pytest tests/`: **325 passed, 12 warnings in 69.69s**, exit 0. Warnings are the existing `tqdm` datetime deprecation.
- Boundaries: this verifies tooling only; no real participant reviewed labels, no agreement result or manual-effort claim was generated, and the self-reported reviewer IDs are not authenticated. A separate explicit adjudication/export is still required before labels become evaluation ground truth.

## TASK-P16 Streamlit visual QA (2026-11-23)

- Opened the running Streamlit dashboard in Brave. Visually exercised navigation, synthetic SRS extraction, MockLLM detection, fixed system-disjoint evaluation, graph visualization, and the human review queue. No review decision was submitted.
- The graph's always-visible requirement IDs overlapped. Changed nodes to show full requirement ID/text on hover instead; the graph is now visually legible while preserving interactive details. Added a focused regression.
- Synthetic UI extraction wrote 3 requirements and 3 provisional pairs to `.pytest-tmp/task-p16-ui/srs_candidates.json`. MockLLM detection returned `REQUIRES` with confidence 5/5. The evaluation UI persisted a new unique report at `data/evaluation/ui_runs/20261123T094742Z-215fe692/system_disjoint_report.json`; it reports 20/20 legacy provenance-free records conditionally eligible, `n_test=5`, and MockLLM. The result is demonstrative only. Existing data files were unchanged during the subsequent test run.
- Focused graph regression: **1 passed in 12.09s**. UI workflow suite: **8 passed in 20.45s**. Full suite: **326 passed, 12 warnings in 77.22s** (pytest; 83.01s outer), exit 0. Warnings are the existing `tqdm` datetime deprecation. The full run compared SHA-256 for every file under `data/` before/after: **0 differences**.
- Limits: the queue was inspected but not annotated; no real human verification was performed. Browser verification covers this local desktop/session, not cross-browser/mobile behavior.

## TASK-P17 Mock ablation validity reporting (2026-11-23)

- Both the primary evaluation report and `ablation_report.json` now include structured `ablation_validity` metadata. Under MockLLM, retrieval/ICL comparisons are marked uninformative because the mock ignores retrieved examples; threshold comparisons are marked uninformative because its confidence is fixed at 5. The limitation is also included in `evaluation_limitations`.
- `tests/test_safe_evaluation.py::TestRunEvaluationIntegration::test_mock_llm_demarcation_in_report` verifies both reports carry the caveat: **1 passed in 16.50s** (2 existing tqdm warnings). The full safe-evaluation module passed **35 tests, 12 warnings in 20.06s**; final repository suite passed **326 tests, 12 warnings in 29.50s**, exit 0. SHA-256 snapshots of all files under `data/` had **0 differences** across the full run.
- This is an interpretation safeguard, not a resolution of ISSUE-009/010's empirical question. MockLLM predictions and threshold behavior were not changed.

## TASK-P18 Requirement identifier collision fix (2026-11-23)

- Requirement extraction now reserves all explicit IDs in the input before allocating synthetic IDs, so an explicit `REQ-ADB-001` does not collide with a generated identifier regardless of order. Existing explicit IDs, data model, and default naming remain unchanged.
- `tests/test_utils.py`: **8 passed in 3.56s**. Full suite: **327 passed, 12 warnings in 89.33s**, exit 0. Existing data-tree hashes had **0 differences** before and after the full suite.
- Scope limit: repeated duplicate IDs explicitly present in a source document are not repaired or rejected by this change.

## TASK-P19 Requirement source/location traceability (2026-11-23)

- File extraction now records the source basename and extracted-text line in separate `Requirement.source` and optional `Requirement.source_location` fields. API text requests can provide `source_name`; UI upload and pasted-SRS paths provide their known source labels. Document identity stays stable for document-level split grouping. Explicit short IDs such as `[REQ-2]` are preserved.
- Backward compatibility: old requirement records omit `source_location` and still load. Calls without source labels retain the existing generic source value. PDF/DOCX location is a line in extracted text, not a physical page coordinate.
- Focused extraction/API/UI/CLI coverage: **43 passed in 24.11s**. Full suite: **329 passed, 12 warnings in 87.12s**, exit 0. Existing data hashes had zero differences. CLI smoke output under `.pytest-tmp` preserved `REQ-DEMO-007` and `REQ-2`, filename `demo-srs.md`, and extracted-text lines 1/2.

## TASK-P9-14 verification evidence (2026-10-13)

Latest fresh unrestricted `python -m pytest tests/` completed with **264 passed, 38 warnings in 170.21s** (exit code 0; outer duration 173.83s). All four `data/replication/*.json` SHA-256 hashes matched their immediately-before-run values, and `git diff -- data` was empty. The sandboxed run stalled at `test_api_root`; the unrestricted run passed it. The warnings are tqdm deprecations for `datetime.datetime.utcfromtimestamp()`.

`LEREDDConfig` reads fields from environment variables (case-insensitive Pydantic settings) and `.env`: `EMBEDDING_MODEL`, `SIMILARITY_METRIC`, `AGGREGATION`, `K_EXAMPLES`, `RETRIEVAL_MODE`, `RRF_K`, `USE_RAG`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`, `RAG_K_CHUNKS`, `LLM_PROVIDER`, `LLM_MODEL`, `TEMPERATURE`, `CONFIDENCE_THRESHOLD`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `DATA_DIR`, and `CACHE_DIR`. Pytest resets the cached settings singleton before and after each test.

- OpenAI package is present in the inspected environment, but no paid/external model request was sent. End-to-end OpenAI behavior remains unverified until credentials and explicit request authorization are provided.
- Ollama service availability and real inference were not tested; no daemon request was sent.
- Cold-cache model download behavior was not exercised. Dense retrieval and embedding code are covered by local tests, but a clean machine may require model weights/network access.
- The dashboard was run with Streamlit `AppTest`; a human-operated browser session and visual cross-browser review were not performed.
- Docker images/containers were not built or launched.
- Imported replication datasets remain `pending_review`, “Implemented by” interpretation remains unresolved, and dataset redistribution rights remain unverified. These are research/data blockers, not software-test passes.
- Cohen’s kappa requires a second independently collected label vector; the page now states this limitation instead of returning synthetic perfect agreement.
- `scripts/train_baselines.py` remains an in-memory-only legacy script; use the CLI `train-bert` command for BERT classifier artifacts. Artifacts use joblib/pickle and must only be loaded from trusted sources.

## TASK-P20 paired annotation-effort tooling (2026-11-23)
- `leredd/effort_study.py` plus three CLIs prepare a hash-bound fixed-sample schedule, collect each pair once in manual and assisted conditions with different reviewers, counterbalance conditions, and compare elapsed time and label disagreements descriptively. Manual mode does not display machine suggestions. Only provisional machine provenance is accepted as input; study labels never become ground truth.
- Study outputs are rejected under all project `data/` directories. README and `docs/ANNOTATION_GUIDE.md` provide the real-use procedure and limitations.
- Focused tests: **27 passed in 1.98s**. Full suite: **341 passed, 12 warnings in 92.81s**, exit 0. Hash comparison found no existing data changes and no added data files.
- This is capability verification only. No genuine human effort comparison, adjudication, correctness assessment, or significance test has been performed. Reviewer IDs remain self-reported; choose a rights-cleared real sample before conducting the pilot.

## TASK-P22 live UI check (2026-11-23)
- API root/health returned the expected payloads and `ok`. Streamlit health and root both returned success, but a live Chrome view showed a blank Streamlit shell with a persistent `Running...` indicator after 7.5 seconds. This supersedes earlier visual evidence as a current runtime blocker; dashboard rendering is not verified.
- Existing captured Streamlit stderr is older than this browser observation. It contains `ModuleNotFoundError: No module named 'torchvision'` while the file watcher inspects Transformers package paths, but correlation and causality are unknown.
- Next diagnostic: run a separately logged Streamlit instance on a free port; capture fresh server logs and browser console/network errors before changing file-watcher or app behavior.

## TASK-P29 live UI verification (2026-11-24)
- The long-running 8501 process displayed an import error for `PredictionValidationStatus`; importing that class in a fresh interpreter succeeded. A fresh Streamlit process rendered the dashboard, completed a synthetic MockLLM detection, wrote one provisional pending review record under `.pytest-tmp/task-goal-live-ui-20261124/`, and loaded that record in the Human Review Queue UI as 1 pending item. No human decision or verified export was performed.
- The stale 8501 process was replaced with a fresh process; `/` and `/_stcore/health` both returned HTTP 200, health body `ok`. The isolated 8503 process was stopped after inspection. This evidence distinguishes the current source/runtime from the stale process and supersedes the unresolved rendering status recorded in TASK-P22.
- Limitations: the live prediction used MockLLM and threshold 5, so it only verifies UI/queue plumbing, not prediction quality or useful confidence screening. Dense first-use took about 40 seconds. Real human annotation, data rights/label direction, real-provider behavior, Docker build, and hosted CI remain open.

## TASK-P30 confidence-screen boundary copy (2026-11-24)
- Because the screen is inclusive (`confidence <= threshold`), a 5/5 MockLLM prediction at threshold 5 is routed for review. The UI and queue rationale now identify the configured cutoff explicitly instead of describing that score as absolutely low. The stable queue reason code remains `low_confidence`.
- Focused UI/queue suite: **16 passed**; full suite: **348 passed, 12 tqdm datetime deprecation warnings**, exit 0. This does not validate calibration or real-provider utility.

## TASK-P31 Streamlit dense embedder lifecycle (2026-11-24)
- The local UI has 10 annotated example pairs available by default. For dense/hybrid retrieval, the UI session now retains the initialized embedder and injects it into subsequent detectors using the same model/cache key. An AppTest performs two detections in one session and verifies object reuse; no model is initialized solely for zero-shot or BM25.

## TASK-P32 metadata-aware example retrieval (2026-11-24)
- `Requirement` now supports optional `component` and `subsystem` fields. `HybridRetriever` adds a normalized exact-overlap ranking over component/subsystem/system values only when target metadata exists and at least one eligible candidate matches; IDs, source locations, and text are excluded from this signal.
- Controlled tests verify that metadata can affect top-k selection and that missing/generic metadata adds no ranking. Focused hybrid/detector/candidate/API tests: **62 passed**; final hybrid suite after the zero-weight guard: **11 passed in 0.27s**; final full suite: **350 passed, 12 warnings in 114.69s**, exit 0 (PowerShell duration 123.08s).
- This verifies implementation, not research utility. A rights-cleared dataset with dependable structured metadata and verified labels is still needed for a leakage-safe retrieval ablation.

## TASK-P33 multi-model provisional labeling (2026-11-24)
- CLI `label --mode multi_llm --models model-a,model-b` invokes each distinct configured model. In hybrid mode the same vote set is also compared with heuristic evidence. Only unanimous valid labels are suggestions; disagreements and failed/invalid votes remain unlabeled and include model-attributed evidence for review. All resulting records remain provisional.
- Model-aware v2 LLM cache keys prevent same-provider model collisions. Controlled tests cover unanimity, disagreement despite a heuristic suggestion, CLI validation, manifest provenance, and cache separation/reuse. Focused tests: **35 passed**; full suite: **355 passed, 12 warnings in 115.92s**, exit 0.
- No remote/local real model evidence was collected. Mock-client behavior is not model independence or quality evidence; API costs, human review, and label validation remain unverified.

## TASK-P34 explicit component/subsystem metadata flow (2026-11-24)
- Metadata can be supplied through CLI `extract`/`candidates --metadata-json`, API `/extract` `metadata_by_id`, Streamlit page 2 mapping JSON, API single-detection fields, and Streamlit target metadata inputs. Typed candidate/target `Requirement` objects preserve component and subsystem fields for hybrid retrieval.
- Exact-ID joins and malformed/unknown metadata validation are covered. Focused utility/CLI/API/UI tests: **50 passed in 29.85s**. Full-suite verification pending. No extraction heuristic infers metadata.
- Existing datasets lack demonstrated structured metadata coverage; this establishes product wiring only, not an empirical retrieval improvement.
- Focused UI/detector/embedding tests: **15 passed**; full suite: **348 passed, 12 tqdm deprecation warnings**, exit 0. This covers resource reuse, not measured inference latency or clean-cache startup.

## TASK-P38 repeated-inference consistency screen (2026-11-24)
- Python, CLI, single/batch API, and Streamlit detection now accept 1–5 repeated runs (default one). Repeated calls bypass the built-in response cache and capture each class/confidence. Any variation recommends human review; the first class is retained. API documentation, README, and architecture describe call/cost and evidence limits.
- Focused detector/LLM/API/CLI/UI tests: **61 passed in 28.25s**. Full suite: **383 passed, 12 existing `tqdm` deprecation warnings in 108.10s**, exit 0 (PowerShell duration 116.29s). An unrestricted run passed `test_api_root`; a sandboxed focused attempt stalled at the known Windows loopback path. All 311 `data/` files retained identical hashes; `git diff --check` passed.
- Evidence limit: controlled clients only. No real provider or human reviewer was used; consistency is not calibrated confidence, accuracy, or verification. Real-provider variability, review usefulness, rights-cleared data, Docker and hosted CI remain unverified.

## TASK-P39 preserve requirements in document tables (2026-11-24)
- DOCX extraction now visits document paragraphs and top-level table rows in source order; cells are separated by tabs so requirement IDs and text in different columns remain on one extracted source line. Text extraction now retains a terminating period. Image-only PDFs fail with an explicit OCR-not-supported message.
- Focused utility/document-script tests: **25 passed in 13.05s**. Full suite with optional documents dependencies in `.venv`: **386 passed, 1 Starlette deprecation warning in 109.83s**, exit 0 (PowerShell duration 115.85s). All **311** `data/` files retained identical paths and SHA-256 hashes; `git diff --check` passed.
- Synthetic documents only. Scanned PDFs, varied real-world table layouts, extraction recall, and requirement segmentation on actual SRS corpora remain unverified. The base system Python still needs the documented `documents` extra for PDF/DOCX support.
