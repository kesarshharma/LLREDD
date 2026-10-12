# LEREDD Feature Traceability and Verification Ledger

Audit date: 2026-10-13. This ledger distinguishes code-path verification from real-service, research, and data-governance evidence. “Verified working” means the documented local contract has passing unit/integration or controlled end-to-end evidence; it does not imply paper reproduction or validity of unverified labels.

## End-to-End workflows

| Workflow / feature | Implementation | Dependencies / inputs | Test and evidence | Status |
|---|---|---|---|---|
| Fresh environment setup | `README.md`, `pyproject.toml`, `requirements.txt` | Python `>=3.10`; core dependencies; optional `.[dev]` and `.[documents]` extras | Install/start instructions inspected; no clean virtual environment was created and no packages were installed during this audit | Unverified on a fresh environment; setup commands are documented |
| Text/Markdown ingestion and requirement extraction | `leredd/utils.py`; `scripts/extract_requirements.py`; CLI `extract`; UI page 2 | UTF-8 text, `.txt` or `.md`; regex rules | `tests/test_utils.py`, `tests/test_data_scripts.py`, UI AppTest extraction and Markdown upload | Verified working |
| PDF and DOCX extraction | `leredd/utils.py` | Optional `pypdf` and `python-docx`, declared as `leredd[documents]` | Missing-package behavior tested; packages are not installed in this environment | Blocked by optional dependencies; clear error is shown |
| Exhaustive pair generation | `leredd/utils.py`; CLI `pairs`; UI page 2 | Extracted `Requirement` records | `tests/test_utils.py`, `tests/test_cli.py`, `tests/test_ui_workflows.py` | Verified working; output stays unlabeled/provisional |
| Candidate generation (ALL_PAIRS/BM25/DENSE/HYBRID) | `leredd/candidate_generator.py`; CLI `candidates`; `scripts/generate_candidates.py` | Requirement corpus; dense strategies use embeddings | `tests/test_candidate_generator.py`, `tests/test_generate_candidates_script.py`; BM25 and hybrid suites | Verified working; dense model quality depends on SBERT availability |
| Provisional heuristic/LLM labeling | `leredd/provisional_labeler.py`; CLI `label`; `scripts/label_candidates.py` | Explicit provider; default is MockLLM | `tests/test_provisional_labeler.py`, `tests/test_label_candidates_script.py` | Verified with heuristic/MockLLM; real OpenAI/Ollama calls not made |
| Human review and verification | `leredd/review_queue.py`; CLI queue commands; UI page 6 | Human reviewer, explicit decision, review queue JSON | `tests/test_review_queue.py`, `tests/test_cli.py`, UI queue AppTest journey | Verified with synthetic fixture; no human decisions were fabricated |
| Pair JSON export / validation | `leredd/utils.py`; CLI `annotate` (copy/validation utility); UI pages 2 and 6 | JSON pair schema | `tests/test_data_scripts.py`, `tests/test_ui_and_queue_integration.py`, `tests/test_ui_workflows.py` | Verified; UI writes are atomic and cannot target source dataset folders |
| System/document/requirement/pair dataset splitting | `leredd/dataset_splitter.py`; CLI `split`; `scripts/split_dataset.py` | Pair JSON inputs; output directory | `tests/test_dataset_splitter.py`, `tests/test_cli.py`, `tests/test_data_scripts.py` | Verified on synthetic inputs; requirement-disjoint split remains empty for the current 20-pair set |
| Dense, BM25, and hybrid example retrieval; optional RAG | `leredd/retrieval.py`, `leredd/bm25_retrieval.py`, `leredd/hybrid_retrieval.py` | SBERT for dense; `rank-bm25` for lexical; corpus and optional SRS text | `tests/test_retrieval.py`, `tests/test_bm25_retrieval.py`, `tests/test_hybrid_retrieval.py`, detector tests | Verified locally; real corpus retrieval quality is not a research claim |
| Dependency detection and response parsing | `leredd/detector.py`, `leredd/prompts.py`, `leredd/llm.py` | MockLLM, OpenAI, or local Ollama; config and optional example pool | `tests/test_detector.py`, `tests/test_prompts.py`, `tests/test_llm.py`, controlled UI journey | Verified with MockLLM and parser fixtures; remote/provider service behavior is unverified |
| Baselines | `leredd/baselines/bert_classifier.py`, `leredd/baselines/tfidf_lsa.py` | Training pairs; BERTBaseline additionally uses embeddings | `tests/test_baselines.py`, safe evaluation tests, BERT joblib save/load roundtrip | Verified when explicitly fit before prediction; BERT artifacts are persisted and reloadable; predictions reject unfit models |
| Leakage-checked evaluation/report | `leredd/evaluation.py`; `scripts/run_full_evaluation.py`; safe `scripts/run_evaluation.py`; UI page 4 | Fixed system-disjoint split; strict eligibility and contamination guards; MockLLM | `tests/test_safe_evaluation.py`, `tests/test_evaluation_entrypoint.py`, controlled UI report-contract test | Verified as demonstrative offline evaluation; n=5 test set and legacy provenance prevent strong research conclusions |
| Replication annotations | `scripts/integrate_replication.py`, `scripts/run_replication_evaluation.py` | Four imported JSON files, local verification status | `tests/test_replication_loader.py`, failure-mode, dataset, and evaluation-runner tests | Loader verified; metrics blocked until `pending_review` records are locally verified |

## Interfaces and user-facing controls

| Surface | Features / controls | Evidence | Status |
|---|---|---|---|
| Typer CLI (`leredd/cli.py`) | `extract`, `pairs`, `annotate`, `detect`, `evaluate`, `train-bert`, `split`, `candidates`, `label`, `queue-create`, `queue-list`, `queue-decide`, `serve` | `tests/test_cli.py`, `tests/test_system_verification.py`, focused script tests; `leredd-cli --help` | Commands covered; `train-bert` writes a versioned, reloadable classifier artifact atomically and protects an existing artifact unless `--overwrite` is supplied. |
| FastAPI (`api/main.py`, `api/routes.py`) | `GET /`, `/api/v1/health`, `/api/v1/config`; `POST /api/v1/detect`, `/batch_detect`, `/extract`; request validation and startup CLI | `tests/test_api.py`; actual local Uvicorn health request returned HTTP 200 and process exited after stop | Verified locally; config endpoint omits API secret and returns only a configured boolean |
| Streamlit page 1 | Home overview and research caveats | `AppTest` all-page render | Verified render; browser visual polish was not separately assessed |
| Streamlit page 2 | System ID, `.txt/.md` uploader, paste area, candidate output path, overwrite confirmation, extraction/pair generation, pair index, label, annotator ID, verified output path, overwrite confirmation; independent second-annotator agreement is explained as unsupported until a second vector exists | `tests/test_ui_workflows.py` triggers upload, extraction, provisional serialization, annotation and verified serialization; `tests/test_ui_and_queue_integration.py` checks persistence safety | Verified local workflow; Cohen’s kappa is intentionally not reported without independent labels |
| Streamlit page 3 | Provider/model, example count, RAG and threshold controls; OpenAI send confirmation; pair text; detection action and raw response expander | Controlled detector AppTest; detector/API suites | Verified with a controlled Mock-like detector. Real OpenAI and Ollama not exercised |
| Streamlit page 4 | Fixed system-disjoint evaluation, report metrics, limitations, confusion matrix; unique output directory | Controlled AppTest verifies split/strict flags and result display; safe evaluation tests exercise actual runner | Verified wiring; UI test substitutes the runner and does not itself retrain real models |
| Streamlit page 5 | Legacy ADB dependency graph and data caveat | AppTest all-page render; graph uses read-only legacy JSON | Verified render only; graph is illustrative and source has no verification provenance |
| Streamlit page 6 | Queue path/reload; status/reason/priority filters; search and item selection; decision/rationale/reviewer fields; explicit confirmation; persisted decision; verified export path and replace confirmation | UI AppTest creates a temporary queue, records a correction, reloads JSON, and exports a verified pair | Verified on synthetic temporary queue; multi-reviewer concurrent writes remain unsupported |

## Python module inventory

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
| `pyproject.toml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml` | Package dependencies, pytest config, container build and API/UI/Redis services; static config inspected, Docker image not built |

Pytest uses `tests/`, files named `test_*.py`, and `--strict-markers -v` from `pyproject.toml`; no custom marker declarations or timeout plugin are configured. The suite includes: `test_ablation_benchmark`, `test_api`, `test_baselines`, `test_bm25_retrieval`, `test_build_review_queue_script`, `test_candidate_generator`, `test_cli`, `test_config`, `test_data_scripts`, `test_dataset_splitter`, `test_detector`, `test_embeddings`, `test_evaluation`, `test_evaluation_entrypoint`, `test_generate_candidates_script`, `test_hybrid_retrieval`, `test_implemented_by_exclusion`, `test_label_candidates_script`, `test_llm`, `test_prompts`, `test_provenance`, `test_provisional_labeler`, four replication tests, `test_retrieval`, `test_review_queue`, `test_safe_evaluation`, `test_system_verification`, `test_ui_and_queue_integration`, `test_ui_workflows`, and `test_utils`.

## Configuration and unverified integrations

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
