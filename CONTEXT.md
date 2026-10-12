# CONTEXT.md — Current System Context & Architecture State

## Project Overview
- **Title:** Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation
- **Repository:** `https://github.com/kesarshharma/LLREDD.git`
- **Local Path:** `c:\Users\DN816\OneDrive\Desktop\My Stuff\minor`
- **Current Git Branch:** `main`; working tree has user changes and untracked replication artifacts (preserved during audit; see current `git status`).

---

## Architectural State & Components

1. **Core Package (`leredd/`):**
   - [`config.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/config.py): Pydantic Settings singleton (`LEREDDConfig`).
   - [`data_models.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/data_models.py): Pydantic models (`Requirement`, `RequirementPair`, `Prediction`, `EvaluationReport`, `DependencyType` Enum) + provenance extensions (`AnnotationSource`, `VerificationStatus`, `AnnotationProvenance`).
   - [`dataset_splitter.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/dataset_splitter.py): Leakage-resistant splitter (`DatasetSplitter`, `SplitConfig`, `DatasetSplit`, `SplitStrategy`, `LeakageReport`).
   - [`bm25_retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/bm25_retrieval.py): Lexical retriever (`BM25Retriever`) using `BM25Plus` from `rank-bm25`.
   - [`hybrid_retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/hybrid_retrieval.py): Hybrid retriever (`HybridRetriever`) combining SBERT dense embeddings and BM25 lexical matching via Reciprocal Rank Fusion (RRF, $rrf\_k=60$).
   - [`candidate_generator.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/candidate_generator.py): Candidate pair generator (`CandidateGenerator`, `CandidateGenerationManifest`, `GenerationStrategy` enum, `get_canonical_pair_key`).
   - [`review_queue.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/review_queue.py): Human review queue (`ReviewQueueManager`, `ReviewItem`, `QueueStatus`, `ReviewReasonCode`, `ReviewPriority`, `HumanDecision`, `ReviewQueueConfig`, `ReviewQueueManifest`).
   - [`provisional_labeler.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/provisional_labeler.py): Provisional dependency labeler (`HeuristicLabeler`, `LLMLabeler`, `ProvisionalLabeler`, `LabelingRunManifest`, `LabelingMode`).
   - [`embeddings.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/embeddings.py): SBERT (`all-MiniLM-L6-v2`) with MD5 pickle disk caching.
   - [`retrieval.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/retrieval.py): `ExampleRetriever` (Paper Eq. 1 Avg / Eq. 2 Max) and `RAGRetriever` (1000-char chunks, 200 overlap).
   - [`prompts.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/prompts.py): `PromptBuilder` matching Paper Figure 2 & Table I definitions.
   - [`llm.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/llm.py): Abstract `LLMClient` with `OpenAILLMClient`, `OllamaLLMClient`, `MockLLMClient`, and `DiskCachedLLMClient`.
   - [`detector.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/detector.py): `ResponseParser` (regex extraction + threshold re-annotation $\le 4 \rightarrow \text{No\_dependency}$) and `LEREDDDetector`.
   - [`evaluation.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/evaluation.py): `Evaluator` for Accuracy, Macro/Weighted F1, McNemar test, Fisher's test.
   - [`utils.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/utils.py): Regex extraction, $n(n-1)/2$ pair generation, Cohen's Kappa.
   - [`baselines/`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/baselines): `TFIDF_LSABaseline` and `BERTBaseline`.

2. **Interfaces & Technical Documentation:**
   - REST API: [`api/main.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/api/main.py), [`api/routes.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/api/routes.py) (FastAPI).
   - CLI Tool: [`leredd/cli.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/cli.py) (`leredd-cli split`, `extract`, `detect`, `evaluate`, etc.).
   - Web Dashboard: [`ui/app.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ui/app.py) (Streamlit).
   - Documentation: [`README.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/README.md), [`docs/ARCHITECTURE.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/ARCHITECTURE.md), [`docs/EXPERIMENTS.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/EXPERIMENTS.md), [`docs/API.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/API.md), [`docs/REPLICATION.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/docs/REPLICATION.md).

---

## Active Environment & Project Status
- **Environment Status (2026-10-13):** Python 3.13.3 and pytest 9.0.2 are installed. Latest unrestricted `python -m pytest tests/`: **264 passed, 38 warnings in 170.21s**, exit code 0 (173.83s outer duration). The sandbox-specific Windows TCP fallback hang remains an execution-boundary limitation; `test_api_root` passed when run unrestricted. Ruff is not installed; no packages were installed.
- **Documentation & Artifact Alignment Status (TASK-P8-02 / P8-04):**
  - **`README.md`**: Fully aligned with project title *"Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation"*. Documents both workflows (Dataset Generation & LEREDD Detection), 6 dependency categories, architecture Mermaid diagrams, CLI/API/UI/Docker instructions, empirical results, and honest limitations.
  - **`docs/ARCHITECTURE.md`**: Created detailed technical specifications covering candidate generation, provisional labeling, human review queue, system-disjoint splitting, dynamic ICL hybrid retrieval, confidence thresholding, and provenance schemas.
  - **`docs/EXPERIMENTS.md`**: Created comprehensive empirical report documenting system-disjoint benchmark results, model comparison tables, per-class metrics, statistical significance testing, and research insights.
  - **`docs/API.md`**: Updated technical reference for REST API endpoints, CLI commands, and Python SDK classes.
- **Current Development Phase:** TASK-P9-14 completed the whole-project workflow audit and TASK-P9-15 fixed false-success behavior in `train-bert` by writing reloadable classifier artifacts. The latest complete suite passes **264 tests** outside the sandbox. Generated pairs remain unlabeled/provisional until human verification; evaluation rejects provenance-bearing ground truth unless `VERIFIED`, while legacy provenance-free records retain a warning-based compatibility path. Replication evaluation remains blocked for current `pending_review` imports. Remaining external/research limits are the legacy benchmark's low power, MockLLM ablation invariance, unresolved `Implemented by` mapping, dataset verification/licensing, absent optional PDF/DOCX packages, and lack of real OpenAI/Ollama credentials/service validation. Next step for replication: confirm authoritative relation direction and data-use rights, then locally review annotations before producing metrics.

## Functional Map (TASK-P9-02, 2026-10-12)
- The product is a research prototype with two connected workflows: (1) convert SRS text into extracted requirements, candidate pairs, provisional labels, human-reviewed/verified annotations, and leakage-aware splits; (2) classify a target requirement pair using retrieved labeled examples and an LLM, then evaluate predictions against labeled pairs.
- Core data flows through `Requirement`, `RequirementPair`, `Prediction`, and `EvaluationReport`. Annotation provenance/status distinguish machine suggestions from reviewed ground truth. The six categories are Requires, Implements, Conflicts, Details, Is similar, and No_dependency.
- Dataset-generation flow: regex extraction (`utils.py`) → scalable candidate selection (`candidate_generator.py`; ALL_PAIRS/BM25/DENSE/HYBRID) → heuristic/LLM provisional labeling (`provisional_labeler.py`) → uncertainty queue and human decisions (`review_queue.py`, Streamlit page 6) → system/document/requirement/pair splitting (`dataset_splitter.py`).
- Detection flow: dense SBERT, BM25, or hybrid RRF retrieval (`retrieval.py`, `bm25_retrieval.py`, `hybrid_retrieval.py`) selects examples; `PromptBuilder` assembles taxonomy/examples/context; configured OpenAI/Ollama/Mock client generates output; `ResponseParser` converts it into a prediction and applies confidence thresholding. Optional RAG retrieves chunks from an SRS document.
- Surfaces: Typer CLI (`leredd/cli.py`), FastAPI (`api/`), Streamlit six-page dashboard (`ui/app.py`), and scripts for extraction, candidate generation, labeling, queue building, splitting, and evaluation.
- Evaluation: existing `run_full_evaluation.py` uses the small legacy 20-pair system-disjoint benchmark (10 ADB train, 5 APA dev, 5 TJA test) and explicitly uses MockLLM for LEREDD. New replication package JSONs (ADB evaluation 313, ADB pilot 100, APA 200, TJA 200) are loaded separately; `run_replication_evaluation.py` is a prototype defaulting to ADB Pilot, synthetic split/IDs, MockLLM, and excludes unresolved “Implemented by” records. It is not the existing paper-scale benchmark.
- Label lifecycle: both legacy CLI `pairs` and `candidates` emit unlabeled (`None`) records with candidate-generator provisional provenance. A human's explicit CLI annotation decision is recorded as `VERIFIED`; unknown explicit labels raise validation errors. Provenance-bearing evaluation ground truth is eligible only at `VERIFIED` status. Existing no-provenance benchmark records retain conditional legacy acceptance with a warning.

## Whole-Project Stabilization Audit (TASK-P9-14, 2026-10-13)
- UI candidate extraction/upload now persists provisional pairs to isolated output paths; explicit per-pair annotations persist a separate verified copy with annotator identity. Source dataset folders are protected, writes are atomic, and overwrites require confirmation. Cohen's kappa is not computed without independent annotation vectors.
- The evaluation page uses the strict system-disjoint runner and a unique report directory. Review-queue construction no longer invents a human decision. Provisional-label script defaults to MockLLM even when remote-provider environment configuration exists. Legacy evaluation now uses the safe split runner and unique output paths.
- Unfit supervised/fitted baselines now raise instead of fitting on their prediction batch. The public API config response redacts the OpenAI key; local API/CLI service bindings default to loopback. PDF/DOCX extraction has an optional extra and reports missing or malformed dependencies instead of silently returning no records.
- `leredd-cli train-bert` now atomically writes a loadable `bert_baseline.joblib` artifact and protects existing artifacts unless `--overwrite` is specified. `BERTBaseline.load()` restores the classifier for prediction.
- `docs/FEATURE_TRACEABILITY.md` tracks modules, CLI commands, API routes, dashboard controls, scripts, tests, external requirements, and limits.
- Verified: latest unrestricted complete suite `python -m pytest tests/` **264 passed, 38 warnings in 170.21s**, exit 0 (outer duration 173.83s); focused script/UI/baseline suite 31 passed/70.70s; UI workflows 7 passed/10.08s; API 9 passed/28.66s; training artifact focused suite 14 passed/1 deselected in 3.48s. FastAPI and Streamlit loopback health checks returned HTTP 200 and both processes exited. Four replication dataset SHA-256 hashes matched immediately-before-run values; `git diff -- data` was empty.
- Active limits: real OpenAI/Ollama requests were not made; PDF/DOCX packages are not installed; Docker was not built; no browser-based visual review was run; replication imports await human review and licensing/direction decisions; the legacy `scripts/train_baselines.py` remains an in-memory-only demonstration. Saved joblib artifacts must be trusted and loaded with compatible dependency versions.
