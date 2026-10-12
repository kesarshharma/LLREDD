# CONTEXT.md — Current System Context & Architecture State

## Project Overview
- **Title:** Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation
- **Repository:** `https://github.com/kesarshharma/LLREDD.git`
- **Local Path:** `c:\Users\DN816\OneDrive\Desktop\My Stuff\minor`
- **Current Git Branch:** `main` (Clean working tree, 3 historical commits)

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
- **Environment Status:** Python virtual environment fully operational. All dependencies installed. `leredd` installed in editable mode. Full test suite passing cleanly with zero regressions.
- **Documentation & Artifact Alignment Status (TASK-P8-02 / P8-04):**
  - **`README.md`**: Fully aligned with project title *"Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation"*. Documents both workflows (Dataset Generation & LEREDD Detection), 6 dependency categories, architecture Mermaid diagrams, CLI/API/UI/Docker instructions, empirical results, and honest limitations.
  - **`docs/ARCHITECTURE.md`**: Created detailed technical specifications covering candidate generation, provisional labeling, human review queue, system-disjoint splitting, dynamic ICL hybrid retrieval, confidence thresholding, and provenance schemas.
  - **`docs/EXPERIMENTS.md`**: Created comprehensive empirical report documenting system-disjoint benchmark results, model comparison tables, per-class metrics, statistical significance testing, and research insights.
  - **`docs/API.md`**: Updated technical reference for REST API endpoints, CLI commands, and Python SDK classes.
- **Current Development Phase:** All project roadmap phases (Phase 0 through TASK-P8-04) are **100% COMPLETE**. Read-only code and artifact audits completed. `ISSUE-011` (redundant duplicate `LEREDD_MockLLM` report entry in `scripts/run_full_evaluation.py`) resolved and verified by 27 passing tests in `tests/test_safe_evaluation.py`. Remaining tracked issues (ISSUE-009, 010, 012–015) documented in [`ISSUES.md`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/ISSUES.md). Codebase and artifacts remain verified and ready for academic submission.
