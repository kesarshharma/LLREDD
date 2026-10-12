# System Architecture & Technical Specifications

This document outlines the detailed system architecture, data models, and component workflows for the project **“Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation”**.

---

## 1. High-Level Architecture Overview

The system consists of two primary operational workflows integrated through common data models, a unified configuration system (`LEREDDConfig`), and a central persistence store:

```mermaid
graph TB
    subgraph Dataset Generation Subsystem
        A[Raw SRS Input] --> B[SRS Extractor / Parser]
        B --> C[Structured Requirements]
        C --> D[Candidate Generator - BM25 / SBERT Filter]
        D --> E[Provisional Labeler - LLM/Rules]
        E --> F[Review Queue Engine]
        F --> G[Streamlit UI - Expert Review]
        G --> H[Verified Ground Truth Dataset]
        H --> I[System-Disjoint Dataset Splitter]
    end

    subgraph Dependency Detection Engine (LEREDD)
        I --> J[ICL Example Pool]
        K[Unseen Requirement Pair] --> L[Hybrid Retriever]
        J --> L
        L --> M[Dynamic Prompt Builder]
        K --> M
        M --> N[LLM Engine Client]
        N --> O[Likert Confidence Filter]
        O --> P[Final Requirement Dependency Pred]
    end

    subgraph Service Interfaces
        P --> Q[FastAPI REST Server]
        P --> R[Typer CLI Interface]
        P --> S[Streamlit Analytics Dashboard]
    end
```

---

## 2. Core Components & Subsystems

### 2.1 Semi-Automated Dataset Generation Pipeline

1. **SRS Extractor (`leredd/cli.py` & `leredd/utils.py`)**:
   - Parses raw Software Requirements Specification (SRS) documents.
   - Extracts discrete requirement statements using key term triggers (`shall`, `must`, `should`, `will`).
   - Assigns unique identifier structures (e.g., `REQ-ADB-001`).

2. **Candidate Pair Generator (`leredd/candidate_generator.py`)**:
   - Reduces the $O(N^2)$ exhaustive requirement pair search space.
   - Applies top-$k$ candidate pair selection using lexical (BM25) and semantic vector distance (SBERT `all-MiniLM-L6-v2`).
   - Preserves high-recall candidate sets while drastically reducing human annotation overhead.

3. **Provisional Labeler (`leredd/provisional_labeler.py`)**:
   - Assigns initial provisional dependency labels and confidence ratings using zero-shot prompting or deterministic rule heuristics.
   - Attaches provenance metadata (`annotator_id`, `annotation_source`, `confidence_score`, `created_at`).

4. **Review Queue Engine & UI (`leredd/review_queue.py` & `ui/app.py`)**:
   - Implements a state-machine review queue (`pending`, `approved`, `modified`, `rejected`).
   - Interactive Streamlit dashboard allows domain experts to rapidly review, refine, or approve candidate pair annotations.

5. **Dataset Splitter (`leredd/dataset_splitter.py`)**:
   - Splits annotated pair datasets into `train`, `dev`, and `test` subsets.
   - Supports both `random` splitting and strict `system-disjoint` splitting (e.g., Train: *ADB*, Dev: *APA*, Test: *TJA*).
   - Audits data leakage across software system boundaries.

---

### 2.2 LEREDD Automated Dependency Detection Engine

1. **Hybrid Retriever (`leredd/hybrid_retrieval.py`)**:
   - Combines BM25 lexical search scores with SBERT embedding cosine similarity.
   - Computes overall pair-to-pair similarity using the paper's 4-way average distance formula:
     $$Score_{avg} = \frac{sim(R_1, R_a) + sim(R_1, R_b) + sim(R_2, R_a) + sim(R_2, R_b)}{4}$$
   - Dynamically selects top $k$ (default 6) annotated examples per dependency class from the training pool.

2. **Dynamic Prompt Builder (`leredd/prompts.py`)**:
   - Constructs contextual prompts adhering strictly to research paper definitions.
   - Includes system instructions, taxonomy guidelines, few-shot dynamic ICL examples, and target pair statements.

3. **LLM Engine Client (`leredd/llm.py`)**:
   - Unified interface supporting `OpenAI` (`gpt-4`, `gpt-3.5-turbo`), `Ollama` local LLMs (`llama3`), and offline `MockLLMClient`.
   - Incorporates prompt-hash caching, token usage calculation, cost estimation, and exponential backoff retry logic.

4. **Confidence Threshold Filter (`leredd/detector.py`)**:
   - Evaluates returned Likert confidence score (1 to 5).
   - If confidence score $\le 4$, automatically overrides prediction to `No_dependency` to eliminate low-confidence false positives.

---

## 3. Data Models & Provenance Schema

All objects conform to strict Pydantic data schemas defined in [`leredd/data_models.py`](file:///c:/Users/DN816/OneDrive/Desktop/My%20Stuff/minor/leredd/data_models.py):

- `Requirement`: ID, system, text, metadata.
- `RequirementPair`: `id`, `req_a`, `req_b`, `dependency_type`, `confidence`, `provenance` (`annotator_id`, `annotation_source`, `timestamp`).
- `Prediction`: `pair_id`, `dependency_type`, `rationale`, `confidence`, `raw_response`, `prompt_tokens`, `completion_tokens`, `estimated_cost_usd`.

---

## 4. Verification & Safety Enforcements

- **Provenanced Dataset Integrity**: All annotated pairs produced by the pipeline maintain audit trails.
- **Data Leakage Auditing**: `SystemDisjointSplitter` explicitly logs and checks system boundary separation to prevent evaluation contamination.
- **Fail-Safe Fallbacks**: If external LLM API endpoints fail, `MockLLMClient` provides fallback functionality with explicit execution warnings.
