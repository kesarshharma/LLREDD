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
        G --> H[Human-Verified Dataset]
        H --> H1[Immutable Versioned Release + Manifest]
        H1 --> I[System-Disjoint Dataset Splitter]
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
   - Reads text-based PDF and DOCX files when the optional `documents` extra is installed; DOCX body paragraphs and table rows are traversed in document order. Image-only PDFs require OCR, which is not provided.
   - Extracts discrete requirement statements using key term triggers (`shall`, `must`, `should`, `will`).
   - Assigns unique identifier structures (e.g., `REQ-ADB-001`).
   - Accepts an optional explicit ID-keyed metadata sidecar/API/UI mapping for `component` and `subsystem`; unknown IDs and unsupported fields fail validation. Values are not inferred from requirement prose.

2. **Candidate Pair Generator (`leredd/candidate_generator.py`)**:
   - Supports exhaustive `ALL_PAIRS` generation and top-$k$ `BM25`, `DENSE`, and `HYBRID` retrieval strategies.
   - Dense generation ranks individual requirements with a requirement-level retriever; hybrid generation fuses that ranking with BM25 using reciprocal-rank fusion.
   - Retrieval limits reduce annotation effort but can omit true dependencies; only untruncated `ALL_PAIRS` runs are marked exhaustive.
   - Each manifest records pair-space recall against the complete unordered pair universe and the count of uncovered pairs. This is coverage of possible candidates, not recall of labeled dependency pairs.

3. **Provisional Labeler (`leredd/provisional_labeler.py`)**:
   - Assigns initial provisional dependency labels and confidence ratings using zero-shot prompting or deterministic rule heuristics.
   - Attaches provenance metadata (`annotator_id`, `annotation_source`, `confidence_score`, `created_at`).

4. **Review Queue Engine & UI (`leredd/review_queue.py` & `ui/app.py`)**:
   - Implements `pending`, `accepted`, `corrected`, `rejected`, `deferred`, and `reviewed` queue outcomes.
   - Accepted and corrected decisions require an explicit reviewer and create verified records; rejection/defer does not produce verified ground truth.
   - Interactive Streamlit dashboard allows domain experts to review, refine, accept, reject, or defer candidate annotations.

5. **Verified Dataset Release (`leredd/dataset_release.py`, CLI `dataset-release`)**:
   - Validates that each record has a final label and coherent human `VERIFIED` provenance; rejects legacy provenance-free records for release.
   - Publishes a new directory containing `pairs.json` and a manifest with the release version, input hashes, released-data hash, record count, and class counts.
   - Updates `provenance.dataset_version` in copied release records without changing the source files. An existing version directory is never overwritten.

6. **Dataset Splitter (`leredd/dataset_splitter.py`)**:
   - Splits annotated pair datasets into `train`, `dev`, and `test` subsets.
   - Supports `SYSTEM`, `DOCUMENT`, `REQUIREMENT`, and `PAIR` strategies. `PAIR` is a random pair-level split and does not prevent requirement leakage.
   - Audits requirement overlaps among train/dev/test/retrieval partitions and system overlaps for system-disjoint splits. A system-disjoint pair must have both requirements from the same system.

---

### 2.2 LEREDD Automated Dependency Detection Engine

1. **Hybrid Retriever (`leredd/hybrid_retrieval.py`)**:
   - Fuses BM25 lexical and SBERT dense rankings with reciprocal-rank fusion. When target and candidate records contain comparable `component`, `subsystem`, or meaningful `system` values, it can add a normalized exact-overlap metadata ranking. The metadata ranking is omitted when no candidate shares a value.
   - Computes overall pair-to-pair similarity using the paper's 4-way average distance formula:
     $$Score_{avg} = \frac{sim(R_1, R_a) + sim(R_1, R_b) + sim(R_2, R_a) + sim(R_2, R_b)}{4}$$
   - Dynamically selects top $k$ (default 6) annotated examples per dependency class from the training pool.

2. **Dynamic Prompt Builder (`leredd/prompts.py`)**:
   - Constructs contextual prompts adhering strictly to research paper definitions.
   - Includes system instructions, taxonomy guidelines, few-shot dynamic ICL examples, and target pair statements.
   - Optional SRS context uses fixed-size overlapping chunks ranked by dense similarity; chunk size must be positive and overlap smaller than chunk size. Requesting zero chunks returns no context without initializing embeddings.

3. **LLM Engine Client (`leredd/llm.py`)**:
   - Unified interface supporting `OpenAI` (`gpt-4`, `gpt-3.5-turbo`), `Ollama` local LLMs (`llama3`), and offline `MockLLMClient`.
   - Incorporates prompt-hash caching, token usage calculation, cost estimation, and exponential backoff retry logic.

4. **Confidence-Aware Validation (`leredd/detector.py`)**:
   - Evaluates the model's self-reported Likert confidence against the configured review threshold.
   - Low-confidence predictions retain their predicted dependency class and are marked `requires_review`; outputs above the threshold are marked `passed_threshold`.
   - An optional 1–5 run consistency screen repeats the same prompt without response-cache reuse. Label or confidence variation recommends human review; the first run's label is retained. The default remains one inference.
   - Screening status is not annotation verification, and Likert confidence is not treated as calibrated probability. Disabling the threshold is recorded explicitly.

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
