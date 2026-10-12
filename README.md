# Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)

A comprehensive research implementation and software framework for **Automated Requirement Dependency Detection (LEREDD)** and **Semi-Automated Dataset Generation**, as presented in the minor project proposal *"Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation"*.

---

## 📌 Executive Summary

Software requirements engineering frequently suffers from hidden requirement dependencies that lead to incomplete system architectures, integration bugs, and costly rework. This repository provides a complete dual-workflow platform:

1. **Automated Requirement Dependency Detection**: Implements the **LEREDD** architecture, utilizing Large Language Models (LLMs) with dynamic In-Context Learning (ICL), hybrid BM25 + SBERT retrieval, confidence thresholding, and multi-class classification across six dependency categories.
2. **Semi-Automated Dataset Generation**: Provides an end-to-end pipeline to transform raw Software Requirements Specification (SRS) text into candidate datasets that remain provisional until an explicit human review decision. The pipeline features SRS parsing, candidate pair generation, hybrid retrieval filtering, provisional labeling, human review queue management via a Streamlit web interface, system-disjoint dataset splitting, and provenance tracking.

---

## 🏷️ Dependency Taxonomy

The framework detects and annotates **six standard requirement dependency categories**:

| Dependency Type | Directional | Description | Example |
| :--- | :---: | :--- | :--- |
| **Requires** | Yes ($R_1 \rightarrow R_2$) | $R_1$ requires the existence or execution of $R_2$ to function. | "BCS shall apply brakes" requires "BCS shall receive camera input". |
| **Implements** | Yes ($R_1 \rightarrow R_2$) | $R_1$ provides a concrete implementation or realization of abstract requirement $R_2$. | High-level architectural specification implemented by component rule. |
| **Conflicts** | No ($R_1 \leftrightarrow R_2$) | $R_1$ and $R_2$ specify mutually exclusive behavior or resource constraints. | Maximum speed capped at 30km/h vs high-speed cruise mode set to 100km/h. |
| **Details** | Yes ($R_1 \rightarrow R_2$) | $R_1$ provides additional context, parameters, or refinement for $R_2$. | Detailed sensor sampling rate parameter refining general sensor rule. |
| **Is_similar** | No ($R_1 \leftrightarrow R_2$) | $R_1$ and $R_2$ share substantial functional overlap without explicit dependency. | Duplicate object detection rules across two separate sub-modules. |
| **No_dependency** | N/A | $R_1$ and $R_2$ are functionally independent. | User login GUI styling pair vs brake pad material specification. |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Workflow 1: Dataset Generation Pipeline
        SRS[Raw SRS Documents] --> Parser[SRS Parser / Extractor]
        Parser --> Reqs[Structured Requirements]
        Reqs --> CandGen[Candidate Pair Generator
(Hybrid BM25 + SBERT Filtering)]
        CandGen --> ProvLabeler[Provisional Labeler
(Zero-Shot / Rule-based)]
        ProvLabeler --> RevQueue[Review Queue Manager]
        RevQueue --> UI[Streamlit Review Dashboard]
        UI --> Annotator[Human Expert Verification]
        Annotator --> VerifiedDS[(Verified Ground-Truth Dataset)]
        VerifiedDS --> Splitter[System-Disjoint Dataset Splitter]
        Splitter --> Splits[(Train / Dev / Test System-Disjoint Splits)]
    end

    subgraph Workflow 2: LEREDD Detection Engine
        Splits --> ICLPool[ICL Example Pool]
        TestPair[Target Requirement Pair] --> HybridRet[Hybrid Retriever
(BM25 + SBERT Embeddings)]
        HybridRet --> ICLPool
        ICLPool --> PromptEngine[Dynamic Prompt Builder
(Taxonomy + Selected Examples)]
        TestPair --> PromptEngine
        PromptEngine --> LLMClient[LLM Client Engine
(OpenAI / Ollama / Mock)]
        LLMClient --> PostFilter[Confidence Threshold Filter
(Likert Score <= 4 -> No_dependency)]
        PostFilter --> Pred[Final Dependency Prediction]
    end
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

- **Python**: 3.10 or higher
- **Git**

```bash
# Clone the repository
git clone https://github.com/kesarshharma/LLREDD.git
cd minor

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install the package and development/test dependencies
pip install -e ".[dev]"

# Optional: enable PDF and DOCX text extraction
pip install -e ".[documents]"
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and adjust setup parameters:

```bash
cp .env.example .env
```

To run completely offline without external API keys (using Mock LLM & local embeddings):
```env
LLM_PROVIDER=mock
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 💻 Usage & Interfaces

The system provides three primary interfaces for execution:

### 1. Command-Line Interface (`leredd-cli`)

```bash
# Workflow 1: Dataset Generation Commands
# Extract requirement statements from SRS text
leredd-cli extract --input data/raw/adb_srs.txt --system ADB --output data/provisional/adb_reqs.json

# Legacy exhaustive pair generation also writes unlabeled candidates
leredd-cli pairs --input data/provisional/adb_reqs.json --output data/candidates/adb_pairs.json

# Train and save a reusable baseline classifier artifact
leredd-cli train-bert --train data/verified/adb_pairs.json --output model/

# Generate exhaustive and BM25 candidates from an SRS (outputs are isolated by run)
python scripts/generate_candidates.py --input data/raw/adb_srs.txt --output-dir data/candidates/runs/demo

# Label candidates with the offline mock provider by default
python scripts/label_candidates.py --input data/candidates/runs/demo/adb_candidates_bm25.json --output-dir data/provisional/runs/demo --provider mock

# Create a pending queue only; this command never creates human decisions
python scripts/build_review_queue.py --input data/provisional/runs/demo/adb_candidates_bm25_provisional_hybrid.json --output data/queue/runs/demo/review_queue.json

# Split verified dataset into system-disjoint train/dev/test sets
python scripts/split_dataset.py --input data/verified/all_annotated.json --output-dir data/split_runs/demo

# Workflow 2: Dependency Detection & Evaluation
# Run LEREDD detection on target requirement pair file
leredd-cli detect --input data/candidates/test_pairs.json --examples data/verified/all_annotated.json --output data/evaluation/predictions.json

# Run the safe system-disjoint demonstration benchmark (writes to a unique run folder)
python scripts/run_evaluation.py
```

The training command writes `model/bert_baseline.joblib`. Existing artifacts are
preserved unless `--overwrite` is supplied. Load only artifacts from trusted
sources; joblib uses Python pickle serialization. The artifact can be loaded with
`BERTBaseline.load(path)` for prediction.

Generated pairs are unlabeled (`dependency_type: null`) until an explicit annotation is recorded. Provenance-bearing ground-truth annotations are eligible for metrics only when their status is `verified`; imported replication records currently marked `pending_review` must pass through local human review first. The replication runner therefore does not produce metrics from the current imported annotations.

### 2. REST API (`api/main.py`)

Launch high-performance FastAPI REST backend:

```bash
python api/main.py --port 8000
# or using CLI:
leredd-cli serve --port 8000
```
- Interactive Swagger UI Documentation: `http://localhost:8000/docs`
- Health check endpoint: `GET /api/v1/health`
- Pair detection endpoint: `POST /api/v1/detect`
- Batch detection endpoint: `POST /api/v1/batch_detect`
- Requirement extraction endpoint: `POST /api/v1/extract`
- Active config endpoint (secrets redacted): `GET /api/v1/config`

### 3. Streamlit Web Dashboard (`ui/app.py`)

Interactive web interface for dataset annotation, pair detection, queue management, and model evaluation visualization:

```bash
streamlit run ui/app.py
```
Open `http://localhost:8501` to access:
- **Interactive Review Queue**: Review, edit, approve, or reject provisionally labeled candidate pairs with full provenance tracking.
- **Single / Batch Detection**: Execute real-time LEREDD detection with customizable confidence thresholding.
- **Model Benchmark Visualizer**: Explore confusion matrices, precision/recall metrics, and statistical significance tests.

---

## 🐳 Docker Deployment

The repository includes complete Docker containerization (`Dockerfile` and `docker-compose.yml`). *(Note: Docker configuration files are statically verified by automated test suites; live container execution requires an active local Docker daemon).*

```bash
# Build and launch API, Streamlit UI, and services via Docker Compose
docker-compose up --build
```

Services exposed:
- **Streamlit Web UI**: `http://localhost:8501`
- **FastAPI REST API**: `http://localhost:8000/docs`

---

## 📊 Evaluation & Empirical Benchmarks

The framework was evaluated under a strict **System-Disjoint Split Strategy** (training on *ADB*, validating on *APA*, and testing on *TJA*) to prevent data leakage between software domains.

### Performance Comparison

| Model Architecture | Evaluation Setting | Accuracy | Macro F1 | Weighted F1 | Primary Characteristics |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **TF-IDF + Cosine Baseline** | System-Disjoint | 0.400 | 0.143 | 0.229 | Unsupervised TF-IDF vector similarity with SVD. |
| **Supervised BERT (RF)** | System-Disjoint | 0.200 | 0.125 | 0.200 | Supervised SBERT + RandomForest classifier with class balancing. |
| **LEREDD (Mock Deterministic)** | System-Disjoint | 0.400 | 0.222 | 0.333 | Keyword heuristic fallback mode without active LLM API. |
| **LEREDD (Paper Reference GPT-4)** | System-Disjoint (Zero/Few-Shot) | **0.880** | **0.860** | **0.875** | Paper reference benchmark using live GPT-4 API (not offline MockLLM). |

*Detailed benchmark metrics, per-class breakdown, and statistical significance results are available in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) and [`data/evaluation/system_disjoint_report.json`](data/evaluation/system_disjoint_report.json).*

---

## ⚠️ Limitations & Open Challenges

1. **System-Disjoint Performance Gap**: Models trained or evaluated on random splits exhibit optimistic performance due to lexical overlap within the same system SRS. System-disjoint evaluation reveals significant domain adaptation challenges across distinct software specifications.
2. **Class Imbalance**: Requirement datasets naturally contain an overwhelmingly high proportion of `No_dependency` pairs (>80%). Precision and recall for rare classes like `Details` and `Conflicts` require targeted candidate sampling.
3. **Candidate Filtering Trade-off**: Aggressive top-$k$ candidate pair retrieval significantly reduces human annotation effort but introduces potential false-negative filtering risk for non-obvious requirement dependencies.

---

## 🛠️ Testing & Verification

Run the full automated test suite (including unit, integration, and provenance verification tests):

```bash
pytest tests/ -v
```

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
