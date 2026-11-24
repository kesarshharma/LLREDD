# Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red.svg)](https://streamlit.io/)

A comprehensive research implementation and software framework for **Automated Requirement Dependency Detection (LEREDD)** and **Semi-Automated Dataset Generation**, as presented in the minor project proposal *"Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation"*.

---

## 📌 Executive Summary

Software requirements engineering frequently suffers from hidden requirement dependencies that lead to incomplete system architectures, integration bugs, and costly rework. This repository provides a complete dual-workflow platform:

1. **Automated Requirement Dependency Detection**: Implements the **LEREDD** architecture, utilizing Large Language Models (LLMs) with dynamic In-Context Learning (ICL), hybrid BM25 + SBERT retrieval with optional component/subsystem metadata alignment, confidence screening, and multi-class classification across six dependency categories.
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
        LLMClient --> Screen[Confidence Review Screen
(low score -> human review cue)]
        Screen --> Pred[Prediction + Screening Status]
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

The local heuristic labeling and MockLLM paths do not require provider
credentials. The default embedding model is `all-MiniLM-L6-v2`; it must be
cached locally or downloaded when network access is available. To force an
offline model-cache check during a run, set `HF_HUB_OFFLINE=1`. Do not add real
provider credentials unless you intend to call that provider.

---

## 💻 Usage & Interfaces

The system provides three primary interfaces for execution:

### 1. Command-Line Interface (`leredd-cli`)

```powershell
# Keep generated examples separate from source datasets and historical reports.
$run = ".pytest-tmp\leredd-demo"
New-Item -ItemType Directory -Force $run | Out-Null
python -m leredd.cli --help
```

The safe end-to-end dataset workflow is shown below. Point `--input` at your
own SRS and use a distinct `--system` value for each system. Every output here
is disposable and stays under `.pytest-tmp`.

Optional component/subsystem metadata can be supplied as a JSON sidecar keyed
by the exact requirement IDs present in the SRS, for example:

```json
{
  "REQ-DEMO-01": {"component": "Brake Controller", "subsystem": "Braking"},
  "REQ-DEMO-02": {"component": "Dashboard"}
}
```

Pass the sidecar with `--metadata-json .pytest-tmp\metadata.json` to `extract`
or `candidates`. The same mapping is available on Streamlit page 2 and the API
`/extract` request. Metadata is joined by exact ID; the software does not infer
component or subsystem from requirement prose. Candidate JSON carries these
values into pair retrieval. CLI `detect`, API batch detection, and the UI
detection form accept metadata on the target requirements as well.

```powershell
$run = ".pytest-tmp\leredd-demo"
New-Item -ItemType Directory -Force "$run\out" | Out-Null
python -m leredd.cli extract --input .\my-srs.txt --system DEMO --output "$run\out\requirements.json"
python -m leredd.cli candidates --input "$run\out\requirements.json" --strategy bm25 --top-k 5 --system DEMO --output "$run\out\candidates.json"
python -m leredd.cli label --input "$run\out\candidates.json" --mode heuristic --output "$run\out\provisional.json"
python -m leredd.cli queue-create --input "$run\out\provisional.json" --output "$run\out\review-queue.json"
python -m leredd.cli queue-list --queue "$run\out\review-queue.json"
```

For an opt-in multi-model vote, use `--mode multi_llm --models model-a,model-b`
or add `--models` to `--mode hybrid`. Each distinct model ID is called through
the configured `LLM_PROVIDER`. Only unanimous valid model votes produce a
provisional label; disagreement or an incomplete vote leaves the pair unlabeled
for review. With OpenAI or Ollama this sends requirement text to the configured
provider and can incur cost or use local compute. The default `mock` provider is
only a software-path demonstration, not independent model evidence. All results
remain provisional.

Review the queue in Streamlit or use `queue-decide` only for an actual human
decision. Supply a real, attributable reviewer identifier; the program records
but cannot authenticate it. Proposals remain provisional until that decision
is saved. A human decision can export eligible pairs with:

```powershell
python -m leredd.cli queue-decide --queue "$run\out\review-queue.json" --item-id "REPLACE_WITH_REVIEWED_ITEM_ID" --decision accepted --reviewer "REPLACE_WITH_YOUR_IDENTIFIER" --verified-out "$run\out\verified.json"
```

Do not run this example without reviewing the selected item and making the
stated decision. For a system-disjoint dataset, repeat extraction and review
for at least three distinct systems. Publish their verified exports as a new
immutable release, then split that released dataset:

```powershell
python -m leredd.cli dataset-release --input "$run\verified-a.json,$run\verified-b.json,$run\verified-c.json" --version v1.0.0 --output-dir "$run\releases\v1.0.0"
python -m leredd.cli split --input "$run\releases\v1.0.0\pairs.json" --strategy system --output-dir "$run\splits"
```

`dataset-release` accepts only labeled, provenance-bearing records whose
provenance passes the human `VERIFIED` eligibility checks. It writes `pairs.json`
and a `manifest.json` with source-file hashes, release hash, class counts, and
version. The output directory must be new; releases are not overwritten. Each
record's `provenance.dataset_version` is set to the requested version. Keep the
source verified exports and inspect the manifest before evaluation.

For single-pair detection, give the CLI a target candidate file and a separate
verified training/example pool:

```powershell
python -m leredd.cli detect --input "$run\target-candidates.json" --examples "$run\verified-train.json" --output "$run\predictions.json"
```

Detection uses one inference by default. To screen for same-prompt run
inconsistency, add `--stability-runs 3` (allowed range 1–5). Multiple runs make
multiple fresh provider calls; any label or self-reported confidence variation
recommends human review while retaining the first run's predicted class. This
screen is not a calibrated confidence estimate or human verification.

Never use test-partition labels to generate predictions or include test
requirements in the retrieval-example pool. The evaluation runner separates
prediction generation from scoring and checks retrieval leakage.

Run the evaluation only on a reviewed, leakage-resistant split and write to a
new report path:

```powershell
python scripts/run_full_evaluation.py --split-dir "$run\splits" --output "$run\out\report.json"
```

The runner uses MockLLM for LEREDD and writes the primary and ablation reports
beside the specified output. Its report describes the supplied data and its
provenance eligibility; synthetic fixture results are software evidence only,
never human annotation or research benchmark evidence.

The training command writes `model/bert_baseline.joblib`. Existing artifacts are
preserved unless `--overwrite` is supplied. Load only artifacts from trusted
sources; joblib uses Python pickle serialization. The artifact can be loaded with
`BERTBaseline.load(path)` for prediction.

#### Isolated, no-service walkthrough (PowerShell)

This example writes its working artifacts only below `.pytest-tmp/p11-demo` and
uses BM25 plus the heuristic labeler; it does not contact an LLM service. The
queue is intentionally left pending until a person reviews each item.

```powershell
$run = ".pytest-tmp\p11-demo"
New-Item -ItemType Directory -Force $run | Out-Null
@'
[REQ-ADB-101] The vehicle controller shall use wheel speed sensor data.
[REQ-ADB-102] The vehicle controller shall report wheel speed sensor faults.
[REQ-ADB-103] The dashboard shall display fault notifications from REQ-ADB-102.
'@ | Set-Content -Encoding utf8 "$run\sample.md"

leredd-cli extract --input "$run\sample.md" --system ADB --synthetic --output "$run\requirements.json"
leredd-cli candidates --input "$run\requirements.json" --strategy bm25 --top-k 1 --system ADB --output "$run\candidates.json"
leredd-cli label --input "$run\candidates.json" --mode heuristic --output "$run\provisional.json"
leredd-cli queue-create --input "$run\provisional.json" --output "$run\review-queue.json"
leredd-cli queue-list --queue "$run\review-queue.json"
```

Inspect the requirements, candidate manifest, provisional provenance, and queue
before deciding labels. Record only decisions actually made by a human using
`leredd-cli queue-decide --reviewer YOUR_IDENTIFIER`; `leredd-cli annotate`
likewise requires `--annotator YOUR_IDENTIFIER`. These commands require the
human operator to supply a non-placeholder identity explicitly; the application
does not authenticate the supplied identifier. Do not treat the heuristic
proposal as ground truth.
The example input above is synthetic/demo data, so `--synthetic` marks every
extracted requirement and generated pair. Such records remain marked after
review and are excluded from evaluation, retrieval examples, and dataset release.

Every candidate-generation manifest also reports `pair_space_recall` and
`uncovered_pair_count`. This compares the emitted pairs with the complete set of
unordered pairs possible for the input requirements. It measures candidate
space coverage, **not** recall of true dependency labels; the latter requires a
verified labeled sample.
The checked-in legacy benchmark can be evaluated separately, with an isolated
report destination, by running:

```powershell
python scripts/run_full_evaluation.py --split-dir data/splits/system_disjoint --output "$run\evaluation\report.json"
```

Use a new report path when reproducing the checked-in example. The checked-in
report uses a 20-pair, provenance-free system-disjoint split and MockLLM. Its
metrics are illustrative, not evidence about new SRS annotations or real-LLM
performance. Provenance-bearing labels require `verified` status, human source,
a non-placeholder reviewer identity, and matching `accepted_label`. Imported
replication records still need local human review before evaluation.

#### Independent annotation and agreement pilot

The assisted review queue is not a substitute for independent annotation. For
a controlled agreement pilot, prepare one fixed candidate sample and optional
UTF-8 SRS context file, then have two reviewers independently run
`scripts/annotate_independent.py` to separate output files. That tool hides
provisional machine labels, records abstentions, short rationales, and elapsed
time per pair, and preserves incomplete work for `--resume`. Compare completed
vectors with `scripts/compare_independent_annotations.py`; its report measures
agreement and time but does not adjudicate labels or mark them verified. Follow
the [independent annotation guide](docs/ANNOTATION_GUIDE.md). Keep the inputs
identical and write outputs to fresh paths outside `data/replication/` and
historical evaluation reports. Synthetic fixtures demonstrate the software
only and cannot be reported as human agreement or annotation-effort evidence.

For the prepared hybrid top-5 coverage pilot, only after real reviewers finish
and disagreements are adjudicated into an exact verified export, run
`scripts/analyze_candidate_coverage.py` with the pinned pilot manifest, hidden
strata map, and verified pairs. It reports a sampling-weighted estimate of
positive dependency recall; it refuses provisional, incomplete, or inconsistent
labels. The 100-pair pilot is exploratory and is not a powered performance claim.
See the [annotation guide](docs/ANNOTATION_GUIDE.md) for the command and blinding
procedure.

For a paired manual-versus-assisted effort study, use the separate
`prepare_annotation_effort_study.py`, `run_annotation_effort_study.py`, and
`compare_annotation_effort.py` commands documented in the [annotation guide](docs/ANNOTATION_GUIDE.md).
The fixed input must contain provisional machine suggestions. Each pair is
reviewed in both conditions by different reviewers, with condition assignments
balanced across the two reviewers. The manual interface hides machine labels;
the assisted interface identifies them as provisional. The resulting comparison
is descriptive software output only: it does not verify labels, establish
annotation quality, or constitute a real human study until reviewers complete
it on an appropriate, rights-cleared sample. Study artifacts are refused under
the project's `data/` tree.

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

## 📊 Evaluation & Empirical Evidence

The checked-in [`data/evaluation/system_disjoint_report.json`](data/evaluation/system_disjoint_report.json)
is a historical local run with 10 ADB train pairs, 5 APA development pairs, and
5 TJA test pairs. It has no annotation provenance and uses MockLLM for LEREDD.
The five-pair test sample and four-of-six label coverage make its metrics
descriptive only. It does not reproduce a live GPT-4 experiment, and no
paper-reference score or statistical-significance claim is made here. See
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the recorded metrics and
limitations. New runs must save to a new path; do not overwrite the historical
report.

---

## ⚠️ Limitations & Open Challenges

1. **System-Disjoint Performance Gap**: Models trained or evaluated on random splits exhibit optimistic performance due to lexical overlap within the same system SRS. System-disjoint evaluation reveals significant domain adaptation challenges across distinct software specifications.
2. **Class Imbalance**: Requirement datasets naturally contain an overwhelmingly high proportion of `No_dependency` pairs (>80%). Precision and recall for rare classes like `Details` and `Conflicts` require targeted candidate sampling.
3. **Candidate Filtering Trade-off**: Aggressive top-$k$ candidate pair retrieval significantly reduces human annotation effort but introduces potential false-negative filtering risk for non-obvious requirement dependencies.

---

## 🛠️ Testing & Verification

Run the full automated test suite (including unit, integration, and provenance verification tests):

```bash
python -m pytest tests/
```

---

## 📜 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
