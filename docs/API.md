# REST API, CLI, and Python Interfaces

This reference describes the interfaces implemented in this checkout. The
FastAPI service exposes prediction, extraction, health, and configuration
routes. Dataset generation and human review are currently performed through
the CLI, Python APIs, or Streamlit; there are no HTTP review-queue routes.

## FastAPI

Start the local service with `leredd-cli serve` or
`python -m api.main --host 127.0.0.1 --port 8000`. Interactive OpenAPI docs
are served at `http://127.0.0.1:8000/docs`.

| Method and path | Purpose |
|---|---|
| `GET /` | API welcome response and links |
| `GET /api/v1/health` | Service health and version |
| `GET /api/v1/config` | Active configuration with API key redacted |
| `POST /api/v1/extract` | Extract `Requirement` records from text |
| `POST /api/v1/detect` | Predict one requirement pair |
| `POST /api/v1/batch_detect` | Predict a list of requirement pairs |

The single-detection request accepts `req_a` and `req_b` strings, an optional
`system`, optional `srs_text`, optional verified example records, and an
optional per-request `retrieval_mode` (`dense`, `bm25`, or `hybrid`). It also
accepts `stability_runs` from 1 to 5 (default 1). Values above one repeat the
same prompt through fresh provider calls, bypassing the built-in response cache.
Any label or self-reported confidence variation marks the result inconsistent
and recommends human review; the first run's label remains the reported class.
Set `use_rag` to `true` to retrieve relevant chunks from the supplied `srs_text`;
if omitted, the server's RAG configuration is used. Set it to `false` to disable
RAG for a request even when the server configuration enables it. Batch detection
accepts the same `srs_text` and `use_rag` fields. Repeated OpenAI runs make
multiple external calls. Example:

```json
{
  "req_a": "The brake controller shall apply braking.",
  "req_b": "The brake controller shall use wheel speed data.",
  "system": "ADB",
  "examples": [],
  "retrieval_mode": "hybrid",
  "srs_text": "The brake controller shall process wheel speed data...",
  "use_rag": true,
  "stability_runs": 3
}
```

Batch detection accepts the same RAG and `stability_runs` options, for example
`{"pairs": [...], "examples": [], "srs_text": null,
"retrieval_mode": "bm25", "stability_runs": 2}`;
each pair follows the `RequirementPair` schema. Extraction accepts
`{"text": "...", "system_name": "ADB"}` with optional `metadata_by_id`, an
object keyed by exact extracted requirement IDs. Each value may contain only
`component` and/or `subsystem` strings. Unknown IDs or malformed metadata return
HTTP 422. For single detection, optional `req_a_component`, `req_a_subsystem`,
`req_b_component`, and `req_b_subsystem` fields populate the target pair. These
values are explicit user-supplied metadata and are never inferred from text.
Pydantic request validation returns HTTP 422 for invalid payloads. Detection
errors are reported as HTTP 500.
Detection responses include `validation_status`: `passed_threshold`,
`requires_review`, `threshold_disabled`, or `not_evaluated` (for predictions
created by another classifier). A low-confidence prediction keeps its predicted
dependency class and signals that human review is recommended. This status does
not make the label verified ground truth, and the model's Likert confidence is
not a calibrated probability. Responses also report `stability_status`,
`stability_run_count`, `stability_labels`, `stability_confidences`, and
`stability_reasons`; stability screening is separate from annotation
verification.

The config endpoint never returns `openai_api_key`; it includes only the
`openai_api_key_configured` boolean. The detection routes use the configured
provider. OpenAI sends requirement text to an external service and requires a
configured key and explicit authorization in the UI. No external provider is
needed for the default MockLLM workflow.

## CLI

Run commands with the installed `leredd-cli` entry point (or use the Python
module invocation shown in the README). Available commands are:

| Command | Purpose |
|---|---|
| `extract` | Extract requirements from a text, Markdown, PDF, or DOCX input; PDF/DOCX support needs the optional documents extra |
| `pairs` | Generate exhaustive unlabeled pairs |
| `candidates` | Generate exhaustive, BM25, dense, or hybrid candidates |
| `label` | Produce provisional heuristic/LLM/hybrid labels; opt-in `multi_llm` mode requires at least two distinct model IDs |
| `queue-create`, `queue-list`, `queue-decide` | Create, inspect, and record explicit review decisions |
| `annotate` | Interactively assign labels to pairs |
| `split` | Write train/dev/test partitions and leakage metadata |
| `detect` | Predict labels for input pairs; `--stability-runs 1..5` optionally checks repeated inference consistency (default 1) |
| `evaluate` | Score predictions against eligible ground truth |
| `train-bert` | Fit and persist the supervised BERT-feature classifier |
| `serve` | Start the FastAPI service |

Machine-labeled records remain provisional. `queue-decide --reviewer IDENTIFIER`
and `annotate --annotator IDENTIFIER` require an explicit, nonblank,
non-placeholder operator identity before recording human-supplied decisions as
verified. The application records but does not authenticate that identity; do
not use synthetic test reviewer identities as real annotation evidence. Evaluation rejects provenance-bearing
ground truth unless status is `verified`, source is human, reviewer or annotator
identity is present, and `accepted_label` matches the scored label. Prediction
coverage is required for every eligible ground-truth pair.

## Python API

The package exports schema and workflow types such as `Requirement`,
`RequirementPair`, `Prediction`, `LEREDDConfig`, `CandidateGenerator`,
`ProvisionalLabeler`, `ReviewQueueManager`, `DatasetSplitter`,
`BM25Retriever`, and `HybridRetriever`. The detection engine is
`leredd.detector.LEREDDDetector` (it is not exported from `leredd`):

```python
from leredd.config import LEREDDConfig
from leredd.data_models import Requirement, RequirementPair
from leredd.detector import LEREDDDetector
from leredd.llm import MockLLMClient

config = LEREDDConfig(llm_provider="mock", k_examples=0)
detector = LEREDDDetector(config=config, llm=MockLLMClient())
pair = RequirementPair(
    id="PAIR-001",
    req_a=Requirement(id="REQ-1", text="The system shall authenticate users."),
    req_b=Requirement(id="REQ-2", text="The system shall verify credentials."),
)
prediction = detector.detect_pair(pair)
print(prediction.dependency_type, prediction.confidence)
```

For repeatable offline calls, inject a deterministic embedder where the
workflow uses dense retrieval and use `MockLLMClient`. Default dense model
initialization may need locally available SentenceTransformer weights or
network access to retrieve them; real OpenAI/Ollama calls need their respective
services and configuration.
