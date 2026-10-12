# REST API, CLI & SDK Technical Reference

This document provides complete reference documentation for the REST API, Command Line Interface (CLI), and Python SDK for **“Improving Automated Requirement Dependency Detection Using Large Language Models and Semi-Automated Dataset Generation”**.

---

## 1. FastAPI REST API Reference

The FastAPI server provides automated dependency detection and dataset management endpoints.

- **Base URL**: `http://localhost:8000/api/v1`
- **Interactive Swagger Documentation**: `http://localhost:8000/docs`

### 1.1 System Endpoints

#### `GET /api/v1/health`
Returns system health status, version, and active LLM provider.
```json
{
  "status": "ok",
  "version": "2.0.0",
  "provider": "mock"
}
```

#### `GET /api/v1/config`

Returns active non-secret configuration fields. The configured OpenAI key is never included; the response contains only `openai_api_key_configured` as a boolean.

---

### 1.2 Detection Endpoints

#### `POST /api/v1/detect`
Analyzes a single requirement pair and returns predicted dependency type, rationale, and confidence rating.

**Request Body**:
```json
{
  "req_a": "The BCS shall apply vehicle braking.",
  "req_b": "The BCS shall receive obstacle distance data from the PCS.",
  "system": "ADB",
  "examples": []
}
```

**Response (200 OK)**:
```json
{
  "pair_id": "PAIR-API-0001",
  "dependency_type": "Requires",
  "rationale": "Requirement A requires camera distance information from Requirement B to activate braking.",
  "confidence": 5,
  "raw_response": "**Dependency_Type: Requires**...",
  "prompt_tokens": 380,
  "completion_tokens": 42,
  "estimated_cost_usd": 0.0012
}
```

#### `POST /api/v1/predict/batch`
Performs batch dependency prediction for a list of requirement pairs.

---

### 1.3 Dataset Generation & Review Queue Endpoints

#### `GET /api/v1/queue`
Retrieves items in the human review queue.

#### `POST /api/v1/queue/{item_id}/approve`
Approves a candidate pair annotation with optional human edits and provenance logging.

---

## 2. Command Line Interface Reference (`leredd-cli`)

The Typer CLI offers full suite controls for data processing and model operations.

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `leredd-cli extract` | `--input <path> --system <name> --output <path>` | Extract requirement statements from SRS text. |
| `leredd-cli pairs` | `--input <path> --output <path>` | Generate exhaustive unlabeled candidate pairs (`dependency_type: null`, candidate-generator provenance). |
| `leredd-cli annotate` | `--input <path> --output <path>` | Collect an explicit human label; records the human annotation as VERIFIED. |
| `leredd-cli detect` | `--input <path> --examples <path> --output <path>` | Execute LEREDD automated dependency detection. |
| `leredd-cli evaluate` | `--predictions <path> --ground-truth <path>` | Compute metrics only with complete prediction coverage and labeled ground truth; provenance-bearing ground truth requires `VERIFIED` status. |
| `leredd-cli serve` | `--host <host> --port <port>` | Start the FastAPI REST API server. |

---

## 3. Python SDK & Core Classes

### `leredd.RequirementDependencyDetector`
Primary detection engine class.

```python
from leredd import RequirementDependencyDetector, LEREDDConfig, RequirementPair, Requirement

config = LEREDDConfig(llm_provider="mock", confidence_threshold=4)
detector = RequirementDependencyDetector(config=config)

pair = RequirementPair(
    id="PAIR-001",
    req_a=Requirement(id="REQ-1", text="System shall authenticate user.", system="ADB"),
    req_b=Requirement(id="REQ-2", text="System shall check password hash.", system="ADB")
)

prediction = detector.detect_dependency(pair)
print(prediction.dependency_type, prediction.confidence)
```

### `leredd.candidate_generator.CandidatePairGenerator`
Generates top-$k$ candidate pairs via hybrid retrieval filtering.

### `leredd.dataset_splitter.SystemDisjointSplitter`
Splits datasets into system-disjoint train, dev, and test partitions with zero data leakage.
