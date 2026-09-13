# LEREDD REST API Reference

The LEREDD REST API is built with FastAPI and provides automated requirement dependency detection services.

## Base URL
Default local endpoint: `http://localhost:8000/api/v1`
Interactive Swagger UI: `http://localhost:8000/docs`

---

## Endpoints

### 1. Single Pair Dependency Detection
**`POST /api/v1/detect`**

Analyzes a single pair of natural language requirement statements and predicts the dependency type, rationale, and confidence score.

#### Request Body
```json
{
  "req_a": "The system shall include the Brake Control Subsystem (BCS).",
  "req_b": "The BCS shall accept camera scenery information provided by the PCS to brake the vehicle if an object is within the vehicle's path.",
  "system": "ADB",
  "srs_text": "Optional full SRS document text for RAG contextual retrieval...",
  "examples": []
}
```

#### Response (200 OK)
```json
{
  "pair_id": "PAIR-API-0001",
  "dependency_type": "Requires",
  "rationale": "Requirement A describes the inclusion of the Brake Control Subsystem (BCS), which is a prerequisite for Requirement B to execute braking functionality.",
  "confidence": 5,
  "raw_response": "**Dependency_Type: Requires**...",
  "prompt_tokens": 420,
  "completion_tokens": 48,
  "estimated_cost_usd": 0.0155
}
```

---

### 2. Batch Pair Dependency Detection
**`POST /api/v1/batch_detect`**

Batch detection across a list of requirement pairs.

#### Request Body
```json
{
  "pairs": [
    {
      "id": "PAIR-001",
      "req_a": { "id": "REQ-1", "text": "The system shall authenticate updates.", "system": "ADB" },
      "req_b": { "id": "REQ-2", "text": "The system shall verify the source of the update.", "system": "ADB" }
    }
  ],
  "srs_text": null,
  "examples": []
}
```

#### Response (200 OK)
Returns a list of `Prediction` objects.

---

### 3. Requirement Extraction
**`POST /api/v1/extract`**

Extracts natural language requirement statements containing keywords (`shall`, `should`, `must`) from raw SRS text.

---

### 4. System Health Check
**`GET /api/v1/health`**

Returns service health status and version.

---

### 5. Active Configuration
**`GET /api/v1/config`**

Returns active `LEREDDConfig` parameters.
