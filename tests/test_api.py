"""Tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["health"] == "/api/v1/health"


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_config():
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert "embedding_model" in data


def test_api_single_detect():
    payload = {
        "req_a": "The system shall include the Brake Control Subsystem (BCS).",
        "req_b": "The BCS shall accept camera input.",
        "system": "ADB"
    }
    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "dependency_type" in data
    assert "rationale" in data
    assert "confidence" in data


def test_api_batch_detect():
    payload = {
        "pairs": [
            {
                "id": "PAIR-API-1",
                "req_a": {"id": "REQ-1", "text": "The system shall control speed.", "system": "ADB"},
                "req_b": {"id": "REQ-2", "text": "The BCS subsystem shall brake.", "system": "ADB"},
                "dependency_type": None
            }
        ]
    }
    response = client.post("/api/v1/batch_detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["pair_id"] == "PAIR-API-1"


def test_api_extract():
    payload = {
        "text": "[REQ-ADB-001] The system shall accelerate safely.\n[REQ-ADB-002] The system shall decelerate.",
        "system_name": "ADB"
    }
    response = client.post("/api/v1/extract", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["system"] == "ADB"


def test_api_invalid_payload_422():
    # Missing required field req_b
    payload = {
        "req_a": "The system shall control speed."
    }
    response = client.post("/api/v1/detect", json=payload)
    assert response.status_code == 422
