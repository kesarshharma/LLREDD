"""FastAPI Route Handlers for requirement dependency detection."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Body
from pydantic import BaseModel, Field
from leredd.config import get_config
from leredd.data_models import Requirement, RequirementPair, Prediction, DependencyType
from leredd.detector import LEREDDDetector
from leredd.utils import extract_requirements_from_text

router = APIRouter()
_detector_instance: Optional[LEREDDDetector] = None


def get_detector() -> LEREDDDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LEREDDDetector()
    return _detector_instance


class SingleDetectRequest(BaseModel):
    req_a: str = Field(..., description="Requirement A text statement")
    req_b: str = Field(..., description="Requirement B text statement")
    system: str = Field(default="ADB", description="System identifier")
    srs_text: Optional[str] = Field(default=None, description="Optional SRS text document for RAG context")
    examples: Optional[List[RequirementPair]] = Field(default=None, description="Optional list of in-context example pairs")


class BatchDetectRequest(BaseModel):
    pairs: List[RequirementPair] = Field(..., description="List of RequirementPair objects")
    srs_text: Optional[str] = Field(default=None, description="Optional SRS text document for RAG context")
    examples: Optional[List[RequirementPair]] = Field(default=None, description="Optional list of in-context example pairs")


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Raw SRS text content")
    system_name: str = Field(default="GenericSystem", description="System identifier")


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "LEREDD API", "version": "1.0.0"}


@router.get("/config")
def get_system_config():
    """Return active system configuration settings."""
    cfg = get_config()
    return cfg.model_dump(mode="json")


@router.post("/detect", response_model=Prediction)
def detect_single(payload: SingleDetectRequest):
    """Detect dependency type, rationale, and confidence for a single requirement pair."""
    try:
        detector = get_detector()
        req_a_obj = Requirement(id="REQ-A", text=payload.req_a, system=payload.system)
        req_b_obj = Requirement(id="REQ-B", text=payload.req_b, system=payload.system)
        target_pair = RequirementPair(id="PAIR-API-0001", req_a=req_a_obj, req_b=req_b_obj)

        prediction = detector.detect_pair(
            target_pair=target_pair,
            annotated_pool=payload.examples or [],
            srs_text=payload.srs_text,
            domain="automotive software engineering"
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/batch_detect", response_model=List[Prediction])
def detect_batch(payload: BatchDetectRequest):
    """Batch requirement dependency detection for a list of pairs."""
    try:
        detector = get_detector()
        predictions = detector.detect_batch(
            pairs=payload.pairs,
            annotated_pool=payload.examples or [],
            srs_text=payload.srs_text
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@router.post("/extract", response_model=List[Requirement])
def extract_reqs(payload: ExtractRequest):
    """Extract natural language requirements from SRS text using regex."""
    try:
        reqs = extract_requirements_from_text(payload.text, system_name=payload.system_name)
        return reqs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
