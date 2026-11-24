"""FastAPI Route Handlers for requirement dependency detection."""

from typing import Any, List, Literal, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from leredd.config import get_config
from leredd.data_models import Requirement, RequirementPair, Prediction
from leredd.detector import LEREDDDetector
from leredd.utils import apply_requirement_metadata, extract_requirements_from_text

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
    req_a_component: Optional[str] = Field(default=None, description="Optional component metadata for Requirement A")
    req_a_subsystem: Optional[str] = Field(default=None, description="Optional subsystem metadata for Requirement A")
    req_b_component: Optional[str] = Field(default=None, description="Optional component metadata for Requirement B")
    req_b_subsystem: Optional[str] = Field(default=None, description="Optional subsystem metadata for Requirement B")
    srs_text: Optional[str] = Field(default=None, description="Optional SRS text document for RAG context")
    use_rag: Optional[bool] = Field(
        default=None,
        description="Enable RAG for this request; if omitted, use the server configuration.",
    )
    examples: Optional[List[RequirementPair]] = Field(default=None, description="Optional list of in-context example pairs")
    retrieval_mode: Optional[Literal["dense", "bm25", "hybrid"]] = Field(
        default=None,
        description="Optional per-request example retrieval mode",
    )
    stability_runs: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Optional number of fresh same-prompt inferences for stability screening (1-5)",
    )


class BatchDetectRequest(BaseModel):
    pairs: List[RequirementPair] = Field(..., description="List of RequirementPair objects")
    srs_text: Optional[str] = Field(default=None, description="Optional SRS text document for RAG context")
    use_rag: Optional[bool] = Field(
        default=None,
        description="Enable RAG for this request; if omitted, use the server configuration.",
    )
    examples: Optional[List[RequirementPair]] = Field(default=None, description="Optional list of in-context example pairs")
    retrieval_mode: Optional[Literal["dense", "bm25", "hybrid"]] = Field(
        default=None,
        description="Optional per-request example retrieval mode",
    )
    stability_runs: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Optional number of fresh same-prompt inferences for stability screening (1-5)",
    )


class ExtractRequest(BaseModel):
    text: str = Field(..., description="Raw SRS text content")
    system_name: str = Field(default="GenericSystem", description="System identifier")
    source_name: Optional[str] = Field(default=None, description="Optional source document name")
    metadata_by_id: Optional[dict[str, dict[str, Any]]] = Field(
        default=None,
        description="Optional explicit component/subsystem objects keyed by extracted requirement ID",
    )


@router.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "LEREDD API", "version": "1.0.0"}


@router.get("/config")
def get_system_config():
    """Return active system configuration settings with all secrets redacted."""
    cfg = get_config()
    public_config = cfg.model_dump(
        mode="json", exclude={"openai_api_key", "gemini_api_key"}
    )
    public_config["openai_api_key_configured"] = bool(cfg.openai_api_key)
    public_config["gemini_api_key_configured"] = bool(cfg.gemini_api_key)
    return public_config


@router.post("/detect", response_model=Prediction)
def detect_single(payload: SingleDetectRequest):
    """Detect dependency type, rationale, and confidence for a single requirement pair."""
    try:
        detector = get_detector()
        req_a_obj = Requirement(
            id="REQ-A", text=payload.req_a, system=payload.system,
            component=payload.req_a_component, subsystem=payload.req_a_subsystem,
        )
        req_b_obj = Requirement(
            id="REQ-B", text=payload.req_b, system=payload.system,
            component=payload.req_b_component, subsystem=payload.req_b_subsystem,
        )
        target_pair = RequirementPair(id="PAIR-API-0001", req_a=req_a_obj, req_b=req_b_obj)

        prediction = detector.detect_pair(
            target_pair=target_pair,
            annotated_pool=payload.examples or [],
            srs_text=payload.srs_text,
            domain="automotive software engineering",
            retrieval_mode=payload.retrieval_mode,
            stability_runs=payload.stability_runs,
            use_rag=payload.use_rag,
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
            srs_text=payload.srs_text,
            retrieval_mode=payload.retrieval_mode,
            stability_runs=payload.stability_runs,
            use_rag=payload.use_rag,
        )
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@router.post("/extract", response_model=List[Requirement])
def extract_reqs(payload: ExtractRequest):
    """Extract natural language requirements from SRS text using regex."""
    try:
        reqs = extract_requirements_from_text(
            payload.text,
            system_name=payload.system_name,
            source_name=payload.source_name,
        )
        if payload.metadata_by_id is not None:
            reqs = apply_requirement_metadata(reqs, payload.metadata_by_id)
        return reqs
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid extraction metadata: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
