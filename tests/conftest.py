"""Pytest fixtures for unit and integration testing."""

import pytest
from leredd.config import LEREDDConfig
from leredd.data_models import Requirement, RequirementPair, DependencyType


@pytest.fixture
def mock_config():
    return LEREDDConfig(
        embedding_model="all-MiniLM-L6-v2",
        similarity_metric="euclidean",
        aggregation="average",
        k_examples=2,
        use_rag=False,
        llm_provider="mock",
        confidence_threshold=4
    )


@pytest.fixture
def sample_req_a():
    return Requirement(
        id="REQ-ADB-001",
        text="The system shall include the Brake Control Subsystem (BCS).",
        system="ADB"
    )


@pytest.fixture
def sample_req_b():
    return Requirement(
        id="REQ-ADB-002",
        text="The BCS shall accept camera scenery information provided by the PCS to brake the vehicle if an object is within the vehicle's path.",
        system="ADB"
    )


@pytest.fixture
def sample_pair(sample_req_a, sample_req_b):
    return RequirementPair(
        id="PAIR-TEST-001",
        req_a=sample_req_a,
        req_b=sample_req_b,
        dependency_type=DependencyType.REQUIRES
    )


@pytest.fixture
def sample_annotated_pool(sample_pair):
    pair2 = RequirementPair(
        id="PAIR-TEST-002",
        req_a=Requirement(id="REQ-A2", text="System shall authenticate updates.", system="ADB"),
        req_b=Requirement(id="REQ-B2", text="System shall verify update source.", system="ADB"),
        dependency_type=DependencyType.IS_SIMILAR
    )
    pair3 = RequirementPair(
        id="PAIR-TEST-003",
        req_a=Requirement(id="REQ-A3", text="System shall not irritate driver.", system="ADB"),
        req_b=Requirement(id="REQ-B3", text="System shall alert driver immediately.", system="ADB"),
        dependency_type=DependencyType.CONFLICTS
    )
    return [sample_pair, pair2, pair3]
