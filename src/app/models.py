from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Evidence(BaseModel):
    evidence_id: str
    source: str
    quote: str
    confidence: float = Field(ge=0.0, le=1.0)


class FixtureRecord(BaseModel):
    record_id: str
    scenario: str
    source_type: str
    timestamp: str
    metric: float
    severity: int = Field(ge=1, le=5)
    expected_status: Literal["pass", "fail", "escalate"]
    evidence: list[Evidence]
    tags: list[str]
    notes: str

    @field_validator("evidence")
    @classmethod
    def require_evidence(cls, value: list[Evidence]) -> list[Evidence]:
        if not value:
            raise ValueError("records must include at least one evidence item")
        return value

    @field_validator("tags")
    @classmethod
    def require_tags(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("records must include at least one tag")
        return value


class ClusterSummary(BaseModel):
    scenario: str
    count: int
    failures: int
    escalations: int
    average_severity: float
    top_evidence_id: str
    recommended_action: str
