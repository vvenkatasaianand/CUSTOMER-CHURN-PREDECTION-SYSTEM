from __future__ import annotations

"""Schemas for single-record prediction requests, explanations, and recommended actions."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import RiskLevel


class PredictRequest(BaseModel):
    # input_data is a generic dict because fields vary by whatever dataset was used for training.
    model_id: str
    input_data: Dict[str, Any] = Field(..., description="Feature inputs keyed by column name")


class PredictionFactor(BaseModel):
    # These fields describe why the model leaned toward or away from churn for one prediction.
    feature: str
    direction: Optional[str] = Field(
        default=None,
        description="Optional: 'increases_risk'/'decreases_risk' etc. Not computed in Stage-1 backend.",
    )
    contribution: Optional[float] = Field(
        default=None,
        description="Signed contribution toward class=1 prediction (if available).",
    )
    reasoning: Optional[str] = None


class PredictionExplanation(BaseModel):
    """
    Placeholder structure for explanation output.
    No LLM integration yet; we return a deterministic summary.
    """
    summary: str
    key_factors: List[PredictionFactor] = Field(default_factory=list)
    confidence_note: Optional[str] = None
    llm_used: bool = False
    llm_model: Optional[str] = None


class RecommendedAction(BaseModel):
    # Action objects turn the prediction into something business users can act on.
    action: str
    reason: str
    priority: int = Field(ge=1, le=5)
    expected_impact: Optional[str] = None


class PredictResponse(BaseModel):
    # Final prediction payload sent to the results screen.
    status: str = "success"
    model_id: str
    prediction: int
    probability: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    explanation: PredictionExplanation
    recommended_actions: List[RecommendedAction] = Field(default_factory=list)
