from __future__ import annotations

"""Shared response/value schemas reused across multiple API modules."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Risk levels are business-friendly buckets derived from raw churn probability.
RiskLevel = Literal["Low", "Medium", "High"]


class MessageResponse(BaseModel):
    # Simple one-field response used for root/admin-style endpoints.
    message: str


class HealthResponse(BaseModel):
    # Health response tells clients whether the backend is alive and which config it is running with.
    status: Literal["ok"] = "ok"
    environment: str
    version: str


class APIError(BaseModel):
    # Standard error payload returned by custom exception handlers.
    message: str
    code: str
    trace_id: str
    details: Optional[Dict[str, Any]] = None


class ColumnInfo(BaseModel):
    # Column metadata helps the frontend show previews and target/feature selection options.
    name: str
    dtype: str
    sample_values: List[Any] = Field(default_factory=list)
    null_count: int = 0
    unique_count: int = 0


class NumericMetric(BaseModel):
    # Carry both raw numeric metric value and preformatted display text.
    value: float
    display: str


class ConfusionMatrix(BaseModel):
    # Confusion matrix is sent to the frontend exactly as rows/columns for binary classes.
    labels: List[str] = Field(default_factory=lambda: ["0", "1"])
    matrix: List[List[int]] = Field(default_factory=list)


class FeatureImportance(BaseModel):
    # Aggregated feature importance is used in training summaries and model detail screens.
    feature: str
    importance: float
