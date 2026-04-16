from __future__ import annotations

"""Schemas for dataset-level and training-level summary/insight endpoints."""

from typing import List, Optional

from pydantic import BaseModel, Field

from app.schemas.common import FeatureImportance


class ClassBalance(BaseModel):
    # One item per target label, showing frequency and percentage share.
    label: str
    count: int
    pct: float = Field(ge=0.0, le=1.0)


class ColumnMissing(BaseModel):
    # Missing-value summary for one column.
    column: str
    null_count: int
    null_pct: float = Field(ge=0.0, le=1.0)


class DatasetSummaryStats(BaseModel):
    # Structured numeric/statistical facts used by both UI cards and the LLM prompts.
    rows: int
    cols: int
    target_column: str
    missing_cells: int
    missing_pct: float = Field(ge=0.0, le=1.0)
    class_balance: List[ClassBalance] = Field(default_factory=list)
    top_missing: List[ColumnMissing] = Field(default_factory=list)


class DatasetSummaryResponse(BaseModel):
    # Full dataset summary payload returned after preprocess/training.
    status: str = "success"
    upload_id: str
    summary: str
    explanation: str
    patterns: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    stats: DatasetSummaryStats
    llm_used: bool = False
    llm_model: Optional[str] = None


class TrainingSummaryResponse(BaseModel):
    # Readable training summary payload built from metrics and top features.
    status: str = "success"
    model_id: str
    summary: str
    metrics_summary: str
    top_features: List[FeatureImportance] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    llm_used: bool = False
    llm_model: Optional[str] = None
