from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.common import HealthResponse, MessageResponse

router = APIRouter()


@router.get("/", response_model=MessageResponse)
def root() -> MessageResponse:
    return MessageResponse(message="CUSTOMER CHURN PREDICTION SYSTEM - Backend is running")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.environment, version=settings.app_version)
