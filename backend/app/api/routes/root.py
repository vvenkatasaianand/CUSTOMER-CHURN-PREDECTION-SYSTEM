from __future__ import annotations

"""Tiny health/info endpoints used to verify the backend is up and configured."""

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas.common import HealthResponse, MessageResponse

router = APIRouter()


@router.get("/", response_model=MessageResponse)
def root() -> MessageResponse:
    # Human-readable landing response for quick manual checks in browser/Postman.
    return MessageResponse(message="CUSTOMER CHURN PREDICTION SYSTEM - Backend is running")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Structured health response is better for frontend checks and monitoring scripts.
    settings = get_settings()
    return HealthResponse(status="ok", environment=settings.environment, version=settings.app_version)
