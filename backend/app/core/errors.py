from __future__ import annotations

"""Shared application error type plus FastAPI exception handlers."""

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette import status

from app.core.logging import get_logger
from app.schemas.common import APIError

logger = get_logger(__name__)


class AppError(Exception):
    """
    Controlled application error that should map to a clean API response.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        code: str = "bad_request",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


def _trace_id() -> str:
    # Trace IDs make it easier to match a client-facing error with a server log entry.
    return str(uuid.uuid4())


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        # Controlled errors preserve the original message/code/details for the client.
        tid = _trace_id()
        logger.warning(
            "app_error",
            extra={
                "trace_id": tid,
                "path": request.url.path,
                "method": request.method,
                "code": exc.code,
                "details": exc.details,
            },
        )
        payload = APIError(
            message=exc.message,
            code=exc.code,
            trace_id=tid,
            details=exc.details or None,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        tid = _trace_id()
        # Don't echo full body; just provide structured validation errors
        logger.info(
            "validation_error",
            extra={
                "trace_id": tid,
                "path": request.url.path,
                "method": request.method,
                "errors": exc.errors(),
            },
        )
        payload = APIError(
            message="Validation failed",
            code="validation_error",
            trace_id=tid,
            details={"errors": exc.errors()},
        )
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        # Unexpected exceptions are hidden behind a generic 500 to avoid leaking internals.
        tid = _trace_id()
        logger.exception(
            "unhandled_error",
            extra={
                "trace_id": tid,
                "path": request.url.path,
                "method": request.method,
            },
        )
        payload = APIError(
            message="Internal server error",
            code="internal_error",
            trace_id=tid,
        )
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=payload.model_dump())
