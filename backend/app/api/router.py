from __future__ import annotations

"""Central place where all API route groups are mounted onto one router."""

from fastapi import APIRouter

from app.api.routes import admin, datasets, models, root

api_router = APIRouter()
# Health/basic info endpoints live at the API root.
api_router.include_router(root.router, tags=["root"])
# Dataset routes own upload, preprocess, and dataset-summary operations.
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
# Model routes own training, schema lookup, summaries, and prediction.
api_router.include_router(models.router, prefix="/models", tags=["models"])
# Admin routes are intentionally separate because they mutate runtime state.
api_router.include_router(admin.router, prefix="/admin", tags=["admin"])
