from __future__ import annotations

"""Disk-backed storage helper for persisted trained model artifacts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from starlette import status

from app.core.config import Settings
from app.core.errors import AppError
from app.utils.files import safe_join


@dataclass(frozen=True)
class ModelStore:
    settings: Settings

    def _model_path(self, model_id: str) -> Path:
        # Each trained model gets one joblib artifact named by its generated model ID.
        return safe_join(Path(self.settings.models_dir), f"{model_id}.joblib")

    def save_model(self, model_id: str, model_object: Any) -> Path:
        path = self._model_path(model_id)
        try:
            # joblib handles sklearn pipelines and XGBoost wrappers well for local persistence.
            joblib.dump(model_object, path)
        except Exception as e:
            raise AppError(
                "Failed to persist model artifact",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="model_save_failed",
                details={"error": str(e)},
            )
        return path

    def load_model(self, model_id: str) -> Any:
        path = self._model_path(model_id)
        if not path.exists():
            raise AppError(
                "Model artifact not found on server. Please retrain the model.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="model_artifact_missing",
                details={"model_id": model_id},
            )
        try:
            # Loading the artifact returns the exact fitted pipeline object saved after training.
            return joblib.load(path)
        except Exception as e:
            raise AppError(
                "Failed to load model artifact",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="model_load_failed",
                details={"error": str(e)},
            )
