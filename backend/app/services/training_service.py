from __future__ import annotations

"""Service layer that turns a preprocessed dataset into a persisted trained model."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from starlette import status

from app.core.config import Settings
from app.core.logging import get_logger
from app.core.errors import AppError
from app.ml.metrics import format_metrics
from app.ml.trainer import train_xgb_pipeline
from app.schemas.common import ConfusionMatrix, FeatureImportance
from app.schemas.training import TrainRequest, TrainResponse, TrainingMetrics
from app.services.insights_service import InsightsService
from app.storage.dataset_store import DatasetStore
from app.storage.metadata_store import MetadataStore
from app.storage.model_store import ModelStore
from app.utils.ids import new_id

logger = get_logger(__name__)

@dataclass(frozen=True)
class TrainingService:
    settings: Settings
    dataset_store: DatasetStore
    model_store: ModelStore
    metadata_store: MetadataStore

    def _train_model(self, req: TrainRequest, progress_cb: Optional[Callable[[int, str], None]] = None) -> TrainResponse:
        def _emit(pct: int, msg: str) -> None:
            if progress_cb is not None:
                progress_cb(pct, msg)

        # Training only works on a dataset that already passed the preprocess step.
        _emit(5, "Validating preprocessing metadata")
        prep = self.metadata_store.read_preprocess_metadata(req.upload_id)
        if prep is None:
            raise AppError(
                "Dataset not preprocessed. Please run preprocess step first.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="preprocess_required",
            )

        target_column = req.target_column
        if target_column != prep.get("target_column"):
            raise AppError(
                "Target column mismatch. Please preprocess again with the selected target column.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="target_mismatch",
                details={"preprocessed_target": prep.get("target_column"), "request_target": target_column},
            )

        feature_columns: List[str] = list(prep.get("feature_columns") or [])
        if not feature_columns:
            raise AppError(
                "No feature columns available. Please preprocess the dataset again.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="no_features",
            )

        processed_path_str = prep.get("processed_path")
        if not processed_path_str:
            raise AppError(
                "Processed dataset path missing. Please preprocess again.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="processed_path_missing",
            )

        processed_path = self.dataset_store.path_from_string(processed_path_str)
        if not processed_path.exists():
            raise AppError(
                "Processed dataset file missing on server. Please preprocess again.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="processed_file_missing",
            )

        _emit(10, "Loading processed dataset")
        df = self.dataset_store.load_processed_dataframe(processed_path)

        # Basic sanity checks protect against mismatched metadata or missing files on disk.
        _emit(15, "Running dataset checks")
        missing_cols = [c for c in ([target_column] + feature_columns) if c not in df.columns]
        if missing_cols:
            raise AppError(
                "Processed dataset is missing expected columns.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="processed_columns_missing",
                details={"missing": missing_cols},
            )

        # Train one persisted pipeline that bundles preprocessing + model so inference matches training exactly.
        model_id = new_id(prefix="mdl")
        model_name = (req.model_name or "xgboost").lower().strip()
        if model_name != "xgboost":
            raise AppError(
                "Only 'xgboost' model_name is supported in this backend version.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="unsupported_model",
                details={"supported": ["xgboost"], "received": model_name},
            )

        X = df[feature_columns].copy()
        y = df[target_column].copy()

        try:
            _emit(20, "Preparing training data")
            _emit(25, "Starting training")
            # Delegate the actual ML work to the trainer helper so this service stays orchestration-focused.
            trained = train_xgb_pipeline(
                X=X,
                y=y,
                random_seed=self.settings.random_seed,
                test_size=self.settings.test_size,
                progress_cb=progress_cb,
                progress_range=(30, 90),
            )
        except AppError:
            raise
        except Exception as e:
            logger.exception(
                "training_failed",
                extra={
                    "upload_id": req.upload_id,
                    "model_name": model_name,
                    "target_column": target_column,
                },
            )
            raise AppError(
                "Training failed due to an internal error.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="training_failed",
                details={"error": str(e)},
            )

        # Persist the fitted sklearn pipeline so later prediction requests can reuse it directly.
        _emit(92, "Saving model artifact")
        artifact_path = self.model_store.save_model(model_id=model_id, model_object=trained.pipeline)

        # Convert raw floats into the API response schema used by the frontend metrics cards.
        _emit(96, "Building metrics")
        metrics_dict = format_metrics(trained.metrics)
        metrics = TrainingMetrics(**metrics_dict)

        cm = ConfusionMatrix(labels=["0", "1"], matrix=trained.confusion_matrix)

        feature_importance = [
            FeatureImportance(feature=f, importance=float(w))
            for f, w in trained.feature_importance
        ]

        # Persist metadata needed for /schema, /predict, and summary endpoints across restarts.
        _emit(98, "Writing model metadata")
        model_metadata: Dict[str, Any] = {
            "model_id": model_id,
            "model_name": model_name,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "artifact_path": str(artifact_path),
            "training": {
                "upload_id": req.upload_id,
                "test_size": float(self.settings.test_size),
                "random_seed": int(self.settings.random_seed),
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            },
            "schema": trained.schema,  # JSON-serializable
            "metrics": metrics_dict,
            "feature_importance": [fi.model_dump() for fi in feature_importance],
        }
        self.metadata_store.write_model_metadata(model_id, model_metadata)

        notes: List[str] = []
        notes.append("Model trained using a consistent preprocessing+model pipeline artifact.")

        _emit(99, "Generating dataset summary")
        try:
            # Precompute the dataset summary once so later UI screens can reload it instantly.
            insights = InsightsService(
                settings=self.settings,
                dataset_store=self.dataset_store,
                metadata_store=self.metadata_store,
                model_store=self.model_store,
            )
            dataset_summary = insights.build_dataset_summary(req.upload_id)
            prep = self.metadata_store.read_preprocess_metadata(req.upload_id) or {}
            prep["dataset_summary"] = dataset_summary.model_dump()
            self.metadata_store.write_preprocess_metadata(req.upload_id, prep)
        except Exception:
            logger.warning("dataset_summary_failed", extra={"upload_id": req.upload_id})

        _emit(100, "Training complete")
        return TrainResponse(
            model_id=model_id,
            model_name=model_name,
            target_column=target_column,
            feature_columns=feature_columns,
            metrics=metrics,
            confusion_matrix=cm,
            feature_importance=feature_importance,
            notes=notes,
        )

    async def train_model(self, req: TrainRequest) -> TrainResponse:
        # Async wrapper exists for API symmetry even though the core implementation is synchronous.
        return self._train_model(req)

    def train_model_with_progress(self, req: TrainRequest, progress_cb: Callable[[int, str], None]) -> TrainResponse:
        # Streaming endpoint calls this version so the route can emit progress updates.
        return self._train_model(req, progress_cb)
