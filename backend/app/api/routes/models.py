from __future__ import annotations

"""Model-facing HTTP routes: training, schema lookup, summaries, and prediction."""

import asyncio
import json
import threading
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.services.prediction_service import PredictionService
from app.services.insights_service import InsightsService
from app.services.schema_service import SchemaService
from app.services.training_service import TrainingService
from app.storage.dataset_store import DatasetStore
from app.storage.metadata_store import MetadataStore
from app.storage.model_store import ModelStore
from app.schemas.datasets import SchemaResponse
from app.schemas.prediction import PredictRequest, PredictResponse
from app.schemas.insights import TrainingSummaryResponse
from app.schemas.training import TrainRequest, TrainResponse

router = APIRouter()
logger = get_logger(__name__)


def get_training_service(settings: Settings = Depends(get_settings)) -> TrainingService:
    # Training needs raw/processed dataset access, model artifact storage, and metadata storage.
    dataset_store = DatasetStore(settings=settings)
    model_store = ModelStore(settings=settings)
    metadata_store = MetadataStore(settings=settings)
    return TrainingService(settings=settings, dataset_store=dataset_store, model_store=model_store, metadata_store=metadata_store)


def get_prediction_service(settings: Settings = Depends(get_settings)) -> PredictionService:
    # Prediction loads the persisted model artifact and the metadata saved during training.
    dataset_store = DatasetStore(settings=settings)
    model_store = ModelStore(settings=settings)
    metadata_store = MetadataStore(settings=settings)
    return PredictionService(settings=settings, dataset_store=dataset_store, model_store=model_store, metadata_store=metadata_store)


def get_schema_service(settings: Settings = Depends(get_settings)) -> SchemaService:
    # Schema lookup is metadata-only; no need to touch the model artifact itself here.
    metadata_store = MetadataStore(settings=settings)
    return SchemaService(settings=settings, metadata_store=metadata_store)


def get_insights_service(settings: Settings = Depends(get_settings)) -> InsightsService:
    # Training summaries read saved metrics/features and may also use dataset metadata.
    dataset_store = DatasetStore(settings=settings)
    model_store = ModelStore(settings=settings)
    metadata_store = MetadataStore(settings=settings)
    return InsightsService(
        settings=settings,
        dataset_store=dataset_store,
        model_store=model_store,
        metadata_store=metadata_store,
    )


@router.post("/train", response_model=TrainResponse)
async def train_model(
    request: TrainRequest,
    svc: TrainingService = Depends(get_training_service),
) -> TrainResponse:
    """
    Train a model from a preprocessed dataset.

    Legacy compatibility:
    - Frontend expects POST /train with:
      { upload_id, target_column, excluded_columns, model_name? }
    - Returns training metrics, confusion matrix, feature importance, and model_id.
    """
    return await svc.train_model(request)


def _sse_event(event: str, data: Dict[str, Any]) -> str:
    # Format events exactly as Server-Sent Events expects: "event" line + "data" line + blank line.
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/train/stream")
async def train_model_stream(
    request: TrainRequest,
    svc: TrainingService = Depends(get_training_service),
):
    """
    Stream training progress via Server-Sent Events (SSE).
    Events:
      - progress: { pct, message }
      - complete: TrainResponse payload
      - error: { message, code?, details? }
    """
    # Use an asyncio queue as the bridge between the worker thread and the async response stream.
    queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit_progress(pct: int, message: str) -> None:
        # loop.call_soon_threadsafe lets the background thread safely push into the async queue.
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"pct": pct, "message": message}))

    def run_training() -> None:
        try:
            result = svc.train_model_with_progress(request, emit_progress)
        except AppError as exc:
            logger.warning(
                "training_stream_app_error",
                extra={"code": exc.code, "details": exc.details},
            )
            payload = {"message": exc.message, "code": exc.code, "details": exc.details}
            loop.call_soon_threadsafe(queue.put_nowait, ("error", payload))
        except Exception as exc:
            logger.exception("training_stream_failed")
            payload = {"message": str(exc) or "Training failed"}
            loop.call_soon_threadsafe(queue.put_nowait, ("error", payload))
        else:
            loop.call_soon_threadsafe(queue.put_nowait, ("complete", result.model_dump()))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", {}))

    # Run CPU-heavy training outside the event loop so streaming stays responsive.
    threading.Thread(target=run_training, daemon=True).start()

    async def event_stream():
        while True:
            event, payload = await queue.get()
            if event == "done":
                break
            yield _sse_event(event, payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/schema/{model_id}", response_model=SchemaResponse)
async def model_schema(
    model_id: str,
    svc: SchemaService = Depends(get_schema_service),
) -> SchemaResponse:
    """
    Return the model schema used for prediction-time form generation.

    Legacy compatibility:
    - Frontend calls GET /schema/{model_id}
    """
    # Frontend uses this schema to build the prediction form dynamically.
    return svc.get_model_schema(model_id)


@router.get("/summary/{model_id}", response_model=TrainingSummaryResponse)
async def model_summary(
    model_id: str,
    svc: InsightsService = Depends(get_insights_service),
) -> TrainingSummaryResponse:
    """
    Generate an LLM-assisted summary of training outcomes and top features.
    """
    # This is a read-only summary endpoint; it does not retrain or mutate the model.
    return await svc.training_summary(model_id)


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    svc: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    """
    Run inference using a trained model.

    Legacy compatibility:
    - Frontend expects POST /predict with:
      { model_id, input_data: { ... } }
    """
    return await svc.predict(request)


@router.post("/predict/stream")
async def predict_stream(
    request: PredictRequest,
    svc: PredictionService = Depends(get_prediction_service),
):
    """
    Stream prediction progress via Server-Sent Events (SSE).
    Events:
      - progress: { pct, message }
      - complete: PredictResponse payload
      - error: { message, code?, details? }
    """
    # Prediction is lighter than training, but we keep the same SSE pattern for consistent UI behavior.
    queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def emit_progress(pct: int, message: str) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"pct": pct, "message": message}))

    def run_prediction() -> None:
        try:
            result = svc.predict_with_progress(request, emit_progress)
        except AppError as exc:
            logger.warning(
                "prediction_stream_app_error",
                extra={"code": exc.code, "details": exc.details},
            )
            payload = {"message": exc.message, "code": exc.code, "details": exc.details}
            loop.call_soon_threadsafe(queue.put_nowait, ("error", payload))
        except Exception as exc:
            logger.exception("prediction_stream_failed")
            payload = {"message": str(exc) or "Prediction failed"}
            loop.call_soon_threadsafe(queue.put_nowait, ("error", payload))
        else:
            loop.call_soon_threadsafe(queue.put_nowait, ("complete", result.model_dump()))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", {}))

    threading.Thread(target=run_prediction, daemon=True).start()

    async def event_stream():
        while True:
            event, payload = await queue.get()
            if event == "done":
                break
            yield _sse_event(event, payload)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
