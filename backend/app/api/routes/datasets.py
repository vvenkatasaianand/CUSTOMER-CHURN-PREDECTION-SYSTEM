from __future__ import annotations

"""Dataset-facing HTTP routes: upload raw data, preprocess it, and summarize it."""

from fastapi import APIRouter, Depends, File, UploadFile
from starlette import status

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.services.dataset_service import DatasetService
from app.services.insights_service import InsightsService
from app.storage.dataset_store import DatasetStore
from app.storage.metadata_store import MetadataStore
from app.storage.model_store import ModelStore
from app.schemas.datasets import DatasetInfo, PreprocessRequest, PreprocessResponse
from app.schemas.insights import DatasetSummaryResponse

router = APIRouter()


def _validate_upload_file(file: UploadFile, settings: Settings) -> None:
    # Reject empty uploads before spending time on storage/parsing.
    if not file or not file.filename:
        raise AppError("No file uploaded", status_code=status.HTTP_400_BAD_REQUEST, code="file_missing")

    filename = file.filename.lower().strip()
    # Keep accepted formats narrow so downstream parsing is predictable.
    if not (filename.endswith(".csv") or filename.endswith(".json")):
        raise AppError(
            "Unsupported file type. Please upload a .csv or .json file.",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="unsupported_file_type",
            details={"allowed": [".csv", ".json"], "received": file.filename},
        )

    # Upload size enforcement is done after writing to disk (streaming) inside DatasetStore.
    # This avoids loading the entire file into memory.


def get_dataset_service(settings: Settings = Depends(get_settings)) -> DatasetService:
    # Build request-scoped service objects from the current application settings.
    dataset_store = DatasetStore(settings=settings)
    metadata_store = MetadataStore(settings=settings)
    return DatasetService(settings=settings, dataset_store=dataset_store, metadata_store=metadata_store)


def get_insights_service(settings: Settings = Depends(get_settings)) -> InsightsService:
    # Dataset summaries need access to dataset files, metadata, and sometimes model metadata.
    dataset_store = DatasetStore(settings=settings)
    metadata_store = MetadataStore(settings=settings)
    model_store = ModelStore(settings=settings)
    return InsightsService(
        settings=settings,
        dataset_store=dataset_store,
        metadata_store=metadata_store,
        model_store=model_store,
    )


@router.post("/upload", response_model=DatasetInfo)
async def upload_dataset(
    file: UploadFile = File(...),
    svc: DatasetService = Depends(get_dataset_service),
    settings: Settings = Depends(get_settings),
) -> DatasetInfo:
    """
    Upload and parse a dataset file (.csv or .json).

    Legacy compatibility:
    - Frontend expects POST /upload returning DatasetInfo:
      { upload_id, filename, shape, columns[{name,type,sample_values,null_count,unique_count}] }
    """
    # Route-level validation keeps obvious bad requests away from the service layer.
    _validate_upload_file(file, settings)
    return await svc.upload_dataset(file)


@router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_dataset(
    request: PreprocessRequest,
    svc: DatasetService = Depends(get_dataset_service),
) -> PreprocessResponse:
    """
    Preprocess the uploaded dataset based on excluded columns + selected target column.

    Legacy compatibility:
    - Frontend expects POST /preprocess with:
      { upload_id, excluded_columns, target_column }
    - Response includes:
      { status, shape, preview, target_column, feature_columns }
    """
    # Service handles all dataset checks and writes the processed dataset/metadata to disk.
    return await svc.preprocess_dataset(request)


@router.get("/summary/{upload_id}", response_model=DatasetSummaryResponse)
async def dataset_summary(
    upload_id: str,
    svc: InsightsService = Depends(get_insights_service),
) -> DatasetSummaryResponse:
    """
    Generate an overall dataset summary (LLM-assisted with safe fallbacks).
    Requires preprocessing to have selected a target column.
    """
    # Summary is generated from saved preprocess metadata so the UI can reload it later.
    return await svc.dataset_summary(upload_id)
