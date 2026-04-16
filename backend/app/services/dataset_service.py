from __future__ import annotations

"""Service layer for dataset upload and preprocessing before model training."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd
from fastapi import UploadFile
from starlette import status

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.common import ColumnInfo
from app.schemas.datasets import DatasetInfo, PreprocessRequest, PreprocessResponse
from app.storage.dataset_store import DatasetStore
from app.storage.metadata_store import MetadataStore
from app.utils.ids import new_id


@dataclass(frozen=True)
class DatasetService:
    settings: Settings
    dataset_store: DatasetStore
    metadata_store: MetadataStore

    async def upload_dataset(self, file: UploadFile) -> DatasetInfo:
        # Create a stable internal ID so later steps can refer to this upload without using filenames.
        upload_id = new_id(prefix="upl")
        saved_path = await self.dataset_store.save_upload(upload_id=upload_id, file=file)

        try:
            # Parse immediately so invalid files fail early instead of later in the workflow.
            df = self.dataset_store.load_dataframe(saved_path)
        except Exception as e:
            # Cleanup invalid uploads
            self.dataset_store.safe_delete(saved_path)
            raise AppError(
                "Failed to parse uploaded file. Ensure it is valid CSV/JSON.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="invalid_dataset",
                details={"error": str(e)},
            )

        columns = self._build_columns_info(df)

        info = DatasetInfo(
            upload_id=upload_id,
            filename=file.filename or "uploaded",
            shape=[int(df.shape[0]), int(df.shape[1])],
            columns=columns,
        )

        # Persist dataset info so preprocess/train steps can reload metadata without reparsing the file.
        self.metadata_store.write_upload_metadata(upload_id, info.model_dump())

        return info

    async def preprocess_dataset(self, req: PreprocessRequest) -> PreprocessResponse:
        # Validate upload exists before touching any file paths on disk.
        upload_meta = self.metadata_store.read_upload_metadata(req.upload_id)
        if upload_meta is None:
            raise AppError(
                "Unknown upload_id. Please upload the dataset again.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="upload_not_found",
            )

        raw_path = self.dataset_store.get_upload_path(req.upload_id)
        if not raw_path.exists():
            raise AppError(
                "Uploaded file not found on server. Please upload again.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="upload_file_missing",
            )

        df = self.dataset_store.load_dataframe(raw_path)

        # Target column is required because the rest of the pipeline assumes supervised learning.
        if req.target_column not in df.columns:
            raise AppError(
                f"Target column '{req.target_column}' not found in dataset.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="invalid_target",
                details={"available_columns": list(map(str, df.columns.tolist()))},
            )

        excluded = set([c for c in req.excluded_columns if c])
        # Never allow excluding the target implicitly; enforce target kept
        if req.target_column in excluded:
            excluded.remove(req.target_column)

        # Features are simply "all remaining columns except target and excluded fields".
        feature_columns = [c for c in df.columns.tolist() if c != req.target_column and c not in excluded]
        if not feature_columns:
            raise AppError(
                "No feature columns remain after excluding columns. Please adjust selection.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="no_features",
            )

        # Basic cleaning: drop rows where target is null
        before_rows = int(df.shape[0])
        df = df.dropna(subset=[req.target_column])
        after_rows = int(df.shape[0])

        notes: List[str] = []
        if after_rows < before_rows:
            # Keep a readable note so the UI can explain why row count changed after preprocess.
            notes.append(f"Dropped {before_rows - after_rows} rows with null target values.")

        # Store processed dataset as Parquet for faster reload and typed storage
        processed_path = self.dataset_store.save_processed(upload_id=req.upload_id, df=df)

        # Persist exactly what training needs so training can be restarted without recomputing preprocess.
        self.metadata_store.write_preprocess_metadata(
            req.upload_id,
            {
                "upload_id": req.upload_id,
                "target_column": req.target_column,
                "excluded_columns": list(excluded),
                "feature_columns": feature_columns,
                "processed_path": str(processed_path),
                "shape": [int(df.shape[0]), int(df.shape[1])],
            },
        )

        preview = self._preview_rows(df, max_rows=10)

        return PreprocessResponse(
            upload_id=req.upload_id,
            target_column=req.target_column,
            feature_columns=feature_columns,
            shape=[int(df.shape[0]), int(df.shape[1])],
            preview=preview,
            notes=notes or None,
        )

    def _preview_rows(self, df: pd.DataFrame, max_rows: int = 10) -> List[Dict[str, Any]]:
        try:
            # Convert NaN to None so preview rows serialize cleanly in JSON responses.
            preview_df = df.head(max_rows).copy()
            return preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records")
        except Exception:
            return []

    def _build_columns_info(self, df: pd.DataFrame) -> List[ColumnInfo]:
        # Build lightweight column metadata so the frontend can guide target/feature selection.
        infos: List[ColumnInfo] = []
        for col in df.columns.tolist():
            series = df[col]
            sample_vals = series.dropna().unique().tolist()[:5]
            # Ensure JSON-friendly samples
            sample_vals = [self._json_safe(v) for v in sample_vals]

            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))

            infos.append(
                ColumnInfo(
                    name=str(col),
                    dtype=dtype,
                    sample_values=sample_vals,
                    null_count=null_count,
                    unique_count=unique_count,
                )
            )
        return infos

    def _json_safe(self, v: Any) -> Any:
        # Handle numpy/pandas scalars before sending values back in API responses.
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass

        # Convert common non-serializable types
        if isinstance(v, (pd.Timestamp,)):
            return v.isoformat()
        # numpy types often have .item()
        if hasattr(v, "item"):
            try:
                return v.item()
            except Exception:
                return str(v)
        return v
