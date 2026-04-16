from __future__ import annotations

"""Disk-backed storage helper for raw uploads and processed dataset files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import UploadFile
from starlette import status

from app.core.config import Settings
from app.core.errors import AppError
from app.utils.files import atomic_write_stream, safe_join


@dataclass(frozen=True)
class DatasetStore:
    settings: Settings

    def get_upload_path(self, upload_id: str) -> Path:
        # Stored as uploads/<upload_id>.csv or uploads/<upload_id>.json (we infer extension later)
        # We don't know ext here, so we resolve by scanning.
        base = Path(self.settings.uploads_dir)
        # Prefer csv then json
        csv_path = safe_join(base, f"{upload_id}.csv")
        json_path = safe_join(base, f"{upload_id}.json")
        if csv_path.exists():
            return csv_path
        if json_path.exists():
            return json_path
        # default to csv path if not found (caller checks existence)
        return csv_path

    async def save_upload(self, upload_id: str, file: UploadFile) -> Path:
        # Preserve only the file extension; the internal upload ID becomes the real stored filename.
        filename = (file.filename or "").lower().strip()
        ext = ".csv" if filename.endswith(".csv") else ".json" if filename.endswith(".json") else ""
        if ext not in (".csv", ".json"):
            raise AppError(
                "Unsupported file type. Upload must be .csv or .json.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="unsupported_file_type",
            )

        dest = safe_join(Path(self.settings.uploads_dir), f"{upload_id}{ext}")
        max_bytes = int(self.settings.max_upload_mb) * 1024 * 1024

        try:
            # Stream the upload straight to disk to avoid holding large files in memory.
            await atomic_write_stream(upload_file=file, dest_path=dest, max_bytes=max_bytes)
        except AppError:
            raise
        except Exception as e:
            raise AppError(
                "Failed to save uploaded file",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="upload_save_failed",
                details={"error": str(e)},
            )

        return dest

    def load_dataframe(self, path: Path) -> pd.DataFrame:
        # Very defensive: only allow known extensions
        ext = path.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(path)
        if ext == ".json":
            # Support JSON records-style or array style as long as pandas can infer it.
            return pd.read_json(path, orient=None)
        raise AppError(
            "Unsupported dataset format on server.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="unsupported_dataset_format",
            details={"path": str(path)},
        )

    def save_processed(self, upload_id: str, df: pd.DataFrame) -> Path:
        # Parquet is compact and reloads with schema information better than CSV.
        dest = safe_join(Path(self.settings.processed_dir), f"{upload_id}.parquet")
        df.to_parquet(dest, index=False)
        return dest

    def load_processed_dataframe(self, path: Path) -> pd.DataFrame:
        # Processed data should always be parquet because preprocess writes it in that format.
        if path.suffix.lower() != ".parquet":
            raise AppError(
                "Processed dataset must be parquet.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="invalid_processed_format",
                details={"path": str(path)},
            )
        return pd.read_parquet(path)

    def safe_delete(self, path: Path) -> None:
        try:
            # Best-effort cleanup only; upload failure should not be blocked by cleanup failure.
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)
        except Exception:
            # best effort
            return

    def path_from_string(self, p: str) -> Path:
        return Path(p)
