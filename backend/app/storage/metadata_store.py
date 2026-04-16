from __future__ import annotations

"""Disk-backed JSON metadata store for uploads, preprocess runs, and trained models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from starlette import status

from app.core.config import Settings
from app.core.errors import AppError
from app.utils.files import atomic_read_json, atomic_write_json, safe_join


@dataclass(frozen=True)
class MetadataStore:
    settings: Settings

    # ---- Upload metadata ----
    def _upload_meta_path(self, upload_id: str) -> Path:
        # Separate subfolders keep upload, preprocess, and model metadata logically distinct on disk.
        return safe_join(Path(self.settings.metadata_dir), "uploads", f"{upload_id}.json")

    def write_upload_metadata(self, upload_id: str, data: Dict[str, Any]) -> None:
        path = self._upload_meta_path(upload_id)
        atomic_write_json(path, data)

    def read_upload_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        path = self._upload_meta_path(upload_id)
        return atomic_read_json(path)

    # ---- Preprocess metadata ----
    def _preprocess_meta_path(self, upload_id: str) -> Path:
        return safe_join(Path(self.settings.metadata_dir), "preprocess", f"{upload_id}.json")

    def write_preprocess_metadata(self, upload_id: str, data: Dict[str, Any]) -> None:
        path = self._preprocess_meta_path(upload_id)
        atomic_write_json(path, data)

    def read_preprocess_metadata(self, upload_id: str) -> Optional[Dict[str, Any]]:
        path = self._preprocess_meta_path(upload_id)
        return atomic_read_json(path)

    # ---- Model metadata ----
    def _model_meta_path(self, model_id: str) -> Path:
        return safe_join(Path(self.settings.metadata_dir), "models", f"{model_id}.json")

    def write_model_metadata(self, model_id: str, data: Dict[str, Any]) -> None:
        path = self._model_meta_path(model_id)
        atomic_write_json(path, data)

    def read_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        path = self._model_meta_path(model_id)
        return atomic_read_json(path)

    # ---- Generic helpers (optional) ----
    def ensure_metadata_dirs(self) -> None:
        """
        Ensure metadata subfolders exist.
        Called indirectly by atomic_write_json() which creates parents, but kept for clarity.
        """
        # This helper is mostly explicit documentation; writes already create parents automatically.
        for sub in ("uploads", "preprocess", "models"):
            p = safe_join(Path(self.settings.metadata_dir), sub)
            p.mkdir(parents=True, exist_ok=True)
