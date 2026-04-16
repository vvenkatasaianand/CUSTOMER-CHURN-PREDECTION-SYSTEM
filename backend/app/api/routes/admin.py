from __future__ import annotations

"""Development/demo maintenance routes such as clearing runtime artifacts."""

import shutil
from pathlib import Path
from typing import Dict, Tuple

from fastapi import APIRouter, Depends, Query
from starlette import status

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.schemas.common import MessageResponse

router = APIRouter()


def _safe_clear_dir(dir_path: Path) -> Tuple[int, int]:
    """
    Remove files and sub-directories inside `dir_path` (but never the dir itself).
    Returns: (files_deleted, dirs_deleted)
    """
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
        return (0, 0)

    if not dir_path.is_dir():
        raise AppError(
            f"Reset path is not a directory: {dir_path}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="reset_invalid_path",
        )

    files_deleted = 0
    dirs_deleted = 0

    # Only delete children; keeping the parent directory avoids later startup surprises.
    for child in dir_path.iterdir():
        # Never follow symlinks out of the sandbox; remove the link itself.
        if child.is_symlink():
            child.unlink(missing_ok=True)
            files_deleted += 1
            continue

        if child.is_file():
            child.unlink(missing_ok=True)
            files_deleted += 1
        elif child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
            dirs_deleted += 1

    return (files_deleted, dirs_deleted)


@router.post("/reset", response_model=MessageResponse)
def reset_runtime_state(
    include_models: bool = Query(
        default=True,
        description="If true, also removes trained model artifacts under MODELS_DIR.",
    ),
    settings: Settings = Depends(get_settings),
) -> MessageResponse:
    """
    Clears runtime artifacts (uploads/processed/metadata and optionally models).

    Notes:
    - This is intended for development/demo resets.
    - It only deletes contents inside configured runtime folders.
    """
    # Build the exact folders we allow this endpoint to clear.
    targets = {
        "uploads": Path(settings.uploads_dir),
        "processed": Path(settings.processed_dir),
        "metadata": Path(settings.metadata_dir),
    }
    if include_models:
        targets["models"] = Path(settings.models_dir)

    summary: Dict[str, Dict[str, int]] = {}
    try:
        # Clear each runtime folder independently so the response can report what changed.
        for name, path in targets.items():
            files_deleted, dirs_deleted = _safe_clear_dir(path)
            summary[name] = {"files_deleted": files_deleted, "dirs_deleted": dirs_deleted}
    except AppError:
        raise
    except Exception as e:
        raise AppError(
            "Failed to reset runtime state",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="reset_failed",
            details={"error": str(e)},
        )

    # Return a compact human-readable summary for the UI toast/message area.
    parts = [f"{k}: {v['files_deleted']} files, {v['dirs_deleted']} dirs" for k, v in summary.items()]
    return MessageResponse(message="Reset completed. " + " | ".join(parts))
