from __future__ import annotations

"""Shared low-level file helpers for safe paths and atomic disk writes."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import UploadFile
from starlette import status

from app.core.errors import AppError


def safe_join(base_dir: Path, *parts: str) -> Path:
    """
    Safely join path parts under base_dir, preventing path traversal.
    """
    # Resolve both paths first so path traversal like "..\\.." cannot escape the base directory.
    base_dir = base_dir.resolve()
    candidate = (base_dir.joinpath(*parts)).resolve()
    if base_dir == candidate or str(candidate).startswith(str(base_dir) + os.sep):
        return candidate
    raise AppError(
        "Invalid path access",
        status_code=status.HTTP_400_BAD_REQUEST,
        code="invalid_path",
        details={"base": str(base_dir), "candidate": str(candidate)},
    )


async def atomic_write_stream(upload_file: UploadFile, dest_path: Path, max_bytes: int) -> None:
    """
    Stream UploadFile to disk with a size limit. Writes atomically using a temp file then rename.
    """
    # Always write into the destination folder first so the final rename stays on the same filesystem.
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_written = 0
    suffix = dest_path.suffix or ".bin"

    with tempfile.NamedTemporaryFile(delete=False, dir=str(dest_path.parent), suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        try:
            while True:
                # Read and write in chunks so large uploads do not spike memory usage.
                chunk = await upload_file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise AppError(
                        f"Upload exceeds max size of {max_bytes // (1024 * 1024)} MB",
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        code="upload_too_large",
                    )
                tmp.write(chunk)
            tmp.flush()
            os.fsync(tmp.fileno())
        except Exception:
            # Cleanup temp on any error
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    # Atomic replace
    tmp_path.replace(dest_path)


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    # JSON writes go through atomic_write_text so readers never observe partial files.
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    atomic_write_text(path, payload)


def atomic_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return None
    try:
        # Returning None on read failure keeps metadata lookup simple for callers.
        raw = path.read_text(encoding="utf-8")
        return json.loads(raw)
    except Exception:
        return None


def atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic text write via temporary file then rename.
    """
    # Write to a temp file first, then rename into place, so readers never see a half-written file.
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp", mode="w", encoding="utf-8") as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(text)
        tmp.flush()
        os.fsync(tmp.fileno())
    tmp_path.replace(path)
