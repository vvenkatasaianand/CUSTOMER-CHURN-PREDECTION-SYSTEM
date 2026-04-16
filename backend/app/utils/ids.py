from __future__ import annotations

"""Tiny helper for short readable IDs used in upload/model metadata."""

import uuid


def new_id(prefix: str) -> str:
    """
    Create a short, URL-safe-ish identifier.
    Example: mdl_9f3a1c2d0e5b4d7a
    """
    # Prefix makes IDs self-describing in logs/UI, while the uuid slice keeps collisions unlikely.
    return f"{prefix}_{uuid.uuid4().hex[:16]}"
