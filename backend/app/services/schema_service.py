from __future__ import annotations

"""Service that exposes the saved model schema for dynamic prediction-form generation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from starlette import status

from app.core.config import Settings
from app.core.errors import AppError
from app.schemas.datasets import SchemaField, SchemaResponse
from app.storage.metadata_store import MetadataStore


@dataclass(frozen=True)
class SchemaService:
    settings: Settings
    metadata_store: MetadataStore

    def get_model_schema(self, model_id: str) -> SchemaResponse:
        # Everything needed for schema lookup lives in model metadata written after training.
        meta = self.metadata_store.read_model_metadata(model_id)
        if meta is None:
            raise AppError(
                "Unknown model_id. Please train a model first.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="model_not_found",
            )

        target_column = str(meta.get("target_column") or "")
        feature_columns: List[str] = list(meta.get("feature_columns") or [])
        schema = meta.get("schema") or {}

        fields: List[SchemaField] = []
        # schema format produced by trainer: { "fields": [ {name, dtype, required, allowed_values, example, description} ] }
        schema_fields = schema.get("fields") if isinstance(schema, dict) else None
        if isinstance(schema_fields, list):
            # Rehydrate raw dict metadata into typed API schema objects.
            for f in schema_fields:
                if not isinstance(f, dict) or "name" not in f:
                    continue
                fields.append(
                    SchemaField(
                        name=str(f.get("name")),
                        dtype=str(f.get("dtype") or "string"),
                        required=bool(f.get("required", True)),
                        allowed_values=f.get("allowed_values"),
                        example=f.get("example"),
                        description=f.get("description"),
                    )
                )
        else:
            # Fallback: still let prediction work even if rich schema metadata is missing.
            for name in feature_columns:
                fields.append(SchemaField(name=name, dtype="string", required=True))

        if not target_column or not feature_columns:
            raise AppError(
                "Model metadata incomplete. Please retrain the model.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="model_metadata_incomplete",
            )

        return SchemaResponse(
            model_id=model_id,
            target_column=target_column,
            feature_columns=feature_columns,
            fields=fields,
        )
