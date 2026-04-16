from __future__ import annotations

"""Helpers for building the training/inference pipeline and the schema used by the UI."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


@dataclass(frozen=True)
class BuiltSchema:
    """
    JSON-serializable schema used for UI form generation.
    """
    fields: List[Dict[str, Any]]


def _infer_dtype(series: pd.Series) -> str:
    # UI schema only needs a small set of input types, not pandas' full dtype vocabulary.
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    return "string"


def _allowed_values(series: pd.Series, max_unique: int = 25) -> Optional[List[str]]:
    # For low-cardinality string columns, expose allowed values so the UI can render dropdowns.
    try:
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            return None
        vals = series.dropna().astype(str).unique().tolist()
        if 0 < len(vals) <= max_unique:
            return sorted(vals)
    except Exception:
        return None
    return None


def build_schema(X: pd.DataFrame) -> BuiltSchema:
    # Convert dataframe columns into a lightweight schema that the React form can render.
    fields: List[Dict[str, Any]] = []
    for col in X.columns.tolist():
        s = X[col]
        dtype = _infer_dtype(s)
        allowed = _allowed_values(s)
        example = None
        try:
            non_null = s.dropna()
            if len(non_null) > 0:
                example = non_null.iloc[0]
                if hasattr(example, "item"):
                    example = example.item()
                if isinstance(example, (pd.Timestamp,)):
                    example = example.isoformat()
        except Exception:
            example = None

        fields.append(
            {
                "name": str(col),
                "dtype": dtype,
                "required": True,
                "allowed_values": allowed,
                "example": example,
                "description": None,
            }
        )
    return BuiltSchema(fields=fields)


def build_xgb_pipeline(X: pd.DataFrame, random_seed: int) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Build a pipeline that is consistent between training and inference:
    - numeric: impute median
    - categorical: impute most_frequent + one-hot (ignore unknowns)
    - model: XGBoost classifier
    """
    # Split columns by type so preprocessing can handle numeric and categorical features differently.
    numeric_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    bool_features = [c for c in X.columns if pd.api.types.is_bool_dtype(X[c])]
    # Treat bools as categorical or numeric? We'll treat as categorical for stability.
    categorical_features = [c for c in X.columns if c not in numeric_features]
    # ensure bool columns are included in categorical_features (already)
    _ = bool_features  # kept for readability/future branching

    # Numeric pipeline only needs imputation because tree models handle scaling well enough here.
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical pipeline fills missing labels, then one-hot encodes them for XGBoost.
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    # ColumnTransformer applies the right preprocessing branch to each feature group.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    # Hyperparameters are fixed here to keep the student project reproducible and easy to explain.
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=random_seed,
        eval_metric="logloss",
        n_jobs=1,  # safer default for student machines; can be configured later
    )

    # Bundle preprocessing and model together so training and prediction always stay in sync.
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    meta: Dict[str, Any] = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    return pipe, meta
