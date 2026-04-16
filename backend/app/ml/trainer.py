from __future__ import annotations

"""Training helper that normalizes labels, fits XGBoost, and computes evaluation artifacts."""

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from starlette import status
from xgboost.callback import TrainingCallback

from app.core.errors import AppError
from app.ml.pipeline import build_schema, build_xgb_pipeline


@dataclass(frozen=True)
class TrainArtifacts:
    pipeline: Any
    schema: Dict[str, Any]
    metrics: Dict[str, float]
    confusion_matrix: List[List[int]]
    feature_importance: List[Tuple[str, float]]


def _normalize_target(y: pd.Series) -> pd.Series:
    """
    Normalize churn target to 0/1.
    Accepts:
      - numeric 0/1
      - boolean
      - strings like Yes/No, True/False, Churn/No Churn (best-effort)
    """
    if pd.api.types.is_bool_dtype(y):
        return y.astype(int)

    if pd.api.types.is_numeric_dtype(y):
        # Numeric targets are assumed to already represent binary labels.
        return y.astype(int)

    # Strings / objects
    s = y.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "churn": 1,
        "no churn": 0,
        "not churn": 0,
        "exited": 1,
        "stayed": 0,
        "stay": 0,
    }
    mapped = s.map(mapping)
    if mapped.isna().any():
        # Fail fast when labels are unfamiliar; silent guessing here would poison the model.
        unknown = sorted(set(s[mapped.isna()].unique().tolist()))[:10]
        raise AppError(
            "Target column contains unsupported labels. Use a binary target (0/1 or Yes/No).",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="invalid_target_labels",
            details={"examples_of_unknown_labels": unknown},
        )
    return mapped.astype(int)


def train_xgb_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_seed: int,
    test_size: float,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    progress_range: Tuple[int, int] = (0, 100),
) -> TrainArtifacts:
    def _emit(pct: float, msg: str) -> None:
        if progress_cb is not None:
            progress_cb(int(pct), msg)

    if X is None or y is None:
        raise AppError(
            "Training data is missing.",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="training_data_missing",
        )
    if len(X) < 10:
        raise AppError(
            "Dataset is too small to train reliably (need at least 10 rows).",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="dataset_too_small",
        )

    y_norm = _normalize_target(y)

    # Stratification keeps the train/test split balanced across the two churn classes.
    unique = sorted(y_norm.unique().tolist())
    if len(unique) != 2:
        raise AppError(
            "Target column must be binary (two classes).",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="target_not_binary",
            details={"unique_values": unique},
        )

    stratify = y_norm if y_norm.nunique() == 2 else None

    start_pct, end_pct = progress_range
    start_pct = max(0, min(100, int(start_pct)))
    end_pct = max(start_pct, min(100, int(end_pct)))

    _emit(start_pct, "Splitting train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_norm,
        test_size=float(test_size),
        random_state=int(random_seed),
        stratify=stratify,
    )
    pipeline, _meta = build_xgb_pipeline(X_train, random_seed=int(random_seed))
    prep_pct = min(start_pct + 1, end_pct)
    _emit(prep_pct, "Preparing training pipeline")

    # XGBoost training callbacks let the streaming endpoint show progress in the UI.
    class _ProgressCallback(TrainingCallback):
        def __init__(self, total_rounds: int) -> None:
            self.total_rounds = max(1, int(total_rounds))
            self.last_pct = -1

        def after_iteration(self, model, epoch: int, evals_log) -> bool:
            frac = float(epoch + 1) / float(self.total_rounds)
            pct = start_pct + (end_pct - start_pct) * frac
            pct_int = int(pct)
            if pct_int != self.last_pct:
                self.last_pct = pct_int
                _emit(pct_int, f"Training model ({epoch + 1}/{self.total_rounds})")
            return False

    fit_params: Dict[str, Any] = {}
    if progress_cb is not None:
        model = pipeline.named_steps.get("model")
        total_rounds = getattr(model, "get_params", lambda: {})().get("n_estimators", 0)
        supports_callbacks = False
        try:
            supports_callbacks = "callbacks" in inspect.signature(model.fit).parameters
        except (TypeError, ValueError):
            supports_callbacks = False
        if supports_callbacks:
            fit_params["model__callbacks"] = [_ProgressCallback(total_rounds)]
            fit_params["model__verbose"] = False
        else:
            _emit(start_pct, "Fitting model (progress callbacks not supported)")

    _emit(start_pct, "Fitting model")
    pipeline.fit(X_train, y_train, **fit_params)
    _emit(end_pct, "Model fit complete")

    # Evaluate on the held-out test split so metrics reflect unseen rows.
    y_pred = pipeline.predict(X_test)
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # Fallback if predict_proba unavailable (should be available for XGBClassifier)
        y_prob = None

    # Collect the main binary-classification metrics used in the UI and report.
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_test, y_prob))
        except Exception:
            metrics["pr_auc"] = float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()

    def aggregate_feature_importance() -> List[Tuple[str, float]]:
        """
        Aggregate model feature importance back to original columns.
        For one-hot encoded columns, sum importances for each source feature.
        """
        columns = [str(c) for c in X.columns.tolist()]
        fallback = [(c, 0.0) for c in columns]

        model = pipeline.named_steps.get("model")
        if model is None or not hasattr(model, "feature_importances_"):
            return fallback

        importances = np.asarray(getattr(model, "feature_importances_", []), dtype=float)
        if importances.size == 0:
            return fallback

        preprocess = pipeline.named_steps.get("preprocess")
        if preprocess is None or not hasattr(preprocess, "transformers_"):
            return fallback

        totals = {c: 0.0 for c in columns}
        idx = 0

        for name, transformer, cols in preprocess.transformers_:
            if name == "remainder" and transformer == "drop":
                continue
            if cols is None:
                continue

            cols_list = [str(c) for c in cols]

            if name == "num":
                for col in cols_list:
                    if idx >= importances.size:
                        break
                    totals[col] += float(importances[idx])
                    idx += 1
            elif name == "cat":
                # One-hot encoding expands one source column into many model columns; sum them back together.
                onehot = None
                if hasattr(transformer, "named_steps"):
                    onehot = transformer.named_steps.get("onehot")
                if onehot is None or not hasattr(onehot, "categories_"):
                    return fallback
                for col, cats in zip(cols_list, onehot.categories_):
                    n = len(cats)
                    if n <= 0:
                        continue
                    if idx >= importances.size:
                        break
                    end = min(idx + n, importances.size)
                    totals[col] += float(np.sum(importances[idx:end]))
                    idx = end
            else:
                return fallback

        return [(c, totals.get(c, 0.0)) for c in columns]

    feature_importance = aggregate_feature_importance()

    # Persist the input schema alongside the model so prediction forms can be built automatically.
    schema = {"fields": build_schema(X).fields}

    return TrainArtifacts(
        pipeline=pipeline,
        schema=schema,
        metrics=metrics,
        confusion_matrix=cm,
        feature_importance=feature_importance,
    )
