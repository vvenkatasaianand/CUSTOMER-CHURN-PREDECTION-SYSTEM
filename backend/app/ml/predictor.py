from __future__ import annotations

"""Prediction helper for running a persisted sklearn pipeline on one dataframe input."""

from typing import Any, Tuple

import pandas as pd
from starlette import status

from app.core.errors import AppError


def predict_with_pipeline(pipeline: Any, X: pd.DataFrame) -> Tuple[int, float]:
    """
    Predict with a persisted sklearn Pipeline.

    Returns:
      (predicted_label, probability_of_class_1)
    """
    if pipeline is None:
        raise AppError(
            "Model pipeline is not loaded.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="pipeline_missing",
        )
    if X is None or not isinstance(X, pd.DataFrame) or X.empty:
        raise AppError(
            "Input features are missing for prediction.",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            code="input_missing",
        )

    try:
        # Pipeline.predict returns the binary class label after running preprocessing + model.
        pred = pipeline.predict(X)
    except Exception as e:
        raise AppError(
            "Failed to run prediction.",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="predict_failed",
            details={"error": str(e)},
        )

    try:
        # Probability is preferred because the UI converts it into risk buckets and confidence display.
        prob = pipeline.predict_proba(X)[:, 1]
        prob_val = float(prob[0])
    except Exception:
        # If probability is unavailable, fall back to a hard 0/1 value based on the predicted class.
        prob_val = 1.0 if int(pred[0]) == 1 else 0.0

    return int(pred[0]), prob_val
