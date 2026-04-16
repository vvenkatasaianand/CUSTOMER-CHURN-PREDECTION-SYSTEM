from __future__ import annotations

"""Small helpers that convert raw metric floats into the response shape used by the frontend."""

from typing import Dict

from app.schemas.common import NumericMetric


def _fmt_pct(v: float) -> str:
    # Frontend shows metrics in percent form, so generate the display string once here.
    return f"{v * 100:.2f}%"


def format_metrics(raw: Dict[str, float]) -> Dict[str, dict]:
    """
    Convert raw metric floats into the TrainingMetrics schema fields (NumericMetric).
    """
    def nm(v: float) -> dict:
        # NumericMetric keeps both the raw float and the already-formatted display text.
        return NumericMetric(value=float(v), display=_fmt_pct(float(v))).model_dump()

    # Only include optional metrics when they were actually computed by the trainer.
    out: Dict[str, dict] = {
        "accuracy": nm(raw.get("accuracy", 0.0)),
        "precision": nm(raw.get("precision", 0.0)),
        "recall": nm(raw.get("recall", 0.0)),
        "f1": nm(raw.get("f1", 0.0)),
        "support": raw.get("support"),
    }
    if "roc_auc" in raw:
        out["roc_auc"] = nm(raw.get("roc_auc", 0.0))
    if "pr_auc" in raw:
        out["pr_auc"] = nm(raw.get("pr_auc", 0.0))
    return out
