from __future__ import annotations

"""Service layer for running a saved model on one customer record and explaining the result."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from xgboost import DMatrix
from starlette import status
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import AppError
from app.ml.predictor import predict_with_pipeline
from app.schemas.common import RiskLevel
from app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
    PredictionExplanation,
    PredictionFactor,
    RecommendedAction,
)
from app.services.llm_service import LLMService
from app.storage.dataset_store import DatasetStore
from app.storage.metadata_store import MetadataStore
from app.storage.model_store import ModelStore


@dataclass(frozen=True)
class PredictionService:
    settings: Settings
    dataset_store: DatasetStore
    model_store: ModelStore
    metadata_store: MetadataStore

    async def predict(self, req: PredictRequest) -> PredictResponse:
        return self._predict_impl(req)

    def predict_with_progress(
        self,
        req: PredictRequest,
        emit_progress: Optional[Callable[[int, str], None]] = None,
    ) -> PredictResponse:
        return self._predict_impl(req, emit_progress=emit_progress)

    def _predict_impl(
        self,
        req: PredictRequest,
        emit_progress: Optional[Callable[[int, str], None]] = None,
    ) -> PredictResponse:
        def _emit(pct: int, message: str) -> None:
            if emit_progress:
                emit_progress(pct, message)

        _emit(5, "Validating model metadata")
        meta = self.metadata_store.read_model_metadata(req.model_id)
        if meta is None:
            raise AppError(
                "Unknown model_id. Please train the model again.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="model_not_found",
            )

        feature_columns: List[str] = list(meta.get("feature_columns") or [])
        target_column = str(meta.get("target_column") or "")

        if not feature_columns or not target_column:
            raise AppError(
                "Model metadata is incomplete. Please retrain the model.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="model_metadata_incomplete",
            )

        _emit(15, "Loading model")
        pipeline = self.model_store.load_model(req.model_id)

        # Build a single-row dataframe in the exact feature order used during training.
        row: Dict[str, Any] = {}
        missing: List[str] = []
        _emit(25, "Preparing input features")

        for col in feature_columns:
            if col in req.input_data:
                row[col] = req.input_data[col]
            else:
                missing.append(col)

        if missing:
            raise AppError(
                "Missing required input fields for prediction.",
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                code="missing_fields",
                details={"missing": missing},
            )

        df = pd.DataFrame([row], columns=feature_columns)

        _emit(45, "Running prediction")
        try:
            # predict_with_pipeline returns both the binary class and probability of churn=1.
            pred_label, prob = predict_with_pipeline(pipeline=pipeline, X=df)
        except AppError:
            raise
        except Exception as e:
            raise AppError(
                "Prediction failed due to an internal error.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="prediction_failed",
                details={"error": str(e)},
            )

        risk_level: RiskLevel = self._risk_from_probability(prob)

        _emit(60, "Computing key factors")
        # Prefer per-prediction contributions so the explanation refers to this exact customer.
        key_factors = self._compute_key_factors(
            pipeline=pipeline,
            df=df,
            feature_columns=feature_columns,
            model_meta=meta,
        )

        _emit(75, "Generating explanation")
        explanation = self._llm_explanation(
            probability=prob,
            risk_level=risk_level,
            key_factors=key_factors,
            llm_model_name=LLMService(settings=self.settings).model_name(),
        )
        _emit(90, "Building recommendations")
        # Recommended actions are deterministic business rules based only on the risk bucket.
        actions = self._deterministic_actions(risk_level=risk_level)

        _emit(95, "Finalizing response")
        return PredictResponse(
            model_id=req.model_id,
            prediction=int(pred_label),
            probability=float(prob),
            risk_level=risk_level,
            explanation=explanation,
            recommended_actions=actions,
        )

    def _risk_from_probability(self, probability: float) -> RiskLevel:
        # Deterministic and easy to justify in viva/report:
        # <0.33 Low, <0.66 Medium, otherwise High
        if probability < 0.33:
            return "Low"
        if probability < 0.66:
            return "Medium"
        return "High"

    def _llm_explanation(
        self,
        probability: float,
        risk_level: RiskLevel,
        key_factors: List[PredictionFactor],
        llm_model_name: str,
    ) -> PredictionExplanation:
        class _PredictionLLMOutput(BaseModel):
            summary: str
            confidence_note: Optional[str] = None

        fallback = self._deterministic_explanation(probability, risk_level, key_factors, llm_model_name)
        llm = LLMService(settings=self.settings)
        if not llm.is_enabled():
            return fallback

        def _word_count(text: str) -> int:
            return len([w for w in text.strip().split() if w])

        facts = {
            "probability": round(float(probability), 4),
            "risk_level": risk_level,
            "key_factors": [k.model_dump() for k in key_factors],
        }
        # prompt = (
        #     "You are a data analyst. Use ONLY the facts provided. "
        #     "Return JSON with keys: summary, confidence_note. "
        #     "summary: 30-40 words, mention risk_level and probability. "
        #     "If key_factors exist, mention up to 2 by name. "
        #     "Do not add new features or numbers. "
        #     f"FACTS={facts}"
        # )

        # The prompt is tightly constrained so the LLM explains the model output instead of inventing new facts.
        prompt = (
            "You are an experienced data analyst interpreting a risk assessment summary. "
            "Use ONLY the facts provided below. Do not introduce new features, metrics, or numbers. "
            "You may explain likely reasons and implications only if they are directly supported by the facts. "
            "Return valid JSON with exactly two keys: summary and confidence_note. "
            "The summary should be 80–100 words and must mention the risk_level and probability. "
            "If key_factors are present in the facts, mention up to two by name and briefly explain how they relate to the issue. "
            "Where supported by the facts, describe why the customer appears to be facing this issue and suggest high-level, non-specific mitigation ideas. "
            "Do not speculate beyond the provided information or assume unseen behavior. "
            f"FACTS={facts}"
        )
        
        data = llm.generate_json(prompt)
        if not isinstance(data, dict):
            return fallback

        try:
            parsed = _PredictionLLMOutput.model_validate(data)
        except ValidationError:
            return fallback

        summary = parsed.summary.strip()
        confidence_note = parsed.confidence_note.strip() if parsed.confidence_note else None
        if not summary:
            return fallback

        return PredictionExplanation(
            summary=summary,
            key_factors=key_factors,
            confidence_note=confidence_note,
            llm_used=True,
            llm_model=llm.model_name(),
        )

    def _deterministic_explanation(
        self,
        probability: float,
        risk_level: RiskLevel,
        key_factors: List[PredictionFactor],
        llm_model_name: str,
    ) -> PredictionExplanation:
        # No LLM available: still return a stable explanation so prediction results remain usable.
        factor_names = [f.feature for f in key_factors[:2]]
        factor_clause = ""
        if factor_names:
            factor_clause = f" Key drivers include {', '.join(factor_names)}."
        summary = (
            f"The model predicts churn risk as {risk_level} with probability {probability:.2f}. "
            "This explanation summarizes learned patterns for this case using available features and statistical evidence "
            "from training data to guide interpretation for analysts."
            f"{factor_clause}"
        )
        confidence_note = "Feature contributions are model-internal signals for this single prediction; treat them as directional, not causal."
        return PredictionExplanation(
            summary=summary,
            key_factors=key_factors,
            confidence_note=confidence_note,
            llm_used=False,
            llm_model=llm_model_name,
        )

    def _deterministic_actions(self, risk_level: RiskLevel) -> List[RecommendedAction]:
        # No LLM here either; actions are simple business rules chosen by risk bucket.
        if risk_level == "High":
            return [
                RecommendedAction(
                    action="Trigger proactive retention outreach within 24 hours",
                    reason="High churn probability indicates urgent intervention is required.",
                    priority=5,
                    expected_impact="Reduce immediate churn risk",
                ),
                RecommendedAction(
                    action="Offer a targeted retention incentive (plan upgrade/discount)",
                    reason="Incentives can increase perceived value and reduce churn intent.",
                    priority=4,
                    expected_impact="Improve short-term retention",
                ),
                RecommendedAction(
                    action="Assign customer success call to identify pain points",
                    reason="Direct feedback helps address dissatisfaction drivers.",
                    priority=4,
                    expected_impact="Increase engagement and satisfaction",
                ),
            ]
        if risk_level == "Medium":
            return [
                RecommendedAction(
                    action="Send personalized engagement campaign",
                    reason="Medium risk customers may respond to engagement and nudges.",
                    priority=3,
                    expected_impact="Increase engagement",
                ),
                RecommendedAction(
                    action="Monitor usage/transactions for early warning signals",
                    reason="Early detection helps prevent movement to high risk.",
                    priority=3,
                    expected_impact="Prevent risk escalation",
                ),
            ]
        return [
            RecommendedAction(
                action="Maintain standard customer communication cadence",
                reason="Low churn probability suggests no immediate intervention is required.",
                priority=2,
                expected_impact="Sustain retention",
            ),
            RecommendedAction(
                action="Continue monitoring periodic risk updates",
                reason="Customer behavior can change over time; periodic checks are recommended.",
                priority=2,
                expected_impact="Early detection of changes",
            ),
        ]

    def _compute_key_factors(
        self,
        pipeline: Any,
        df: pd.DataFrame,
        feature_columns: List[str],
        model_meta: Dict[str, Any],
        top_k: int = 5,
    ) -> List[PredictionFactor]:
        # Prefer per-prediction contributions (tree SHAP-style values via XGBoost pred_contribs).
        contribs = self._xgb_pred_contribs(pipeline, df, feature_columns, top_k=top_k)
        if contribs:
            return contribs

        # Fallback: use global feature importance saved during training if local contributions fail.
        fi_raw = model_meta.get("feature_importance") or []
        factors: List[PredictionFactor] = []
        for item in fi_raw[:top_k]:
            if not isinstance(item, dict):
                continue
            feature = str(item.get("feature") or "")
            if not feature:
                continue
            factors.append(
                PredictionFactor(
                    feature=feature,
                    direction=None,
                    contribution=float(item.get("importance", 0.0)),
                    reasoning="Global feature importance (not prediction-specific).",
                )
            )
        return factors

    def _xgb_pred_contribs(
        self,
        pipeline: Any,
        df: pd.DataFrame,
        feature_columns: List[str],
        top_k: int = 5,
    ) -> List[PredictionFactor]:
        try:
            pre = pipeline.named_steps.get("preprocess")
            model = pipeline.named_steps.get("model")
        except Exception:
            return []
        if pre is None or model is None:
            return []

        try:
            X_trans = pre.transform(df)
            feature_names = list(pre.get_feature_names_out())
        except Exception:
            return []

        try:
            booster = model.get_booster()
        except Exception:
            return []

        try:
            dmat = DMatrix(X_trans, feature_names=feature_names)
            contribs = booster.predict(dmat, pred_contribs=True)
            row = contribs[0]
        except Exception:
            return []

        if len(row) <= 1:
            return []
        # The last contribution term is the model bias, not a feature, so exclude it.
        feature_contribs = row[:-1]

        aggregated: Dict[str, float] = {}
        for name, val in zip(feature_names, feature_contribs):
            # One-hot encoded columns are mapped back to their original base feature for readability.
            base = self._map_base_feature(name, feature_columns)
            aggregated[base] = aggregated.get(base, 0.0) + float(val)

        ranked = sorted(aggregated.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        factors: List[PredictionFactor] = []
        for feat, contrib in ranked:
            direction = "increases_risk" if contrib >= 0 else "decreases_risk"
            factors.append(
                PredictionFactor(
                    feature=feat,
                    direction=direction,
                    contribution=float(contrib),
                    reasoning=f"Contribution {contrib:+.4f} toward churn prediction.",
                )
            )
        return factors

    def _map_base_feature(self, name: str, feature_columns: List[str]) -> str:
        base = name.split("__", 1)[-1]
        if base in feature_columns:
            return base
        # Prefer longest prefix match (helps one-hot encoded features)
        for feat in sorted(feature_columns, key=len, reverse=True):
            if base.startswith(f"{feat}_") or base.startswith(f"{feat}=") or base.startswith(f"{feat}__"):
                return feat
        return base.split("_", 1)[0] if "_" in base else base
