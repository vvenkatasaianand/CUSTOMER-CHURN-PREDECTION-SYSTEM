from __future__ import annotations

"""Builds dataset and training summaries, mixing computed stats with optional LLM narration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from starlette import status
from pydantic import BaseModel, ValidationError

from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import get_logger
from app.schemas.common import FeatureImportance
from app.schemas.insights import (
    ClassBalance,
    ColumnMissing,
    DatasetSummaryResponse,
    DatasetSummaryStats,
    TrainingSummaryResponse,
)
from app.services.llm_service import LLMService
from app.storage.dataset_store import DatasetStore
from app.storage.metadata_store import MetadataStore
from app.storage.model_store import ModelStore


logger = get_logger(__name__)


@dataclass(frozen=True)
class InsightsService:
    settings: Settings
    dataset_store: DatasetStore
    metadata_store: MetadataStore
    model_store: ModelStore

    async def dataset_summary(self, upload_id: str) -> DatasetSummaryResponse:
        # Dataset summary depends on preprocess metadata because that metadata defines target/excluded columns.
        prep = self.metadata_store.read_preprocess_metadata(upload_id)
        if prep is None:
            raise AppError(
                "Dataset not preprocessed. Please run preprocess step first.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="preprocess_required",
            )

        cached = prep.get("dataset_summary")
        if isinstance(cached, dict):
            try:
                # Reuse cached summaries to avoid repeated LLM calls for the same dataset.
                return DatasetSummaryResponse(**cached)
            except Exception:
                pass

        return self.build_dataset_summary(upload_id)

    def build_dataset_summary(self, upload_id: str) -> DatasetSummaryResponse:
        # This method recomputes the full dataset summary from saved preprocess output.
        prep = self.metadata_store.read_preprocess_metadata(upload_id)
        if prep is None:
            raise AppError(
                "Dataset not preprocessed. Please run preprocess step first.",
                status_code=status.HTTP_400_BAD_REQUEST,
                code="preprocess_required",
            )

        processed_path_str = prep.get("processed_path")
        if not processed_path_str:
            raise AppError(
                "Processed dataset path missing. Please preprocess again.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="processed_path_missing",
            )

        processed_path = self.dataset_store.path_from_string(processed_path_str)
        if not processed_path.exists():
            raise AppError(
                "Processed dataset file missing on server. Please preprocess again.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="processed_file_missing",
            )

        df = self.dataset_store.load_processed_dataframe(processed_path)
        target_column = str(prep.get("target_column") or "")
        if not target_column or target_column not in df.columns:
            raise AppError(
                "Target column not found in processed dataset.",
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="target_missing",
            )

        excluded = set([str(x) for x in (prep.get("excluded_columns") or []) if x])
        if target_column in excluded:
            excluded.remove(target_column)
        # Summary should reflect the actual training view of the dataset, not excluded columns.
        df_summary = df.drop(columns=list(excluded), errors="ignore")

        stats, patterns, risks = self._build_dataset_insights(
            df_summary, target_column, upload_id, excluded
        )

        llm = LLMService(settings=self.settings)
        logger.info(
            "dataset_summary_llm_start",
            extra={"upload_id": upload_id, "llm_enabled": llm.is_enabled(), "llm_model": llm.model_name()},
        )
        summary, explanation, llm_used = self._dataset_llm_summary(
            llm=llm,
            stats=stats,
            patterns=patterns,
            risks=risks,
        )
        logger.info(
            "dataset_summary_llm_result",
            extra={"upload_id": upload_id, "llm_used": llm_used, "llm_model": llm.model_name()},
        )

        return DatasetSummaryResponse(
            upload_id=upload_id,
            summary=summary,
            explanation=explanation,
            patterns=patterns,
            risks=risks,
            stats=stats,
            llm_used=llm_used,
            llm_model=llm.model_name(),
        )

    async def training_summary(self, model_id: str) -> TrainingSummaryResponse:
        # Training summaries are generated entirely from saved model metadata.
        meta = self.metadata_store.read_model_metadata(model_id)
        if meta is None:
            raise AppError(
                "Unknown model_id. Please train the model again.",
                status_code=status.HTTP_404_NOT_FOUND,
                code="model_not_found",
            )

        metrics = meta.get("metrics") or {}
        fi_raw = meta.get("feature_importance") or []
        top_features = [
            FeatureImportance(feature=str(x.get("feature")), importance=float(x.get("importance", 0.0)))
            for x in fi_raw
            if isinstance(x, dict) and x.get("feature") is not None
        ]
        # Keep only the top few features because the UI summary is meant to be read quickly.
        top_features = sorted(top_features, key=lambda x: x.importance, reverse=True)[:5]

        risks = self._training_risks(metrics)

        llm = LLMService(settings=self.settings)
        summary, metrics_summary, llm_used = self._training_llm_summary(
            llm=llm,
            model_name=str(meta.get("model_name") or "model"),
            target=str(meta.get("target_column") or "target"),
            metrics=metrics,
            top_features=top_features,
            risks=risks,
        )

        return TrainingSummaryResponse(
            model_id=model_id,
            summary=summary,
            metrics_summary=metrics_summary,
            top_features=top_features,
            risks=risks,
            llm_used=llm_used,
            llm_model=llm.model_name(),
        )

    def _build_dataset_insights(
        self,
        df: pd.DataFrame,
        target_column: str,
        upload_id: str,
        excluded: Optional[set] = None,
    ) -> Tuple[DatasetSummaryStats, List[str], List[str]]:
        rows, cols = int(df.shape[0]), int(df.shape[1])
        missing_cells = int(df.isna().sum().sum())
        total_cells = max(1, rows * cols)
        missing_pct = float(missing_cells / total_cells)

        # These computed stats are the factual base used both by the UI and by the LLM prompt.
        target_series = df[target_column]
        class_balance = self._class_balance(target_series)

        missing_by_col = df.isna().sum().sort_values(ascending=False)
        top_missing = [
            ColumnMissing(
                column=str(col),
                null_count=int(cnt),
                null_pct=float(cnt / max(1, rows)),
            )
            for col, cnt in missing_by_col.head(5).items()
            if int(cnt) > 0
        ]

        patterns = self._detect_patterns(df, target_column)
        risks = self._dataset_risks(
            df=df,
            target_column=target_column,
            missing_pct=missing_pct,
            class_balance=class_balance,
            rows=rows,
            upload_id=upload_id,
            excluded=excluded,
        )

        stats = DatasetSummaryStats(
            rows=rows,
            cols=cols,
            target_column=target_column,
            missing_cells=missing_cells,
            missing_pct=missing_pct,
            class_balance=class_balance,
            top_missing=top_missing,
        )
        return stats, patterns, risks

    def _class_balance(self, series: pd.Series) -> List[ClassBalance]:
        clean = series.dropna()
        if clean.empty:
            return []
        counts = clean.value_counts(dropna=True)
        total = int(counts.sum())
        return [
            ClassBalance(label=str(label), count=int(cnt), pct=float(cnt / total))
            for label, cnt in counts.items()
        ]

    def _target_as_numeric(self, series: pd.Series) -> Optional[pd.Series]:
        clean = series.dropna()
        if clean.empty:
            return None
        if pd.api.types.is_bool_dtype(clean):
            return clean.astype(int)
        if pd.api.types.is_numeric_dtype(clean):
            return clean.astype(float)
        # Handle simple binary categories
        uniques = clean.astype(str).unique().tolist()
        if len(uniques) == 2:
            mapping = {uniques[0]: 0.0, uniques[1]: 1.0}
            return clean.astype(str).map(mapping).astype(float)
        return None

    def _detect_patterns(self, df: pd.DataFrame, target_column: str) -> List[str]:
        # Keep this heuristic on purpose: it surfaces interesting patterns without pretending to be causal analysis.
        patterns: List[str] = []
        target_num = self._target_as_numeric(df[target_column])
        overall_rate = None
        if target_num is not None:
            try:
                overall_rate = float(target_num.mean())
            except Exception:
                overall_rate = None

        # Numeric correlations give a quick sense of directional relationship with the target.
        if target_num is not None:
            numeric_cols = [
                c
                for c in df.columns
                if c != target_column and pd.api.types.is_numeric_dtype(df[c])
            ]
            corrs: List[Tuple[str, float]] = []
            for col in numeric_cols:
                try:
                    corr = float(df[col].corr(target_num))
                except Exception:
                    continue
                if np.isnan(corr):
                    continue
                corrs.append((col, corr))
            corrs.sort(key=lambda x: abs(x[1]), reverse=True)
            for col, corr in corrs[:3]:
                patterns.append(f"{col} correlates with target at {corr:.2f}.")

        # For categorical columns, compare each common category's churn rate to the overall churn rate.
        if overall_rate is not None:
            cat_cols = [
                c
                for c in df.columns
                if c != target_column and not pd.api.types.is_numeric_dtype(df[c])
            ]
            cat_patterns: List[Tuple[str, float]] = []
            for col in cat_cols:
                try:
                    counts = df[col].astype(str).value_counts()
                    for val, cnt in counts.head(5).items():
                        if int(cnt) < 5:
                            continue
                        mask = df[col].astype(str) == val
                        rate = float(target_num[mask].mean())
                        diff = rate - overall_rate
                        cat_patterns.append((f"{col}={val} shows churn rate {rate:.2f}.", diff))
                except Exception:
                    continue
            cat_patterns.sort(key=lambda x: abs(x[1]), reverse=True)
            for text, _diff in cat_patterns[:3]:
                patterns.append(text)

        return patterns[:5]

    def _dataset_risks(
        self,
        df: pd.DataFrame,
        target_column: str,
        missing_pct: float,
        class_balance: List[ClassBalance],
        rows: int,
        upload_id: str,
        excluded: Optional[set] = None,
    ) -> List[str]:
        risks: List[str] = []
        excluded = excluded or set()

        # These checks flag common data quality/model quality problems for student projects.
        if rows < 100:
            risks.append(f"Small dataset size (rows={rows}) may reduce generalization.")

        if missing_pct >= 0.1:
            risks.append(f"Overall missing rate is {missing_pct:.2%}; consider imputation or cleaning.")

        if class_balance:
            max_share = max(x.pct for x in class_balance)
            if max_share >= 0.75:
                risks.append("Class imbalance detected; consider stratified evaluation or rebalancing.")

        upload_meta = self.metadata_store.read_upload_metadata(upload_id) or {}
        cols_info = upload_meta.get("columns") or []
        for col in cols_info:
            try:
                name = str(col.get("name") or "")
                unique_count = int(col.get("unique_count") or 0)
            except Exception:
                continue
            if name in excluded:
                continue
            if unique_count >= 100:
                risks.append(f"High-cardinality feature '{name}' may overfit.")
                break

        target_lower = target_column.lower()
        for col in df.columns:
            name = str(col).lower()
            if col == target_column:
                continue
            if target_lower in name:
                risks.append(f"Potential leakage: feature '{col}' includes target keyword.")
                break
            if any(k in name for k in ("id", "email", "phone")):
                risks.append(f"Identifier-like feature '{col}' may leak or reduce generalization.")
                break

        return risks[:5]

    def _training_risks(self, metrics: Dict[str, Any]) -> List[str]:
        # Convert numeric metrics into plain-language warnings the UI can show directly.
        risks: List[str] = []
        acc = (metrics.get("accuracy") or {}).get("value")
        recall = (metrics.get("recall") or {}).get("value")
        precision = (metrics.get("precision") or {}).get("value")

        try:
            if acc is not None and float(acc) < 0.7:
                risks.append("Overall accuracy is modest; consider more data or tuning.")
        except Exception:
            pass
        try:
            if recall is not None and float(recall) < 0.6:
                risks.append("Recall is low; high-risk churn cases may be missed.")
        except Exception:
            pass
        try:
            if precision is not None and float(precision) < 0.6:
                risks.append("Precision is low; false positives may be high.")
        except Exception:
            pass
        return risks[:4]

    def _dataset_llm_summary(
        self,
        llm: LLMService,
        stats: DatasetSummaryStats,
        patterns: List[str],
        risks: List[str],
    ) -> Tuple[str, str, bool]:
        class _DatasetLLMOutput(BaseModel):
            summary: str
            explanation: str

        def _word_count(text: str) -> int:
            return len([w for w in text.strip().split() if w])

        # Only pass structured facts so the LLM is constrained to describe what we already computed.
        facts = {
            "rows": stats.rows,
            "cols": stats.cols,
            "target_column": stats.target_column,
            "missing_pct": round(stats.missing_pct, 4),
            "class_balance": [cb.model_dump() for cb in stats.class_balance],
            "patterns": patterns,
            "risks": risks,
        }

        # prompt = (
        #     "You are a data analyst. Use ONLY the facts provided. "
        #     "Return JSON with keys: summary, explanation. "
        #     "summary: about 50 words (45-55). "
        #     "explanation: 2-4 sentences, must mention rows, cols, target, missing_pct, class balance. "
        #     "Do not add new numbers or features. "
        #     f"FACTS={facts}"
        # )

        prompt = (
            "You are an experienced data analyst reviewing a dataset summary. "
            "Use only the facts provided below. Do not infer, assume, or introduce any information that is not explicitly stated. "
            "Your task is to describe both the structure of the dataset and what it appears to be about, based strictly on the given facts. "
            "Return valid JSON with exactly two keys: summary and explanation. "
            "The summary should be a clear, high-level overview of the dataset in about 100 words (80–120 words), "
            "including what the dataset seems to represent or analyze. "
            "The explanation should be 2 to 4 concise sentences and must explicitly mention: number of rows, number of columns, "
            "the target variable, percentage of missing values, and class balance. "
            "Do not add new metrics, features, or numbers, and do not speculate beyond the provided facts. "
            f"FACTS={facts}"
        )

        # Fallback text keeps the feature usable even when the LLM is disabled or malformed.
        fallback_summary = (
            f"Dataset has {stats.rows} rows and {stats.cols} columns with target '{stats.target_column}'. "
            f"Overall missing rate is {stats.missing_pct:.2%}, and class balance is {', '.join([f'{c.label}:{c.pct:.0%}' for c in stats.class_balance]) or 'unknown'}. "
            "This summary reflects computed statistics only, emphasizing dataset scale, missingness, target distribution, and quality for clear reporting. "
            "It avoids subjective interpretation and supports academic documentation and enables comparison across versions and cohorts for audit readiness."
        )
        balance_text = ", ".join([f"{c.label}:{c.pct:.0%}" for c in stats.class_balance]) or "unknown"
        fallback_expl = (
            f"The dataset contains {stats.rows} rows and {stats.cols} columns. "
            f"Target is '{stats.target_column}'. Missing rate is {stats.missing_pct:.2%}. "
            f"Class balance is {balance_text}."
        )

        data = llm.generate_json(prompt) if llm.is_enabled() else None
        if not llm.is_enabled():
            logger.info("dataset_summary_llm_disabled")
        if not isinstance(data, dict):
            logger.warning("dataset_summary_llm_invalid_response", extra={"data_type": type(data).__name__})
            return fallback_summary, fallback_expl, False
        try:
            parsed = _DatasetLLMOutput.model_validate(data)
        except ValidationError:
            logger.warning("dataset_summary_llm_validation_error")
            return fallback_summary, fallback_expl, False
        summary = parsed.summary.strip()
        explanation = parsed.explanation.strip()
        if not summary or not explanation:
            logger.warning("dataset_summary_llm_empty_fields")
            return fallback_summary, fallback_expl, False
        return summary, explanation, True

    def _training_llm_summary(
        self,
        llm: LLMService,
        model_name: str,
        target: str,
        metrics: Dict[str, Any],
        top_features: List[FeatureImportance],
        risks: List[str],
    ) -> Tuple[str, str, bool]:
        class _TrainingLLMOutput(BaseModel):
            summary: str
            metrics_summary: str

        # Training summary prompt also stays strictly grounded in saved metrics and top features.
        facts = {
            "model_name": model_name,
            "target": target,
            "metrics": metrics,
            "top_features": [x.model_dump() for x in top_features],
            "risks": risks,
        }

        prompt = (
            "You are a data analyst. Use ONLY the facts provided. "
            "Return JSON with keys: summary, metrics_summary. "
            "summary: <=2 sentences. "
            "metrics_summary: 1-2 sentences and must cite accuracy/precision/recall/f1 if present. "
            "Do not add new numbers or features. "
            f"FACTS={facts}"
        )

        fallback_summary = (
            f"The {model_name} model predicts '{target}' using {len(top_features)} top-ranked features."
        )
        acc = (metrics.get("accuracy") or {}).get("value")
        prec = (metrics.get("precision") or {}).get("value")
        rec = (metrics.get("recall") or {}).get("value")
        f1 = (metrics.get("f1") or {}).get("value")
        fallback_metrics = (
            "Metrics available: "
            f"accuracy={acc:.3f}, precision={prec:.3f}, recall={rec:.3f}, f1={f1:.3f}."
            if all(v is not None for v in (acc, prec, rec, f1))
            else "Metrics are not fully available for this run."
        )

        data = llm.generate_json(prompt) if llm.is_enabled() else None
        if not isinstance(data, dict):
            return fallback_summary, fallback_metrics, False
        try:
            parsed = _TrainingLLMOutput.model_validate(data)
        except ValidationError:
            return fallback_summary, fallback_metrics, False
        summary = parsed.summary.strip()
        metrics_summary = parsed.metrics_summary.strip()
        if not summary or not metrics_summary:
            return fallback_summary, fallback_metrics, False
        return summary, metrics_summary, True
