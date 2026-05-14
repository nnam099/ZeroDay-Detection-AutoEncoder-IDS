from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np
import pandas as pd

from inference_runtime import traffic_verdict


Normalizer = Callable[[pd.DataFrame], tuple[pd.DataFrame, Any]]


@dataclass
class DashboardPreprocessResult:
    features: np.ndarray
    normalization_report: dict[str, Any] | None = None


def preprocess_dashboard_df(
    df_raw: pd.DataFrame,
    feature_names: list[str],
    model_version: str,
    pipeline_meta: dict[str, Any] | None = None,
    normalizer: Normalizer | None = None,
) -> DashboardPreprocessResult:
    """
    Align dashboard input rows to the exact feature contract expected by the scaler.

    The function is deliberately Streamlit-free so CSV preprocessing can be tested
    without importing the dashboard application.
    """
    train_mod = __import__(f"ids_{model_version}_unswnb15", fromlist=["engineer_features"])
    engineer_features = train_mod.engineer_features

    df = df_raw.copy()
    report_dict: dict[str, Any] | None = None
    if normalizer is not None:
        try:
            df, report = normalizer(df)
            report_dict = report.as_dict() if hasattr(report, "as_dict") else dict(report)
        except Exception as exc:
            report_dict = {
                "schema": "normalization_failed",
                "error": str(exc),
            }

    categorical_maps = (pipeline_meta or {}).get("categorical_maps", {})
    for cat in ["proto", "service", "state"]:
        if cat in df.columns:
            df[f"{cat}_num"] = _encode_categorical_column(df[cat], categorical_maps.get(cat))

    existing_numeric = [
        col for col in df.columns
        if col not in {
            "attack_cat", "label", "label_binary", "srcip", "dstip",
            "sport", "dsport", "stime", "ltime", "id", "proto", "service", "state",
        }
    ]
    try:
        df, _ = engineer_features(df, existing_numeric)
    except Exception:
        pass

    rows = len(df)
    out = np.zeros((rows, len(feature_names)), dtype=np.float32)
    for index, col in enumerate(feature_names):
        if col not in df.columns:
            continue
        try:
            out[:, index] = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        except Exception:
            out[:, index] = 0.0

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return DashboardPreprocessResult(features=out, normalization_report=report_dict)


def build_alert_context_from_log(
    row_scores: dict[str, Any],
    timestamp: str | None = None,
) -> dict[str, Any]:
    source_row = int(row_scores.get("source_row", 0))
    family = str(row_scores.get("zero_day_family") or "")
    classifier_class = str(row_scores.get("classifier_class", row_scores.get("predicted_class", "Unknown")))
    detection = str(row_scores.get("detection") or traffic_verdict(row_scores.get("is_zeroday"), classifier_class))
    return {
        "alert_id": f"ZD-{source_row:06d}",
        "timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_row": source_row,
        "hybrid_score": float(row_scores.get("hybrid_score", 0)),
        "ae_score": float(row_scores.get("ae_score", 0)),
        "max_prob": float(row_scores.get("max_prob", 0)),
        "predicted_class": detection,
        "classifier_class": classifier_class,
        "zero_day_family": family or "",
        "is_zeroday": bool(row_scores.get("is_zeroday", False)),
        "shap_summary": "Batch log context - SHAP explanation not computed for this row.",
        "mitre_summary": "",
        "top_features": [],
        "probs": [],
        "demo_mode": False,
        "raw_scores": {key: str(value) for key, value in row_scores.items()},
    }


def _encode_categorical_column(series: pd.Series, mapping=None) -> np.ndarray:
    values = series.astype(str).fillna("unk")
    if mapping:
        return values.map(lambda x: mapping.get(x, mapping.get("unk", -1))).astype(np.float32).values
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.float32)
