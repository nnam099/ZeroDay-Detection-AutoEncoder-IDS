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


@dataclass
class AIContextOption:
    label: str
    context: dict[str, Any]


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


def build_ai_context_options(
    alert_history: list[dict[str, Any]] | None = None,
    bulk_logs: pd.DataFrame | None = None,
    max_history: int = 20,
    max_bulk_logs: int = 50,
) -> list[AIContextOption]:
    options: list[AIContextOption] = []

    for item in (alert_history or [])[-max_history:]:
        options.append(AIContextOption(
            label=(
                f"Alert history | {item.get('alert_id')} | {item.get('predicted_class')} | "
                f"score={float(item.get('hybrid_score', 0)):.3g}"
            ),
            context=item,
        ))

    if isinstance(bulk_logs, pd.DataFrame) and not bulk_logs.empty:
        logs_for_ai = bulk_logs.copy()
        if "source_row" not in logs_for_ai.columns:
            logs_for_ai.insert(0, "source_row", np.arange(len(logs_for_ai)))
        sort_cols = [col for col in ["is_zeroday", "hybrid_score", "ae_score"] if col in logs_for_ai.columns]
        if sort_cols:
            logs_for_ai = logs_for_ai.sort_values(sort_cols, ascending=False)
        for _, row in logs_for_ai.head(max_bulk_logs).iterrows():
            row_dict = row.to_dict()
            options.append(AIContextOption(
                label=(
                    f"Zero-day log | row={int(row_dict.get('source_row', 0))} | "
                    f"{row_dict.get('detection', row_dict.get('predicted_class'))} | "
                    f"hybrid={float(row_dict.get('hybrid_score', 0)):.3g}"
                ),
                context=build_alert_context_from_log(row_dict),
            ))

    return options


def default_ai_context_index(options: list[AIContextOption], current_alert_id: str | None) -> int:
    for index, option in enumerate(options):
        if option.context.get("alert_id") == current_alert_id:
            return index
    return 0


def triage_alert_with_fallback(result: dict[str, Any], agent: Any | None = None) -> dict[str, Any]:
    if agent is None:
        return {
            "severity": "HIGH" if result["hybrid_score"] > 0.6 else "MEDIUM",
            "verdict": (
                f"{'Zero-Day Candidate' if result['is_zeroday'] else result['predicted_class']} "
                f"detected - hybrid score: {result['hybrid_score']:.3f}"
            ),
            "attack_summary": (
                f"AE reconstruction error cao ({result['ae_score']:.3f}) cho thay traffic bat thuong. "
                "Can kiem tra thu cong."
            ),
            "recommended_actions": ["Kiem tra nguon IP", "Xem xet block traffic", "Escalate len Tier 2"],
            "false_positive_risk": "MEDIUM",
            "false_positive_reason": "Chua co LLM de phan tich sau hon",
            "analyst_note": "Kich hoat LLM agent de co phan tich chi tiet hon",
        }
    try:
        return agent.triage_alert(result)
    except Exception as exc:
        return {
            "severity": "HIGH",
            "verdict": "LLM analysis loi - can review thu cong",
            "attack_summary": str(exc),
            "recommended_actions": ["Review thu cong"],
            "false_positive_risk": "UNKNOWN",
            "false_positive_reason": "N/A",
            "analyst_note": f"Loi: {exc}",
        }


def answer_analyst_question(
    question: str,
    alert_context: dict[str, Any],
    has_llm: bool,
    agent_factory: Callable[[], Any] | None = None,
    llm_provider: str | None = None,
    llm_dependency: str | None = None,
    has_llm_dependency: bool = True,
) -> str:
    if has_llm and agent_factory is not None:
        try:
            return str(agent_factory().explain_to_analyst(question, alert_context))
        except Exception as exc:
            return f"Loi LLM Agent: {exc}. Vui long kiem tra lai provider va API Key trong file .env."

    if llm_dependency and not has_llm_dependency:
        return f"Chua cai thu vien cho LLM provider '{llm_provider}'. Can cai: {llm_dependency}."
    return "Khong tim thay llm_agent.py. Vui long kiem tra lai source code."


def _encode_categorical_column(series: pd.Series, mapping=None) -> np.ndarray:
    values = series.astype(str).fillna("unk")
    if mapping:
        return values.map(lambda x: mapping.get(x, mapping.get("unk", -1))).astype(np.float32).values
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.float32)
