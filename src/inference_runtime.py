from __future__ import annotations

import numpy as np


def zero_day_decision(
    ae_score,
    max_prob,
    hybrid_score,
    thresholds: dict | None = None,
    ae_threshold: float = 0.5,
    hybrid_threshold: float = 0.5,
):
    if thresholds and thresholds.get("decision_mode") == "vote":
        min_votes = int(thresholds.get("min_votes", 2))
        votes = []
        if "hybrid" in thresholds:
            votes.append(np.asarray(hybrid_score) > float(thresholds["hybrid"]))
        if "ae_re" in thresholds:
            votes.append(np.asarray(ae_score) > float(thresholds["ae_re"]))
        if "softmax" in thresholds:
            votes.append((1.0 - np.asarray(max_prob)) > float(thresholds["softmax"]))
        if votes:
            vote_count = np.sum(np.stack(votes, axis=0), axis=0)
            return vote_count >= min_votes, f"vote_{min_votes}_of_{len(votes)}"

    if thresholds and "hybrid" in thresholds:
        return np.asarray(hybrid_score) > hybrid_threshold, "hybrid_calibrated"
    return (np.asarray(ae_score) > ae_threshold) & (np.asarray(max_prob) < 0.6), "ae_plus_confidence_fallback"


def traffic_verdict(is_zeroday, classifier_class) -> str:
    if bool(is_zeroday):
        return "Zero-Day"
    if str(classifier_class).strip().lower() == "normal":
        return "Normal"
    return "Known-Attack"


def ground_truth_verdict(label) -> str:
    value = str(label).strip().lower()
    if value in {"", "nan", "none", "-", "unknown"}:
        return ""
    if value in {"0", "benign", "normal"}:
        return "Normal"
    return "Known-Attack"


def severity_rank(severity: str) -> int:
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(str(severity).upper(), 0)


def severity_class(severity: str) -> str:
    return {
        "CRITICAL": "sev-critical",
        "HIGH": "sev-high",
        "MEDIUM": "sev-medium",
        "LOW": "sev-low",
    }.get(str(severity).upper(), "sev-medium")


def risk_score(result: dict, severity: str | None = None) -> int:
    sev_weight = {"CRITICAL": 35, "HIGH": 25, "MEDIUM": 15, "LOW": 5}.get(str(severity or "").upper(), 10)
    hybrid = float(result.get("hybrid_score", 0)) * 45
    ae = min(float(result.get("ae_score", 0)), 1.0) * 15
    zd = 15 if result.get("is_zeroday") else 0
    return int(min(100, round(sev_weight + hybrid + ae + zd)))


def assess_normalization_quality(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {
            "level": "UNKNOWN",
            "message": "No CSV normalization report is available.",
            "warnings": ["Normalization did not run or did not return metadata."],
        }

    coverage = report.get("feature_coverage")
    try:
        coverage = float(coverage)
    except (TypeError, ValueError):
        coverage = None

    missing = list(report.get("missing_core_features") or [])
    mapped_count = len(report.get("mapped_columns") or {})
    warnings = []

    if coverage is None:
        level = "UNKNOWN"
        warnings.append("Feature coverage is unavailable.")
    elif coverage >= 0.80:
        level = "GOOD"
    elif coverage >= 0.60:
        level = "MEDIUM"
        warnings.append("Feature coverage is moderate; treat scores as triage signals.")
    else:
        level = "LOW"
        warnings.append("Feature coverage is low; inference confidence is reduced.")

    directional = {"sbytes", "dbytes", "spkts", "dpkts"}
    missing_directional = sorted(directional & set(missing))
    if missing_directional:
        warnings.append("Missing directional counters: " + ", ".join(missing_directional))

    timing = {"dur", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat"}
    missing_timing = sorted(timing & set(missing))
    if missing_timing:
        warnings.append("Missing timing/TCP fields: " + ", ".join(missing_timing[:6]))

    if mapped_count < 5:
        warnings.append("Few source columns were mapped; verify CSV schema manually.")

    if not warnings:
        warnings.append("CSV schema has enough core fields for prototype inference.")

    cov_text = "unknown" if coverage is None else f"{coverage * 100:.1f}%"
    return {
        "level": level,
        "coverage": coverage,
        "mapped_columns": mapped_count,
        "message": f"Normalization quality: {level} (coverage {cov_text}, mapped columns {mapped_count}).",
        "warnings": warnings,
    }
