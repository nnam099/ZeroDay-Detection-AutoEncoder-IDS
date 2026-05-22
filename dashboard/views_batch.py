from __future__ import annotations

import pandas as pd
import streamlit as st

from ui_safety import render_safety_notice


def render_batch_safety_notice() -> None:
    render_safety_notice()
    st.caption(
        "Batch CSV analysis is best used for prioritization. Low feature coverage or missing flow counters can change OOD rates."
    )


def render_bulk_detection_summary(result_df: pd.DataFrame) -> None:
    total = len(result_df)
    zd_cnt = int(result_df["is_zeroday"].sum())
    st.metric("OOD candidates", zd_cnt)
    st.metric("OOD candidate rate", f"{(zd_cnt / total * 100):.2f}%" if total else "0.00%")
    verdict_counts = (
        result_df["detection"]
        .value_counts()
        .reindex(["Normal", "Known-Attack", "Zero-Day Candidate"], fill_value=0)
        .rename_axis("Label")
        .reset_index(name="Count")
    )
    st.dataframe(verdict_counts, width="stretch", hide_index=True)


def render_ground_truth_summary(result_df: pd.DataFrame) -> None:
    if "ground_truth" not in result_df.columns or not result_df["ground_truth"].astype(str).str.len().any():
        return
    gt_counts = (
        result_df["ground_truth"]
        .value_counts()
        .reindex(["Normal", "Known-Attack"], fill_value=0)
        .rename_axis("Ground Truth")
        .reset_index(name="Count")
    )
    comparable = result_df["correct_vs_ground_truth"].dropna()
    gt_acc = float(comparable.mean()) if len(comparable) else 0.0
    st.metric("Accuracy vs CSV Label", f"{gt_acc * 100:.2f}%")
    st.dataframe(gt_counts, width="stretch", hide_index=True)
