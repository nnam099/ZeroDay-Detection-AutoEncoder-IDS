from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_evaluator import load_ids_artifacts, preprocess_raw_df, run_batch_scores, summarize_scores  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate v14 artifact evaluation metrics and plots.")
    parser.add_argument("--csv-path", default=str(ROOT_DIR / "data" / "UNSW_NB15_testing-set.csv"))
    parser.add_argument("--label-col", default="attack_cat")
    parser.add_argument("--model-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_model.pth"))
    parser.add_argument("--pipeline-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_pipeline.pkl"))
    parser.add_argument("--output-json", default=str(ROOT_DIR / "results" / "ids_v14_results.json"))
    parser.add_argument("--plots-dir", default=str(ROOT_DIR / "plots"))
    parser.add_argument("--scores-csv", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.csv_path, nrows=args.max_rows)
    artifacts = load_ids_artifacts(args.model_path, args.pipeline_path, "v14")
    raw_features, normalization_report = preprocess_raw_df(raw_df, artifacts)
    scores = run_batch_scores(raw_features, artifacts)
    summary = summarize_scores(
        scores,
        raw_df=raw_df,
        label_col=args.label_col,
        class_names=artifacts.class_names,
        zero_day_labels=list(artifacts.pipeline.get("zd_cats", [])),
        thresholds=artifacts.thresholds,
    )
    plots = write_evaluation_plots(scores, summary, plots_dir)

    if args.scores_csv:
        scores.to_csv(args.scores_csv, index=False)

    report = {
        "version": "v14.0",
        "status": "artifact_evaluation_regenerated",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "regeneration_mode": "current_artifact_evaluation",
        "input_csv": str(Path(args.csv_path).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "pipeline_path": str(Path(args.pipeline_path).resolve()),
        "n_features": len(artifacts.feature_names),
        "n_classes": len(artifacts.class_names),
        "known_cats": list(artifacts.pipeline.get("known_cats", [])),
        "zd_cats": list(artifacts.pipeline.get("zd_cats", [])),
        "metrics": summary,
        "normalization_report": normalization_report,
        "threshold_profile": summary.get("threshold_profile", {}),
        "plots": plots,
        "limitations_and_safety": {
            "zero_day_candidate_meaning": "OOD hypothesis for analyst review; not a confirmed novel attack.",
            "regeneration_note": "This report evaluates the current saved v14 artifacts. It does not retrain model weights.",
            "required_validation": "Confirm suspicious rows with SIEM, firewall, endpoint and packet evidence.",
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    print(f"Report: {args.output_json}")
    print(f"Rows: {summary.get('rows', 0):,}")
    print(f"Accuracy: {summary.get('accuracy')}")
    print(f"Normal FPR: {summary.get('false_positive_rate')}")
    print(f"OOD detection rate: {summary.get('ood_detection_rate')}")
    return 0


def write_evaluation_plots(scores: pd.DataFrame, summary: dict, plots_dir: Path) -> list[str]:
    paths = []

    verdict_counts = scores["predicted_class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    verdict_counts.plot(kind="bar", ax=ax, color="#39b7e8")
    ax.set_title("v14 Verdict Distribution")
    ax.set_xlabel("Verdict")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=25)
    path = plots_dir / "v14_eval_verdict_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    scores[["hybrid", "ae_re", "softmax"]].plot(kind="box", ax=ax)
    ax.set_title("v14 Score Distribution")
    ax.set_ylabel("Score")
    path = plots_dir / "v14_eval_score_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    recall = summary.get("recall_per_class") or {}
    if recall:
        recall_df = pd.DataFrame([
            {"Class": key, "Recall": value.get("recall", 0.0), "Support": value.get("support", 0)}
            for key, value in recall.items()
        ]).sort_values("Class")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(recall_df["Class"], recall_df["Recall"], color="#42d392")
        ax.set_ylim(0, 1.0)
        ax.set_title("v14 Known-Class Recall")
        ax.set_xlabel("Class")
        ax.set_ylabel("Recall")
        ax.tick_params(axis="x", rotation=25)
        path = plots_dir / "v14_eval_known_class_recall.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    return paths


if __name__ == "__main__":
    raise SystemExit(main())
