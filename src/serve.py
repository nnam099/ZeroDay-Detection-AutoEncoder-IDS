from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_evaluator import IDSArtifacts, load_ids_artifacts, run_batch_scores

MODEL_VERSION = "v14"


class PredictRequest(BaseModel):
    features: list[float]


def _required_env_path(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    if not os.path.exists(value):
        raise RuntimeError(f"{name} does not exist: {value}")
    return value


def load_artifacts_from_env() -> IDSArtifacts:
    model_path = _required_env_path("IDS_MODEL_PATH")
    pipeline_path = _required_env_path("IDS_PIPELINE_PATH")
    return load_ids_artifacts(model_path, pipeline_path, MODEL_VERSION)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts_from_env()
    yield


app = FastAPI(title="IDS v14 Inference API", version=MODEL_VERSION, lifespan=lifespan)


def _artifacts() -> IDSArtifacts:
    artifacts = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="IDS artifacts are not loaded")
    return artifacts


def _prediction_row(features: list[float], artifacts: IDSArtifacts) -> dict[str, Any]:
    if not features:
        raise HTTPException(status_code=400, detail="features must contain at least one value")

    expected = int(artifacts.checkpoint.get("n_features", len(artifacts.feature_names)))
    if len(features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"expected {expected} features, received {len(features)}",
        )

    raw = np.asarray(features, dtype=np.float32).reshape(1, -1)
    scores = run_batch_scores(raw, artifacts, batch_size=1)
    if scores.empty:
        raise HTTPException(status_code=500, detail="model produced no prediction")
    return scores.iloc[0].to_dict()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, Any]:
    row = _prediction_row(payload.features, _artifacts())
    return {
        "label": str(row.get("predicted_class", "Unknown")),
        "confidence": float(row.get("max_prob", 0.0)),
        "ae_re": float(row.get("ae_re", row.get("ae_score", 0.0))),
        "hybrid_score": float(row.get("hybrid_score", row.get("hybrid", 0.0))),
        "is_anomaly": bool(row.get("is_zeroday", False)),
    }
