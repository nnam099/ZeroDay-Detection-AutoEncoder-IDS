import asyncio
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import httpx
import torch
from fastapi.testclient import TestClient
from sklearn.preprocessing import LabelEncoder, RobustScaler


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ids.models import IDSModel
from src.serve import app, load_artifacts_from_env


class ServeApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_model_path = os.environ.get("IDS_MODEL_PATH")
        self.old_pipeline_path = os.environ.get("IDS_PIPELINE_PATH")

        model_path = Path(self.tmp.name) / "ids_v14_model.pth"
        pipeline_path = Path(self.tmp.name) / "ids_v14_pipeline.pkl"
        self._write_dummy_artifacts(model_path, pipeline_path)

        os.environ["IDS_MODEL_PATH"] = str(model_path)
        os.environ["IDS_PIPELINE_PATH"] = str(pipeline_path)

    def tearDown(self):
        if self.old_model_path is None:
            os.environ.pop("IDS_MODEL_PATH", None)
        else:
            os.environ["IDS_MODEL_PATH"] = self.old_model_path

        if self.old_pipeline_path is None:
            os.environ.pop("IDS_PIPELINE_PATH", None)
        else:
            os.environ["IDS_PIPELINE_PATH"] = self.old_pipeline_path

        self.tmp.cleanup()

    def test_health_with_test_client(self):
        with TestClient(app) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "model_version": "v14"})

    def test_predict_with_async_client(self):
        async def run_request():
            with TestClient(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    return await client.post("/predict", json={"features": [0.0] * 55})

        response = asyncio.run(run_request())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(set(payload), {"label", "confidence", "ae_re", "hybrid_score", "is_anomaly", "uncertainty"})
        self.assertIsInstance(payload["label"], str)
        self.assertIsInstance(payload["confidence"], float)
        self.assertIsInstance(payload["ae_re"], float)
        self.assertIsInstance(payload["hybrid_score"], float)
        self.assertIsInstance(payload["is_anomaly"], bool)
        self.assertEqual(set(payload["uncertainty"]), {"entropy", "std_max_class"})
        self.assertIsInstance(payload["uncertainty"]["entropy"], float)
        self.assertIsInstance(payload["uncertainty"]["std_max_class"], float)

    def test_predict_rejects_wrong_feature_count(self):
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": [0.0] * 54})

        self.assertEqual(response.status_code, 400)
        self.assertIn("expected 55 features", response.json()["detail"])

    def test_predict_rejects_non_finite_features(self):
        with TestClient(app) as client:
            response = client.post(
                "/predict",
                content='{"features":[NaN,0.0,0.0]}',
                headers={"content-type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("finite numeric", response.json()["detail"])

    def test_predict_rejects_non_numeric_feature_values(self):
        with TestClient(app) as client:
            response = client.post("/predict", json={"features": ["not-a-number"] * 55})

        self.assertEqual(response.status_code, 422)

    def test_artifact_loader_rejects_missing_paths(self):
        missing_model = Path(self.tmp.name) / "missing-model.pth"
        os.environ["IDS_MODEL_PATH"] = str(missing_model)

        with self.assertRaisesRegex(RuntimeError, "IDS_MODEL_PATH does not exist"):
            load_artifacts_from_env()

    def _write_dummy_artifacts(self, model_path: Path, pipeline_path: Path):
        n_features = 55
        n_classes = 2
        model = IDSModel(n_features=n_features, n_classes=n_classes, hidden=16, ae_hidden=8)
        scaler = RobustScaler().fit([[0.0] * n_features, [1.0] * n_features])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "n_features": n_features,
                "n_classes": n_classes,
                "hidden": 16,
                "ae_hidden": 8,
                "thresholds": {"decision_mode": "vote", "min_votes": 2, "hybrid": 999.0, "ae_re": 999.0},
                "version": "v14.0",
            },
            model_path,
        )
        with open(pipeline_path, "wb") as handle:
            pickle.dump(
                {
                    "scaler": scaler,
                    "label_encoder": label_encoder,
                    "feature_names": [f"f{i}" for i in range(n_features)],
                    "thresholds": {"decision_mode": "vote", "min_votes": 2, "hybrid": 999.0, "ae_re": 999.0},
                    "version": "v14.0",
                },
                handle,
            )


if __name__ == "__main__":
    unittest.main()
