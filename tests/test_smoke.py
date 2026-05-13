import os
import pickle
import sys
import unittest

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, RobustScaler


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class CoreSmokeTests(unittest.TestCase):
    def test_artifact_validator_accepts_matching_metadata(self):
        from artifact_validator import validate_artifact_contract

        checkpoint = {
            "model_state_dict": {},
            "n_features": 3,
            "n_classes": 2,
            "version": "v14.0",
            "thresholds": {"hybrid": 0.5},
        }
        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        pipeline = {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_names": ["dur", "sbytes", "dbytes"],
            "version": "v14.0",
        }

        result = validate_artifact_contract(checkpoint, pipeline)

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_artifact_validator_rejects_feature_mismatch(self):
        from artifact_validator import validate_artifact_contract

        checkpoint = {"model_state_dict": {}, "n_features": 4, "n_classes": 2}
        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        pipeline = {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_names": ["dur", "sbytes", "dbytes"],
        }

        result = validate_artifact_contract(checkpoint, pipeline)

        self.assertFalse(result.ok)
        self.assertTrue(any("feature count mismatch" in err for err in result.errors))

    def test_log_normalizer_maps_common_firewall_csv(self):
        from log_normalizer import normalize_real_world_logs

        df = pd.DataFrame(
            {
                "src_ip": ["10.0.0.5", "10.0.0.5"],
                "dst_ip": ["172.16.1.10", "172.16.1.11"],
                "src_port": [52512, 52513],
                "dst_port": [443, 443],
                "protocol": ["tcp", "tcp"],
                "duration": [1.5, 0.2],
                "sent_bytes": [1200, 80],
                "received_bytes": [3200, 40],
                "sent_pkts": [10, 2],
                "received_pkts": [14, 1],
                "action": ["CON", "RST"],
            }
        )

        normalized, report = normalize_real_world_logs(df)

        self.assertEqual(len(normalized), 2)
        for col in ["proto", "state", "dur", "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload"]:
            self.assertIn(col, normalized.columns)
        self.assertGreaterEqual(report.feature_coverage, 0.7)
        self.assertEqual(report.schema, "firewall_or_flow_csv")

    def test_mitre_mapper_known_attack(self):
        from mitre_mapper import MITREMapper

        result = MITREMapper().map_known_attack(1, class_names=["Normal", "DoS"])

        self.assertEqual(result["attack_class"], "DoS")
        self.assertEqual(result["mapping_mode"], "known_attack")
        self.assertTrue(result["techniques"])

    def test_v14_artifacts_load_when_present(self):
        model_path = os.path.join(ROOT_DIR, "checkpoints", "ids_v14_model.pth")
        pipeline_path = os.path.join(ROOT_DIR, "checkpoints", "ids_v14_pipeline.pkl")
        if not (os.path.exists(model_path) and os.path.exists(pipeline_path)):
            self.skipTest("v14 checkpoint/pipeline artifacts are not present")

        from ids_v14_unswnb15 import IDSModel

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        feature_names = pipeline.get("feature_names", pipeline.get("feat_cols", []))
        self.assertEqual(checkpoint["n_features"], len(feature_names))
        self.assertEqual(checkpoint["n_classes"], len(pipeline["label_encoder"].classes_))

        model = IDSModel(
            n_features=checkpoint["n_features"],
            n_classes=checkpoint["n_classes"],
            hidden=checkpoint.get("hidden", 256),
            ae_hidden=checkpoint.get("ae_hidden", 128),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        x = torch.zeros(2, checkpoint["n_features"])
        with torch.no_grad():
            logits, features = model(x)
            ae_score = model.ae.recon_error(x)

        self.assertEqual(tuple(logits.shape), (2, checkpoint["n_classes"]))
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(ae_score.shape), (2,))


if __name__ == "__main__":
    unittest.main()
