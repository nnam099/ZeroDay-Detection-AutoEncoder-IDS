import os
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

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

    def test_artifact_validator_rejects_duplicate_feature_names(self):
        from artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {"model_state_dict": {}, "n_features": 3, "n_classes": 2},
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "dur", "dbytes"],
            },
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("duplicates" in err for err in result.errors))

    def test_artifact_validator_rejects_invalid_thresholds(self):
        from artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {"model_state_dict": {}, "n_features": 3, "n_classes": 2, "thresholds": {"hybrid": -0.1}},
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "sbytes", "dbytes"],
            },
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("non-negative" in err for err in result.errors))

    def test_artifact_validator_allows_negative_energy_threshold(self):
        from artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {
                "model_state_dict": {},
                "n_features": 3,
                "n_classes": 2,
                "thresholds": {"hybrid": 0.5, "energy": -1.2},
            },
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "sbytes", "dbytes"],
            },
        )

        self.assertTrue(result.ok)

    def test_patch_checkpoint_infers_dims_from_state_dict(self):
        from patch_checkpoint import infer_dims

        checkpoint = {
            "model_state_dict": {
                "ae.enc.4.weight": torch.zeros(64, 128),
                "backbone.input_proj.0.weight": torch.zeros(256, 61),
            }
        }

        self.assertEqual(infer_dims(checkpoint), (128, 256))

    def test_export_model_build_config_overrides_defaults(self):
        from export_model import build_config

        args = type(
            "Args",
            (),
            {
                "data_dir": "data",
                "save_dir": "checkpoints",
                "plot_dir": "plots",
                "epochs": 2,
                "patience": 1,
                "batch_size": 16,
                "num_workers": 0,
                "demo": True,
                "seed": 123,
            },
        )()

        cfg = build_config(args)

        self.assertEqual(cfg.data_dir, "data")
        self.assertEqual(cfg.epochs, 2)
        self.assertTrue(cfg.demo)
        self.assertEqual(cfg.seed, 123)

    def test_environment_check_does_not_expose_secret_values(self):
        from scripts.check_environment import assess_readiness, collect_environment

        old_provider = os.environ.get("LLM_PROVIDER")
        old_key = os.environ.get("GROQ_API_KEY")
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "secret-value"
        try:
            env = collect_environment()
        finally:
            if old_provider is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = old_provider
            if old_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old_key

        self.assertTrue(env["llm"]["key_present"])
        self.assertNotIn("secret-value", json.dumps(env))

        readiness = assess_readiness(
            {
                "packages": {"torch": False, "numpy": True, "pandas": True, "sklearn": True, "matplotlib": True, "dotenv": True},
                "artifacts": {},
                "data": {"data_dir": True},
                "llm": {},
            }
        )
        self.assertEqual(readiness["status"], "BLOCKED")
        self.assertTrue(any("torch" in item for item in readiness["blockers"]))

    def test_artifact_manifest_detects_hash_changes(self):
        from scripts.artifact_manifest import build_manifest, verify_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.abspath(tmp)
            os.makedirs(os.path.join(root, "checkpoints"))
            artifact = os.path.join(root, "checkpoints", "ids_v14_model.pth")
            with open(artifact, "wb") as f:
                f.write(b"model-v1")

            manifest = build_manifest(Path(root), ["checkpoints/ids_v14_model.pth"])
            self.assertTrue(verify_manifest(Path(root), manifest)["ok"])

            with open(artifact, "wb") as f:
                f.write(b"model-v2")

            result = verify_manifest(Path(root), manifest)
            self.assertFalse(result["ok"])
            self.assertTrue(any("sha256" in err for err in result["errors"]))

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

    def test_input_guard_rejects_bad_csv_shape(self):
        from input_guard import CSVInputPolicy, validate_uploaded_csv

        empty_df = pd.DataFrame()
        result = validate_uploaded_csv(empty_df, size_bytes=10, policy=CSVInputPolicy(max_rows=10, min_columns=3))

        self.assertFalse(result.ok)
        self.assertTrue(any("no rows" in err for err in result.errors))
        self.assertTrue(any("too few columns" in err for err in result.errors))

        large_df = pd.DataFrame({"a": range(12), "b": range(12), "c": range(12)})
        result = validate_uploaded_csv(large_df, size_bytes=10, policy=CSVInputPolicy(max_rows=10))

        self.assertFalse(result.ok)
        self.assertTrue(any("too many rows" in err for err in result.errors))

        dup_df = pd.DataFrame([[1, 2, 3]], columns=["src_ip", "src_ip", "dst_ip"])
        result = validate_uploaded_csv(dup_df)

        self.assertFalse(result.ok)
        self.assertTrue(any("duplicate column" in err for err in result.errors))

    def test_mitre_mapper_known_attack(self):
        from mitre_mapper import MITREMapper

        result = MITREMapper().map_known_attack(1, class_names=["Normal", "DoS"])

        self.assertEqual(result["attack_class"], "DoS")
        self.assertEqual(result["mapping_mode"], "known_attack")
        self.assertTrue(result["techniques"])

    def test_inference_runtime_verdict_and_risk_helpers(self):
        from inference_runtime import (
            assess_normalization_quality,
            ground_truth_verdict,
            risk_score,
            traffic_verdict,
            zero_day_decision,
        )

        decision, rule = zero_day_decision(0.8, 0.4, 0.7)
        self.assertTrue(bool(decision))
        self.assertEqual(rule, "ae_plus_confidence_fallback")

        decision, rule = zero_day_decision(0.1, 0.9, 0.8, thresholds={"hybrid": 0.5}, hybrid_threshold=0.6)
        self.assertTrue(bool(decision))
        self.assertEqual(rule, "hybrid_calibrated")

        self.assertEqual(traffic_verdict(True, "Normal"), "Zero-Day")
        self.assertEqual(traffic_verdict(False, "Normal"), "Normal")
        self.assertEqual(traffic_verdict(False, "DoS"), "Known-Attack")
        self.assertEqual(ground_truth_verdict("benign"), "Normal")
        self.assertEqual(ground_truth_verdict("Exploit"), "Known-Attack")
        self.assertGreater(risk_score({"hybrid_score": 0.9, "ae_score": 0.8, "is_zeroday": True}, "HIGH"), 80)

        quality = assess_normalization_quality(
            {
                "feature_coverage": 0.52,
                "mapped_columns": {"src_ip": "src_ip", "dst_ip": "dst_ip"},
                "missing_core_features": ["sbytes", "dbytes", "dur"],
            }
        )
        self.assertEqual(quality["level"], "LOW")
        self.assertTrue(any("directional" in warning for warning in quality["warnings"]))

    def test_llm_agent_import_has_no_provider_side_effect(self):
        sys.modules.pop("llm_agent", None)
        buf = StringIO()

        with redirect_stdout(buf):
            import llm_agent

        self.assertEqual(buf.getvalue(), "")
        status = llm_agent.get_llm_status()
        self.assertFalse(status["initialized"])
        self.assertIsNone(status["error"])

    def test_llm_agent_falls_back_without_api_key(self):
        old_provider = os.environ.get("LLM_PROVIDER")
        old_key = os.environ.get("GROQ_API_KEY")
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = ""
        sys.modules.pop("llm_agent", None)
        try:
            from llm_agent import SOCTriageAgent

            result = SOCTriageAgent().triage_alert(
                {
                    "alert_id": "A1",
                    "timestamp": "2026-05-13 00:00:00",
                    "hybrid_score": 0.7,
                    "ae_score": 0.6,
                    "max_prob": 0.4,
                    "predicted_class": "Zero-Day",
                    "is_zeroday": True,
                }
            )
        finally:
            sys.modules.pop("llm_agent", None)
            if old_provider is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = old_provider
            if old_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old_key

        self.assertEqual(result["alert_id"], "A1")
        self.assertIn("severity", result)
        self.assertIn("LLM", result["verdict"])

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
