# Architecture - IDS v14

## Data Flow

1. Load UNSW-NB15 CSV files from `data/`.
2. Normalize labels and engineer numeric/categorical flow features.
3. Split known classes for supervised training and hold out selected classes as zero-day/OOD traffic.
4. Train a hybrid model with classifier, contrastive projection and autoencoder heads.
5. Calibrate thresholds on validation data.
6. Save model weights plus preprocessing pipeline metadata in `checkpoints/`.
7. Dashboard loads the artifacts, validates their contract and runs single-alert or batch inference.

## Model

`IDSBackbone`

```text
Linear(n_features -> 256) -> LayerNorm -> GELU -> ResBlock x 3
```

Main heads:

| Component | Output | Purpose |
| --- | --- | --- |
| `classifier` | `n_classes` | Known-attack classification |
| `proj_head` | 64 | Supervised contrastive representation |
| `autoencoder` | `n_features` | Reconstruction error for anomaly/OOD scoring |

Training objective:

```text
FocalLoss + 0.3 * SupConLoss + 0.5 * AE_MSE
```

## Runtime Guards

- `artifact_validator.py` checks feature count, class count, scaler metadata and threshold metadata before inference.
- `inference_runtime.py` contains pure verdict, zero-day decision and risk helpers used by the dashboard.
- `scripts/smoke_check.py` compiles the code and runs unit smoke tests.
- `llm_agent.py` is lazy-initialized so importing the dashboard does not require an API key.
