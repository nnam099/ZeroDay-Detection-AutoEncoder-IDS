# Zero-Day Detection AutoEncoder IDS

Prototype IDS for UNSW-NB15 research and SOC-style demonstrations. The project combines known-attack classification, reconstruction-error based zero-day/OOD detection, Streamlit dashboard analysis, FastAPI inference serving, SHAP explainability, heuristic MITRE ATT&CK mapping and optional LLM triage.

> This is a research/demo project, not a production IDS. Analyst review and supporting SIEM/firewall/endpoint evidence are still required.

## Overview

The training pipeline learns from selected known attack classes and holds out other classes to simulate zero-day or out-of-distribution traffic. The dashboard and FastAPI server load saved artifacts and support single-alert analysis, CSV batch uploads and real-time JSON prediction.

Main flow:

1. Load and clean UNSW-NB15 CSV files.
2. Engineer flow features and encode categorical fields.
3. Train a hybrid supervised classifier + contrastive representation + autoencoder model.
4. Calibrate anomaly/zero-day thresholds.
5. Save checkpoint and preprocessing pipeline artifacts.
6. Analyze alerts in the Streamlit dashboard or call the FastAPI `/predict` endpoint.
7. Attach SHAP, MITRE, uncertainty and optional LLM context for SOC triage.

## Current Status

- v14 is the operational default because local artifacts exist in `checkpoints/`.
- Runtime dependencies are pinned in `requirements.txt`; test-only dependencies are in `requirements-dev.txt`.
- `scripts/smoke_check.py` passes locally with 39 tests.
- Smoke coverage includes artifact contract validation, threshold metadata validation, artifact manifest hashing, duplicate feature-name rejection, environment readiness checks, export config handling, checkpoint metadata patch logic, SQLite alert store persistence, CSV input guardrails, CSV normalization quality checks, dashboard preprocessing/context contracts, AI context selection, alert queue filtering, top-N batch alert selection, alert entity enrichment, lightweight correlation, Recon/DoS prototype separation, LLM fallback behavior, MITRE mapping and v14 artifact loading.
- `llm_agent.py` lazy-loads provider clients, so importing dashboard code does not require an API key.
- A Windows GitHub Actions smoke workflow is available at `.github/workflows/smoke.yml`.
- A Dockerfile is available for FastAPI inference on port `8080`.
- v14 performance metrics have not been regenerated after the latest operational fixes; `results/ids_v14_results.json` records artifact smoke verification only.

## Features

- Known-attack classification for classes such as `Normal`, `DoS`, `Exploits`, `Reconnaissance` and `Generic`.
- Zero-day/OOD detection using reconstruction error, optional adaptive AE thresholding, classifier confidence and a learned logistic-regression hybrid score.
- Streamlit SOC dashboard for single alert review, CSV batch analysis, alert history and AI follow-up.
- FastAPI inference server for real-time JSON predictions from saved v14 artifacts.
- Monte Carlo Dropout uncertainty for served predictions, with `LOW_CONFIDENCE` labeling when entropy is high.
- SQLite alert store for persisted queue history, status, analyst notes, queue filtering, top-N batch alert persistence and lightweight correlation groups.
- Real-world CSV normalization for common firewall/flow/Zeek/Suricata-like exports.
- CSV upload guardrails for empty, oversized or malformed files.
- SHAP top-feature explanation.
- Heuristic MITRE ATT&CK mapping.
- Optional LLM triage through Groq, Gemini, OpenAI or Anthropic.

## Repository Layout

```text
src/
  train.py                 # v14 training entry point
  ids/                     # v14 model, dataset, losses, trainer, evaluator, thresholds and plots
  ids_v14_unswnb15.py      # compatibility wrapper for older imports/commands
  ids_v15_unswnb15.py      # experimental v15 pipeline
  inference_runtime.py     # pure verdict/zero-day/risk/CSV-quality helpers used by dashboard
  serve.py                 # FastAPI v14 inference server
  dashboard_runtime.py     # Streamlit-free dashboard preprocessing, AI context and fallback helpers
  alert_store.py           # SQLite alert history, status and analyst note persistence
  batch_evaluator.py       # CLI-friendly batch inference, CSV reporting and threshold calibration
  input_guard.py           # uploaded CSV size/shape validation
  artifact_validator.py    # checkpoint/pipeline contract validation
  explainer.py             # SHAP explainer
  log_normalizer.py        # real-world CSV normalization
  mitre_mapper.py          # heuristic MITRE ATT&CK mapping
  llm_agent.py             # optional lazy-loaded LLM triage
dashboard/
  app.py                   # Streamlit SOC dashboard
configs/
  config_default.yaml      # default v15 config
docs/
  architecture.md
  operations.md
  real_world_csv.md
  project_audit.md
tests/
  test_smoke.py
scripts/
  check_environment.py
  smoke_check.py
  install_requirements.ps1
  quality_check.ps1
  run_dashboard.ps1
  evaluate_csv.py
  train_v14.ps1
  train_v15.ps1
  train.sh
  train_v15.sh
```

## Setup

Recommended:

- Python 3.9+ for local work. Python 3.11 is used by CI.
- Windows PowerShell or a Unix-like shell.

Clone the correct repository:

```powershell
git clone https://github.com/nnam099/ZeroDay-Detection-AutoEncoder-IDS.git
cd ZeroDay-Detection-AutoEncoder-IDS
```

Create/activate a virtual environment, then install dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\install_requirements.ps1
```

Manual install:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python -m pip install --progress-bar off -r requirements.txt
```

Minimal dependencies for smoke checks only:

```powershell
python -m pip install --progress-bar off -r requirements-smoke.txt
```

Developer test dependencies:

```powershell
python -m pip install --progress-bar off -r requirements-dev.txt
```

The UTF-8 environment variables avoid console encoding failures when the project path contains Vietnamese characters.

Optional LLM setup:

```powershell
Copy-Item .env.example .env
# then fill the provider key you actually use
```

## Data

Place UNSW-NB15 CSV files in `data/`, for example:

```text
data/
  UNSW-NB15_1.csv
  UNSW-NB15_2.csv
  UNSW-NB15_3.csv
  UNSW-NB15_4.csv
  UNSW_NB15_training-set.csv
  UNSW_NB15_testing-set.csv
```

`data/` is ignored by git because it is large and local.

## Reproducibility

- Runtime packages are pinned in `requirements.txt`; test-only packages are pinned in `requirements-dev.txt`.
- v14 training defaults to seed `42`. Override it with `--seed` when running `train.py` or `src/ids_v14_unswnb15.py`.
- Data splits, PyTorch, NumPy, Python hashing and DataLoader workers are seeded through `seed_everything()` and seeded worker initialization.
- Local artifacts are intentionally not committed: `checkpoints/*.pth`, `checkpoints/*.pkl`, CSV datasets and SQLite files can be large or environment-specific.
- Use `scripts/artifact_manifest.py` after training to record local artifact hashes.

Pretrained artifacts are not currently published in this repository. For demos, train/export artifacts locally or attach a trusted release asset, Hugging Face artifact, Google Drive file or internal model registry object, then point runtime tools at it with `IDS_MODEL_PATH` and `IDS_PIPELINE_PATH`.

## Run Dashboard

Default dashboard mode uses v14:

```powershell
$env:IDS_MODEL_VERSION="v14"
streamlit run dashboard/app.py
```

PowerShell launcher:

```powershell
.\scripts\run_dashboard.ps1
```

Important environment variables:

| Variable | Meaning |
| --- | --- |
| `IDS_MODEL_VERSION` | `v14` or `v15` |
| `IDS_MODEL_PATH` | Path to model `.pth` |
| `IDS_PIPELINE_PATH` | Path to pipeline `.pkl` |
| `IDS_DATA_DIR` | Dataset directory |
| `IDS_SAMPLE_DATA_PATH` | Sample CSV path for dashboard |
| `IDS_ALERT_DB_PATH` | SQLite alert history path, default `results/alerts.sqlite3` |
| `LLM_PROVIDER` | `none`, `groq`, `gemini`, `openai` or `anthropic` |

If model or pipeline artifacts are missing, the dashboard falls back to demo mode. Alert history is persisted locally in SQLite; database files are ignored by git.

## Run Inference API

The FastAPI server is inference-only. It loads the saved v14 checkpoint and preprocessing pipeline at startup from environment variables:

```powershell
$env:IDS_MODEL_PATH="checkpoints/ids_v14_model.pth"
$env:IDS_PIPELINE_PATH="checkpoints/ids_v14_pipeline.pkl"
uvicorn src.serve:app --host 0.0.0.0 --port 8080
```

Health check:

```bash
curl http://localhost:8080/health
```

Prediction request. Replace `61` with the feature count in your checkpoint if you retrain with a different feature set:

```powershell
$body = @{ features = @(0..60 | ForEach-Object { 0.0 }) } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri "http://localhost:8080/predict" -Method Post -ContentType "application/json" -Body $body
```

The `features` array must match the checkpoint feature count. The server applies the saved `RobustScaler` from `ids_v14_pipeline.pkl`, then returns:

```json
{
  "label": "Normal",
  "confidence": 0.99,
  "ae_re": 0.01,
  "hybrid_score": 0.02,
  "is_anomaly": false,
  "uncertainty": {
    "entropy": 0.1,
    "std_max_class": 0.01
  }
}
```

Uncertainty is estimated with Monte Carlo Dropout over 30 stochastic forward passes. If `uncertainty.entropy > 1.5`, the API response uses `"label": "LOW_CONFIDENCE"` so clients can route the alert for analyst review.

Docker:

```bash
docker build -t ids-v14-serve .
docker run --rm -p 8080:8080 \
  -v "$(pwd)/checkpoints:/app/checkpoints:ro" \
  -e IDS_MODEL_PATH=/app/checkpoints/ids_v14_model.pth \
  -e IDS_PIPELINE_PATH=/app/checkpoints/ids_v14_pipeline.pkl \
  ids-v14-serve
```

Local checkpoint files are ignored by git and excluded by `.dockerignore`, so mount artifacts into the container or copy them into a deployment-specific image.

## Train

Train v14:

```bash
python train.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
```

PowerShell launcher:

```powershell
.\scripts\train_v14.ps1
```

Quick export helper for dashboard artifacts:

```bash
python export_model.py --data_dir data --epochs 5 --patience 3
```

Train v15:

```bash
python src/ids_v15_unswnb15.py --config configs/config_default.yaml
```

PowerShell launcher:

```powershell
.\scripts\train_v15.ps1
```

Training outputs:

- model weights `.pth`
- pipeline `.pkl`
- plots in `plots/`
- metric summary in `results/`

### Kaggle Retraining

On Kaggle, upload this repository as a working notebook dataset or clone it into
`/kaggle/working`, then attach UNSW-NB15/CICIDS datasets as Kaggle inputs. Copy
or symlink the CSV files into one training directory before running:

```bash
cd /kaggle/working/ZeroDay-Detection-AutoEncoder-IDS
python train.py \
  --data_dir /kaggle/working/data \
  --save_dir /kaggle/working/checkpoints \
  --plot_dir /kaggle/working/plots \
  --target_fpr 0.01
```

After training, download the generated `.pth`, `.pkl`, plots and result JSONs,
then point the dashboard at those artifacts with `IDS_MODEL_PATH` and
`IDS_PIPELINE_PATH`.

## Verify

Check local environment and artifact availability:

```powershell
python scripts/check_environment.py
```

Create or verify local artifact hashes:

```powershell
python scripts/artifact_manifest.py
python scripts/artifact_manifest.py --verify
```

Run the full smoke check:

```powershell
$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
python scripts/smoke_check.py
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `IDS_MODEL_PATH must be set` | Set `IDS_MODEL_PATH` and `IDS_PIPELINE_PATH` before starting `uvicorn src.serve:app`. |
| `expected N features, received M` | Send exactly the feature count saved in the checkpoint/pipeline; current local v14 artifacts use 61 features. |
| Dashboard starts in demo mode | Confirm `checkpoints/ids_v14_model.pth` and `checkpoints/ids_v14_pipeline.pkl` exist or set explicit artifact paths. |
| Too many zero-day candidates on a new CSV | Run `scripts/evaluate_csv.py --calibrate-thresholds` on labeled benign rows and load the generated threshold profile. |
| Import or console encoding errors on Windows paths | Set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8`; `scripts/smoke_check.py` already does this for subprocesses. |
| Docker container cannot find artifacts | Mount `checkpoints/` into `/app/checkpoints` or build a deployment-specific image that includes trusted artifacts. |

## Evaluate and Calibrate CSV Drift

Use the CSV evaluator before retraining when a new dataset produces too many
OOD/zero-day candidate hypotheses. It runs the saved model/pipeline outside Streamlit, writes a
score distribution report and can calibrate a local threshold profile.

```powershell
python scripts/evaluate_csv.py "path\to\Tuesday-WorkingHours.pcap_ISCX.csv" --scores-csv
```

If the CSV has a benign/attack label column, calibrate thresholds from benign
rows at the requested false-positive rate:

```powershell
python scripts/evaluate_csv.py "path\to\Tuesday-WorkingHours.pcap_ISCX.csv" `
  --label-col Label `
  --calibrate-thresholds `
  --target-fpr 0.01 `
  --threshold-output checkpoints\local_thresholds.json
```

The dashboard automatically loads `checkpoints/local_thresholds.json` when it is
present. Override with `IDS_THRESHOLD_PROFILE` if you want to test another
profile. Local profiles use vote-based OOD candidate decisions by default, so a row
must cross multiple calibrated signals instead of only the hybrid score.

Run the full local quality gate:

```powershell
.\scripts\quality_check.ps1
```

Or run commands separately:

```bash
python -m compileall src dashboard scripts export_model.py patch_checkpoint.py tests
python -m unittest discover -s tests
```

For demo and runtime procedures, see [docs/operations.md](docs/operations.md).

## Limitations

- MITRE mapping is heuristic and should be treated as triage support.
- Zero-day results depend strongly on feature quality, scaler compatibility and threshold calibration.
- Real-world CSV normalization is approximate when directional counters or timing fields are missing.
- v15 is experimental until v15 artifacts are trained/exported and smoke-tested.
- The dashboard is not hardened for production deployment: no authentication, no realtime packet capture and no deployment security controls.

## License

See [LICENSE](LICENSE).
