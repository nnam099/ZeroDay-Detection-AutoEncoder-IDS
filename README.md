# Zero-Day Detection AutoEncoder IDS

Prototype IDS for UNSW-NB15 research and SOC-style demonstrations. The project combines known-attack classification, reconstruction-error based zero-day/OOD detection, Streamlit dashboard analysis, SHAP explainability, heuristic MITRE ATT&CK mapping and optional LLM triage.

> This is a research/demo project, not a production IDS. Analyst review and supporting SIEM/firewall/endpoint evidence are still required.

## Overview

The training pipeline learns from selected known attack classes and holds out other classes to simulate zero-day or out-of-distribution traffic. The dashboard loads saved artifacts and supports both single-alert analysis and CSV batch uploads.

Main flow:

1. Load and clean UNSW-NB15 CSV files.
2. Engineer flow features and encode categorical fields.
3. Train a hybrid supervised classifier + contrastive representation + autoencoder model.
4. Calibrate anomaly/zero-day thresholds.
5. Save checkpoint and preprocessing pipeline artifacts.
6. Analyze alerts in the Streamlit dashboard.
7. Attach SHAP, MITRE and optional LLM context for SOC triage.

## Current Status

- v14 is the operational default because local artifacts exist in `checkpoints/`.
- `scripts/smoke_check.py` passes locally with 30 tests.
- Smoke coverage includes artifact contract validation, threshold metadata validation, artifact manifest hashing, duplicate feature-name rejection, environment readiness checks, export config handling, checkpoint metadata patch logic, SQLite alert store persistence, CSV input guardrails, CSV normalization quality checks, dashboard preprocessing/context contracts, AI context selection, alert queue filtering, top-N batch alert selection, alert entity enrichment, lightweight correlation, LLM fallback behavior, MITRE mapping and v14 artifact loading.
- `llm_agent.py` lazy-loads provider clients, so importing dashboard code does not require an API key.
- A Windows GitHub Actions smoke workflow is available at `.github/workflows/smoke.yml`.
- A basic Dockerfile is available for dashboard deployment experiments.
- v14 performance metrics have not been regenerated after the latest operational fixes; `results/ids_v14_results.json` records artifact smoke verification only.

## Features

- Known-attack classification for classes such as `Normal`, `DoS`, `Exploits`, `Reconnaissance` and `Generic`.
- Zero-day/OOD detection using reconstruction error, classifier confidence and calibrated hybrid thresholds.
- Streamlit SOC dashboard for single alert review, CSV batch analysis, alert history and AI follow-up.
- SQLite alert store for persisted queue history, status, analyst notes, queue filtering, top-N batch alert persistence and lightweight correlation groups.
- Real-world CSV normalization for common firewall/flow/Zeek/Suricata-like exports.
- CSV upload guardrails for empty, oversized or malformed files.
- SHAP top-feature explanation.
- Heuristic MITRE ATT&CK mapping.
- Optional LLM triage through Groq, Gemini, OpenAI or Anthropic.

## Repository Layout

```text
src/
  ids_v14_unswnb15.py      # train/evaluate/export pipeline v14
  ids_v15_unswnb15.py      # experimental v15 pipeline
  inference_runtime.py     # pure verdict/zero-day/risk/CSV-quality helpers used by dashboard
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

## Train

Train v14:

```bash
python src/ids_v14_unswnb15.py --data_dir data/ --save_dir checkpoints/ --plot_dir plots/
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
python src/ids_v14_unswnb15.py \
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
