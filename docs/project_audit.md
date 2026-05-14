# Project Audit

Updated: 2026-05-14

## Verified

- `scripts/smoke_check.py` now runs with UTF-8 subprocess settings, so it works on Windows paths containing Vietnamese characters.
- `scripts/check_environment.py` now configures UTF-8 console output before printing JSON, so direct execution works from Windows paths containing Vietnamese characters.
- Python compile passes for `src/`, `dashboard/`, `export_model.py`, `patch_checkpoint.py` and `tests/`.
- Smoke tests pass locally: 28 tests.
- Checkpoint `ids_v14_model.pth` loads into `IDSModel` v14 without missing or unexpected weights.
- Pipeline v14 has 61 features and matches `n_features` in the checkpoint.
- `log_normalizer.py` maps common firewall/flow CSV columns into an UNSW-like flow schema.
- Dashboard validates the checkpoint/pipeline contract before inference.
- Pure inference verdict/risk/CSV-quality helpers are split into `src/inference_runtime.py` and covered by smoke tests.
- Dashboard batch inference is split into `src/inference_runtime.py` and covered by a dashboard contract smoke test.
- Dashboard preprocessing, batch alert context construction, AI context option selection and LLM fallback handling are split into `src/dashboard_runtime.py` and covered by smoke tests.
- Alert history, status and analyst notes can persist across Streamlit reloads through `src/alert_store.py` backed by local SQLite.
- The alert queue supports status/severity/OOD/search filters, and batch uploads can persist the top N highest-risk rows.
- `llm_agent.py` no longer initializes the provider client on import; it initializes lazily when LLM output is requested.
- `patch_checkpoint.py` now validates checkpoint structure, accepts a path argument and creates a backup by default.
- `artifact_validator.py` rejects duplicate/empty feature names and invalid threshold metadata.
- `export_model.py` now exposes a CLI instead of hard-coding `data/quick_train`.
- PowerShell train launchers are available for v14 and v15.
- CI uses `requirements-smoke.txt` so optional dashboard/explainability packages do not slow down core smoke checks.
- `scripts/check_environment.py` reports package/artifact/data readiness without exposing secret values and is covered for UTF-8 console output.
- `input_guard.py` validates uploaded CSV size/shape before dashboard batch inference.
- `.env.example` documents optional provider/runtime environment variables without storing secrets.
- `scripts/artifact_manifest.py` can create/verify SHA-256 manifests for local model artifacts.
- A Windows GitHub Actions smoke workflow is available in `.github/workflows/smoke.yml`.

## Strengths

- Training, inference support modules and dashboard are separated at the folder level.
- Artifact v14 stores the metadata needed for inference: scaler, label encoder, feature list and thresholds.
- Dashboard has demo fallback when artifacts are missing and supports common real-world CSV uploads through the normalizer.
- Dashboard alert queue can now survive reloads through a local SQLite store.
- MITRE mapping is packaged separately and is easy to extend with more techniques/evidence rules.
- Smoke tests cover the highest-risk runtime contracts: artifact compatibility, threshold metadata validation, artifact hash drift detection, duplicate feature metadata rejection, environment readiness reporting, export config handling, checkpoint metadata patching, SQLite alert persistence, CSV input guardrails, normalizer behavior, dashboard preprocessing/context helpers, AI context selection, alert queue filtering, top-N batch alert selection, CSV quality classification, MITRE mapping and LLM import/fallback behavior.

## Current Risks

- `dashboard/app.py` is still a large file. Verdict/risk, batch inference, preprocessing, persistence, batch alert context helpers and AI/LLM fallback logic have been split out, but UI rendering and session-state orchestration should be split further for unit testing.
- Some code comments/docstrings still contain mojibake or ASCII-only Vietnamese text from earlier encoding issues.
- v14 artifact compatibility is verified, but full model performance metrics have not been regenerated after the latest operational fixes.
- v15 is experimental. The dashboard can select it, but stable v15 use requires separately trained/exported artifacts.
- LLM provider packages remain optional and are not installed unless the selected provider is needed.

## Next Priorities

1. Start storing related event groups for correlation across source IPs, services, attack families and time windows.
2. Regenerate v14 metrics/plots from the current code and update `results/ids_v14_results.json`.
3. Fix remaining mojibake in source comments/docstrings that are used in reports or presentations.
4. Add schema-quality warnings for uploaded CSVs with low feature coverage or missing directional counters.
5. Train/export v15 artifacts and add v15 artifact smoke tests if v15 will be demonstrated.
