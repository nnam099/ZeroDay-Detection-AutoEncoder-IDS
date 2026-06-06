# Real-World Readiness Roadmap

This project should be framed as a SOC-oriented anomaly and OOD triage IDS, not
as a standalone proof of novel zero-day compromise. The practical path is to
combine multi-source data, local threshold calibration, drift monitoring and
analyst feedback.

## Phase 0: Keep The Demo Honest

- Keep the README wording clear: `Zero-Day Candidate` means an OOD hypothesis
  for analyst review.
- Run smoke checks and dependency audit before publishing changes.
- Verify artifact hashes after retraining or replacing checkpoints.

## Phase 1: Baseline Comparison

Use `scripts/evaluate_baselines.py` to compare the saved IDS scores against
classical anomaly detectors:

```powershell
python scripts/evaluate_baselines.py data\UNSW_NB15_testing-set.csv `
  --label-col attack_cat `
  --target-fpr 0.05 `
  --scores-csv
```

The report includes:

- active IDS threshold behavior
- IDS hybrid, AE-only and softmax-confidence scores
- Isolation Forest
- Local Outlier Factor
- One-Class SVM, unless skipped
- AUROC/AUPRC when labels are available
- normal false-positive rate and OOD detection rate when labels support them

Use this as a sanity baseline before claiming that the hybrid model adds value.

## Phase 2: Cross-Dataset Evaluation

Add datasets beyond UNSW-NB15:

- CICIDS2017 or CSE-CIC-IDS2018 for common enterprise attack families
- CIC-DDoS2019 for DoS/DDoS stress
- TON_IoT or BoT-IoT for IoT-style traffic
- Zeek, Suricata, firewall or NetFlow exports from a lab network

Avoid random mixing when measuring generalization. Prefer splits such as:

```text
Train: UNSW-NB15 known classes
Validation: UNSW-NB15 known/normal traffic
Test: CICIDS or lab traffic unseen during calibration
```

## Phase 3: AutoEncoder Ablation

Run latent-size and loss-function sweeps:

```text
latent_dim: 8, 16, 32
variants: plain AE, denoising AE, sparse AE
```

Recommended ablations:

- classifier only
- AE reconstruction only
- classifier + AE
- classifier + AE + contrastive loss
- classifier + AE + contrastive loss + hybrid meta-learner

Report macro-F1, per-class recall, OOD AUROC, OOD AUPRC and latency.

## Phase 4: Local Threshold Calibration

Each environment needs its own threshold profile. Calibrate from representative
benign traffic:

```powershell
python scripts/evaluate_csv.py path\to\benign_or_labeled_flows.csv `
  --label-col Label `
  --calibrate-thresholds `
  --target-fpr 0.01 `
  --threshold-output checkpoints\local_thresholds.json
```

Keep local profiles separate when possible:

```text
thresholds/lab_network.json
thresholds/office_network.json
thresholds/cloud_vpc.json
thresholds/iot_vlan.json
```

## Phase 5: Drift Monitoring

Use `scripts/drift_report.py` before trusting results on a new CSV:

```powershell
python scripts/drift_report.py path\to\new_environment.csv `
  --label-col Label `
  --scores-csv
```

The drift report checks:

- normalization feature coverage
- zero-day candidate rate changes
- score distribution changes for hybrid, AE, softmax and optional cluster scores

If drift is high, inspect mapped columns and recalibrate thresholds on benign
traffic from that environment.

## Phase 6: Analyst Feedback Loop

The dashboard already persists alert status and analyst notes. The next useful
step is to export feedback labels such as:

```text
confirmed_attack
benign_false_positive
suspicious
unknown
```

Feedback should feed:

- false-positive reports
- local threshold recalibration
- local validation sets
- future fine-tuning runs

## Phase 7: Production Hardening

Before deploying outside a controlled lab:

- add API/dashboard authentication
- verify artifact hashes before loading
- avoid untrusted pickle artifacts
- log inference metadata, artifact version and threshold profile
- rate-limit public API endpoints
- keep the service behind an internal gateway or VPN

## Success Criteria

The project is closer to practical use when it has:

- cross-dataset evaluation reports
- baseline comparison reports
- local threshold profiles
- drift reports for new environments
- analyst feedback export/recalibration
- artifact hash verification
- clear model-card style documentation
