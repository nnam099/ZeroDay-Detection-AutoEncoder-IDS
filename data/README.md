# Data Directory

This folder is for local UNSW-NB15, CICFlowMeter and other flow CSV files.
Large data files are ignored by git.

Expected examples:

```text
UNSW-NB15_1.csv
UNSW-NB15_2.csv
UNSW-NB15_3.csv
UNSW-NB15_4.csv
UNSW_NB15_training-set.csv
UNSW_NB15_testing-set.csv
```

Source: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

Small committed samples live in `data/samples/` for parser and CI tests. They
are not training datasets.

For production-like flow data preparation, export CICFlowMeter CSVs locally and
run:

```powershell
python scripts/prepare_production_flow_data.py data\your_cicflowmeter.csv `
  --output-dir results\production_flow_data
```

The script writes:

- `production_flows.csv`
- `train.csv`
- `validation.csv`
- `test.csv`
- `manifest.json`
