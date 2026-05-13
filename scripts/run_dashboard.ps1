$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not $env:IDS_MODEL_VERSION) {
    $env:IDS_MODEL_VERSION = "v14"
}

streamlit run dashboard/app.py @args
