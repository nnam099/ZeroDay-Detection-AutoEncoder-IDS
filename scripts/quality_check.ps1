$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

& $Python scripts/check_environment.py
& $Python scripts/artifact_manifest.py --verify
& $Python scripts/smoke_check.py
& $Python -m pip check
