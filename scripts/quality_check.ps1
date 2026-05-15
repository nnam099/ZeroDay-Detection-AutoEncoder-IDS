$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$Python = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string[]]$Command)
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

Invoke-Checked @($Python, "scripts/check_environment.py")
Invoke-Checked @($Python, "-m", "ruff", "check", ".")
Invoke-Checked @($Python, "scripts/artifact_manifest.py", "--verify")
Invoke-Checked @($Python, "scripts/smoke_check.py")
Invoke-Checked @($Python, "-m", "pip", "check")
