$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$UsingProjectVenv = Test-Path ".\.venv\Scripts\python.exe"
$Python = if ($UsingProjectVenv) { ".\.venv\Scripts\python.exe" } else { "python" }
if (-not $UsingProjectVenv) {
    Write-Warning "No .venv found; quality checks will use the active Python and may report unrelated global package conflicts."
}

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][string[]]$Command)
    & $Command[0] @($Command | Select-Object -Skip 1)
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $($Command -join ' ')"
    }
}

Invoke-Checked @($Python, "scripts/check_environment.py", "--fail-on-blocked")
Invoke-Checked @($Python, "-m", "ruff", "check", ".")
Invoke-Checked @($Python, "scripts/artifact_manifest.py", "--verify")
Invoke-Checked @($Python, "scripts/smoke_check.py")
& $Python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "pip check failed for the active environment. Create/use .venv and reinstall project requirements if this reports unrelated global packages."
}
