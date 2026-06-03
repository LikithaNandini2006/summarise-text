$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$python = Join-Path $projectRoot "..\venv\Scripts\python.exe"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Virtual environment Python not found: $python"
}

& $python (Join-Path $projectRoot "recognise.py")
exit $LASTEXITCODE
