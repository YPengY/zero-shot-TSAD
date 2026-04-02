param(
    [switch]$Fix
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $projectRoot ".venv\\Scripts\\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found: $pythonExe`nRun .\\scripts\\setup_env.ps1 first."
}

$targets = @("src", "apps", "scripts")
if (Test-Path (Join-Path $projectRoot "tests")) {
    $targets += "tests"
}

Push-Location $projectRoot
try {
    if ($Fix) {
        & $pythonExe -m ruff format @targets
        & $pythonExe -m ruff check --fix @targets
    } else {
        & $pythonExe -m ruff format --check @targets
        & $pythonExe -m ruff check @targets
    }

    & $pythonExe -m pyright src
    & $pythonExe -m pytest -q
} finally {
    Pop-Location
}
