param(
    [switch]$Fix
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment python not found: $pythonExe`nRun .\\scripts\\setup_env.ps1 first."
}

function Invoke-ProjectChecks {
    param(
        [string]$ProjectPath,
        [string[]]$Targets,
        [string]$PyrightTarget = ""
    )

    $projectRoot = Join-Path $repoRoot $ProjectPath
    Push-Location $projectRoot
    try {
        if ($Fix) {
            & $pythonExe -m ruff format @Targets
            & $pythonExe -m ruff check --fix @Targets
        } else {
            & $pythonExe -m ruff format --check @Targets
            & $pythonExe -m ruff check @Targets
        }

        if ([string]::IsNullOrWhiteSpace($PyrightTarget)) {
            & $pythonExe -m pyright
        } else {
            & $pythonExe -m pyright $PyrightTarget
        }

        & $pythonExe -m pytest -q
    } finally {
        Pop-Location
    }
}

Invoke-ProjectChecks -ProjectPath "synthetic_tsad" -Targets @("src", "apps", "scripts", "tests")
Invoke-ProjectChecks -ProjectPath "train_tsad" -Targets @("src", "apps", "scripts", "tests") -PyrightTarget "src"
