param(
    [switch]$Fix
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue

if ($null -eq $pythonCommand) {
    throw "Python runtime not found on PATH."
}

$targets = @("src", "apps", "scripts")
if (Test-Path (Join-Path $projectRoot "tests")) {
    $targets += "tests"
}

Push-Location $projectRoot
try {
    if ($Fix) {
        & $pythonCommand.Source -m ruff format @targets
        & $pythonCommand.Source -m ruff check --fix @targets
    } else {
        & $pythonCommand.Source -m ruff format --check @targets
        & $pythonCommand.Source -m ruff check @targets
    }

    & $pythonCommand.Source -m pyright src
    & $pythonCommand.Source -m pytest -q
} finally {
    Pop-Location
}
