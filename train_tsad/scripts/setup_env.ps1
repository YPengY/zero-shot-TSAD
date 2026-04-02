param(
    [string]$PythonExe = "python",
    [switch]$Recreate,
    [switch]$SkipSyntheticTsad
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $projectRoot
$syntheticRoot = Join-Path $repoRoot "synthetic_tsad"
$venvPath = Join-Path $projectRoot ".venv"

if ($Recreate -and (Test-Path $venvPath)) {
    Write-Host "Removing existing .venv ..."
    Remove-Item -Recurse -Force $venvPath
}

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment ..."
    & $PythonExe -m venv $venvPath
}

$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment python not found: $venvPython"
}

function Test-PipAvailable {
    param([string]$PythonPath)
    try {
        & $PythonPath -c "import pip" | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-PipAvailable -PythonPath $venvPython)) {
    Write-Host "pip missing in .venv, trying ensurepip ..."
    try {
        & $venvPython -m ensurepip --upgrade
    } catch {
        Write-Host "ensurepip failed, falling back to include-system-site-packages=true"
    }
}

if (-not (Test-PipAvailable -PythonPath $venvPython)) {
    $cfg = Join-Path $venvPath "pyvenv.cfg"
    (Get-Content $cfg -Raw).Replace("include-system-site-packages = false", "include-system-site-packages = true") | Set-Content $cfg -Encoding UTF8
}

if (-not (Test-PipAvailable -PythonPath $venvPython)) {
    throw "pip is still unavailable in .venv after fallback"
}

& $venvPython -m pip install -U pip

Push-Location $projectRoot
try {
    & $venvPython -m pip install -e ".[dev]"
    if ((-not $SkipSyntheticTsad) -and (Test-Path $syntheticRoot)) {
        & $venvPython -m pip install -e $syntheticRoot
    }
} finally {
    Pop-Location
}

Write-Host "Environment ready: $venvPython"
& $venvPython -c "import sys, numpy, torch, yaml; print(sys.executable); print('numpy', numpy.__version__); print('torch', torch.__version__); print('yaml', yaml.__version__)"
