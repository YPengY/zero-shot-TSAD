param(
    [string]$RunRoot = 'D:\tsad_runs\mini_run_002',
    [int]$TrainSamples = 10000,
    [int]$ValSamples = 1500,
    [int]$TestSamples = 1500,
    [int]$NumSeries = 6,
    [int]$SequenceLengthMultiple = 2,
    [double]$AnomalySampleRatio = 1.0,
    [int]$LocalEventsMin = 2,
    [int]$LocalEventsMax = 4,
    [int]$SeasonalEventsMin = 1,
    [int]$SeasonalEventsMax = 1,
    [int]$LocalWindowMin = 16,
    [int]$LocalWindowMax = 128,
    [int]$SeasonalWindowMin = 16,
    [int]$SeasonalWindowMax = 128,
    [int]$MinGap = 4,
    [int]$MaxEventsPerNode = 3,
    [double]$EndogenousProbability = 0.25,
    [switch]$DisableTrend,
    [switch]$DisableSeasonality,
    [switch]$DisableNoise,
    [switch]$DisableCausal,
    [switch]$DisableLocalAnomaly,
    [switch]$DisableSeasonalAnomaly,
    [switch]$KeepRawData
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$PythonExe = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$UseFallbackPython = $false
if (Test-Path $PythonExe) {
    try {
        & $PythonExe '--version' *> $null
        if ($LASTEXITCODE -ne 0) {
            $UseFallbackPython = $true
        }
    }
    catch {
        $UseFallbackPython = $true
    }
}
else {
    $UseFallbackPython = $true
}

if ($UseFallbackPython) {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $PythonCommand) {
        throw 'Python runtime not found in .venv or PATH.'
    }
    $PythonExe = $PythonCommand.Source
}

Write-Host "Using Python runtime: $PythonExe"

$RawRoot = Join-Path $RunRoot 'data_raw'
$PackedRoot = Join-Path $RunRoot 'data_packed'
$ConfigDir = Join-Path $RunRoot 'configs'
$TrainCfg = Join-Path $ConfigDir 'train_mini.json'
$SyntheticOverridesPath = Join-Path $ConfigDir 'synthetic_mini_overrides.json'
$TrainOut = Join-Path $RunRoot 'train_out'
$EvalOut = Join-Path $RunRoot 'eval'
$TrainTemplatePath = Join-Path $RepoRoot 'train_tsad\configs\timercd_small.json'

function Assert-Range {
    param(
        [string]$Name,
        [double]$Min,
        [double]$Max
    )

    if ($Min -gt $Max) {
        throw "Invalid range for ${Name}: min=$Min max=$Max"
    }
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $cmdLine = "$PythonExe $($Args -join ' ')"
    Write-Host "[run] $cmdLine"
    & $PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $cmdLine"
    }
}

function Merge-Hashtable {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.IDictionary]$Base,
        [Parameter(Mandatory = $true)]
        [System.Collections.IDictionary]$Override
    )

    $merged = [ordered]@{}
    foreach ($key in $Base.Keys) {
        $merged[$key] = $Base[$key]
    }

    foreach ($key in $Override.Keys) {
        $baseValue = if ($merged.Contains($key)) { $merged[$key] } else { $null }
        $overrideValue = $Override[$key]
        if (
            $baseValue -is [System.Collections.IDictionary] -and
            $overrideValue -is [System.Collections.IDictionary]
        ) {
            $merged[$key] = Merge-Hashtable -Base $baseValue -Override $overrideValue
        }
        else {
            $merged[$key] = $overrideValue
        }
    }

    return $merged
}

if ($SequenceLengthMultiple -le 0) {
    throw '`SequenceLengthMultiple` must be positive.'
}
if ($AnomalySampleRatio -lt 0.0 -or $AnomalySampleRatio -gt 1.0) {
    throw '`AnomalySampleRatio` must be in [0, 1].'
}
if ($EndogenousProbability -lt 0.0 -or $EndogenousProbability -gt 1.0) {
    throw '`EndogenousProbability` must be in [0, 1].'
}
Assert-Range -Name 'LocalEvents' -Min $LocalEventsMin -Max $LocalEventsMax
Assert-Range -Name 'SeasonalEvents' -Min $SeasonalEventsMin -Max $SeasonalEventsMax
Assert-Range -Name 'LocalWindow' -Min $LocalWindowMin -Max $LocalWindowMax
Assert-Range -Name 'SeasonalWindow' -Min $SeasonalWindowMin -Max $SeasonalWindowMax
if ($MinGap -lt 0) {
    throw '`MinGap` cannot be negative.'
}
if ($MaxEventsPerNode -le 0) {
    throw '`MaxEventsPerNode` must be positive.'
}
if (-not $DisableLocalAnomaly -and $LocalEventsMin -le 0) {
    throw '`LocalEventsMin` must be >= 1 when local anomalies are enabled.'
}
if (-not $DisableSeasonalAnomaly -and $SeasonalEventsMin -le 0) {
    throw '`SeasonalEventsMin` must be >= 1 when seasonal anomalies are enabled.'
}

$TrainTemplate = Get-Content $TrainTemplatePath -Raw | ConvertFrom-Json
$SequenceLength = [int]$TrainTemplate.data.context_size * $SequenceLengthMultiple
Write-Host "Using fixed synthetic sequence length: $SequenceLength ($SequenceLengthMultiple x context_size=$($TrainTemplate.data.context_size))"

# Optional raw overrides that follow the native `synthetic_tsad` config structure.
# Add any nested fields here when the named parameters above are not enough.
$SyntheticExtraOverrides = [ordered]@{
    # Example:
    # stage1 = [ordered]@{
    #     noise = [ordered]@{
    #         sigma = [ordered]@{ high = 0.15 }
    #     }
    # }
}

$AnomalyOverrides = [ordered]@{
    defaults = [ordered]@{
        min_gap = $MinGap
        max_events_per_node = $MaxEventsPerNode
    }
}
if (-not $DisableLocalAnomaly) {
    $AnomalyOverrides.local = [ordered]@{
        budget = [ordered]@{
            events_per_sample = [ordered]@{
                min = $LocalEventsMin
                max = $LocalEventsMax
            }
        }
        defaults = [ordered]@{
            window_length = [ordered]@{
                min = $LocalWindowMin
                max = $LocalWindowMax
            }
            endogenous_p = $EndogenousProbability
        }
    }
}
if (-not $DisableSeasonalAnomaly) {
    $AnomalyOverrides.seasonal = [ordered]@{
        budget = [ordered]@{
            events_per_sample = [ordered]@{
                min = $SeasonalEventsMin
                max = $SeasonalEventsMax
            }
        }
        defaults = [ordered]@{
            window_length = [ordered]@{
                min = $SeasonalWindowMin
                max = $SeasonalWindowMax
            }
        }
    }
}

$SyntheticOverrides = [ordered]@{
    sequence_length = [ordered]@{
        min = $SequenceLength
        max = $SequenceLength
    }
    num_series = [ordered]@{
        min = $NumSeries
        max = $NumSeries
    }
    anomaly_sample_ratio = $AnomalySampleRatio
    anomaly = $AnomalyOverrides
    debug = [ordered]@{
        enable_trend = (-not $DisableTrend)
        enable_seasonality = (-not $DisableSeasonality)
        enable_noise = (-not $DisableNoise)
        enable_causal = (-not $DisableCausal)
        enable_local_anomaly = (-not $DisableLocalAnomaly)
        enable_seasonal_anomaly = (-not $DisableSeasonalAnomaly)
    }
}

if ($SyntheticExtraOverrides.Count -gt 0) {
    $SyntheticOverrides = Merge-Hashtable -Base $SyntheticOverrides -Override $SyntheticExtraOverrides
}

if (Test-Path $RunRoot) {
    Remove-Item $RunRoot -Recurse -Force
}

New-Item -ItemType Directory -Force -Path `
    (Join-Path $RawRoot 'train'), `
    (Join-Path $RawRoot 'val'), `
    (Join-Path $RawRoot 'test'), `
    $ConfigDir, `
    $TrainOut, `
    $EvalOut | Out-Null

$SyntheticOverridesJson = $SyntheticOverrides | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($SyntheticOverridesPath, $SyntheticOverridesJson, [System.Text.UTF8Encoding]::new($false))
Write-Host "Synthetic overrides: $SyntheticOverridesPath"

function Invoke-GenerateSplit {
    param(
        [string]$Split,
        [int]$Count,
        [int]$Seed
    )

    Invoke-Python -Args @(
        '-B'
        (Join-Path $RepoRoot 'synthetic_tsad\scripts\generate_dataset.py')
        '--config'
        (Join-Path $RepoRoot 'synthetic_tsad\configs\default.json')
        '--raw-overrides'
        $SyntheticOverridesPath
        '--output'
        (Join-Path $RawRoot $Split)
        '--num-samples'
        $Count
        '--seed'
        $Seed
    )
}

Invoke-GenerateSplit -Split 'train' -Count $TrainSamples -Seed 101
Invoke-GenerateSplit -Split 'val' -Count $ValSamples -Seed 102
Invoke-GenerateSplit -Split 'test' -Count $TestSamples -Seed 103

Invoke-Python -Args @(
    '-B'
    (Join-Path $RepoRoot 'synthetic_tsad\scripts\pack_dataset.py')
    '--input'
    $RawRoot
    '--output'
    $PackedRoot
    '--samples-per-shard'
    128
    '--overwrite'
    '--dataset-name'
    'mini_tsad'
    '--dataset-version'
    'v1'
)

$Cfg = $TrainTemplate
$Cfg.data.dataset_root = $PackedRoot
$Cfg.train.output_dir = $TrainOut
$Cfg.train.device = 'cuda'
$Cfg.train.max_epochs = 15
$Cfg.train.early_stopping_patience = 4
$Cfg.data.batch_size = 16
$Cfg.data.eval_batch_size = 16
$CfgJson = $Cfg | ConvertTo-Json -Depth 100
[System.IO.File]::WriteAllText($TrainCfg, $CfgJson, [System.Text.UTF8Encoding]::new($false))

Invoke-Python -Args @(
    '-B'
    (Join-Path $RepoRoot 'train_tsad\scripts\train.py')
    '--config'
    $TrainCfg
)

Invoke-Python -Args @(
    '-B'
    (Join-Path $RepoRoot 'train_tsad\scripts\evaluate.py')
    '--config'
    $TrainCfg
    '--checkpoint'
    (Join-Path $TrainOut 'best.pt')
    '--split'
    'test'
    '--device'
    'cpu'
    '--output'
    (Join-Path $EvalOut 'metrics_test.json')
)

if (-not $KeepRawData) {
    Remove-Item $RawRoot -Recurse -Force
}

Write-Host 'Run completed.'
Write-Host "Artifacts root: $RunRoot"
Write-Host "Synthetic overrides: $SyntheticOverridesPath"
Write-Host "Packed dataset: $PackedRoot"
Write-Host "Training outputs: $TrainOut"
Write-Host "Evaluation outputs: $EvalOut"
