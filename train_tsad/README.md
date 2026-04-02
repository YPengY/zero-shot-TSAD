# train_tsad

Training and evaluation utilities for zero-shot TSAD experiments, plus the training workbench app.

## Dependencies

Runtime:

- Python `>=3.10`
- `numpy>=1.26`
- `PyYAML>=6.0`
- `torch>=2.2`

Dev / quality gate:

- `pytest>=8.0`
- `pyright>=1.1.390`
- `ruff>=0.11.0`
- `types-PyYAML>=6.0.12`

Workbench integration:

- If you want the workbench backend to call dataset generation and packing helpers, keep the sibling `synthetic_tsad/` folder in the same repository and install it into the same environment.

Install from this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Environment Setup

For repository-wide development, use the root bootstrap:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

If you only need `train_tsad`, you can create a local environment inside this folder:

```powershell
cd .\train_tsad
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

When `synthetic_tsad` exists as a sibling folder in the repository, the local setup script installs it into the same environment automatically. That keeps the workbench backend import bridge working without extra manual steps.

## Entry Points

Editable install registers these console commands:

- `train-tsad-train`
- `train-tsad-evaluate`
- `train-tsad-inspect`

## Quality Checks

Run the local project self-check after setup:

```powershell
cd .\train_tsad
powershell -ExecutionPolicy Bypass -File .\scripts\check_quality.ps1
```

The current gate covers formatting, linting, type-checking, and smoke tests under `train_tsad/tests`.
