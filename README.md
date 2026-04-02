# TSAD Workspace

This repository contains two related Python projects:

- `synthetic_tsad/`: synthetic time-series anomaly dataset generation, packing, and the dataset studio app.
- `train_tsad/`: model training, evaluation, and the training workbench that can consume `synthetic_tsad` outputs.

## Dependency Summary

Repository-wide baseline:

- Python `>=3.10`

`synthetic_tsad` runtime dependencies:

- `numpy>=1.26`
- `PyYAML>=6.0`

`synthetic_tsad` dev dependencies:

- `pytest>=8.0`
- `pyright>=1.1.390`
- `ruff>=0.11.0`
- `types-PyYAML>=6.0.12`

`train_tsad` runtime dependencies:

- `numpy>=1.26`
- `PyYAML>=6.0`
- `torch>=2.2`

`train_tsad` dev dependencies:

- `pytest>=8.0`
- `pyright>=1.1.390`
- `ruff>=0.11.0`
- `types-PyYAML>=6.0.12`

Integration note:

- `train_tsad` workbench can bridge to the sibling `synthetic_tsad` project when both folders exist in the same repository.

## Quick Start

Create a repository-level virtual environment and install both projects in editable mode with dev tools:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".\synthetic_tsad[dev]" -e ".\train_tsad[dev]"
```

If you only want one project, install it directly from its own folder instead of pulling both.

## Quality Checks

Run the repository-level quality gate for both subprojects:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_quality.ps1
```

Use `-Fix` to apply Ruff formatting and auto-fixable lint changes before re-running the checks.

## Project-Specific Setup

If you only need one subproject, each package also has its own local setup script:

- `synthetic_tsad/scripts/setup_env.ps1`
- `train_tsad/scripts/setup_env.ps1`

`train_tsad`'s local setup script installs the sibling `synthetic_tsad` package automatically when the repository keeps the default layout, because the workbench backend bridges into the generator code.
