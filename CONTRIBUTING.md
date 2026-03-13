# Contributing

## Repository Layout

- `synthetic_tsad/`: synthetic data generator, configs, scripts, app, and tests
- `train_tsad/`: reserved for training code
- `datasets/`: local generated or downloaded data, not committed

## Development Setup

All current runnable code lives under `synthetic_tsad/`.

```powershell
cd .\synthetic_tsad
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

## Local Checks

```powershell
cd .\synthetic_tsad
powershell -ExecutionPolicy Bypass -File .\scripts\check_quality.ps1
```

## Pull Requests

- Keep each pull request focused on one concern.
- Do not commit local `.venv`, generated `outputs*`, or `datasets/`.
- Update documentation when command-line usage or config behavior changes.
