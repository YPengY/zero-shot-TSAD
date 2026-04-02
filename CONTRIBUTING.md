# Contributing

## Repository Layout

- `synthetic_tsad/`: synthetic data generator, configs, scripts, app, and tests
- `train_tsad/`: training code, evaluation tools, workbench app, and tests
- `datasets/`: local generated or downloaded data, not committed

## Development Setup

For day-to-day work across the repository, prefer the root setup script:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
```

If you only need one subproject, you can still use its local setup script under that folder's `scripts/` directory.

## Local Checks

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_quality.ps1
```

You can also run project-local checks inside `synthetic_tsad/` or `train_tsad/` when working on just one package.

## Pull Requests

- Keep each pull request focused on one concern.
- Do not commit local `.venv`, generated `outputs*`, or `datasets/`.
- Update documentation when command-line usage or config behavior changes.
