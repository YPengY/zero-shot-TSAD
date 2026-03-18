# TSAD Workbench

Interactive frontend for the end-to-end workflow in this repo:

1. configure synthetic generation parameters
2. generate packed dataset (direct-window-pack by default)
3. inspect sample/feature/slice with labels and windowing
4. launch model training and stream logs live

## Run

```powershell
cd C:\Users\Administrator\Desktop\TSAD
python train_tsad\apps\tsad_workbench\server.py
```

Open:

```text
http://127.0.0.1:8777
```

## Notes

- Generation reuses `synthetic_tsad/scripts/generate_dataset.py`.
- Default path is direct window-packed generation per split into `data_packed_windows/` (no intermediate `data_raw`, no sample-packed post-pass).
- Default window packing uses `context_size=1024`, `stride=1024`, `patch_size=16`.
- Train split applies a light window filter (`min patch_positive_ratio = 1/192`) to reduce all-negative windows by default.
- Workbench regenerates `dataset_meta.json` from manifests after generation; window-packed mode writes storage format `npz_window_shards_minimal_v1`.
- You can still fall back to legacy modes via payload switches (`direct_window_pack=false`, `window_pack=false`, or `direct_pack=false`).
- Training reuses `train_tsad/scripts/train.py` and supports `--inspect-data` options.
- Default run outputs are written under `D:\TSAD\runs\<run_name>` when `D:` is available, otherwise they fall back to the repo-local `runs\<run_name>`.
