# TSAD Workbench

Interactive frontend for the end-to-end workflow in this repo:

1. configure synthetic generation parameters
2. generate + pack dataset
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

- Generation reuses `synthetic_tsad/scripts/generate_dataset.py` and `pack_dataset.py`.
- Training reuses `train_tsad/scripts/train.py` and supports `--inspect-data` options.
- Default run outputs are written under `C:\Users\Administrator\Desktop\TSAD\runs\<run_name>`.
