# Reproducibility Guide

This guide records the commands needed to move from a fresh checkout to benchmark outputs.

## 1. Environment

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

CPU-only machines are supported. XGBoost and LightGBM automatically use CPU parameters when CUDA/GPU-enabled builds are unavailable.

On Apple Silicon or constrained environments, set `FORCE_CPU=1` before a run if PyTorch reports an MPS device but model execution fails:

```bash
FORCE_CPU=1 python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke
```

## 2. Verify the Checkout

```bash
python run.py --help
python run.py config
python -m pytest -q
```

If `pytest` is unavailable, install it with `python -m pip install pytest`.

## 3. Smoke Test

Run the fastest model on one included plant:

```bash
python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke
```

Expected artifact:

```text
results/smoke/results_171_all.csv
```

The file should contain two successful Linear Regression NWP/NWP+ smoke-test rows. Check it with:

```bash
python run.py status --output-dir results/smoke
```

## 4. Full Single-Plant Benchmark

```bash
python run.py forecast --plant-id 171 --output-dir results/plant171
```

Expected artifact:

```text
results/plant171/results_171_all.csv
```

The full grid contains 284 configurations per plant. The runner appends results after each experiment, so rerunning the command resumes from the existing CSV.

The required time depends strongly on hardware and the selected models. The Linear smoke test is intended as a correctness check; full single-plant and multi-plant runs are benchmark jobs and can take substantially longer.

## 5. Multi-Plant Benchmark

```bash
python run.py multi_plant --plants 171 172 186 --output-dir results/sample_plants
```

For the full release, place all `Project<ID>.csv` files under `data/`, regenerate configs with `python run.py config`, and omit `--plants`.

Full 100-plant data release: [https://doi.org/10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM)

The paper states that the full release is for non-commercial research use and should be cited together with the paper. The code license is MIT; the data terms are separate.

## 6. Reporting

The main result CSV columns are:

| Column | Meaning |
| --- | --- |
| `experiment_name` | Unique model/input/lookback/time-encoding configuration |
| `model` | Forecasting model |
| `complexity` | Low/high hyperparameter setting, or `N/A` for Linear |
| `feature_combo` or `scenario` | Input set |
| `lookback_hours` | Historical window length |
| `use_time_encoding` | Whether cyclic month/hour features are included |
| `mae`, `rmse`, `r2`, `nrmse` | Test-set metrics |
| `train_time_sec` | Wall-clock training time |
| `test_samples` | Number of forecasted hourly targets in the test set |
| `param_count` | Trainable parameter count or ML proxy count |
| `status` | `SUCCESS` or `FAILED`; only successful rows count toward completion |
| `error` | Error message for failed configurations |

Keep generated outputs under `results/`; they are ignored by Git.
