# NYSolarForecastLab

[![CI](https://github.com/zhesun-0209/NYSolarForecastLab/actions/workflows/ci.yml/badge.svg)](https://github.com/zhesun-0209/NYSolarForecastLab/actions/workflows/ci.yml)
[![DOI](https://img.shields.io/badge/paper-10.1016%2Fj.tra.2026.105040-blue)](https://doi.org/10.1016/j.tra.2026.105040)
[![Data](https://img.shields.io/badge/data-10.7910%2FDVN%2F3VKAGM-green)](https://doi.org/10.7910/DVN/3VKAGM)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

Reference implementation for the day-ahead photovoltaic (PV) power forecasting benchmark in:

**Toward better integration of solar energy in transportation systems: machine learning benchmarks for day-ahead photovoltaic power forecasting**

Zhaoyao Bao, Zhe Sun, Yishuo Jiang, Chi Xie, Lijun Sun, and H. Oliver Gao. *Transportation Research Part A: Policy and Practice*, 210, 105040, 2026. DOI: [10.1016/j.tra.2026.105040](https://doi.org/10.1016/j.tra.2026.105040).

This repository provides a compact command-line workflow for preparing plant configs, running model benchmarks, resuming interrupted experiments, checking progress, and exporting result CSVs.

## What Is Included

| Item | Details |
| --- | --- |
| Task | Day-ahead hourly PV capacity-factor forecasting |
| Models | Linear Regression, Random Forest, XGBoost, LightGBM, LSTM, GRU, Transformer, TCN |
| Inputs | PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+ |
| Full grid | 284 configurations per plant |
| Example data | Plants 171, 172, and 186 in `data/` |
| Full data | [Harvard Dataverse DOI: 10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM) |
| Main command | `python run.py ...` |

The GitHub repository includes three example plant files so reviewers can run the code immediately. The paper-scale 100-plant dataset is released separately on Dataverse.

## Quick Start

Use Python 3.10 or newer. The commands below run only on the included sample data.

1. Install dependencies:

```bash
git clone https://github.com/zhesun-0209/NYSolarForecastLab.git
cd NYSolarForecastLab
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Generate plant configuration files:

```bash
python run.py config
```

Expected output:

```text
config/plants/Plant171.yaml
config/plants/Plant172.yaml
config/plants/Plant186.yaml
```

3. Run a two-row Linear Regression smoke test:

```bash
FORCE_CPU=1 python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke
python run.py status --output-dir results/smoke
```

Expected result file:

```text
results/smoke/results_171_all.csv
```

The smoke CSV should contain two successful rows: `Linear_NWP_noTE` and `Linear_NWP+_noTE`.

4. Inspect the smoke result:

```bash
python -c "import pandas as pd; print(pd.read_csv('results/smoke/results_171_all.csv')[['experiment_name','rmse','status']])"
```

5. Plot the included example plants:

```bash
python examples/plot_monthly_generation.py
```

This writes `figures/example_monthly_generation.png`.

## Common Commands

| Goal | Command |
| --- | --- |
| Show CLI help | `python run.py --help` |
| Regenerate configs | `python run.py config` |
| Single-plant full grid | `python run.py forecast --plant-id 171 --output-dir results/plant171` |
| Three included plants | `python run.py multi_plant --plants 171 172 186 --output-dir results/sample_plants` |
| Check progress | `python run.py status --output-dir results/sample_plants` |
| Run one sensitivity analysis | `python run.py sensitivity --analysis lookback_window` |

The runner appends one row after each configuration. Interrupted runs can be resumed. Rows with `status=FAILED` remain in the CSV for auditability and are not counted as completed.

## Full Data

Download the full 100-plant release from [10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM), place the `Project<ID>.csv` files under `data/`, then run:

```bash
python run.py config
python run.py multi_plant --output-dir results/full_release
```

The code is MIT licensed. Dataset access and reuse follow the paper's data availability statement and the Dataverse record; the data are for non-commercial research use and should be cited together with the paper.

## Data Format

Files must be named `Project<ID>.csv`. The three included examples cover plants 171, 172, and 186, each with hourly records from 2020-01-01 to 2024-09-28; experiments use the default benchmark period 2022-01-01 to 2024-09-28.

Required columns:

- `Year`, `Month`, `Day`, `Hour`
- `Capacity Factor`
- weather features such as `global_tilted_irradiance`, `vapour_pressure_deficit`, `relative_humidity_2m`, `temperature_2m`, `wind_gusts_10m`, `cloud_cover_low`, `wind_speed_100m`, `snow_depth`, `dew_point_2m`, `surface_pressure`, and `precipitation`
- matching day-ahead forecast columns with the `_pred` suffix

## Repository Layout

```text
NYSolarForecastLab/
├── run.py                 # Thin CLI wrapper
├── nysolarforecastlab/    # Source package: experiments, models, training, evaluation
├── data/                  # Example plant CSVs
├── config/plants/         # Generated plant configs
├── examples/              # Lightweight plotting helpers
└── tests/                 # Unit and CLI-facing tests
```

## Development Checks

```bash
python run.py --help
python run.py config
python -m pytest -q
```

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{BAO2026105040,
  title = {Toward better integration of solar energy in transportation systems: machine learning benchmarks for day-ahead photovoltaic power forecasting},
  journal = {Transportation Research Part A: Policy and Practice},
  volume = {210},
  pages = {105040},
  year = {2026},
  issn = {0965-8564},
  doi = {10.1016/j.tra.2026.105040},
  url = {https://www.sciencedirect.com/science/article/pii/S0965856426001813},
  author = {Zhaoyao Bao and Zhe Sun and Yishuo Jiang and Chi Xie and Lijun Sun and H. {Oliver Gao}},
  keywords = {Transportation, Solar power, Forecasting, Photovoltaic, Machine learning, Benchmark}
}
```

## License

This project's code is released under the MIT License. See [LICENSE](LICENSE).
