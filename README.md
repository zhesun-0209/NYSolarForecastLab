# NYSolarForecastLab

Code and benchmark data utilities for the paper:

**Toward better integration of solar energy in transportation systems: machine learning benchmarks for day-ahead photovoltaic power forecasting**

Zhaoyao Bao, Zhe Sun, Yishuo Jiang, Chi Xie, Lijun Sun, and H. Oliver Gao

*Transportation Research Part A: Policy and Practice*, 210, 105040, 2026
DOI: [10.1016/j.tra.2026.105040](https://doi.org/10.1016/j.tra.2026.105040)

This repository supports reproducible day-ahead photovoltaic (PV) power forecasting experiments for transportation energy-management and planning applications. It includes unified data preprocessing, model training, evaluation, multi-plant experiment runners, and sensitivity-analysis scripts.

## What Is Included

- Eight forecasting methods: Linear Regression, Random Forest, XGBoost, LightGBM, LSTM, GRU, Transformer, and TCN.
- Six input settings: PV, PV+HW, PV+NWP, PV+NWP+, NWP, and NWP+.
- Two model-complexity levels, two lookback windows, and time-encoding ablations.
- A reproducible 284-configuration experiment grid per plant.
- Example plant CSV files in `data/` with the same schema expected by the benchmark pipeline.

The committed CSV files are example benchmark-format plant files that make the repository runnable without external downloads. Download the full 100-plant data release from [https://doi.org/10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM), place the `Project<ID>.csv` files under `data/`, and run the same commands below.

## Data And Code Availability

The paper states that the benchmark code is hosted in this GitHub repository and that the full dataset is available through Harvard Dataverse:

- Code: [https://github.com/zhesun-0209/NYSolarForecastLab](https://github.com/zhesun-0209/NYSolarForecastLab)
- Full 100-plant dataset: [https://doi.org/10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM)

Because the full data release is larger than 1.5 GB, this repository includes only three example PV plants for quick verification. The dataset is released for non-commercial research use; cite the paper when using either the code or data.

## Installation

Use Python 3.10 or newer.

```bash
git clone https://github.com/zhesun-0209/NYSolarForecastLab.git
cd NYSolarForecastLab
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For tests and documentation tooling, install `requirements-dev.txt`.

On Apple Silicon or constrained environments, prefix commands with `FORCE_CPU=1` if PyTorch advertises MPS but a model run fails.

## Quick Start

Generate plant configuration files for every `data/Project*.csv` file:

```bash
python run.py config
```

Run a fast smoke test on the included Plant 171 data:

```bash
python run.py forecast --plant-id 171 --test-mode --test-model Linear --output-dir results/smoke
```

Run the full 284-configuration grid for one plant:

```bash
python run.py forecast --plant-id 171 --output-dir results/plant171
```

Run multiple configured plants:

```bash
python run.py multi_plant --plants 171 172 186 --output-dir results/sample_plants
```

Check progress:

```bash
python run.py status --output-dir results/sample_plants
```

## Experiment Grid

Each full plant benchmark contains 280 non-Linear model configurations plus 4 Linear Regression configurations:

| Component | Values |
| --- | --- |
| Deep learning models | LSTM, GRU, Transformer, TCN |
| Machine-learning models | Random Forest, XGBoost, LightGBM |
| Baseline | Linear Regression |
| Complexity | low, high |
| Lookback | 24 h, 72 h for PV-based inputs; 0 h for NWP-only inputs |
| Time encoding | enabled, disabled |
| Input sets | PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+ |

The experiment runner writes CSV summaries incrementally, so interrupted runs can be resumed.

## Data Format

Plant files must be named `Project<ID>.csv` and include:

- Time columns: `Year`, `Month`, `Day`, `Hour`
- Target: `Capacity Factor` as a percentage-scale capacity factor
- Historical weather columns such as `global_tilted_irradiance`, `temperature_2m`, `relative_humidity_2m`
- Numerical weather prediction columns with the `_pred` suffix, for example `global_tilted_irradiance_pred`

See [data/README.md](data/README.md) for the complete schema notes and feature groups used by the code.

## Sensitivity Analyses

```bash
python run.py sensitivity --analysis lookback_window
python run.py sensitivity --analysis model_complexity
python run.py sensitivity --analysis training_scale
python run.py sensitivity --analysis seasonal_effect
python run.py sensitivity --analysis hourly_effect
python run.py sensitivity --analysis weather_feature
```

## Repository Layout

```text
NYSolarForecastLab/
├── config/                  # Plant and experiment configuration
├── data/                    # Example plant CSVs and data documentation
├── docs/                    # Sphinx documentation
├── models/                  # Forecasting model implementations
├── sensitivity_analysis/    # Ablation and sensitivity-analysis scripts
├── tests/                   # Unit tests and interface smoke tests
├── train/                   # Training pipelines for DL and ML models
├── eval.py                  # Metrics and result export helpers
├── experiments.py           # Experiment grid and batch runners
└── run.py                   # Command-line entry point
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

This project is released under the MIT License. See [LICENSE](LICENSE).
