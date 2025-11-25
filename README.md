# PV-Forecasting

A comprehensive multi-plant solar power forecasting system supporting both deep learning and machine learning models.

## Features

- **Multiple Models**: Support for LSTM, GRU, Transformer, TCN (DL) and Random Forest, XGBoost, LightGBM, Linear Regression (ML)
- **Multi-Plant Support**: Batch processing for multiple solar plants with resume capability
- **Flexible Features**: Support for PV historical data, historical weather, and NWP forecasts
- **Comprehensive Experiments**: 284 experiment configurations per plant
- **Sensitivity Analysis**: 8 different sensitivity analysis experiments
- **Unified Interface**: Single entry point (`run.py`) for all operations

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Generate Configuration Files

```bash
python run.py config
```

This automatically generates configuration files for all CSV files in the `data/` directory.

### 2. Run Single Plant Experiments

```bash
python run.py forecast --plant-id 1140
```

Runs all 284 experiments for plant 1140.

### 3. Run Multi-Plant Batch

```bash
# Run first 25 plants
python run.py multi_plant --max-plants 25

# Run specific plants
python run.py multi_plant --plants 1140 1141 1142

# Skip first 10 plants, run next 20
python run.py multi_plant --skip 10 --max-plants 20
```

### 4. Check Experiment Status

```bash
python run.py status
```

### 5. Run Sensitivity Analysis

```bash
python run.py sensitivity --analysis lookback_window
```

Available analysis types:
- `lookback_window`: Analyze effect of lookback window length
- `model_complexity`: Analyze effect of model complexity
- `training_scale`: Analyze effect of training dataset size
- `seasonal_effect`: Analyze seasonal performance variations
- `hourly_effect`: Analyze hourly performance variations
- `weather_feature`: Analyze effect of weather feature tiers

## Project Structure

```
PV-Forecasting/
├── config/              # Configuration management
│   ├── config_manager.py
│   ├── plant_template.yaml
│   └── plants/          # Plant-specific configs
├── data/                # Data utilities
│   └── data_utils.py
├── eval.py              # Evaluation metrics and result saving
├── experiments.py       # Experiment running logic
├── models/              # Model implementations
│   ├── rnn_models.py    # LSTM, GRU
│   ├── transformer.py   # Transformer
│   ├── tcn.py           # TCN
│   └── ml_models.py     # RF, XGB, LGBM, Linear
├── run.py               # Unified entry point
├── sensitivity_analysis/ # Sensitivity analysis experiments
├── train/               # Training modules
│   ├── train_dl.py      # Deep learning training
│   └── train_ml.py      # Machine learning training
└── utils/               # Utility functions
    └── normalization.py # Unified scaler
```

## Experiment Configuration

Each plant runs 284 experiments covering:
- **Models**: 4 DL + 3 ML + 1 Linear = 8 models
- **Complexity**: Low, High (2 levels)
- **Lookback**: 24h, 72h (2 options)
- **Time Encoding**: True, False (2 options)
- **Features**: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+ (6 combinations)

Total: 8 × 2 × 2 × 2 × 6 = 192 (adjusted to 284 with specific rules)

## Data Format

CSV files should contain:
- `Year`, `Month`, `Day`, `Hour`: Time columns
- `Capacity_Factor`: Target variable (PV power output)
- Weather features: `global_tilted_irradiance`, `temperature_2m`, etc.
- Forecast features: `*_pred` suffix for NWP forecasts

## Output

Results are saved as:
- `summary.csv`: Experiment summary metrics
- `predictions.csv`: Detailed predictions
- `training_log.csv`: Training history (DL models)
- Excel files: Aggregated results per plant

## License

MIT License - see LICENSE file for details

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pv_forecasting,
  title={PV-Forecasting: Multi-Plant Solar Power Prediction System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/PV-Forecasting}
}
```

