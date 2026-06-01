# Data Notes

This directory contains benchmark-format PV plant CSV files. The current repository snapshot includes:

| File | Plant ID | Rows | Period in file | Default benchmark period |
| --- | ---: | ---: | --- | --- |
| `Project171.csv` | 171 | 41,609 | 2020-01-01 to 2024-09-28 | 2022-01-01 to 2024-09-28 |
| `Project172.csv` | 172 | 41,609 | 2020-01-01 to 2024-09-28 | 2022-01-01 to 2024-09-28 |
| `Project186.csv` | 186 | 41,609 | 2020-01-01 to 2024-09-28 | 2022-01-01 to 2024-09-28 |

The experiment code filters all plant files to the default benchmark period unless a plant configuration specifies a later start or earlier end date. These committed files are example fixtures for code verification, not a substitute for the paper-scale benchmark release.

## Full Data Release

The full 100-plant benchmark dataset is available at:

[https://doi.org/10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM)

The paper notes that the complete data volume is larger than 1.5 GB, so the GitHub repository carries only three example plant files. The full data release is for non-commercial research use and should be cited together with the paper.

The MIT license in this repository applies to the code. Dataset access and reuse are governed by the paper's data availability statement and the Dataverse record.

## Required Columns

Every `Project<ID>.csv` file must include:

| Column | Meaning |
| --- | --- |
| `Year`, `Month`, `Day`, `Hour` | Local hourly timestamp fields |
| `Capacity Factor` | PV generation target on a percentage-capacity-factor scale |
| `global_tilted_irradiance` | Historical irradiance feature |
| `vapour_pressure_deficit`, `relative_humidity_2m`, `temperature_2m` | High-correlation weather features |
| `wind_gusts_10m`, `cloud_cover_low`, `wind_speed_100m` | Medium-correlation weather features |
| `snow_depth`, `dew_point_2m`, `surface_pressure`, `precipitation` | Lower-correlation weather features |
| `<weather_feature>_pred` | Day-ahead numerical weather prediction version of each weather feature |

Additional metadata columns such as plant coordinates and raw date strings are preserved by the CSVs but are not required by the training pipeline.

## Feature Groups

The code uses the following weather groups:

| Config value | Included features |
| --- | --- |
| `none` | no weather features |
| `solar_irradiance_only` | `global_tilted_irradiance` |
| `high_weather` | irradiance, vapour pressure deficit, relative humidity |
| `medium_weather` | high-weather features plus temperature, wind gusts, low cloud cover, 100 m wind speed |
| `low_weather` | all 11 weather features listed above |

Forecast features use the same group names with the `_pred` suffix unless `use_ideal_nwp: true`, which uses realized target-day weather as an idealized forecast upper-bound setting.

## Adding the Full Benchmark Release

Download additional plant files from [https://doi.org/10.7910/DVN/3VKAGM](https://doi.org/10.7910/DVN/3VKAGM), place them in this directory using the same `Project<ID>.csv` naming convention, then run:

```bash
python run.py config
python run.py multi_plant --output-dir results/full_release
python run.py status --output-dir results/full_release
```

The generated plant configs will appear under `config/plants/`.

Each plant writes `results_<PlantID>_all.csv` under the selected output directory.
