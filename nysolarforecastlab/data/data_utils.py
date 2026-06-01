#!/usr/bin/env python3
"""
Data preprocessing and windowing utilities

Provides functions for:
- Loading and preprocessing PV plant data
- Creating daily and sliding windows
- Feature engineering and normalization
- Data splitting for train/val/test
"""

import pandas as pd
import numpy as np
from nysolarforecastlab.utils.normalization import UnifiedScaler

# Based on actual weather features in data, simplified into two categories
# Solar irradiance features - most important features
IRRADIANCE_FEATURES = [
    'global_tilted_irradiance',    # Global tilted irradiance (most important irradiance feature)
]

# All weather features - removed low correlation features, only keep high correlation features
ALL_WEATHER_FEATURES = [
    'global_tilted_irradiance',    # Global tilted irradiance (most important)
    'vapour_pressure_deficit',     # Vapour pressure deficit
    'relative_humidity_2m',        # Relative humidity at 2m
    'temperature_2m',              # Temperature at 2m
    'wind_gusts_10m',             # Wind gusts at 10m
    'cloud_cover_low',            # Low cloud cover
    'wind_speed_100m',            # Wind speed at 100m
    # Removed low correlation features:
    # Snow depth - low correlation
    # Dew point at 2m - low correlation
    # Surface pressure - low correlation
    # Precipitation - low correlation
]

# Weather feature definitions for sensitivity analysis (11 meteorological variables)
# Based on correlation with PV output
SOLAR_IRRADIANCE_FEATURES = ['global_tilted_irradiance']  # SI: Solar Irradiance only (1 feature)
HIGH_WEATHER_FEATURES = ['global_tilted_irradiance', 'vapour_pressure_deficit', 'relative_humidity_2m']  # H: High correlation (3 features)
MEDIUM_WEATHER_FEATURES = HIGH_WEATHER_FEATURES + ['temperature_2m', 'wind_gusts_10m', 'cloud_cover_low', 'wind_speed_100m']  # H+M (7 features)
LOW_WEATHER_FEATURES = MEDIUM_WEATHER_FEATURES + ['snow_depth', 'dew_point_2m', 'surface_pressure', 'precipitation']  # H+M+L: All 11 features

# Update ALL_WEATHER_FEATURES to include all 11 meteorological variables
ALL_WEATHER_FEATURES = LOW_WEATHER_FEATURES

def get_weather_features_by_category(weather_category):
    """
    Return weather features based on category
    
    Args:
        weather_category: 'none', 'solar_irradiance_only', 'high_weather',
            'medium_weather', or 'low_weather'. A few short aliases are accepted
            for backward compatibility with early config files.
    
    Returns:
        list: List of selected weather features
    """
    aliases = {
        'irradiance': 'solar_irradiance_only',
        'solar': 'solar_irradiance_only',
        'high': 'high_weather',
        'medium': 'medium_weather',
        'low': 'low_weather',
        'all': 'low_weather',
        'all_weather': 'low_weather',
    }
    weather_category = aliases.get(weather_category, weather_category)

    if weather_category == 'none':
        return []  # Return no weather features
    elif weather_category == 'solar_irradiance_only':
        return SOLAR_IRRADIANCE_FEATURES
    elif weather_category == 'high_weather':
        return HIGH_WEATHER_FEATURES
    elif weather_category == 'medium_weather':
        return MEDIUM_WEATHER_FEATURES
    elif weather_category == 'low_weather':
        return LOW_WEATHER_FEATURES
    else:
        raise ValueError(f"Invalid weather_category: {weather_category}")

# Maintain backward compatibility
BASE_HIST_FEATURES = IRRADIANCE_FEATURES
BASE_FCST_FEATURES = IRRADIANCE_FEATURES

# Time encoding features
TIME_FEATURES = ['month_cos', 'month_sin', 'hour_cos', 'hour_sin']

TARGET_COL = 'Capacity Factor'

# Statistical feature functions removed

def load_raw_data(path: str) -> pd.DataFrame:
    """Load a plant CSV file and add a normalized Datetime column."""
    df = pd.read_csv(path)
    required_columns = ['Year', 'Month', 'Day', 'Hour', TARGET_COL]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    return df

def preprocess_features(df: pd.DataFrame, config: dict):
    df_clean = df.dropna(subset=[TARGET_COL]).copy()

    # Date filtering: CRITICAL - ensure all experiments use data starting from 2022-01-01
    # This is a double-check to ensure date filtering even if data was not pre-filtered
    start_date = config.get('start_date', '2022-01-01')
    end_date = config.get('end_date', '2024-09-28')
    
    # Force start_date to be at least 2022-01-01 if not specified or earlier
    if not start_date:
        start_date = '2022-01-01'
    else:
        start_dt_check = pd.to_datetime(start_date)
        min_start_dt = pd.to_datetime('2022-01-01')
        if start_dt_check < min_start_dt:
            start_date = '2022-01-01'
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        rows_before = len(df_clean)
        df_clean = df_clean[df_clean['Datetime'] >= start_dt].copy()
        if len(df_clean) < rows_before:
            print(f"  [Date Filter] Filtered {rows_before - len(df_clean)} rows before {start_date}")
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        rows_before = len(df_clean)
        df_clean = df_clean[df_clean['Datetime'] <= end_dt].copy()
        if len(df_clean) < rows_before:
            print(f"  [Date Filter] Filtered {rows_before - len(df_clean)} rows after {end_date}")

    # Add time encoding features (based on switch)
    use_time_encoding = config.get('use_time_encoding', True)
    if use_time_encoding:
        df_clean['month_cos'] = np.cos(2 * np.pi * df_clean['Month'] / 12)
        df_clean['month_sin'] = np.sin(2 * np.pi * df_clean['Month'] / 12)
        df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['Hour'] / 24)
        df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['Hour'] / 24)

    # Build feature lists
    hist_feats = []
    fcst_feats = []

    # Get weather feature category
    weather_category = config.get('weather_category', 'none')

    # PV features (historical power generation)
    if config.get('use_pv', False):
        # Create historical power features: use shift operation to get past power generation
        # Don't directly copy target values, use shift(1) to get previous time point values
        # In sliding windows, this will provide true historical power generation data
        df_clean['Capacity_Factor_hist'] = df_clean[TARGET_COL].shift(1)
        hist_feats.append('Capacity_Factor_hist')

    # Historical weather features (HW) - without _pred suffix
    if config.get('use_hist_weather', False):
        hist_feats += get_weather_features_by_category(weather_category)

    # Time encoding features (based on switch)
    if use_time_encoding:
        hist_feats += TIME_FEATURES

    # Forecast features (NWP)
    if config.get('use_forecast', False):
        if config.get('use_ideal_nwp', False):
            # Ideal NWP+: use target day HW features (without _pred suffix)
            base_weather_features = get_weather_features_by_category(weather_category)
            fcst_feats += base_weather_features
            # Add time encoding features to forecast features
            if use_time_encoding:
                fcst_feats += TIME_FEATURES
        else:
            # Normal NWP: use forecast features with _pred suffix
            base_weather_features = get_weather_features_by_category(weather_category)
            forecast_features = [f + '_pred' for f in base_weather_features]
            fcst_feats += forecast_features
            # Add time encoding features to forecast features
            if use_time_encoding:
                fcst_feats += TIME_FEATURES

    # Check if pure NWP mode (no historical data)
    no_hist_power = config.get('no_hist_power', False)
    # Auto-detect only when user hasn't explicitly set no_hist_power
    if 'no_hist_power' not in config:
        # Auto-detect: if neither historical power nor historical weather, then pure NWP mode
        no_hist_power = not config.get('use_pv', False) and not config.get('use_hist_weather', False)

    available_hist_feats = [f for f in hist_feats if f in df_clean.columns]
    available_fcst_feats = [f for f in fcst_feats if f in df_clean.columns]

    na_check_feats = available_fcst_feats + [TARGET_COL]
    df_clean = df_clean.dropna(subset=na_check_feats).reset_index(drop=True)
    
    for feat in available_hist_feats:
        df_clean[feat] = df_clean[feat].fillna(0.0)

    # Get normalization method (default to minmax for backward compatibility)
    normalization_method = config.get('normalization_method', 'minmax')
    
    # Normalize historical features
    scaler_hist = UnifiedScaler(method=normalization_method)
    if available_hist_feats:
        # UnifiedScaler will automatically handle zero variance features, but check and warn first
        for feat in available_hist_feats:
            if df_clean[feat].std() < 1e-8:
                print(f"  [Warning] Feature {feat} has zero std, will be handled by UnifiedScaler")
        df_clean[available_hist_feats] = scaler_hist.fit_transform(df_clean[available_hist_feats].values)

    # Normalize forecast features
    scaler_fcst = UnifiedScaler(method=normalization_method)
    if available_fcst_feats:
        for feat in available_fcst_feats:
            if df_clean[feat].std() < 1e-8:
                print(f"  [Warning] Feature {feat} has zero std, will be handled by UnifiedScaler")
        df_clean[available_fcst_feats] = scaler_fcst.fit_transform(df_clean[available_fcst_feats].values)

    # Normalize target variable
    scaler_target = UnifiedScaler(method=normalization_method)
    # Check if target variable has zero variance
    target_values = df_clean[[TARGET_COL]].values
    if np.std(target_values) < 1e-8:
        print(f"  [Warning] Target variable has zero std, will be handled by UnifiedScaler")
    df_clean[TARGET_COL] = scaler_target.fit_transform(target_values).flatten()

    df_clean = df_clean.sort_values('Datetime').reset_index(drop=True)

    return df_clean, available_hist_feats, available_fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power

def create_daily_windows(df, future_hours, hist_feats, fcst_feats, no_hist_power=False, past_hours=24):
    """
    Create daily prediction samples (one sample per day)
    
    Each sample represents a day-ahead forecast:
        Input: Previous N hours (historical) + Next day's NWP (forecast)
        Output: Next day's 24 hours (target)
    
    Args:
        df: Preprocessed dataframe with hourly data
        future_hours: Should be 24 (full day prediction)
        hist_feats: Historical feature columns
        fcst_feats: Forecast feature columns  
        no_hist_power: If True, only use forecast features
        past_hours: Lookback window in hours (24 or 72, default 24)
    
    Returns:
        X_hist: (n_days, past_hours, n_hist_feats)
        X_fcst: (n_days, 24, n_fcst_feats) or None
        y: (n_days, 24)
        hours: (n_days,) - all 23 (prediction made at end of day)
        dates: (n_days,) - next day's date
    """
    TARGET_COL = 'Capacity Factor'
    if future_hours != 24:
        raise ValueError("create_daily_windows currently supports a 24-hour day-ahead horizon only.")
    df = df.copy()
    
    # Group by date
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    unique_dates = sorted(df['date'].unique())
    
    X_hist_list = []
    X_fcst_list = []
    y_list = []
    dates_list = []
    
    # Calculate how many days we need for lookback
    past_days = past_hours // 24  # 24h=1day, 72h=3days
    
    # For NWP-only mode with past_hours=0, we don't need historical days
    if no_hist_power and past_hours == 0:
        past_days = 0
    
    # Create samples: use past N days to predict next day
    for i in range(len(unique_dates) - 1):
        # Need at least past_days of history before current day (unless NWP-only with past_hours=0)
        if not (no_hist_power and past_hours == 0) and i < past_days:
            continue
        
        # Get historical days (past_days before prediction day)
        if no_hist_power and past_hours == 0:
            # NWP-only mode: no historical data needed
            hist_dates = []
            next_date = unique_dates[i + 1]
            hist_data_list = []
            valid_hist = True
        else:
            hist_dates = unique_dates[i - past_days + 1 : i + 1]
            next_date = unique_dates[i + 1]
            
            # Collect historical data
            hist_data_list = []
            valid_hist = True
            for hist_date in hist_dates:
                hist_day = df[df['date'] == hist_date].copy()
                if len(hist_day) != 24:
                    valid_hist = False
                    break
                hist_day = hist_day.sort_values('Hour')
                hist_data_list.append(hist_day)
            
            if not valid_hist:
                continue
        
        # Get forecast day (next day)
        next_day = df[df['date'] == next_date].copy()
        if len(next_day) != 24:
            continue
        next_day = next_day.sort_values('Hour')
        
        # Historical features: concatenate all historical days
        if hist_feats and not no_hist_power and past_hours > 0:
            hist_arrays = []
            for hist_day in hist_data_list:
                if hist_day[hist_feats].isnull().any().any():
                    valid_hist = False
                    break
                hist_arrays.append(hist_day[hist_feats].values)  # (24, n_hist)
            
            if not valid_hist:
                continue
            
            X_hist = np.vstack(hist_arrays)  # (past_hours, n_hist)
        elif no_hist_power and past_hours == 0:
            # NWP-only mode with no historical data: create empty historical features
            # Use shape (1, 0) to avoid RNN sequence length error
            X_hist = np.zeros((1, 0))  # (1, 0) - minimal shape for compatibility
        else:
            # For other cases: create dummy historical features
            X_hist = np.zeros((max(1, past_hours), 1 if not hist_feats else len(hist_feats)))
        
        # Forecast features: Next day's 24 hours (NWP for day to predict)
        if fcst_feats:
            if next_day[fcst_feats].isnull().any().any():
                continue
            X_fcst = next_day[fcst_feats].values  # (24, n_fcst)
        else:
            X_fcst = None
        
        # Target: Next day's 24 hours
        if next_day[TARGET_COL].isnull().any():
            continue
        y = next_day[TARGET_COL].values  # (24,)
        
        X_hist_list.append(X_hist)
        if X_fcst is not None:
            X_fcst_list.append(X_fcst)
        y_list.append(y)
        dates_list.append(str(next_date.date()))
    
    if len(X_hist_list) == 0:
        raise ValueError("Unable to create any valid daily samples")
    
    X_hist = np.array(X_hist_list)
    y = np.array(y_list)
    
    if fcst_feats and len(X_fcst_list) > 0:
        X_fcst = np.array(X_fcst_list)
    else:
        X_fcst = None
    
    # All predictions made at 23:00 (end of day)
    hours = np.full(len(X_hist), 23, dtype=np.int64)
    dates = dates_list
    
    return X_hist, X_fcst, y, hours, dates


def create_sliding_windows(df, past_hours, future_hours, hist_feats, fcst_feats, no_hist_power=False):
    """
    Create sliding window sequences for time series data
    
    Args:
        df: Preprocessed dataframe
        past_hours: Historical time window (hours)
        future_hours: Prediction time window (hours)
        hist_feats: List of historical features
        fcst_feats: List of forecast features
        no_hist_power: If True, don't use historical power data, only use forecast weather
    """
    X_hist, y, hours, dates = [], [], [], []
    X_fcst = [] if fcst_feats else None
    n = len(df)
    
    for i in range(past_hours, n - future_hours + 1):
        hist_start = i - past_hours
        hist_end = i
        hist_data = df.iloc[hist_start:hist_end]
        
        fut_start = i
        fut_end = i + future_hours
        fut_data = df.iloc[fut_start:fut_end]
        
        if no_hist_power:
            if fcst_feats:
                X_fcst.append(fut_data[fcst_feats].values)
            
            y.append(fut_data[TARGET_COL].values)
            hours.append(fut_data['Hour'].values)
            dates.append(fut_data['Datetime'].iloc[-1])
            
            if past_hours > 0 and len(hist_feats) > 0:
                empty_hist = np.zeros((past_hours, len(hist_feats)))
            elif past_hours == 0 and len(hist_feats) > 0:
                empty_hist = np.zeros((1, len(hist_feats)))
            elif past_hours > 0 and len(hist_feats) == 0:
                empty_hist = np.zeros((past_hours, 0))
            else:
                empty_hist = np.zeros((1, 0))
            X_hist.append(empty_hist)
        else:
            if len(hist_data) < past_hours:
                continue
            
            if len(fut_data) < future_hours:
                continue
            
            if hist_data[hist_feats].isnull().any().any():
                continue
            
            X_hist.append(hist_data[hist_feats].values)
            
            if fcst_feats:
                X_fcst.append(fut_data[fcst_feats].values)
            
            y.append(fut_data[TARGET_COL].values)
            hours.append(fut_data['Hour'].values)
            dates.append(fut_data['Datetime'].iloc[-1])
    
    if len(X_hist) == 0:
        raise ValueError("Unable to create any valid samples")
    
    X_hist = np.array(X_hist)
    y = np.array(y)
    
    if X_fcst is not None:
        X_fcst = np.array(X_fcst)
    
    return X_hist, X_fcst, y, hours, dates

def split_data(X_hist, X_fcst, y, hours, dates, train_ratio=0.8, val_ratio=0.1, shuffle=True, random_state=42):
    """
    Split data into train, validation and test sets
    """
    N = X_hist.shape[0]
    
    if shuffle:
        np.random.seed(random_state)
        indices = np.random.permutation(N)
    else:
        indices = np.arange(N)
    
    i_tr = int(N * train_ratio)
    i_val = int(N * (train_ratio + val_ratio))
    
    train_idx = indices[:i_tr]
    val_idx = indices[i_tr:i_val]
    test_idx = indices[i_val:]
    
    def slice_array(arr, indices):
        if isinstance(arr, np.ndarray):
            return arr[indices]
        else:
            return [arr[i] for i in indices]

    Xh_tr, Xh_va, Xh_te = slice_array(X_hist, train_idx), slice_array(X_hist, val_idx), slice_array(X_hist, test_idx)
    y_tr, y_va, y_te = slice_array(y, train_idx), slice_array(y, val_idx), slice_array(y, test_idx)
    hrs_tr, hrs_va, hrs_te = slice_array(hours, train_idx), slice_array(hours, val_idx), slice_array(hours, test_idx)
    
    dates_tr = [dates[i] for i in train_idx]
    dates_va = [dates[i] for i in val_idx]
    dates_te = [dates[i] for i in test_idx]

    if X_fcst is not None:
        Xf_tr, Xf_va, Xf_te = slice_array(X_fcst, train_idx), slice_array(X_fcst, val_idx), slice_array(X_fcst, test_idx)
    else:
        Xf_tr = Xf_va = Xf_te = None

    return (
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr,
        Xh_va, Xf_va, y_va, hrs_va, dates_va,
        Xh_te, Xf_te, y_te, hrs_te, dates_te
    )
