#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common utilities for sensitivity analysis experiments
Shared functions for running experiments and computing metrics
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress specific library warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Suppress pandas warnings
warnings.filterwarnings('ignore', module='pandas')

# Suppress numpy warnings
warnings.filterwarnings('ignore', module='numpy')

# Suppress sklearn warnings
warnings.filterwarnings('ignore', module='sklearn')

# Suppress XGBoost warnings
warnings.filterwarnings('ignore', module='xgboost')

# Suppress LightGBM warnings
warnings.filterwarnings('ignore', module='lightgbm')

# Suppress cuML warnings
warnings.filterwarnings('ignore', module='cuml')

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', module='torch')

# Set environment variables to suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Suppress specific library output
import logging
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.getLogger('cuml').setLevel(logging.ERROR)

# Suppress LightGBM initialization output
import os
os.environ['LIGHTGBM_VERBOSE'] = '0'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nysolarforecastlab.config.config_manager import PlantConfigManager


def set_global_seed(seed=42):
    """
    Set global random seed for full reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    import torch
    
    # Python random seed
    random.seed(seed)
    
    # NumPy random seed
    np.random.seed(seed)
    
    # PyTorch random seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Set global random seed: {seed}")

from nysolarforecastlab.data.data_utils import load_raw_data, preprocess_features, create_daily_windows, create_sliding_windows, split_data
from nysolarforecastlab.train.train_dl import train_dl_model
from nysolarforecastlab.train.train_ml import train_ml_model
import numpy as np
import pandas as pd


def load_and_filter_data(data_path: str, plant_config: Dict) -> pd.DataFrame:
    """
    Load and filter data by date range
    CRITICAL: Ensures all data starts from 2022-01-01
    
    Args:
        data_path: Path to data CSV file
        plant_config: Plant configuration dictionary
        
    Returns:
        Filtered DataFrame with Datetime column
    """
    df = load_raw_data(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    # CRITICAL: Filter data by date range BEFORE any processing
    # Ensure all experiments use data starting from 2022-01-01
    start_date = plant_config.get('start_date', '2022-01-01')
    end_date = plant_config.get('end_date', '2024-09-28')
    
    # Force start_date to be at least 2022-01-01
    if not start_date or start_date < '2022-01-01':
        start_date = '2022-01-01'
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df['Datetime'] >= start_dt].copy()
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df['Datetime'] <= end_dt].copy()
    
    return df


# Model definitions
DL_MODELS = ['LSTM', 'GRU', 'Transformer', 'TCN']
ML_MODELS = ['RF', 'XGB', 'LGBM']
ALL_MODELS_NO_LINEAR = DL_MODELS + ML_MODELS  # 7 models excluding Linear


def get_season(month: int) -> str:
    """
    Get season from month
    
    Args:
        month: Month number (1-12)
        
    Returns:
        Season name: Spring, Summer, Fall, Winter
    """
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:  # 12, 1, 2
        return 'Winter'


def compute_nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Normalized RMSE (RMSE / range of true values)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        NRMSE value (as percentage of data range)
    """
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_range = np.max(y_true) - np.min(y_true)
    if y_range == 0:
        return np.inf
    return rmse / y_range


def create_base_config(plant_config: Dict, model: str, complexity: str = 'high', 
                       lookback: int = 24, use_te: bool = False) -> Dict:
    """
    Create base experiment configuration
    
    Args:
        plant_config: Plant configuration
        model: Model name
        complexity: Model complexity ('low', 'mid_low', 'mid_high', 'high')
        lookback: Lookback window hours
        use_te: Use time encoding
        
    Returns:
        Experiment configuration dictionary
    """
    config = {
        'plant_id': plant_config['plant_id'],
        'data_path': plant_config['data_path'],
        'model': model,
        'model_complexity': complexity,
        'use_pv': True,
        'use_hist_weather': False,
        'use_forecast': True,
        'use_ideal_nwp': False,
        'use_time_encoding': use_te,
        'past_hours': lookback,
        'past_days': lookback // 24,
        'no_hist_power': False,  # PV+NWP uses historical power data
        'future_hours': plant_config.get('future_hours', 24),
        'train_ratio': plant_config.get('train_ratio', 0.8),
        'val_ratio': plant_config.get('val_ratio', 0.1),
        'test_ratio': plant_config.get('test_ratio', 0.1),
        'shuffle_split': plant_config.get('shuffle_split', True),  # Default to shuffle for robust evaluation
        'random_seed': plant_config.get('random_seed', 42),
        'weather_category': plant_config.get('weather_category', 'medium_weather'),
        'start_date': plant_config.get('start_date', '2022-01-01'),  # Default to 2022-01-01
        'end_date': plant_config.get('end_date', '2024-09-28'),
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_training_log': False,
            'save_excel_results': False
        }
    }
    
    # CRITICAL: Force start_date to be at least 2022-01-01
    if not config['start_date'] or config['start_date'] < '2022-01-01':
        config['start_date'] = '2022-01-01'
    
    # Add model parameters
    if model in DL_MODELS:
        # Get DL parameters from plant config
        dl_params = plant_config.get('dl_params', {}).get(complexity, {})
        config['train_params'] = {
            'epochs': dl_params.get('epochs', 20 if complexity == 'low' else 50),
            'batch_size': dl_params.get('batch_size', 64),
            'learning_rate': dl_params.get('learning_rate', 0.001),
            'patience': dl_params.get('patience', 10),
            'min_delta': dl_params.get('min_delta', 0.001),
            'weight_decay': dl_params.get('weight_decay', 0.0001)
        }
        config['model_params'] = {
            'd_model': dl_params.get('d_model', 16 if complexity == 'low' else 32),
            'hidden_dim': dl_params.get('hidden_dim', 8 if complexity == 'low' else 16),
            'num_heads': dl_params.get('num_heads', 2),
            'num_layers': dl_params.get('num_layers', 1 if complexity == 'low' else 2),
            'dropout': dl_params.get('dropout', 0.1),
            'tcn_channels': dl_params.get('tcn_channels', [8, 16] if complexity == 'low' else [16, 32]),
            'kernel_size': dl_params.get('kernel_size', 3)
        }
    elif model in ML_MODELS:
        # Get ML parameters from plant config (same as multi-plant)
        ml_params = plant_config.get('ml_params', {}).get(complexity, {})
        config['model_params'] = {
            'n_estimators': ml_params.get('n_estimators', 10 if complexity == 'low' else 30),
            'max_depth': ml_params.get('max_depth', 1 if complexity == 'low' else 3),
            'learning_rate': ml_params.get('learning_rate', 0.2 if complexity == 'low' else 0.1),
            'random_state': ml_params.get('random_state', 42),
            'verbosity': ml_params.get('verbosity', -1)  # Silent mode: suppress warnings
        }
    elif model == 'Linear':
        config['model_complexity'] = None
        config['model_params'] = {}
    
    return config


def run_single_experiment(config: Dict, df: pd.DataFrame, use_sliding_windows: bool = False) -> Dict:
    """
    Run a single experiment with complete data preparation and training
    Uses the same processing pipeline as multi-plant experiments
    
    Args:
        config: Experiment configuration
        df: Raw dataframe with Datetime column
        use_sliding_windows: If True, use hourly sliding windows; if False, use daily windows
        
    Returns:
        Result dictionary with metrics and predictions
    """
    import time
    
    # Start timing (includes data processing, same as main experiments)
    start_time = time.time()
    
    # Set random seed for reproducibility
    random_seed = config.get('random_seed', 42)
    set_global_seed(random_seed)
    
    try:
        # Data preprocessing (same as multi-plant)
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = preprocess_features(df, config)
        
        # Create windows (same as multi-plant)
        if use_sliding_windows:
            # Hourly sliding windows for dataset extension experiment
            X_hist, X_fcst, y, hours, dates = create_sliding_windows(
                df_clean,
                past_hours=config.get('past_hours', 24),
                future_hours=config.get('future_hours', 24),
                hist_feats=hist_feats,
                fcst_feats=fcst_feats,
                no_hist_power=no_hist_power
            )
        else:
            # Daily windows (default, same as multi-plant)
            past_hours = config.get('past_hours', 24)
            X_hist, X_fcst, y, hours, dates = create_daily_windows(
                df_clean, config['future_hours'], hist_feats, fcst_feats, no_hist_power, past_hours
            )
        
        # Data splitting (same as multi-plant)
        total_samples = len(X_hist)
        indices = np.arange(total_samples)
        
        shuffle_split = config.get('shuffle_split', True)  # Default to shuffle for robust evaluation
        random_seed = config.get('random_seed', 42)
        
        if shuffle_split:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        
        train_size = int(total_samples * train_ratio)
        val_size = int(total_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        # Removed verbose data split and test period output
        
        X_hist_train, y_train = X_hist[train_idx], y[train_idx]
        X_hist_val, y_val = X_hist[val_idx], y[val_idx]
        X_hist_test, y_test = X_hist[test_idx], y[test_idx]
        
        if X_fcst is not None:
            X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
        else:
            X_fcst_train = X_fcst_val = X_fcst_test = None
        
        # Split hours and dates
        train_hours = np.array([hours[i] for i in train_idx])
        val_hours = np.array([hours[i] for i in val_idx])
        test_hours = np.array([hours[i] for i in test_idx])
        test_dates = [dates[i] for i in test_idx]
        
        train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
        val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
        test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
        scalers = (scaler_hist, scaler_fcst, scaler_target)
        
        # Train model
        if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
        else:
            model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
        
        # Calculate total time including data processing (same as main experiments)
        # This matches the timing in experiments.py: training_time = time.time() - start_time
        total_time = time.time() - start_time
        
        # Return result with predictions
        return {
            'model': config['model'],
            'complexity': config.get('model_complexity', 'N/A'),
            'mae': metrics.get('mae', 0.0),
            'rmse': metrics.get('rmse', 0.0),
            'r2': metrics.get('r2', 0.0),
            'nrmse': metrics.get('nrmse', 0.0),
            'train_time': round(total_time, 2),  # Total time including data processing (same as main experiments)
            'test_samples': metrics.get('samples_count', 0),
            'status': 'SUCCESS',
            'y_test_pred': metrics.get('predictions_all', None),  # Full predictions matrix
            'y_test': metrics.get('y_true_all', y_test),  # Full ground truth matrix
            'test_dates': test_dates,  # Dates for season/hour analysis
            'test_hours': test_hours  # Hours for hourly analysis
        }
        
    except Exception as e:
        import traceback
        print(f"Error in experiment: {str(e)}")
        traceback.print_exc()
        return {
            'model': config.get('model', 'Unknown'),
            'complexity': config.get('model_complexity', 'N/A'),
            'mae': np.nan,
            'rmse': np.nan,
            'r2': np.nan,
            'nrmse': np.nan,
            'train_time': 0,
            'test_samples': 0,
            'status': 'FAILED',
            'error': str(e)
        }


def aggregate_results(results: List[Dict], group_by: str = None) -> pd.DataFrame:
    """
    Aggregate results across multiple plants
    
    Args:
        results: List of result dictionaries
        group_by: Column to group by (e.g., 'model', 'season', 'hour')
        
    Returns:
        DataFrame with mean and std for each group
    """
    df = pd.DataFrame(results)
    
    # Remove failed experiments
    df = df[df['status'] == 'SUCCESS']
    
    if len(df) == 0:
        print("Warning: No successful experiments to aggregate")
        return pd.DataFrame()
    
    # Compute NRMSE
    if 'rmse' in df.columns and 'mean_true' in df.columns:
        df['nrmse'] = df['rmse'] / df['mean_true']
    
    # Metrics to aggregate
    metrics = ['mae', 'rmse', 'r2', 'nrmse', 'train_time']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if group_by and group_by in df.columns:
        # Group by specified column
        grouped = df.groupby(group_by)[available_metrics]
        
        # Compute mean and std
        mean_df = grouped.mean().round(2)
        std_df = grouped.std().round(2)
        
        # Combine mean and std
        result_df = pd.DataFrame()
        for metric in available_metrics:
            result_df[f'{metric}_mean'] = mean_df[metric]
            result_df[f'{metric}_std'] = std_df[metric]
        
        return result_df
    else:
        # Overall statistics
        mean_series = df[available_metrics].mean().round(2)
        std_series = df[available_metrics].std().round(2)
        
        result_df = pd.DataFrame({
            'mean': mean_series,
            'std': std_series
        })
        
        return result_df


def load_all_plant_configs(data_dir: str = 'data') -> List[Dict]:
    """
    Load all plant configurations from CSV files in data directory
    Tries to load from config files first, then falls back to auto-detection
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        List of plant configuration dictionaries
    """
    plant_configs = []
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found")
        return plant_configs
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    # Try to use PlantConfigManager to load configs if available
    try:
        manager = PlantConfigManager()
        config_dir = 'config/plants'
        
        for csv_file in csv_files:
            # Extract plant ID from filename (e.g., Project1140.csv -> 1140)
            plant_id = csv_file.replace('Project', '').replace('.csv', '').replace('Plant', '').replace('plant', '')
            
            config_path = os.path.join(config_dir, f'Plant{plant_id}.yaml')
            
            if os.path.exists(config_path):
                # Load from config file (same as multi_plant)
                try:
                    plant_config = manager.load_plant_config(config_path)
                    plant_config['data_path'] = os.path.join(data_dir, csv_file)  # Use actual data path
                    plant_configs.append(plant_config)
                    continue
                except Exception as e:
                    print(f"  Warning: Failed to load config for {plant_id}: {e}, using auto-detection")
            
            # Fall back to auto-detection from data file
            data_path = os.path.join(data_dir, csv_file)
            
            # Auto-detect date range from data file
            # CRITICAL: Force start_date to be at least 2022-01-01
            try:
                df = pd.read_csv(data_path)
                if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
                    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
                    detected_start = df['Datetime'].min().strftime('%Y-%m-%d')
                    end_date = df['Datetime'].max().strftime('%Y-%m-%d')
                    # Force start_date to be at least 2022-01-01
                    if detected_start < '2022-01-01':
                        start_date = '2022-01-01'
                    else:
                        start_date = detected_start
                else:
                    start_date = '2022-01-01'
                    end_date = '2024-09-28'
            except Exception as e:
                print(f"  Warning: Failed to detect date range for {plant_id}: {e}, using defaults")
                start_date = '2022-01-01'
                end_date = '2024-09-28'
            
            # Create minimal plant config with detected dates
            # CRITICAL: Ensure start_date is at least 2022-01-01
            if not start_date or start_date < '2022-01-01':
                start_date = '2022-01-01'
            
            plant_config = {
                'plant_id': plant_id,
                'data_path': data_path,
                'future_hours': 24,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'random_seed': 42,
                'shuffle_split': True,  # Default to shuffle for robust evaluation
                'weather_category': 'medium_weather',
                'start_date': start_date,  # Force >= 2022-01-01
                'end_date': end_date,
                # DL parameters - consistent with multiplant
                'dl_params': {
                    'low': {
                        'epochs': 20,
                        'batch_size': 64,
                        'learning_rate': 0.001,
                        'patience': 10,
                        'min_delta': 0.001,
                        'weight_decay': 0.0001,
                        'd_model': 16,
                        'hidden_dim': 8,
                        'num_heads': 2,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'tcn_channels': [8, 16],
                        'kernel_size': 3
                    },
                    'mid_low': {
                        'epochs': 35,
                        'batch_size': 64,
                        'learning_rate': 0.001,
                        'patience': 10,
                        'min_delta': 0.001,
                        'weight_decay': 0.0001,
                        'd_model': 24,
                        'hidden_dim': 12,
                        'num_heads': 2,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'tcn_channels': [12, 24],
                        'kernel_size': 3
                    },
                    'mid_high': {
                        'epochs': 50,
                        'batch_size': 64,
                        'learning_rate': 0.001,
                        'patience': 10,
                        'min_delta': 0.001,
                        'weight_decay': 0.0001,
                        'd_model': 28,
                        'hidden_dim': 14,
                        'num_heads': 2,
                        'num_layers': 1,
                        'dropout': 0.1,
                        'tcn_channels': [14, 28],
                        'kernel_size': 3
                    },
                    'high': {
                        'epochs': 50,
                        'batch_size': 64,
                        'learning_rate': 0.001,
                        'patience': 10,
                        'min_delta': 0.001,
                        'weight_decay': 0.0001,
                        'd_model': 32,
                        'hidden_dim': 16,
                        'num_heads': 2,
                        'num_layers': 2,
                        'dropout': 0.1,
                        'tcn_channels': [16, 32],
                        'kernel_size': 3
                    }
                },
                # ML parameters - consistent with multiplant
                'ml_params': {
                    'low': {
                        'n_estimators': 10,
                        'max_depth': 1,
                        'learning_rate': 0.2,
                        'random_state': 42,
                        'verbosity': -1
                    },
                    'mid_low': {
                        'n_estimators': 20,
                        'max_depth': 2,
                        'learning_rate': 0.15,
                        'random_state': 42,
                        'verbosity': -1
                    },
                    'mid_high': {
                        'n_estimators': 25,
                        'max_depth': 2,
                        'learning_rate': 0.12,
                        'random_state': 42,
                        'verbosity': -1
                    },
                    'high': {
                        'n_estimators': 30,
                        'max_depth': 3,
                        'learning_rate': 0.1,
                        'random_state': 42,
                        'verbosity': -1
                    }
                }
            }
            
            plant_configs.append(plant_config)
        
    except Exception as e:
        print(f"  Warning: Failed to use PlantConfigManager: {e}")
        # Fall back to simple auto-detection for all files
        for csv_file in csv_files:
            plant_id = csv_file.replace('Project', '').replace('.csv', '').replace('Plant', '').replace('plant', '')
            data_path = os.path.join(data_dir, csv_file)
            
            # Auto-detect date range
            # CRITICAL: Force start_date to be at least 2022-01-01
            try:
                df = pd.read_csv(data_path)
                if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
                    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
                    detected_start = df['Datetime'].min().strftime('%Y-%m-%d')
                    end_date = df['Datetime'].max().strftime('%Y-%m-%d')
                    # Force start_date to be at least 2022-01-01
                    if detected_start < '2022-01-01':
                        start_date = '2022-01-01'
                    else:
                        start_date = detected_start
                else:
                    start_date = '2022-01-01'
                    end_date = '2024-09-28'
            except Exception:
                start_date = '2022-01-01'
                end_date = '2024-09-28'
            
            # CRITICAL: Ensure start_date is at least 2022-01-01
            if not start_date or start_date < '2022-01-01':
                start_date = '2022-01-01'
            
            # Use default config (same structure as above)
            plant_config = {
                'plant_id': plant_id,
                'data_path': data_path,
                'future_hours': 24,
                'train_ratio': 0.8,
                'val_ratio': 0.1,
                'test_ratio': 0.1,
                'random_seed': 42,
                'shuffle_split': True,
                'weather_category': 'medium_weather',
                'start_date': start_date,  # Force >= 2022-01-01
                'end_date': end_date,
                'dl_params': {},
                'ml_params': {}
            }
            plant_configs.append(plant_config)
    
    return plant_configs


def save_results(results_df: pd.DataFrame, output_file: str, local_output_dir: str = None, experiment_name: str = None):
    """
    Save results to CSV file with model ordering and local backup
    Creates separate folders for each experiment
    
    Args:
        results_df: Results DataFrame
        output_file: Output CSV file path (usually Google Drive)
        local_output_dir: Local output directory for backup (optional)
        experiment_name: Name of the experiment (used for folder creation)
    """
    # Define model order: LSR (if exists) RF XGB LGBM LSTM GRU TCN Transformer
    model_order = ['Linear', 'RF', 'XGB', 'LGBM', 'LSTM', 'GRU', 'TCN', 'Transformer']
    
    # Sort results by model order if 'model' column exists
    if 'model' in results_df.columns:
        # Create a categorical column for proper sorting
        results_df['model'] = pd.Categorical(results_df['model'], categories=model_order, ordered=True)
        results_df = results_df.sort_values('model')
        # Convert back to string
        results_df['model'] = results_df['model'].astype(str)
    
    # Create experiment-specific folder
    if experiment_name:
        # Extract base directory and create experiment folder
        base_dir = os.path.dirname(output_file)
        experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Update output file to be inside experiment folder
        filename = os.path.basename(output_file)
        output_file = os.path.join(experiment_dir, filename)
    else:
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    # Save to primary location (usually Google Drive)
    results_df.to_csv(output_file, index=True, encoding='utf-8-sig')
    print(f"\nResults saved to: {output_file}")
    
    # Also save to local directory if specified
    if local_output_dir:
        if experiment_name:
            local_experiment_dir = os.path.join(local_output_dir, experiment_name)
            os.makedirs(local_experiment_dir, exist_ok=True)
            local_output_file = os.path.join(local_experiment_dir, os.path.basename(output_file))
        else:
            local_output_file = os.path.join(local_output_dir, os.path.basename(output_file))
            os.makedirs(local_output_dir, exist_ok=True)
        
        results_df.to_csv(local_output_file, index=True, encoding='utf-8-sig')
        print(f"Local backup saved to: {local_output_file}")


def create_formatted_pivot(agg_df: pd.DataFrame, index_col: str, metric_cols: list, 
                          model_order: list = None) -> dict:
    """
    Create a formatted pivot table with mean±std format and proper model ordering
    
    Args:
        agg_df: Aggregated DataFrame with mean and std columns
        index_col: Column to use as index (e.g., 'season', 'hour', 'lookback_hours')
        metric_cols: List of metric names (e.g., ['mae', 'rmse', 'r2'])
        model_order: List of models in desired order
        
    Returns:
        Dictionary of formatted pivot DataFrames for each metric
    """
    if model_order is None:
        model_order = ['Linear', 'RF', 'XGB', 'LGBM', 'LSTM', 'GRU', 'TCN', 'Transformer']
    
    # Create formatted pivot for each metric
    formatted_pivots = {}
    
    for metric in metric_cols:
        mean_col = f'{metric}_mean'
        std_col = f'{metric}_std'
        
        if mean_col in agg_df.columns and std_col in agg_df.columns:
            # Create pivot with mean values
            mean_pivot = agg_df.pivot(index=index_col, columns='model', values=mean_col)
            
            # Create pivot with std values
            std_pivot = agg_df.pivot(index=index_col, columns='model', values=std_col)
            
            # Reorder columns according to model_order
            available_models = [m for m in model_order if m in mean_pivot.columns]
            mean_pivot = mean_pivot[available_models]
            std_pivot = std_pivot[available_models]
            
            # Create formatted strings: mean±std
            formatted_pivot = mean_pivot.copy()
            for col in formatted_pivot.columns:
                for idx in formatted_pivot.index:
                    mean_val = mean_pivot.loc[idx, col]
                    std_val = std_pivot.loc[idx, col]
                    if pd.notna(mean_val) and pd.notna(std_val):
                        formatted_pivot.loc[idx, col] = f"{mean_val:.2f}±{std_val:.2f}"
                    elif pd.notna(mean_val):
                        formatted_pivot.loc[idx, col] = f"{mean_val:.2f}"
                    else:
                        formatted_pivot.loc[idx, col] = "N/A"
            
            formatted_pivots[metric] = formatted_pivot
        else:
            print(f"Warning: Columns {mean_col} or {std_col} not found in aggregated data")
    
    return formatted_pivots


def run_experiments_for_plants(plant_configs: List[Dict], models: List[str], 
                               config_modifier=None, use_sliding_windows=False) -> List[Dict]:
    """
    Run experiments for multiple plants and models
    
    Args:
        plant_configs: List of plant configurations
        models: List of model names to test
        config_modifier: Optional function to modify config before running (takes config dict, returns config dict)
        use_sliding_windows: Whether to use hourly sliding windows
        
    Returns:
        List of result dictionaries
    """
    from tqdm import tqdm
    
    all_results = []
    
    for plant_idx, plant_config in enumerate(plant_configs, 1):
        plant_id = plant_config['plant_id']
        data_path = plant_config['data_path']
        
        print(f"\n{'=' * 80}")
        print(f"Plant {plant_idx}/{len(plant_configs)}: {plant_id}")
        print(f"{'=' * 80}")
        
        # Load and filter data (ensures data starts from 2022-01-01)
        try:
            df = load_and_filter_data(data_path, plant_config)
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
        # Run experiments for each model
        for model in tqdm(models, desc=f"Plant {plant_id}"):
            # Create base configuration
            if model == 'Linear':
                config = create_base_config(plant_config, model, complexity='high', 
                                          lookback=24, use_te=False)
                config['use_pv'] = False
                config['use_hist_weather'] = False
                config['no_hist_power'] = True
                config['past_hours'] = 0
            else:
                config = create_base_config(plant_config, model, complexity='high', 
                                          lookback=24, use_te=False)
            
            # Apply custom modifications if provided
            if config_modifier:
                config = config_modifier(config, model, plant_config)
            
            try:
                # Run experiment
                result = run_single_experiment(config, df.copy(), use_sliding_windows=use_sliding_windows)
                
                if result['status'] != 'SUCCESS':
                    print(f"  Error running {model}: {result.get('error', 'Unknown')}")
                    continue
                
                # Add plant ID to result
                result['plant_id'] = plant_id
                all_results.append(result)
                
            except Exception as e:
                print(f"  Error running {model}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    return all_results
