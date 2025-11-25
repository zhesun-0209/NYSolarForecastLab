#!/usr/bin/env python3
"""
Unified experiments module
Consolidates all experiment running functionality
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys
import time
import glob
from datetime import datetime
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.append(script_dir)

from config.config_manager import PlantConfigManager
from data.data_utils import preprocess_features, create_daily_windows, split_data
from train.train_dl import train_dl_model
from train.train_ml import train_ml_model

# Import sensitivity analysis utilities
try:
    from sensitivity_analysis.common_utils import set_global_seed
except ImportError:
    def set_global_seed(seed=42):
        import random
        import torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)


def create_config(data_path, model, complexity, lookback, feat_combo, use_te, is_nwp_only):
    """Create experiment configuration"""
    # Parse feature combination name
    feat_name = feat_combo.get('name', 'Unknown')
    te_suffix = '_TE' if use_te else '_noTE'
    
    config = {
        'data_path': data_path,
        'model': model,
        'model_complexity': complexity,
        'past_hours': lookback,
        'future_hours': 24,
        'use_pv': feat_combo.get('use_pv', False),
        'use_hist_weather': feat_combo.get('use_hist_weather', False),
        'use_forecast': feat_combo.get('use_forecast', False),
        'use_ideal_nwp': feat_combo.get('use_ideal_nwp', False),
        'use_time_encoding': use_te,
        'no_hist_power': is_nwp_only,
        'weather_category': 'medium_weather',  # Default weather category
        'normalization_method': 'minmax',
        'inverse_transform': True,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'shuffle_split': False,  # Sequential split for time series
        'random_seed': 42,
        'start_date': '2022-01-01',  # Default start date
        'end_date': '2024-09-28',     # Default end date
        'train_params': {
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 50 if complexity == 'high' else 20,
            'patience': 10,
            'min_delta': 0.001,
        },
        'model_params': {
            'low': {'d_model': 16, 'num_layers': 1, 'hidden_dim': 16, 'dropout': 0.1},
            'high': {'d_model': 32, 'num_layers': 2, 'hidden_dim': 32, 'dropout': 0.1},
        }
    }
    
    # Create experiment name
    if model == 'Linear':
        config['experiment_name'] = f"{model}_{feat_name}{te_suffix}"
    else:
        config['experiment_name'] = f"{model}_{complexity}_{feat_name}{te_suffix}"
    
    return config


def generate_all_configs(data_path: str, test_mode: bool = False, test_model: str = 'LSTM'):
    """Generate all 284 experiment configurations
    
    Args:
        data_path: Path to data file
        test_mode: If True, only generate configs for test_model
        test_model: Model to use in test mode (default: 'LSTM')
    """
    configs = []
    if test_mode:
        # Test mode: only generate configs for specified model
        dl_models = [test_model] if test_model in ['LSTM', 'GRU', 'Transformer', 'TCN'] else []
        ml_models = [test_model] if test_model in ['RF', 'XGB', 'LGBM'] else []
        complexities = ['low']  # Only test low complexity in test mode
        lookbacks = [24]  # Only test 24h lookback in test mode
        te_options = [False]  # Only test without time encoding in test mode
    else:
        dl_models = ['LSTM', 'GRU', 'Transformer', 'TCN']
        ml_models = ['RF', 'XGB', 'LGBM']
        complexities = ['low', 'high']
        lookbacks = [24, 72]
        te_options = [True, False]

    feature_combos_pv = [
        {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
        {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    feature_combos_nwp = [
        {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
        {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
    ]

    # DL models: PV-based
    for model in dl_models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        configs.append(create_config(data_path, model, complexity, lookback, feat_combo, use_te, False))

    # DL models: NWP-only
    for model in dl_models:
        for complexity in complexities:
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    configs.append(create_config(data_path, model, complexity, 0, feat_combo, use_te, True))

    # ML models
    for model in ml_models:
        for complexity in complexities:
            for lookback in lookbacks:
                for feat_combo in feature_combos_pv:
                    for use_te in te_options:
                        configs.append(create_config(data_path, model, complexity, lookback, feat_combo, use_te, False))
            for feat_combo in feature_combos_nwp:
                for use_te in te_options:
                    configs.append(create_config(data_path, model, complexity, 0, feat_combo, use_te, True))

    # Linear model
    for feat_combo in feature_combos_nwp:
        for use_te in te_options:
            configs.append(create_config(data_path, 'Linear', None, 0, feat_combo, use_te, True))

    return configs


def run_single_experiment(config: Dict, df: pd.DataFrame) -> Dict:
    """Run a single experiment"""
    try:
        set_global_seed(config.get('random_seed', 42))
        
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = \
            preprocess_features(df, config)
        
        past_hours = config.get('past_hours', 24)
        future_hours = config.get('future_hours', 24)
        
        X_hist, X_fcst, y, hours, dates = create_daily_windows(
            df_clean, future_hours, hist_feats, fcst_feats, no_hist_power, past_hours
        )
        
        train_ratio = config.get('train_ratio', 0.8)
        val_ratio = config.get('val_ratio', 0.1)
        shuffle = config.get('shuffle_split', False)
        
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = split_data(
            X_hist, X_fcst, y, hours, dates, train_ratio, val_ratio, shuffle
        )
        
        train_data = (Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr)
        val_data = (Xh_va, Xf_va, y_va, hrs_va, dates_va)
        test_data = (Xh_te, Xf_te, y_te, hrs_te, dates_te)
        scalers = (scaler_hist, scaler_fcst, scaler_target)
        
        if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
            model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
        else:
            model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
        
        return {
            'status': 'SUCCESS',
            'metrics': metrics,
            'config': config
        }
    except Exception as e:
        return {
            'status': 'FAILED',
            'error': str(e),
            'config': config
        }


def run_forecast_experiments(plant_id: str = '1140', output_dir: Optional[str] = None, 
                             test_mode: bool = False, test_model: str = 'LSTM'):
    """Run all 284 experiments for a single plant with resume support
    
    Args:
        plant_id: Plant ID
        output_dir: Output directory for results
        test_mode: If True, only run test_model experiments
        test_model: Model to use in test mode (default: 'LSTM')
    """
    if test_mode:
        print("=" * 80)
        print(f"PV Forecasting: Test Mode - Running {test_model} experiments only")
        print("=" * 80)
    else:
        print("=" * 80)
        print("PV Forecasting: Running 284 Experiments (with resume support)")
        print("=" * 80)
    
    data_path = os.path.join(script_dir, "data", f"Project{plant_id}.csv")
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    # CRITICAL: Filter data to start from 2022-01-01 before generating configs
    # This ensures all experiments use the same date range
    start_date = '2022-01-01'
    end_date = '2024-09-28'
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['Datetime'] >= start_dt) & (df['Datetime'] <= end_dt)].copy()
    print(f"Data filtered: {start_date} to {end_date} ({len(df)} rows)")
    
    configs = generate_all_configs(data_path, test_mode=test_mode, test_model=test_model)
    
    print(f"Total configurations generated: {len(configs)}")
    
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Set output directory
    if output_dir is None:
        output_dir = script_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Check for existing results (resume support)
    output_file = os.path.join(output_dir, f"results_{plant_id}_all.csv")
    
    if os.path.exists(output_file):
        print(f"Found existing result file: {output_file}")
        results_df = pd.read_csv(output_file)
        # Handle old format files that might not have 'experiment_name' column
        if 'experiment_name' in results_df.columns:
            done_experiments = set(results_df["experiment_name"].tolist())
        else:
            # Old format: try to reconstruct experiment_name from other columns
            if len(results_df) > 0:
                print("Warning: Old format result file detected. Will recreate experiment names.")
                done_experiments = set()
            else:
                done_experiments = set()
    else:
        results_df = pd.DataFrame(columns=[
            'experiment_name', 'model', 'complexity', 'feature_combo',
            'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2', 'nrmse',
            'train_time_sec', 'test_samples', 'best_epoch', 'param_count'
        ])
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        done_experiments = set()
        print(f"Created new result file: {output_file}")
    
    print(f"[OK] Already completed: {len(done_experiments)}")
    print(f"[INFO] Remaining: {len(configs) - len(done_experiments)}")
    
    # Main loop
    for idx, config in enumerate(configs, 1):
        exp_name = config.get('experiment_name', f"{config['model']}_{config.get('model_complexity', 'N/A')}")
        
        if exp_name in done_experiments:
            print(f"[SKIP] {exp_name} already completed.")
            continue
        
        print(f"\n{'='*80}")
        print(f"Experiment {idx}/{len(configs)}: {exp_name}")
        print(f"{'='*80}")
        
        try:
            start_time = time.time()
            result = run_single_experiment(config, df)
            training_time = time.time() - start_time
            
            if result['status'] == 'SUCCESS':
                metrics = result['metrics']
                
                # Parse feature combination name
                use_pv = config.get('use_pv', False)
                use_hist_weather = config.get('use_hist_weather', False)
                use_forecast = config.get('use_forecast', False)
                use_ideal_nwp = config.get('use_ideal_nwp', False)
                
                if use_pv and use_hist_weather:
                    feat_name_str = 'PV+HW'
                elif use_pv and use_forecast and use_ideal_nwp:
                    feat_name_str = 'PV+NWP+'
                elif use_pv and use_forecast:
                    feat_name_str = 'PV+NWP'
                elif use_pv:
                    feat_name_str = 'PV'
                elif use_forecast and use_ideal_nwp:
                    feat_name_str = 'NWP+'
                elif use_forecast:
                    feat_name_str = 'NWP'
                else:
                    feat_name_str = 'Unknown'
                
                result_row = {
                    'experiment_name': exp_name,
                    'model': config['model'],
                    'complexity': config.get('model_complexity', 'N/A'),
                    'feature_combo': feat_name_str,
                    'lookback_hours': config.get('past_hours', 0),
                    'use_time_encoding': config.get('use_time_encoding', False),
                    'mae': metrics.get('mae', 0.0),
                    'rmse': metrics.get('rmse', 0.0),
                    'r2': metrics.get('r2', 0.0),
                    'nrmse': metrics.get('nrmse', 0.0),  # Unified metric calculation
                    'train_time_sec': round(training_time, 2),
                    'test_samples': metrics.get('samples_count', 0),
                    'best_epoch': int(metrics.get('best_epoch', 0)) if not pd.isna(metrics.get('best_epoch', 0)) else 0,
                    'param_count': int(metrics.get('param_count', 0))
                }
                
                print(f"  [OK] MAE: {metrics.get('mae', 0):.4f}, RMSE: {metrics.get('rmse', 0):.4f}")
                pd.DataFrame([result_row]).to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
                done_experiments.add(exp_name)
            else:
                print(f"  [ERROR] {exp_name} failed: {result.get('error', 'Unknown error')}")
                error_row = {
                    'experiment_name': exp_name,
                    'model': config['model'],
                    'complexity': config.get('model_complexity', 'N/A'),
                    'feature_combo': 'FAILED',
                    'lookback_hours': config.get('past_hours', 0),
                    'use_time_encoding': config.get('use_time_encoding', False),
                    'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'nrmse': np.nan,
                    'train_time_sec': 0, 'test_samples': 0,
                    'best_epoch': 0, 'param_count': 0
                }
                pd.DataFrame([error_row]).to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        except Exception as e:
            print(f"  [ERROR] {exp_name} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print("[OK] All Experiments Completed or Skipped!")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


def check_plant_status(plant_id: str, output_dir: str = '.') -> Dict:
    """Check completion status of a plant"""
    pattern = os.path.join(output_dir, f"results_{plant_id}_*.csv")
    result_files = glob.glob(pattern)
    
    if not result_files:
        return {'plant_id': plant_id, 'status': 'NOT_STARTED', 'completed': 0, 'total': 284}
    
    result_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_file = result_files[0]
    
    try:
        df = pd.read_csv(latest_file)
        completed = len(df[df.get('status', '') == 'SUCCESS']) if 'status' in df.columns else len(df)
        progress = completed / 284 * 100
        
        return {
            'plant_id': plant_id,
            'status': 'COMPLETE' if completed >= 284 else 'IN_PROGRESS',
            'completed': completed,
            'total': 284,
            'progress': progress,
            'result_file': latest_file
        }
    except Exception as e:
        return {'plant_id': plant_id, 'status': 'ERROR', 'error': str(e)}


def check_all_plants_status(output_dir: str = '.'):
    """Check status of all plants"""
    data_dir = os.path.join(script_dir, 'data')
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    plant_ids = []
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        import re
        match = re.search(r'(\d+)', basename)
        if match:
            plant_ids.append(match.group(1))
    
    statuses = []
    for plant_id in plant_ids:
        status = check_plant_status(plant_id, output_dir)
        statuses.append(status)
    
    df = pd.DataFrame(statuses)
    print(df.to_string(index=False))
    return df


def batch_create_configs(data_dir: str = 'data', config_dir: str = 'config/plants'):
    """Auto-generate configuration files for all CSV files"""
    import re
    from pathlib import Path
    
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    os.makedirs(config_dir, exist_ok=True)
    
    template_path = os.path.join(script_dir, 'config', 'plant_template.yaml')
    with open(template_path, 'r') as f:
        template = yaml.safe_load(f)
    
    for csv_file in csv_files:
        basename = os.path.basename(csv_file)
        match = re.search(r'(\d+)', basename)
        plant_id = match.group(1) if match else basename.replace('.csv', '')
        
        # Detect date range
        try:
            df = pd.read_csv(csv_file)
            if all(col in df.columns for col in ['Year', 'Month', 'Day', 'Hour']):
                df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
                start_date = df['Datetime'].min().strftime('%Y-%m-%d')
                end_date = df['Datetime'].max().strftime('%Y-%m-%d')
            else:
                start_date = template.get('start_date', '2022-01-01')
                end_date = template.get('end_date', '2024-09-28')
        except:
            start_date = template.get('start_date', '2022-01-01')
            end_date = template.get('end_date', '2024-09-28')
        
        config = template.copy()
        config['plant_id'] = plant_id
        config['data_path'] = csv_file
        config['start_date'] = start_date
        config['end_date'] = end_date
        
        config_file = os.path.join(config_dir, f"Plant{plant_id}.yaml")
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"Created config: {config_file}")
    
    print(f"\nGenerated {len(csv_files)} configuration files")


# =============================================================================
# Multi-Plant Batch Functions (from run_experiments_multi_plant.py)
# =============================================================================

def is_drive_path(path: str) -> bool:
    """Check if path is a Google Drive path"""
    return path and path.startswith('/content/drive/')


def check_drive_path(output_dir: str = None) -> bool:
    """Check if Drive path exists (assumes Drive is already mounted)"""
    if output_dir and is_drive_path(output_dir):
        if not os.path.exists(output_dir):
            print(f"Warning: Drive path does not exist: {output_dir}")
            print("Please ensure Google Drive is mounted manually")
            return False
        else:
            print(f"Drive path detected and accessible: {output_dir}")
    return True


def check_plant_completion(plant_id: str, output_dir: str = None) -> tuple:
    """Check if a plant's experiments are complete"""
    if output_dir is None:
        output_dir = script_dir
    
    if not check_drive_path(output_dir):
        return False, 0, None
    
    if not os.path.exists(output_dir):
        print(f"  Warning: Output directory does not exist: {output_dir}")
        return False, 0, None
    
    result_file = os.path.join(output_dir, f"results_{plant_id}.csv")
    
    if not os.path.exists(result_file):
        print(f"  Info: Result file not found: {result_file}")
        return False, 0, None
    
    try:
        df = pd.read_csv(result_file)
        
        if 'status' in df.columns:
            completed = len(df[df['status'] == 'SUCCESS'])
        else:
            completed = len(df[df['experiment_name'].notna()])
        
        is_complete = (completed >= 284)
        print(f"  Found {completed}/284 completed experiments in {result_file}")
        return is_complete, completed, result_file
    
    except Exception as e:
        print(f"  Warning: Error reading {result_file}: {str(e)}")
        return False, 0, None


def run_plant_experiments(plant_config_path: str, resume: bool = True, output_dir: str = None):
    """Run all experiments for a single plant with resume support"""
    print("\n" + "=" * 80)
    print(f"Running experiments for plant: {plant_config_path}")
    print("=" * 80)
    
    manager = PlantConfigManager()
    plant_config = manager.load_plant_config(plant_config_path)
    plant_id = plant_config['plant_id']
    
    if output_dir is None:
        output_dir = script_dir
    else:
        if not check_drive_path(output_dir):
            return 0
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return 0
    
    is_complete, completed_count, existing_file = check_plant_completion(plant_id, output_dir)
    
    if is_complete and resume:
        print(f"[OK] Plant {plant_id} already complete: {completed_count}/284 experiments")
        print(f"  Result file: {existing_file}")
        print(f"  Skipping to next plant...\n")
        return completed_count
    
    all_configs = manager.generate_experiment_configs(plant_config)
    print(f"Total configurations: {len(all_configs)}")
    
    data_path = plant_config['data_path']
    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        return 0
    
    df = pd.read_csv(data_path)
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    
    start_date = plant_config.get('start_date', '2022-01-01')
    end_date = plant_config.get('end_date', '2024-09-28')
    
    if start_date:
        start_dt = pd.to_datetime(start_date)
        df = df[df['Datetime'] >= start_dt].copy()
        print(f"  Data filtered: Start date = {start_date} ({len(df)} rows remain)")
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        df = df[df['Datetime'] <= end_dt].copy()
        print(f"  Data filtered: End date = {end_date} ({len(df)} rows remain)")
    
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    output_file = os.path.join(output_dir, f"results_{plant_id}.csv")
    
    done_experiments = set()
    
    if resume and existing_file:
        print(f"Resuming from: {output_file}")
        results_df = pd.read_csv(output_file)
        
        if 'status' in results_df.columns:
            done_experiments = set(results_df[results_df['status'] == 'SUCCESS']["experiment_name"].tolist())
        else:
            done_experiments = set(results_df["experiment_name"].dropna().tolist())
        
        print(f"Already completed: {len(done_experiments)}/284")
        print(f"Remaining: {len(all_configs) - len(done_experiments)}/284")
    else:
        results_df = pd.DataFrame(columns=[
            'plant_id', 'experiment_name', 'model', 'complexity', 'scenario',
            'lookback_hours', 'use_time_encoding', 'mae', 'rmse', 'r2', 'nrmse',
            'train_time_sec', 'test_samples', 'best_epoch', 'param_count', 'status'
        ])
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"Created new result file: {output_file}")
    
    # Use run_single_experiment from run_experiments_multi_plant.py logic
    from data.data_utils import preprocess_features, create_daily_windows
    
    def run_single_experiment_multi_plant(config: Dict, df: pd.DataFrame) -> Dict:
        """Run single experiment (multi-plant version)"""
        try:
            random_seed = config.get('random_seed', 42)
            set_global_seed(random_seed)
            
            df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = \
                preprocess_features(df, config)
            
            past_hours = config.get('past_hours', 24)
            X_hist, X_fcst, y, hours, dates = create_daily_windows(
                df_clean, config['future_hours'], hist_feats, fcst_feats, no_hist_power, past_hours
            )
            
            total_samples = len(X_hist)
            indices = np.arange(total_samples)
            
            shuffle_split = config.get('shuffle_split', True)
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
            
            X_hist_train, y_train = X_hist[train_idx], y[train_idx]
            X_hist_val, y_val = X_hist[val_idx], y[val_idx]
            X_hist_test, y_test = X_hist[test_idx], y[test_idx]
            
            if X_fcst is not None:
                X_fcst_train, X_fcst_val, X_fcst_test = X_fcst[train_idx], X_fcst[val_idx], X_fcst[test_idx]
            else:
                X_fcst_train = X_fcst_val = X_fcst_test = None
            
            train_hours = np.array([hours[i] for i in train_idx])
            val_hours = np.array([hours[i] for i in val_idx])
            test_hours = np.array([hours[i] for i in test_idx])
            test_dates = [dates[i] for i in test_idx]
            
            train_data = (X_hist_train, X_fcst_train, y_train, train_hours, [])
            val_data = (X_hist_val, X_fcst_val, y_val, val_hours, [])
            test_data = (X_hist_test, X_fcst_test, y_test, test_hours, test_dates)
            scalers = (scaler_hist, scaler_fcst, scaler_target)
            
            if config['model'] in ['LSTM', 'GRU', 'Transformer', 'TCN']:
                model, metrics = train_dl_model(config, train_data, val_data, test_data, scalers)
            else:
                model, metrics = train_ml_model(config, train_data, val_data, test_data, scalers)
            
            training_time = metrics.get('train_time_sec', 0.0)
            
            use_pv = config.get('use_pv', False)
            use_hist_weather = config.get('use_hist_weather', False)
            use_forecast = config.get('use_forecast', False)
            use_ideal_nwp = config.get('use_ideal_nwp', False)
            
            if use_pv and use_hist_weather:
                scenario = 'PV+HW'
            elif use_pv and use_forecast and use_ideal_nwp:
                scenario = 'PV+NWP+'
            elif use_pv and use_forecast:
                scenario = 'PV+NWP'
            elif use_pv:
                scenario = 'PV'
            elif use_forecast and use_ideal_nwp:
                scenario = 'NWP+'
            elif use_forecast:
                scenario = 'NWP'
            else:
                scenario = 'Unknown'
            
            return {
                'plant_id': config['plant_id'],
                'experiment_name': config['experiment_name'],
                'model': config['model'],
                'complexity': config.get('model_complexity', 'N/A'),
                'scenario': scenario,
                'lookback_hours': config['past_hours'],
                'use_time_encoding': config['use_time_encoding'],
                'mae': metrics.get('mae', 0.0),
                'rmse': metrics.get('rmse', 0.0),
                'r2': metrics.get('r2', 0.0),
                'nrmse': metrics.get('nrmse', 0.0),
                'train_time_sec': training_time,
                'test_samples': metrics.get('samples_count', 0),
                'best_epoch': int(metrics.get('best_epoch', 0)) if not np.isnan(metrics.get('best_epoch', 0)) else 0,
                'param_count': int(metrics.get('param_count', 0)),
                'status': 'SUCCESS'
            }
        except Exception as e:
            print(f"  [ERROR] Experiment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                'plant_id': config.get('plant_id', 'Unknown'),
                'experiment_name': config.get('experiment_name', 'Unknown'),
                'model': config.get('model', 'Unknown'),
                'complexity': config.get('model_complexity', 'N/A'),
                'scenario': 'FAILED',
                'lookback_hours': config.get('past_hours', 0),
                'use_time_encoding': config.get('use_time_encoding', False),
                'mae': np.nan, 'rmse': np.nan, 'r2': np.nan, 'nrmse': np.nan,
                'train_time_sec': 0, 'test_samples': 0, 'best_epoch': 0, 'param_count': 0,
                'status': 'FAILED', 'error': str(e)
            }
    
    success_count = 0
    for idx, config in enumerate(all_configs, 1):
        exp_name = config['experiment_name']
        
        if exp_name in done_experiments:
            print(f"[{idx}/{len(all_configs)}] SKIP: {exp_name} (already completed)")
            success_count += 1
            continue
        
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(all_configs)}] Running: {exp_name}")
        print(f"{'=' * 80}")
        
        result = run_single_experiment_multi_plant(config, df)
        
        pd.DataFrame([result]).to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8-sig')
        
        if result['status'] == 'SUCCESS':
            print(f"  [OK] MAE: {result['mae']:.4f}, RMSE: {result['rmse']:.4f}")
            success_count += 1
        else:
            print(f"  [FAILED] {result.get('error', 'Unknown error')}")
    
    print(f"\n{'=' * 80}")
    print(f"Plant {plant_id} Experiments Completed!")
    print(f"Success: {success_count}/{len(all_configs)}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 80}\n")
    
    return success_count


def scan_all_plants_status(output_dir: str = None) -> List[Dict]:
    """Scan all plants and their completion status"""
    manager = PlantConfigManager()
    plants = manager.get_all_plants()
    
    plant_statuses = []
    
    for plant in plants:
        plant_id = plant['plant_id']
        is_complete, completed, result_file = check_plant_completion(plant_id, output_dir)
        
        plant_statuses.append({
            'plant_id': plant_id,
            'data_path': plant['data_path'],
            'is_complete': is_complete,
            'completed_experiments': completed,
            'remaining_experiments': 284 - completed,
            'result_file': result_file,
            'status': 'COMPLETE' if is_complete else ('IN_PROGRESS' if completed > 0 else 'NOT_STARTED')
        })
    
    return plant_statuses


def run_all_plants(resume: bool = True, skip: int = 0, max_plants: int = None, 
                   plants: List[str] = None, output_dir: str = None):
    """Run experiments for all plants with advanced filtering"""
    print("=" * 80)
    print("Multi-Plant Batch Experiment Runner")
    print("=" * 80)
    
    manager = PlantConfigManager()
    all_plants = manager.get_all_plants()
    
    if not all_plants:
        print("[ERROR] No plant configurations found in config/plants/")
        print("Please run: python run.py config")
        return
    
    print(f"Total plants available: {len(all_plants)} (sorted by plant_id)")
    if len(all_plants) > 0:
        plant_ids = [p['plant_id'] for p in all_plants]
        print(f"Plant order: {', '.join(str(pid) for pid in plant_ids[:10])}" + 
              (f" ... (and {len(plant_ids) - 10} more)" if len(plant_ids) > 10 else ""))
    
    if plants:
        filtered_plants = [p for p in all_plants if p['plant_id'] in plants]
        print(f"Running specified plants: {plants}")
    else:
        filtered_plants = all_plants[skip:]
        if max_plants:
            filtered_plants = filtered_plants[:max_plants]
        if skip > 0:
            print(f"Skipping first: {skip} plants (will start from plant_id: {all_plants[skip]['plant_id']})")
        if max_plants:
            print(f"Running maximum: {max_plants} plants")
            if len(filtered_plants) > 0:
                print(f"  Will process plants: {filtered_plants[0]['plant_id']} to {filtered_plants[-1]['plant_id']}")
    
    if output_dir is None:
        output_dir = script_dir
    else:
        if not check_drive_path(output_dir):
            return
        
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating output directory {output_dir}: {e}")
            return
    
    print(f"\nOutput directory: {output_dir}")
    if is_drive_path(output_dir):
        print("  (Google Drive path detected)")
    print(f"\n{'='*80}")
    print(f"Plants to process: {len(filtered_plants)}")
    print(f"{'='*80}")
    
    if resume:
        print("\n[Scanning existing results...]")
        plant_statuses = []
        for plant in filtered_plants:
            is_complete, completed, result_file = check_plant_completion(plant['plant_id'], output_dir)
            plant_statuses.append({
                'plant_id': plant['plant_id'],
                'completed': completed,
                'is_complete': is_complete
            })
        
        complete_count = sum(1 for s in plant_statuses if s['is_complete'])
        in_progress_count = sum(1 for s in plant_statuses if 0 < s['completed'] < 284)
        not_started_count = sum(1 for s in plant_statuses if s['completed'] == 0)
        
        print(f"\nStatus Summary:")
        print(f"  [COMPLETE]:     {complete_count} plants")
        print(f"  [IN_PROGRESS]:  {in_progress_count} plants")
        print(f"  [NOT_STARTED]:  {not_started_count} plants")
        print(f"  [TO_RUN]:       {len(filtered_plants) - complete_count} plants")
    
    print(f"\n{'='*80}\n")
    
    total_success = 0
    total_experiments = 0
    plants_processed = 0
    
    start_time = time.time()
    
    for i, plant in enumerate(filtered_plants, 1):
        plant_id = plant['plant_id']
        plant_config_path = f"config/plants/Plant{plant_id}.yaml"
        
        print(f"\n{'#' * 80}")
        print(f"Plant {i}/{len(filtered_plants)}: {plant_id}")
        print(f"Progress: {i/len(filtered_plants)*100:.1f}%")
        print(f"{'#' * 80}")
        
        success = run_plant_experiments(plant_config_path, resume=resume, output_dir=output_dir)
        total_success += success
        total_experiments += 284
        plants_processed += 1
        
        elapsed = time.time() - start_time
        avg_time_per_plant = elapsed / plants_processed
        remaining_plants = len(filtered_plants) - plants_processed
        estimated_remaining = avg_time_per_plant * remaining_plants
        
        print(f"\nProgress Summary:")
        print(f"  Plants processed: {plants_processed}/{len(filtered_plants)}")
        print(f"  Experiments done: {total_success}/{total_experiments}")
        print(f"  Time elapsed:     {elapsed/3600:.2f} hours")
        print(f"  Time remaining:   {estimated_remaining/3600:.2f} hours")
        print(f"  Est. completion:  {(elapsed + estimated_remaining)/3600:.2f} hours total")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("[COMPLETE] Batch Experiments Finished!")
    print("=" * 80)
    print(f"Plants processed: {plants_processed}")
    print(f"Experiments successful: {total_success}/{total_experiments} ({total_success/total_experiments*100:.1f}%)")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg per plant: {total_time/plants_processed/60:.1f} minutes" if plants_processed > 0 else "")
    print("=" * 80)

