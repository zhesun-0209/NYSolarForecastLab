#!/usr/bin/env python3
"""
Unified evaluation module
Consolidates all evaluation utilities: metrics, predictions, Excel export
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.metrics import r2_score


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return {
            'mae': np.nan, 'rmse': np.nan, 'nrmse': np.nan,
            'r_square': np.nan, 'r2': np.nan, 'smape': np.nan
        }
    
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
    
    y_range = np.max(y_true_clean) - np.min(y_true_clean)
    nrmse = rmse / y_range if y_range != 0 else np.nan
    
    r_square = r2_score(y_true_clean, y_pred_clean)
    
    # sMAPE
    nonzero_mask = (y_true_clean > 0) | (y_pred_clean > 0)
    if np.any(nonzero_mask):
        y_true_nonzero = y_true_clean[nonzero_mask]
        y_pred_nonzero = y_pred_clean[nonzero_mask]
        denominator = np.abs(y_true_nonzero) + np.abs(y_pred_nonzero)
        smape_mask = denominator > 0
        if np.any(smape_mask):
            smape = np.mean(2 * np.abs(y_true_nonzero[smape_mask] - y_pred_nonzero[smape_mask]) / 
                           denominator[smape_mask])
        else:
            smape = np.nan
    else:
        smape = np.nan
    
    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'nrmse': round(nrmse, 4),
        'r_square': round(r_square, 4),
        'r2': round(r_square, 4),
        'smape': round(smape, 4)
    }


def calculate_mse(y_true, y_pred):
    """Calculate MSE"""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    return round(np.mean((y_true_clean - y_pred_clean) ** 2), 4)


def calculate_daily_avg_metrics(y_true, y_pred):
    """
    Calculate metrics using daily average method (recommended for day-ahead forecasting)
    This calculates RMSE/MAE for each day, then averages across days.
    All metrics (MAE, RMSE, R2, NRMSE) are calculated consistently.
    """
    n_days = y_true.shape[0]
    daily_rmses = []
    daily_maes = []
    
    for i in range(n_days):
        day_true = y_true[i]
        day_pred = y_pred[i]
        
        mask = ~(np.isnan(day_true) | np.isnan(day_pred))
        day_true_clean = day_true[mask]
        day_pred_clean = day_pred[mask]
        
        if len(day_true_clean) > 0:
            daily_rmse = np.sqrt(np.mean((day_true_clean - day_pred_clean) ** 2))
            daily_mae = np.mean(np.abs(day_true_clean - day_pred_clean))
            daily_rmses.append(daily_rmse)
            daily_maes.append(daily_mae)
    
    rmse_daily_avg = np.mean(daily_rmses) if len(daily_rmses) > 0 else np.nan
    mae_daily_avg = np.mean(daily_maes) if len(daily_maes) > 0 else np.nan
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]
    
    if len(y_true_clean) > 0:
        r_square = r2_score(y_true_clean, y_pred_clean)
        # Calculate NRMSE: RMSE / (max - min) of true values
        y_range = np.max(y_true_clean) - np.min(y_true_clean)
        nrmse = rmse_daily_avg / y_range if y_range != 0 else np.nan
    else:
        r_square = np.nan
        nrmse = np.nan
    
    return {
        'mae': round(mae_daily_avg, 4),
        'rmse': round(rmse_daily_avg, 4),
        'r2': round(r_square, 4),
        'r_square': round(r_square, 4),
        'nrmse': round(nrmse, 4),
        'n_days': len(daily_rmses)
    }


# =============================================================================
# Prediction Utilities
# =============================================================================

def extract_one_hour_ahead_predictions(predictions, ground_truth):
    """Extract 24-hour ahead predictions for day-ahead forecasting"""
    final_preds = predictions.flatten()
    final_gt = ground_truth.flatten()
    return final_preds, final_gt


# =============================================================================
# Result Saving
# =============================================================================

# =============================================================================
# Deep Learning Model Names
# =============================================================================

DL_MODELS = {"Transformer", "LSTM", "GRU", "TCN"}


# =============================================================================
# Result Saving (Enhanced Version)
# =============================================================================

def save_results(model, metrics: dict, dates: list, y_true: np.ndarray,
                Xh_test: np.ndarray, Xf_test: np.ndarray, config: dict):
    """
    Save experiment results (summary, predictions, training log, Excel export)
    
    Args:
        model: Trained DL or sklearn model
        metrics: Dictionary containing:
                 'test_loss', 'train_time_sec', 'param_count', 'rmse', 'mae',
                 'predictions' (n,h), 'y_true' (n,h),
                 'dates' (n), 'epoch_logs' (list of dicts)
        dates: List of datetime strings
        y_true, Xh_test, Xf_test: Used for legacy or optional plots
        config: Dictionary with keys like 'save_dir', 'model', 'scaler_target'
    """
    save_dir = config.get('save_dir', './results')
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract predictions and ground truth (already computed in metrics)
    preds = metrics.get('predictions', y_true)
    yts = metrics.get('y_true', y_true)
    
    # Use pre-computed metrics instead of recalculating
    test_mse = metrics.get('mse', 0)
    test_rmse = metrics.get('rmse', 0)
    test_mae = metrics.get('mae', 0)
    r_square = metrics.get('r_square', 0)
    
    save_options = config.get('save_options', {})
    
    summary = {
        'model': config['model'],
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast': config.get('use_forecast', False),
        'past_days': config.get('past_days', 1),
        'model_complexity': config.get('model_complexity', 'low'),
        'correlation_level': config.get('correlation_level', 'high'),
        'use_time_encoding': config.get('use_time_encoding', True),
        'past_hours': config.get('past_hours', 24),
        'future_hours': config.get('future_hours', 24),
        'mse': test_mse,
        'rmse': test_rmse,
        'mae': test_mae,
        'r_square': r_square,
        'train_time_sec': metrics.get('train_time_sec', 0),
        'inference_time_sec': metrics.get('inference_time_sec', np.nan),
        'param_count': metrics.get('param_count', 0),
        'samples_count': len(preds) if hasattr(preds, '__len__') else 0,
    }
    
    # Save summary.csv
    if save_options.get('save_summary', True):
        pd.DataFrame([summary]).to_csv(
            os.path.join(save_dir, 'summary.csv'), index=False
        )
    
    # Save predictions.csv
    if save_options.get('save_predictions', True):
        hrs = metrics.get('hours')
        dates_list = metrics.get('dates', dates)
        records = []
        
        if preds.ndim == 2:
            n_samples, horizon = preds.shape
        else:
            n_samples, horizon = 1, len(preds)
            preds = preds.reshape(1, -1)
            yts = yts.reshape(1, -1) if yts.ndim == 1 else yts
        
        # Handle case where hours information is not available
        if hrs is None:
            hrs = np.tile(np.arange(horizon), (n_samples, 1))
        
        for i in range(n_samples):
            start = pd.to_datetime(dates_list[i]) - pd.Timedelta(hours=horizon - 1)
            for h in range(horizon):
                dt = start + pd.Timedelta(hours=h)
                records.append({
                    'window_index': i,
                    'forecast_datetime': dt,
                    'hour': int(hrs[i, h]) if hrs is not None else dt.hour,
                    'y_true': float(yts[i, h]),
                    'y_pred': float(preds[i, h])
                })
        pd.DataFrame(records).to_csv(
            os.path.join(save_dir, "predictions.csv"), index=False
        )
    
    # Save training log (only if DL)
    is_dl = config['model'] in DL_MODELS
    if is_dl and 'epoch_logs' in metrics and save_options.get('save_training_log', True):
        pd.DataFrame(metrics['epoch_logs']).to_csv(
            os.path.join(save_dir, "training_log.csv"), index=False
        )
    
    # Save Excel results
    if save_options.get('save_excel_results', True):
        result_data = {
            'config': {
                'model': config['model'],
                'use_pv': config.get('use_pv', True),
                'use_hist_weather': config.get('use_hist_weather', False),
                'use_forecast': config.get('use_forecast', False),
                'weather_category': config.get('weather_category', 'irradiance'),
                'use_time_encoding': config.get('use_time_encoding', True),
                'past_days': config.get('past_days', 1),
                'model_complexity': config.get('model_complexity', 'low'),
                'epochs': config.get('epochs', 15),
                'batch_size': config.get('batch_size', 32),
                'learning_rate': config.get('learning_rate', 0.001)
            },
            'metrics': {
                'train_time_sec': summary['train_time_sec'],
                'inference_time_sec': summary['inference_time_sec'],
                'param_count': summary['param_count'],
                'samples_count': summary['samples_count'],
                'mse': summary['mse'],
                'rmse': summary['rmse'],
                'mae': summary['mae'],
                'nrmse': metrics.get('nrmse', np.nan),
                'r_square': summary['r_square'],
                'smape': metrics.get('smape', np.nan),
                'best_epoch': metrics.get('best_epoch', np.nan),
                'final_lr': metrics.get('final_lr', np.nan),
                'gpu_memory_used': metrics.get('gpu_memory_used', 0)
            }
        }
        
        append_plant_excel_results(
            plant_id=config.get('plant_id', 'unknown'),
            result=result_data,
            save_dir=save_dir
        )
    else:
        print(f"[INFO] Results saved in {save_dir}")


def append_plant_excel_results(plant_id: str, result: Dict[str, Any], save_dir: str) -> str:
    """Append single result to Excel file"""
    save_dir = save_dir or "/content/drive/MyDrive/Solar PV electricity/ablation results"
    os.makedirs(save_dir, exist_ok=True)
    
    excel_path = os.path.join(save_dir, f"{plant_id}_results.xlsx")
    
    config = result.get('config', {})
    metrics = result.get('metrics', {})
    
    row_data = {
        'model': config.get('model', ''),
        'use_pv': config.get('use_pv', True),
        'use_hist_weather': config.get('use_hist_weather', False),
        'use_forecast': config.get('use_forecast', False),
        'weather_category': config.get('weather_category', 'irradiance'),
        'use_time_encoding': config.get('use_time_encoding', True),
        'past_days': config.get('past_days', 1),
        'model_complexity': config.get('model_complexity', 'low'),
        'epochs': config.get('epochs', 15),
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('learning_rate', 0.001),
        'use_ideal_nwp': config.get('use_ideal_nwp', False),
        'train_time_sec': round(metrics.get('train_time_sec', 0), 4),
        'inference_time_sec': round(metrics.get('inference_time_sec', 0), 4),
        'param_count': metrics.get('param_count', 0),
        'samples_count': metrics.get('samples_count', 0),
        'best_epoch': metrics.get('best_epoch', np.nan),
        'final_lr': metrics.get('final_lr', np.nan),
        'mse': round(metrics.get('mse', 0), 4),
        'rmse': round(metrics.get('rmse', 0), 4),
        'mae': round(metrics.get('mae', 0), 4),
        'nrmse': round(metrics.get('nrmse', 0), 4),
        'r_square': round(metrics.get('r_square', 0), 4),
        'smape': round(metrics.get('smape', 0), 4),
        'gpu_memory_used': metrics.get('gpu_memory_used', 0)
    }
    
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])
    
    df.to_excel(excel_path, index=False)
    return excel_path

