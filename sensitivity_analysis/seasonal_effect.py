#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Experiment 1: Seasonal Effect

Analyze model performance across four seasons (Spring, Summer, Fall, Winter)
- Models: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) + Linear (NWP only)
- Configuration: PV+NWP, 24-hour lookback, no TE, high complexity
- Seasons: Spring (Mar-May), Summer (Jun-Aug), Fall (Sep-Nov), Winter (Dec-Feb)
- Metrics: MAE, RMSE, R2, NRMSE, train_time (mean and std across 100 plants)
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sensitivity_analysis.common_utils import (
    DL_MODELS, ML_MODELS, ALL_MODELS_NO_LINEAR,
    get_season, compute_nrmse,
    create_base_config, run_single_experiment,
    load_all_plant_configs, save_results, create_formatted_pivot,
    set_global_seed,
    load_and_filter_data
)
from data.data_utils import preprocess_features, create_daily_windows, split_data
from eval import calculate_daily_avg_metrics


def run_seasonal_analysis(data_dir: str = 'data', output_dir: str = 'sensitivity_analysis/results', local_output_dir: str = None):
    """
    Run seasonal effect analysis across all plants
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Sensitivity Analysis Experiment 1: Seasonal Effect")
    print("=" * 80)
    
    # Set global random seed for reproducibility
    set_global_seed(42)
    
    # Load all plant configurations
    plant_configs = load_all_plant_configs(data_dir)
    print(f"\nLoaded {len(plant_configs)} plant configurations")
    
    if len(plant_configs) == 0:
        print("Error: No plant configurations found")
        return
    
    # Models to test
    models_to_test = ALL_MODELS_NO_LINEAR + ['Linear']  # 7 + 1 = 8 models
    
    # Store results for each plant
    all_results = []
    
    # Run experiments for each plant
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
        for model in tqdm(models_to_test, desc=f"Plant {plant_id}"):
            # Create configuration
            if model == 'Linear':
                # Linear model: NWP only (no PV, no lookback)
                config = create_base_config(plant_config, model, complexity='high', 
                                          lookback=24, use_te=False)
                config['use_pv'] = False
                config['use_hist_weather'] = False
                config['no_hist_power'] = True
                config['past_hours'] = 0
            else:
                # Other models: PV+NWP, 24h lookback, no TE, high complexity
                config = create_base_config(plant_config, model, complexity='high', 
                                          lookback=24, use_te=False)
            
            try:
                # Run experiment using the corrected function
                result = run_single_experiment(config, df.copy(), use_sliding_windows=False)
                
                # Check if experiment succeeded
                if result['status'] != 'SUCCESS':
                    print(f"  Error running {model}: {result.get('error', 'Unknown error')}")
                    continue
                
                # Get predictions and test data
                y_pred = result.get('y_test_pred')
                y_test = result.get('y_test')
                test_dates = result.get('test_dates')
                
                if y_pred is None or y_test is None or test_dates is None:
                    print(f"  Warning: Missing data for {model}")
                    continue
                
                # Store base metrics (overall performance)
                base_mae = result['mae']
                base_rmse = result['rmse']
                base_r2 = result['r2']
                base_nrmse = result.get('nrmse', compute_nrmse(y_test.flatten(), y_pred.flatten()))
                train_time = result['train_time']
                test_samples = result['test_samples']
                
                # Store base result
                all_results.append({
                    'plant_id': plant_id,
                    'model': model,
                    'season': 'Overall',
                    'mae': base_mae,
                    'rmse': base_rmse,
                    'r2': base_r2,
                    'nrmse': base_nrmse,
                    'train_time': train_time,
                    'samples': test_samples
                })
                
                # Group test results by season using daily average method (same as main experiments)
                # y_test and y_pred are shape (n_days, 24)
                n_days = y_test.shape[0]
                
                # Create date array for each day
                test_dates_array = pd.to_datetime(test_dates)
                test_months = test_dates_array.month
                test_seasons = [get_season(m) for m in test_months]
                
                # Compute metrics for each season using daily average method
                for season in ['Spring', 'Summer', 'Fall', 'Winter']:
                    # Find days belonging to this season
                    season_day_indices = [i for i, s in enumerate(test_seasons) if s == season]
                    
                    if len(season_day_indices) == 0:
                        continue
                    
                    # Extract season data: (n_season_days, 24)
                    y_true_season = y_test[season_day_indices]
                    y_pred_season = y_pred[season_day_indices]
                    
                    # Use calculate_daily_avg_metrics (same as main experiments)
                    # All metrics (MAE, RMSE, R2, NRMSE) are calculated consistently
                    season_metrics = calculate_daily_avg_metrics(y_true_season, y_pred_season)
                    
                    mae = season_metrics.get('mae', np.nan)
                    rmse = season_metrics.get('rmse', np.nan)
                    r2 = season_metrics.get('r2', np.nan)
                    nrmse = season_metrics.get('nrmse', np.nan)  # From unified calculation
                    
                    # Store result
                    all_results.append({
                        'plant_id': plant_id,
                        'model': model,
                        'season': season,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'nrmse': nrmse,
                        'train_time': result.get('train_time', 0),
                        'samples': len(season_day_indices) * 24  # Total hourly samples
                    })
                
            except Exception as e:
                print(f"  Error running {model}: {e}")
                continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nError: No results generated")
        return
    
    print(f"\nTotal results: {len(results_df)}")
    
    # Aggregate results by season and model
    print("\n" + "=" * 80)
    print("Aggregating results across plants...")
    print("=" * 80)
    
    # Group by season and model
    grouped = results_df.groupby(['season', 'model'])
    
    # Compute mean and std
    agg_results = []
    for (season, model), group in grouped:
        agg_results.append({
            'season': season,
            'model': model,
            'mae_mean': group['mae'].mean(),
            'mae_std': group['mae'].std(),
            'rmse_mean': group['rmse'].mean(),
            'rmse_std': group['rmse'].std(),
            'r2_mean': group['r2'].mean(),
            'r2_std': group['r2'].std(),
            'nrmse_mean': group['nrmse'].mean(),
            'nrmse_std': group['nrmse'].std(),
            'train_time_mean': group['train_time'].mean(),
            'train_time_std': group['train_time'].std(),
            'n_plants': len(group)
        })
    
    agg_df = pd.DataFrame(agg_results)
    
    # Round to 2 decimals
    for col in agg_df.columns:
        if col not in ['season', 'model', 'n_plants']:
            agg_df[col] = agg_df[col].round(2)
    
    # Create formatted pivot tables with mean±std format
    formatted_pivots = create_formatted_pivot(agg_df, 'season', ['mae', 'rmse', 'r2', 'nrmse', 'train_time'])
    
    # Save results with model ordering and local backup
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file_detailed = os.path.join(output_dir, 'seasonal_effect_detailed.csv')
    save_results(results_df, output_file_detailed, local_output_dir, 'seasonal_effect')
    
    # Save aggregated results
    output_file_agg = os.path.join(output_dir, 'seasonal_effect_aggregated.csv')
    save_results(agg_df, output_file_agg, local_output_dir, 'seasonal_effect')
    
    # Save formatted pivot tables for each metric
    for metric, pivot_df in formatted_pivots.items():
        output_file_pivot = os.path.join(output_dir, f'seasonal_effect_pivot_{metric}.csv')
        save_results(pivot_df, output_file_pivot, local_output_dir, 'seasonal_effect')
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary (MAE by season and model):")
    print("=" * 80)
    summary = agg_df.pivot(index='season', columns='model', values='mae_mean')
    print(summary)
    
    print("\n" + "=" * 80)
    print("Seasonal Effect Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis: Seasonal Effect')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    parser.add_argument('--local-output', type=str, default=None,
                       help='Local backup directory for results')
    
    args = parser.parse_args()
    
    run_seasonal_analysis(data_dir=args.data_dir, output_dir=args.output_dir, local_output_dir=args.local_output)

