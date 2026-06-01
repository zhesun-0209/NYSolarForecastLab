#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Experiment 4: Lookback Window Length

Analyze model performance with different lookback window lengths
- Models: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) - Linear NOT included
- Configuration: PV+NWP, various lookback windows, no TE, high complexity
- Lookback windows: 24h, 72h, 120h, 168h
- Metrics: MAE, RMSE, R2, NRMSE, train_time (mean and std across 100 plants)
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nysolarforecastlab.sensitivity_analysis.common_utils import (
    DL_MODELS, ML_MODELS, ALL_MODELS_NO_LINEAR,
    compute_nrmse,
    create_base_config,
    load_all_plant_configs,
    run_single_experiment,
    save_results, create_formatted_pivot,
    set_global_seed,
    load_and_filter_data
)
from nysolarforecastlab.data.data_utils import preprocess_features, create_daily_windows, split_data


# Lookback windows to test
LOOKBACK_WINDOWS = [24, 72, 120, 168]


def run_lookback_window_analysis(data_dir: str = 'data', output_dir: str = 'sensitivity_analysis/results', local_output_dir: str = None):
    """
    Run lookback window analysis across all plants
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Sensitivity Analysis Experiment 4: Lookback Window Length")
    print("=" * 80)
    
    # Set global random seed for reproducibility
    set_global_seed(42)
    
    # Load all plant configurations
    plant_configs = load_all_plant_configs(data_dir)
    print(f"\nLoaded {len(plant_configs)} plant configurations")
    
    if len(plant_configs) == 0:
        print("Error: No plant configurations found")
        return
    
    # Models to test (7 models, no Linear)
    models_to_test = ALL_MODELS_NO_LINEAR
    
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
        
        # Run experiments for each model and lookback window
        for model in tqdm(models_to_test, desc=f"Plant {plant_id}"):
            for lookback in LOOKBACK_WINDOWS:
                # Create configuration: PV+NWP, lookback hours, no TE, high complexity
                config = create_base_config(plant_config, model, complexity='high', 
                                          lookback=lookback, use_te=False)
                # Set weather category to medium_weather (7 features) to match H+M baseline
                config['weather_category'] = 'medium_weather'
                
                try:
                    # Train and evaluate
                    result = run_single_experiment(config, df.copy(), use_sliding_windows=False)
                    
                    # Check if experiment succeeded
                    if result['status'] != 'SUCCESS':
                        print(f"  Error running {model}: {result.get('error', 'Unknown error')}")
                        continue
                    
                    # Extract metrics
                    mae = result['mae']
                    rmse = result['rmse']
                    r2 = result['r2']
                    nrmse = result.get('nrmse', compute_nrmse(result['y_test'].flatten(), result['y_test_pred'].flatten()))
                    train_time = result['train_time']
                    test_samples = result['test_samples']
                    
                    # Store result
                    all_results.append({
                        'plant_id': plant_id,
                        'model': model,
                        'lookback_hours': lookback,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'nrmse': nrmse,
                        'train_time': train_time,
                        'test_samples': test_samples
                    })

                except Exception as e:
                    print(f"  Error running {model} - lookback {lookback}h: {e}")
                    continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nError: No results generated")
        return
    
    print(f"\nTotal results: {len(results_df)}")
    
    # Aggregate results by lookback window and model
    print("\n" + "=" * 80)
    print("Aggregating results across plants...")
    print("=" * 80)
    
    # Group by lookback_hours and model
    grouped = results_df.groupby(['lookback_hours', 'model'])
    
    # Compute mean and std
    agg_results = []
    for (lookback, model), group in grouped:
        agg_results.append({
            'lookback_hours': lookback,
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
        if col not in ['lookback_hours', 'model', 'n_plants']:
            agg_df[col] = agg_df[col].round(2)
    
    # Sort by lookback hours
    agg_df = agg_df.sort_values(['lookback_hours', 'model'])
    
    # Create formatted pivot tables with mean±std format
    formatted_pivots = create_formatted_pivot(agg_df, 'lookback_hours', ['mae', 'rmse', 'r2', 'nrmse', 'train_time'])
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file_detailed = os.path.join(output_dir, 'lookback_window_detailed.csv')
    save_results(results_df, output_file_detailed, local_output_dir, 'lookback_window')
    
    # Save aggregated results
    output_file_agg = os.path.join(output_dir, 'lookback_window_aggregated.csv')
    save_results(agg_df, output_file_agg, local_output_dir, 'lookback_window')
    
    # Save formatted pivot tables for each metric
    for metric, pivot_df in formatted_pivots.items():
        output_file_pivot = os.path.join(output_dir, f'lookback_window_pivot_{metric}.csv')
        save_results(pivot_df, output_file_pivot, local_output_dir, 'lookback_window')
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary (MAE by lookback window and model):")
    print("=" * 80)
    summary = agg_df.pivot(index='lookback_hours', columns='model', values='mae_mean')
    print(summary)
    
    print("\n" + "=" * 80)
    print("Lookback Window Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis: Lookback Window Length')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    parser.add_argument('--local-output', type=str, default=None,
                       help='Local backup directory for results')
    
    args = parser.parse_args()
    
    run_lookback_window_analysis(data_dir=args.data_dir, output_dir=args.output_dir, local_output_dir=args.local_output)

