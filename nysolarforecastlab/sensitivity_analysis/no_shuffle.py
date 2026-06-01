#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Experiment 7: No Shuffle Training

Analyze model performance with sequential (non-shuffled) data splitting
- Models: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) + Linear (NWP only)
- Configuration: PV+NWP, 24-hour lookback, no TE, high complexity
- Data split: Sequential (no shuffle) - preserves temporal order
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
from nysolarforecastlab.data.data_utils import preprocess_features


def run_no_shuffle_analysis(data_dir: str = 'data', output_dir: str = 'sensitivity_analysis/results', local_output_dir: str = None):
    """
    Run no-shuffle analysis across all plants (tests sequential data splitting)
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Sensitivity Analysis Experiment 7: No Shuffle Training")
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
            
            # Set shuffle_split to False for this experiment (testing no shuffle / sequential split)
            config['shuffle_split'] = False
            
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
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'nrmse': nrmse,
                    'train_time': train_time,
                    'test_samples': test_samples
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
    
    # Aggregate results by model
    print("\n" + "=" * 80)
    print("Aggregating results across plants...")
    print("=" * 80)
    
    # Group by model
    grouped = results_df.groupby('model')
    
    # Compute mean and std
    agg_results = []
    for model, group in grouped:
        agg_results.append({
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
        if col not in ['model', 'n_plants']:
            agg_df[col] = agg_df[col].round(2)
    
    # Save results with model ordering and local backup
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file_detailed = os.path.join(output_dir, 'no_shuffle_detailed.csv')
    save_results(results_df, output_file_detailed, local_output_dir, 'no_shuffle')
    
    # Save aggregated results
    output_file_agg = os.path.join(output_dir, 'no_shuffle_aggregated.csv')
    save_results(agg_df, output_file_agg, local_output_dir, 'no_shuffle')
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary (No Shuffle - Sequential Split):")
    print("=" * 80)
    print(agg_df[['model', 'mae_mean', 'mae_std', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']])
    
    print("\n" + "=" * 80)
    print("No Shuffle Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis: No Shuffle Training')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    parser.add_argument('--local-output', type=str, default=None,
                       help='Local backup directory for results')
    
    args = parser.parse_args()
    
    run_no_shuffle_analysis(data_dir=args.data_dir, output_dir=args.output_dir, local_output_dir=args.local_output)

