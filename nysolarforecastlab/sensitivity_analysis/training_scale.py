#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity Analysis Experiment 6: Training Dataset Scale

Analyze model performance with different training data scales
- Models: 7 models (LSTM, GRU, Transformer, TCN, RF, XGB, LGBM) + Linear (NWP only)
- Configuration: PV+NWP, 24-hour lookback, no TE, high complexity
- Training scales: Low (20%), Medium (40%), High (60%), Full (80%)
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


# Training scales to test
TRAINING_SCALES = {
    'Low': 0.2,
    'Medium': 0.4,
    'High': 0.6,
    'Full': 0.8
}


def run_training_scale_analysis(data_dir: str = 'data', output_dir: str = 'sensitivity_analysis/results', local_output_dir: str = None):
    """
    Run training scale analysis across all plants
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
    """
    print("=" * 80)
    print("Sensitivity Analysis Experiment 6: Training Dataset Scale")
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
        
        # Run experiments for each model and training scale
        for model in tqdm(models_to_test, desc=f"Plant {plant_id}"):
            for scale_name, train_ratio in TRAINING_SCALES.items():
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
                
                # Set training ratio
                config['train_ratio'] = train_ratio
                # Adjust val/test ratio to maintain same test size
                remaining = 1.0 - train_ratio
                config['val_ratio'] = 0.1
                config['test_ratio'] = remaining - 0.1
                
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
                        'training_scale': scale_name,
                        'mae': mae,
                        'rmse': rmse,
                        'r2': r2,
                        'nrmse': nrmse,
                        'train_time': train_time,
                        'test_samples': test_samples
                    })

                except Exception as e:
                    print(f"  Error running {model} - {scale_name}: {e}")
                    continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("\nError: No results generated")
        return
    
    print(f"\nTotal results: {len(results_df)}")
    
    # Aggregate results
    print("\n" + "=" * 80)
    print("Aggregating results across plants...")
    print("=" * 80)
    
    # Group by training_scale and model
    grouped = results_df.groupby(['training_scale', 'model'])
    
    # Compute mean and std
    agg_results = []
    for (scale, model), group in grouped:
        agg_results.append({
            'training_scale': scale,
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
        if col not in ['training_scale', 'model', 'n_plants']:
            agg_df[col] = agg_df[col].round(2)
    
    # Order training scales
    scale_order = ['Low', 'Medium', 'High', 'Full']
    agg_df['training_scale'] = pd.Categorical(
        agg_df['training_scale'], 
        categories=scale_order, 
        ordered=True
    )
    agg_df = agg_df.sort_values(['training_scale', 'model'])
    
    # Create formatted pivot tables with mean±std format
    formatted_pivots = create_formatted_pivot(agg_df, 'training_scale', ['mae', 'rmse', 'r2', 'nrmse', 'train_time'])
    
    # Save results with model ordering and local backup
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    output_file_detailed = os.path.join(output_dir, 'training_scale_detailed.csv')
    save_results(results_df, output_file_detailed, local_output_dir, 'training_scale')
    
    # Save aggregated results
    output_file_agg = os.path.join(output_dir, 'training_scale_aggregated.csv')
    save_results(agg_df, output_file_agg, local_output_dir, 'training_scale')
    
    # Save formatted pivot tables for each metric
    for metric, pivot_df in formatted_pivots.items():
        output_file_pivot = os.path.join(output_dir, f'training_scale_pivot_{metric}.csv')
        save_results(pivot_df, output_file_pivot, local_output_dir, 'training_scale')
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary (MAE by training scale and model):")
    print("=" * 80)
    summary = agg_df.pivot(index='training_scale', columns='model', values='mae_mean')
    print(summary)
    
    print("\n" + "=" * 80)
    print("Training Scale Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Sensitivity Analysis: Training Dataset Scale')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    parser.add_argument('--local-output', type=str, default=None,
                       help='Local backup directory for results')
    
    args = parser.parse_args()
    
    run_training_scale_analysis(data_dir=args.data_dir, output_dir=args.output_dir, local_output_dir=args.local_output)

