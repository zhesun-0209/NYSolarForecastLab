#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all sensitivity analysis experiments

This script runs all 8 sensitivity analysis experiments:
1. Seasonal Effect
2. Hourly Effect
3. Weather Feature Adoption
4. Lookback Window Length
5. Model Complexity
6. Training Dataset Scale
7. No Shuffle Training
8. Dataset Extension (Hourly Sliding Windows)
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nysolarforecastlab.sensitivity_analysis.common_utils import set_global_seed


def run_all_experiments(data_dir='data', output_dir='results/sensitivity_analysis', experiments=None):
    """
    Run all or selected sensitivity analysis experiments
    
    Args:
        data_dir: Directory containing plant CSV files
        output_dir: Directory to save results
        experiments: List of experiment numbers to run (1-8), or None for all
    """
    print("=" * 80)
    print("Sensitivity Analysis - Run All Experiments")
    print("=" * 80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Set global random seed for reproducibility across all experiments
    set_global_seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define all experiments
    all_experiments = {
        1: ('Seasonal Effect', 'seasonal_effect'),
        2: ('Hourly Effect', 'hourly_effect'),
        3: ('Weather Feature Adoption', 'weather_feature_adoption'),
        4: ('Lookback Window Length', 'lookback_window'),
        5: ('Model Complexity', 'model_complexity'),
        6: ('Training Dataset Scale', 'training_scale'),
        7: ('No Shuffle Training', 'no_shuffle'),
        8: ('Dataset Extension (Hourly Sliding Windows)', 'dataset_extension')
    }
    
    # Determine which experiments to run
    if experiments is None:
        experiments_to_run = list(all_experiments.keys())
    else:
        experiments_to_run = [int(e) for e in experiments if int(e) in all_experiments]
    
    print(f"\nRunning {len(experiments_to_run)} experiments:")
    for exp_num in experiments_to_run:
        print(f"  {exp_num}. {all_experiments[exp_num][0]}")
    print()
    
    # Run experiments
    results = {}
    for exp_num in experiments_to_run:
        exp_name, exp_module = all_experiments[exp_num]
        
        print("\n" + "=" * 80)
        print(f"Experiment {exp_num}/{len(all_experiments)}: {exp_name}")
        print("=" * 80)
        
        try:
            # Import and run experiment
            module = __import__(
                f'nysolarforecastlab.sensitivity_analysis.{exp_module}',
                fromlist=['']
            )
            run_func = getattr(module, f'run_{exp_module}_analysis')
            
            start_time = datetime.now()
            run_func(data_dir=data_dir, output_dir=output_dir)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds() / 60
            results[exp_num] = {'status': 'SUCCESS', 'duration_min': duration}
            
            print(f"\n[OK] Experiment {exp_num} completed in {duration:.1f} minutes")
            
        except Exception as e:
            print(f"\n[ERROR] Experiment {exp_num} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            results[exp_num] = {'status': 'FAILED', 'error': str(e)}
    
    # Print summary
    print("\n" + "=" * 80)
    print("Sensitivity Analysis - Summary")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    for exp_num in experiments_to_run:
        exp_name = all_experiments[exp_num][0]
        result = results.get(exp_num, {})
        status = result.get('status', 'UNKNOWN')
        
        if status == 'SUCCESS':
            duration = result.get('duration_min', 0)
            print(f"[OK] Experiment {exp_num}: {exp_name} ({duration:.1f} min)")
        else:
            print(f"[FAILED] Experiment {exp_num}: {exp_name}")
    
    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sensitivity analysis experiments')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing plant CSV files')
    parser.add_argument('--output-dir', type=str, default='sensitivity_analysis/results',
                       help='Directory to save results')
    parser.add_argument('--experiments', type=int, nargs='+', choices=range(1, 9),
                       help='Experiment numbers to run (1-8). If not specified, run all.')
    
    args = parser.parse_args()
    
    run_all_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        experiments=args.experiments
    )
