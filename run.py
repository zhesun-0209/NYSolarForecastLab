#!/usr/bin/env python3
"""
Unified entry point for PV-Forecasting experiments

All experiment functionality is consolidated in this single script.
"""

import argparse
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
os.chdir(script_dir)

from experiments import (
    run_forecast_experiments,
    check_all_plants_status,
    batch_create_configs,
    run_all_plants
)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='PV-Forecasting: Multi-Plant Solar Power Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create configs for all CSV files
  python run.py config

  # Run single plant experiments
  python run.py forecast --plant-id 1140

  # Run multi-plant batch
  python run.py multi_plant --max-plants 25

  # Check status
  python run.py status

  # Run sensitivity analysis
  python run.py sensitivity --analysis lookback_window
        """
    )
    
    subparsers = parser.add_subparsers(dest='task', help='Task to perform')
    
    # Config generation
    config_parser = subparsers.add_parser('config', help='Generate configuration files')
    config_parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    config_parser.add_argument('--config-dir', type=str, default='config/plants', help='Config directory')
    
    # Forecast task
    forecast_parser = subparsers.add_parser('forecast', help='Run forecasting experiments')
    forecast_parser.add_argument('--plant-id', type=str, default='1140', help='Plant ID')
    forecast_parser.add_argument('--output-dir', type=str, help='Output directory')
    forecast_parser.add_argument('--test-mode', action='store_true', help='Test mode: only run LSTM model')
    forecast_parser.add_argument('--test-model', type=str, default='LSTM', help='Model to use in test mode (default: LSTM)')
    
    # Multi-plant task
    multi_parser = subparsers.add_parser('multi_plant', help='Run multi-plant batch experiments')
    multi_parser.add_argument('--max-plants', type=int, help='Maximum number of plants')
    multi_parser.add_argument('--skip', type=int, default=0, help='Number of plants to skip')
    multi_parser.add_argument('--plants', nargs='+', help='Specific plant IDs')
    multi_parser.add_argument('--output-dir', type=str, help='Output directory')
    multi_parser.add_argument('--no-resume', action='store_true', help='Start fresh')
    
    # Status check
    status_parser = subparsers.add_parser('status', help='Check experiment status')
    status_parser.add_argument('--output-dir', type=str, default='.', help='Results directory')
    
    # Sensitivity analysis
    sens_parser = subparsers.add_parser('sensitivity', help='Run sensitivity analysis')
    sens_parser.add_argument('--analysis', type=str, 
                            choices=['lookback_window', 'model_complexity', 'training_scale',
                                    'seasonal_effect', 'hourly_effect', 'weather_feature'],
                            help='Analysis type')
    sens_parser.add_argument('--output-dir', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    if not args.task:
        parser.print_help()
        sys.exit(1)
    
    # Route to handlers
    if args.task == 'config':
        batch_create_configs(args.data_dir, args.config_dir)
    
    elif args.task == 'forecast':
        run_forecast_experiments(args.plant_id, args.output_dir, 
                                test_mode=args.test_mode, test_model=args.test_model)
    
    elif args.task == 'multi_plant':
        # run_all_plants is now imported from experiments
        run_all_plants(
            resume=not args.no_resume,
            skip=args.skip,
            max_plants=args.max_plants,
            plants=args.plants,
            output_dir=args.output_dir
        )
    
    elif args.task == 'status':
        check_all_plants_status(args.output_dir)
    
    elif args.task == 'sensitivity':
        from sensitivity_analysis.run_all_experiments import run_all_experiments
        # Map analysis type to experiment number
        analysis_map = {
            'lookback_window': 4,
            'model_complexity': 5,
            'training_scale': 6,
            'seasonal_effect': 1,
            'hourly_effect': 2,
            'weather_feature': 3
        }
        if args.analysis:
            exp_num = analysis_map.get(args.analysis)
            if exp_num:
                run_all_experiments(experiments=[exp_num], output_dir=args.output_dir)
            else:
                print(f"Unknown analysis type: {args.analysis}")
                sys.exit(1)
        else:
            run_all_experiments(output_dir=args.output_dir)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

