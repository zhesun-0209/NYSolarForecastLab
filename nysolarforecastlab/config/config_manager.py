#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Manager - Multi-plant configuration management
Supports:
1. Load plant configurations
2. Generate 284 experiment configurations
3. Manage multi-plant experiments
"""

import yaml
import os
from typing import Dict, List
from pathlib import Path


PACKAGE_TEMPLATE_PATH = Path(__file__).with_name("plant_template.yaml")


class PlantConfigManager:
    """Plant Configuration Manager"""
    
    def __init__(self, template_path: str = None):
        """
        Initialize configuration manager
        
        Args:
            template_path: Template configuration file path
        """
        self.template_path = str(template_path or PACKAGE_TEMPLATE_PATH)
        self.template_config = self._load_template()
    
    def _load_template(self) -> Dict:
        """Load template configuration"""
        if os.path.exists(self.template_path):
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Template file {self.template_path} not found")
            return {}
    
    def load_plant_config(self, plant_config_path: str) -> Dict:
        """
        Load plant configuration file and merge with template
        
        Args:
            plant_config_path: Plant configuration file path
            
        Returns:
            Complete plant configuration dictionary
        """
        # Load plant-specific configuration
        with open(plant_config_path, 'r', encoding='utf-8') as f:
            plant_config = yaml.safe_load(f)
        
        # Merge template configuration (plant config overrides template)
        merged_config = self.template_config.copy()
        merged_config.update(plant_config)
        
        return merged_config
    
    def get_all_plants(self, plants_dir: str = "config/plants") -> List[Dict]:
        """
        Get all plant configurations, sorted by plant ID (numeric order)
        This ensures consistent ordering for skip/max_plants functionality
        
        Args:
            plants_dir: Plant configuration directory
            
        Returns:
            List of plant configurations, sorted by plant_id (numeric order)
        """
        plants = []
        
        if not os.path.exists(plants_dir):
            print(f"Warning: Plants directory {plants_dir} not found")
            return plants
        
        for filename in os.listdir(plants_dir):
            if filename.endswith('.yaml'):
                plant_path = os.path.join(plants_dir, filename)
                try:
                    config = self.load_plant_config(plant_path)
                    plants.append(config)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
        
        # Sort by plant_id (numeric order) to ensure consistent ordering
        # This is critical for skip/max_plants functionality to work correctly
        def sort_key(plant_config):
            plant_id = plant_config.get('plant_id', '0')
            # Try to convert to int for numeric sorting, fallback to string comparison
            try:
                return int(plant_id)
            except (ValueError, TypeError):
                return plant_id
        
        plants.sort(key=sort_key)
        
        return plants
    
    def generate_experiment_configs(self, plant_config: Dict) -> List[Dict]:
        """
        Generate 284 experiment configurations for a plant
        
        Experiment combinations:
        - DL models: LSTM, GRU, Transformer, TCN (4 types)
        - ML models: RF, XGB, LGBM (3 types)
        - Linear model: 1 type
        - Complexity: low, high (2 types, except Linear)
        - Lookback: 24h, 72h (2 types)
        - Time Encoding: True, False (2 types)
        - Feature combinations: PV, PV+HW, PV+NWP, PV+NWP+, NWP, NWP+ (6 types)
        
        Total:
        - DL: 4 models × 2 complexity × 2 lookback × 2 TE × 4 PV-features = 128
          + 4 models × 2 complexity × 2 TE × 2 NWP-only = 32
          = 160
        - ML: 3 models × 2 complexity × 2 lookback × 2 TE × 4 PV-features = 96
          + 3 models × 2 complexity × 2 TE × 2 NWP-only = 24
          = 120
        - Linear: 1 model × 2 TE × 2 NWP-only = 4
        
        Total: 160 + 120 + 4 = 284 experiments
        
        Args:
            plant_config: Plant configuration dictionary
            
        Returns:
            List of experiment configurations
        """
        configs = []
        plant_id = plant_config['plant_id']
        
        # Base configuration
        # CRITICAL: Ensure all experiments use data starting from 2022-01-01
        start_date = plant_config.get('start_date', '2022-01-01')
        end_date = plant_config.get('end_date', '2024-09-28')
        
        # Force start_date to be at least 2022-01-01
        if not start_date or start_date < '2022-01-01':
            start_date = '2022-01-01'
        
        base_config = {
            'plant_id': plant_id,
            'data_path': plant_config['data_path'],
            'start_date': start_date,
            'end_date': end_date,
            'future_hours': plant_config['future_hours'],
            'train_ratio': plant_config['train_ratio'],
            'val_ratio': plant_config['val_ratio'],
            'test_ratio': plant_config.get('test_ratio', 0.1),
            'shuffle_split': plant_config.get('shuffle_split', True),  # Random shuffle for robust evaluation
            'random_seed': plant_config.get('random_seed', 42),  # Fixed seed for reproducibility
            'weather_category': plant_config['weather_category'],
            'save_options': plant_config.get('save_options', {
                'save_model': False,
                'save_predictions': False,
                'save_training_log': False,
                'save_excel_results': False
            })
        }
        
        # Model lists
        dl_models = ['LSTM', 'GRU', 'Transformer', 'TCN']
        ml_models = ['RF', 'XGB', 'LGBM']
        complexities = ['low', 'high']
        lookbacks = [24, 72]
        te_options = [True, False]
        
        # Feature combinations
        pv_features = [
            {'name': 'PV', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': False, 'use_ideal_nwp': False},
            {'name': 'PV+HW', 'use_pv': True, 'use_hist_weather': True, 'use_forecast': False, 'use_ideal_nwp': False},
            {'name': 'PV+NWP', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
            {'name': 'PV+NWP+', 'use_pv': True, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
        ]
        
        nwp_features = [
            {'name': 'NWP', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': False},
            {'name': 'NWP+', 'use_pv': False, 'use_hist_weather': False, 'use_forecast': True, 'use_ideal_nwp': True},
        ]
        
        # Get model parameters
        dl_params = plant_config.get('dl_params', self.template_config.get('dl_params', {}))
        ml_params = plant_config.get('ml_params', self.template_config.get('ml_params', {}))
        
        # === 1. DL models: PV-related experiments ===
        for model in dl_models:
            for complexity in complexities:
                for lookback in lookbacks:
                    for feat in pv_features:
                        for use_te in te_options:
                            config = self._create_dl_config(
                                base_config, model, complexity, lookback, feat, use_te, False, dl_params
                            )
                            configs.append(config)
        
        # === 2. DL models: NWP-only experiments ===
        for model in dl_models:
            for complexity in complexities:
                for feat in nwp_features:
                    for use_te in te_options:
                        config = self._create_dl_config(
                            base_config, model, complexity, 0, feat, use_te, True, dl_params
                        )
                        configs.append(config)
        
        # === 3. ML models: PV-related experiments ===
        for model in ml_models:
            for complexity in complexities:
                for lookback in lookbacks:
                    for feat in pv_features:
                        for use_te in te_options:
                            config = self._create_ml_config(
                                base_config, model, complexity, lookback, feat, use_te, False, ml_params
                            )
                            configs.append(config)
        
        # === 4. ML models: NWP-only experiments ===
        for model in ml_models:
            for complexity in complexities:
                for feat in nwp_features:
                    for use_te in te_options:
                        config = self._create_ml_config(
                            base_config, model, complexity, 0, feat, use_te, True, ml_params
                        )
                        configs.append(config)
        
        # === 5. Linear model ===
        for feat in nwp_features:
            for use_te in te_options:
                config = self._create_linear_config(base_config, feat, use_te)
                configs.append(config)
        
        print(f"Generated {len(configs)} experiment configs for plant {plant_id}")
        return configs
    
    def _create_dl_config(self, base: Dict, model: str, complexity: str, 
                          lookback: int, feat: Dict, use_te: bool, 
                          is_nwp_only: bool, dl_params: Dict) -> Dict:
        """Create deep learning model configuration"""
        config = base.copy()
        config.update({
            'model': model,
            'model_complexity': complexity,
            'use_pv': feat['use_pv'],
            'use_hist_weather': feat['use_hist_weather'],
            'use_forecast': feat['use_forecast'],
            'use_ideal_nwp': feat['use_ideal_nwp'],
            'use_time_encoding': use_te,
            'past_hours': 0 if is_nwp_only else lookback,
            'past_days': 0 if is_nwp_only else (lookback // 24),
            'no_hist_power': is_nwp_only,
        })
        
        # Training parameters
        params = dl_params.get(complexity, {})
        config['train_params'] = {
            'epochs': params.get('epochs', 20 if complexity == 'low' else 50),
            'batch_size': params.get('batch_size', 64),
            'learning_rate': params.get('learning_rate', 0.001),
            'patience': params.get('patience', 10),
            'min_delta': params.get('min_delta', 0.001),
            'weight_decay': params.get('weight_decay', 0.0001)
        }
        
        # Model parameters
        config['model_params'] = {
            'd_model': params.get('d_model', 16 if complexity == 'low' else 32),
            'hidden_dim': params.get('hidden_dim', 8 if complexity == 'low' else 16),
            'num_heads': params.get('num_heads', 2),
            'num_layers': params.get('num_layers', 1 if complexity == 'low' else 2),
            'dropout': params.get('dropout', 0.1),
            'tcn_channels': params.get('tcn_channels', [8, 16] if complexity == 'low' else [16, 32]),
            'kernel_size': params.get('kernel_size', 3)
        }
        
        # Experiment name
        feat_str = f"{feat['name']}_{lookback}h" if not is_nwp_only else feat['name']
        te_str = "TE" if use_te else "noTE"
        config['experiment_name'] = f"{model}_{complexity}_{feat_str}_{te_str}"
        config['save_dir'] = f"{base['plant_id']}_results/{config['experiment_name']}"
        
        return config
    
    def _create_ml_config(self, base: Dict, model: str, complexity: str,
                          lookback: int, feat: Dict, use_te: bool,
                          is_nwp_only: bool, ml_params: Dict) -> Dict:
        """Create machine learning model configuration"""
        config = base.copy()
        config.update({
            'model': model,
            'model_complexity': complexity,
            'use_pv': feat['use_pv'],
            'use_hist_weather': feat['use_hist_weather'],
            'use_forecast': feat['use_forecast'],
            'use_ideal_nwp': feat['use_ideal_nwp'],
            'use_time_encoding': use_te,
            'past_hours': 0 if is_nwp_only else lookback,
            'past_days': 0 if is_nwp_only else (lookback // 24),
            'no_hist_power': is_nwp_only,
        })
        
        # ML model parameters - validated optimal config
        params = ml_params.get(complexity, {})
        config['model_params'] = {
            'n_estimators': params.get('n_estimators', 10 if complexity == 'low' else 30),
            'max_depth': params.get('max_depth', 1 if complexity == 'low' else 3),
            'learning_rate': params.get('learning_rate', 0.2 if complexity == 'low' else 0.1),
            'random_state': params.get('random_state', 42),
            'verbosity': params.get('verbosity', -1)  # Silent mode: suppress warnings
        }
        
        # Experiment name
        feat_str = f"{feat['name']}_{lookback}h" if not is_nwp_only else feat['name']
        te_str = "TE" if use_te else "noTE"
        config['experiment_name'] = f"{model}_{complexity}_{feat_str}_{te_str}"
        config['save_dir'] = f"{base['plant_id']}_results/{config['experiment_name']}"
        
        return config
    
    def _create_linear_config(self, base: Dict, feat: Dict, use_te: bool) -> Dict:
        """Create linear regression model configuration"""
        config = base.copy()
        config.update({
            'model': 'Linear',
            'model_complexity': None,
            'use_pv': feat['use_pv'],
            'use_hist_weather': feat['use_hist_weather'],
            'use_forecast': feat['use_forecast'],
            'use_ideal_nwp': feat['use_ideal_nwp'],
            'use_time_encoding': use_te,
            'past_hours': 0,
            'past_days': 0,
            'no_hist_power': True,
            'model_params': {}
        })
        
        # Experiment name
        te_str = "TE" if use_te else "noTE"
        config['experiment_name'] = f"Linear_{feat['name']}_{te_str}"
        config['save_dir'] = f"{base['plant_id']}_results/{config['experiment_name']}"
        
        return config


# ========================================
# Command-line utility functions
# ========================================

def create_new_plant_config(plant_id: str, data_path: str, start_date: str, end_date: str):
    """
    Create new plant configuration file
    
    Args:
        plant_id: Plant ID
        data_path: Data file path
        start_date: Start date
        end_date: End date
    """
    manager = PlantConfigManager()
    
    # Create configuration
    plant_config = {
        'plant_id': plant_id,
        'plant_name': f"Project {plant_id}",
        'data_path': data_path,
        'start_date': start_date,
        'end_date': end_date,
        'future_hours': 24,
        'past_hours': 24,  # Default lookback window (can be 24 or 72)
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'shuffle_split': True,  # Random shuffle for robust evaluation
        'random_seed': 42,  # Fixed seed for reproducibility
        'weather_category': "medium_weather",
        'has_ideal_nwp': True,
        'results_base_dir': "results",
        'save_options': {
            'save_model': False,
            'save_predictions': False,
            'save_training_log': False,
            'save_excel_results': False
        },
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
        'ml_params': {
            'low': {
                'n_estimators': 10,
                'max_depth': 1,
                'learning_rate': 0.2,
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
    
    # Save configuration
    os.makedirs("config/plants", exist_ok=True)
    config_path = f"config/plants/Plant{plant_id}.yaml"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(plant_config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"[OK] Created plant config: {config_path}")
    return config_path


def list_all_plants():
    """List all configured plants"""
    manager = PlantConfigManager()
    plants = manager.get_all_plants()
    
    print("\n" + "=" * 60)
    print("Configured Plants:")
    print("=" * 60)
    
    for i, plant in enumerate(plants, 1):
        print(f"{i}. Plant ID: {plant['plant_id']}")
        print(f"   Name: {plant.get('plant_name', 'N/A')}")
        print(f"   Data: {plant['data_path']}")
        print(f"   Period: {plant['start_date']} to {plant['end_date']}")
        print()
    
    print(f"Total: {len(plants)} plants")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python config_manager.py list                              # List all plants")
        print("  python config_manager.py create <id> <data> <start> <end>  # Create new plant config")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_all_plants()
    
    elif command == "create":
        if len(sys.argv) != 6:
            print("Usage: python config_manager.py create <plant_id> <data_path> <start_date> <end_date>")
            print("Example: python config_manager.py create 1141 data/Project1141.csv 2022-01-01 2024-12-31")
            sys.exit(1)
        
        plant_id = sys.argv[2]
        data_path = sys.argv[3]
        start_date = sys.argv[4]
        end_date = sys.argv[5]
        
        create_new_plant_config(plant_id, data_path, start_date, end_date)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
