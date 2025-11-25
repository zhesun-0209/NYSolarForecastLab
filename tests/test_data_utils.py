#!/usr/bin/env python3
"""
Unit tests for data utilities
"""

import pytest
import pandas as pd
import numpy as np
from data.data_utils import (
    load_raw_data,
    preprocess_features,
    create_daily_windows,
    split_data
)


class TestDataUtils:
    """Test cases for data utilities"""
    
    def test_load_raw_data(self, tmp_path):
        """Test loading raw data from CSV"""
        # Create a temporary CSV file
        data = {
            'Year': [2022, 2022, 2022],
            'Month': [1, 1, 1],
            'Day': [1, 2, 3],
            'Hour': [0, 1, 2],
            'Capacity Factor': [0.5, 0.6, 0.7]
        }
        df = pd.DataFrame(data)
        csv_path = tmp_path / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        loaded_df = load_raw_data(str(csv_path))
        assert 'Datetime' in loaded_df.columns
        assert len(loaded_df) == 3
    
    def test_preprocess_features_basic(self):
        """Test basic feature preprocessing"""
        data = {
            'Year': [2022] * 24,
            'Month': [1] * 24,
            'Day': [1] * 24,
            'Hour': list(range(24)),
            'Capacity Factor': np.random.rand(24),
            'global_tilted_irradiance': np.random.rand(24)
        }
        df = pd.DataFrame(data)
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        
        config = {
            'use_pv': True,
            'use_hist_weather': False,
            'use_forecast': False,
            'use_time_encoding': True,
            'start_date': '2022-01-01',
            'end_date': '2022-01-02',
            'normalization_method': 'minmax'
        }
        
        df_clean, hist_feats, fcst_feats, scaler_hist, scaler_fcst, scaler_target, no_hist_power = \
            preprocess_features(df, config)
        
        assert len(df_clean) > 0
        assert scaler_target is not None
        assert not no_hist_power
    
    def test_create_daily_windows(self):
        """Test creating daily windows"""
        # Create sample data
        n_days = 10
        data = []
        for day in range(n_days):
            for hour in range(24):
                data.append({
                    'Year': 2022,
                    'Month': 1,
                    'Day': day + 1,
                    'Hour': hour,
                    'Capacity Factor': np.random.rand(),
                    'global_tilted_irradiance': np.random.rand()
                })
        df = pd.DataFrame(data)
        df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
        
        hist_feats = ['Capacity_Factor_hist']
        fcst_feats = ['global_tilted_irradiance']
        
        X_hist, X_fcst, y, hours, dates = create_daily_windows(
            df, future_hours=24, hist_feats=hist_feats, 
            fcst_feats=fcst_feats, past_hours=24
        )
        
        assert len(X_hist) > 0
        assert len(y) > 0
        assert X_hist.shape[1] == 24  # past_hours
        assert y.shape[1] == 24  # future_hours
    
    def test_split_data(self):
        """Test data splitting"""
        n_samples = 100
        X_hist = np.random.randn(n_samples, 24, 5)
        y = np.random.randn(n_samples, 24)
        hours = np.random.randint(0, 24, n_samples)
        dates = [f"2022-01-{i%30+1}" for i in range(n_samples)]
        
        result = split_data(X_hist, None, y, hours, dates, 
                           train_ratio=0.8, val_ratio=0.1, shuffle=False)
        
        Xh_tr, Xf_tr, y_tr, hrs_tr, dates_tr, \
        Xh_va, Xf_va, y_va, hrs_va, dates_va, \
        Xh_te, Xf_te, y_te, hrs_te, dates_te = result
        
        assert len(y_tr) == 80
        assert len(y_va) == 10
        assert len(y_te) == 10

