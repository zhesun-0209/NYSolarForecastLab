#!/usr/bin/env python3
"""
Unit tests for model implementations
"""

import pytest
import torch
import numpy as np
from models.rnn_models import LSTM, GRU
from models.transformer import Transformer
from models.tcn import TCNModel


class TestModels:
    """Test cases for model implementations"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        batch_size = 4
        seq_len = 24
        hist_dim = 5
        fcst_dim = 3
        
        X_hist = torch.randn(batch_size, seq_len, hist_dim)
        X_fcst = torch.randn(batch_size, 24, fcst_dim)
        hours = torch.randint(0, 24, (batch_size,))
        
        return X_hist, X_fcst, hours
    
    def test_lstm_forward(self, sample_data):
        """Test LSTM forward pass"""
        X_hist, X_fcst, hours = sample_data
        config = {
            'd_model': 16,
            'hidden_dim': 8,
            'num_layers': 1,
            'dropout': 0.1,
            'use_forecast': True,
            'past_hours': 24,
            'future_hours': 24
        }
        
        model = LSTM(hist_dim=5, fcst_dim=3, config=config)
        output = model(X_hist, X_fcst, hours)
        
        assert output.shape == (4, 24, 1)
        assert not torch.isnan(output).any()
    
    def test_gru_forward(self, sample_data):
        """Test GRU forward pass"""
        X_hist, X_fcst, hours = sample_data
        config = {
            'd_model': 16,
            'hidden_dim': 8,
            'num_layers': 1,
            'dropout': 0.1,
            'use_forecast': True,
            'past_hours': 24,
            'future_hours': 24
        }
        
        model = GRU(hist_dim=5, fcst_dim=3, config=config)
        output = model(X_hist, X_fcst, hours)
        
        assert output.shape == (4, 24, 1)
        assert not torch.isnan(output).any()
    
    def test_transformer_forward(self, sample_data):
        """Test Transformer forward pass"""
        X_hist, X_fcst, hours = sample_data
        config = {
            'd_model': 16,
            'num_heads': 2,
            'num_layers': 1,
            'dropout': 0.1,
            'use_forecast': True,
            'past_hours': 24,
            'future_hours': 24
        }
        
        model = Transformer(hist_dim=5, fcst_dim=3, config=config)
        output = model(X_hist, X_fcst, hours)
        
        assert output.shape == (4, 24, 1)
        assert not torch.isnan(output).any()
    
    def test_tcn_forward(self, sample_data):
        """Test TCN forward pass"""
        X_hist, X_fcst, hours = sample_data
        config = {
            'd_model': 16,
            'tcn_channels': [8, 16],
            'kernel_size': 3,
            'dropout': 0.1,
            'use_forecast': True,
            'past_hours': 24,
            'future_hours': 24
        }
        
        model = TCNModel(hist_dim=5, fcst_dim=3, config=config)
        output = model(X_hist, X_fcst, hours)
        
        assert output.shape == (4, 24, 1)
        assert not torch.isnan(output).any()

