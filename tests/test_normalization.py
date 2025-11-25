#!/usr/bin/env python3
"""
Unit tests for normalization utilities
"""

import pytest
import numpy as np
from utils.normalization import UnifiedScaler


class TestUnifiedScaler:
    """Test cases for UnifiedScaler"""
    
    def test_minmax_scaler_basic(self):
        """Test basic MinMaxScaler functionality"""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        scaler = UnifiedScaler(method='minmax')
        X_scaled = scaler.fit_transform(X)
        X_original = scaler.inverse_transform(X_scaled)
        
        assert X_scaled.shape == X.shape
        assert np.allclose(X, X_original, atol=1e-6)
        assert np.min(X_scaled) >= 0
        assert np.max(X_scaled) <= 1
    
    def test_standard_scaler_basic(self):
        """Test basic StandardScaler functionality"""
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        scaler = UnifiedScaler(method='standard')
        X_scaled = scaler.fit_transform(X)
        X_original = scaler.inverse_transform(X_scaled)
        
        assert X_scaled.shape == X.shape
        assert np.allclose(X, X_original, atol=1e-6)
        # Check that mean is approximately 0
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-6)
    
    def test_1d_array(self):
        """Test with 1D array"""
        X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        scaler = UnifiedScaler(method='minmax')
        X_scaled = scaler.fit_transform(X)
        X_original = scaler.inverse_transform(X_scaled)
        
        assert X_scaled.shape == X.shape
        assert np.allclose(X, X_original, atol=1e-6)
    
    def test_zero_variance_handling(self):
        """Test handling of zero variance features"""
        X = np.array([[1, 2, 1], [2, 3, 1], [3, 4, 1]], dtype=np.float32)
        scaler = UnifiedScaler(method='minmax')
        # Should not raise an error
        X_scaled = scaler.fit_transform(X)
        assert X_scaled.shape == X.shape
    
    def test_not_fitted_error(self):
        """Test that transform raises error if not fitted"""
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        scaler = UnifiedScaler(method='minmax')
        
        with pytest.raises(ValueError, match="must be fitted"):
            scaler.transform(X)
        
        with pytest.raises(ValueError, match="must be fitted"):
            scaler.inverse_transform(X)
    
    def test_invalid_method(self):
        """Test that invalid method raises error"""
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            UnifiedScaler(method='invalid')
    
    def test_repr(self):
        """Test string representation"""
        scaler = UnifiedScaler(method='minmax')
        assert 'UnifiedScaler' in repr(scaler)
        assert 'minmax' in repr(scaler)
        assert 'fitted=False' in repr(scaler)
        
        X = np.array([[1, 2], [3, 4]], dtype=np.float32)
        scaler.fit(X)
        assert 'fitted=True' in repr(scaler)

