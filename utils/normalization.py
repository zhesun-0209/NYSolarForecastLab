#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified normalization utility class
Supports MinMaxScaler and StandardScaler
Provides unified interface and error handling
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Literal


class UnifiedScaler:
    """
    Unified scaler supporting multiple normalization methods
    
    Usage example:
        scaler = UnifiedScaler(method='minmax')
        X_scaled = scaler.fit_transform(X)
        X_original = scaler.inverse_transform(X_scaled)
    """
    
    def __init__(self, method: Literal['minmax', 'standard'] = 'minmax', 
                 feature_range: tuple = (0, 1)):
        """
        Initialize the scaler
        
        Args:
            method: Normalization method, 'minmax' or 'standard'
                - 'minmax': MinMaxScaler, scales data to [0, 1] or specified range
                - 'standard': StandardScaler, Z-score normalization (mean=0, std=1)
            feature_range: Only for MinMaxScaler, specifies scaling range, default (0, 1)
        """
        self.method = method
        self.feature_range = feature_range
        
        if method == 'minmax':
            self.scaler = MinMaxScaler(feature_range=feature_range)
        elif method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}. "
                           f"Supported methods: 'minmax', 'standard'")
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray):
        """
        Fit the scaler (compute statistics)
        
        Args:
            X: Input data, can be 1D or 2D array
        
        Returns:
            self: Returns self for method chaining
        """
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError("UnifiedScaler expects a 1D or 2D numeric array.")

        # sklearn scalers already handle constant columns. Avoid adding noise:
        # reproducibility is more important than suppressing harmless zeros.
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted statistics
        
        Args:
            X: Input data, can be 1D or 2D array
        
        Returns:
            Transformed data, preserving original shape
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform. "
                           "Call fit() or fit_transform() first.")
        
        X = np.asarray(X)
        original_shape = X.shape
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_transformed = self.scaler.transform(X)
        
        # Restore original shape
        if len(original_shape) == 1:
            return X_transformed.flatten()
        return X_transformed.reshape(original_shape)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform (restore data to original scale)
        
        Args:
            X: Transformed data, can be 1D or 2D array
        
        Returns:
            Inverse transformed data, preserving original shape
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform. "
                           "Call fit() or fit_transform() first.")
        
        X = np.asarray(X)
        original_shape = X.shape
        
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        X_inv = self.scaler.inverse_transform(X)
        
        # Restore original shape
        if len(original_shape) == 1:
            return X_inv.flatten()
        return X_inv.reshape(original_shape)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform data in one step
        
        Args:
            X: Input data, can be 1D or 2D array
        
        Returns:
            Transformed data, preserving original shape
        """
        return self.fit(X).transform(X)
    
    def __repr__(self):
        """String representation"""
        return f"UnifiedScaler(method='{self.method}', fitted={self.is_fitted})"
