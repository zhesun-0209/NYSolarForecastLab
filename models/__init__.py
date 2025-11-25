"""
Model registry and imports
"""

# Model registry dictionary
MODEL_REGISTRY = {}


def register_model(name: str):
    """
    Decorator to register a model in the registry
    
    Usage:
        @register_model('MyModel')
        class MyModel(nn.Module):
            ...
    
    Args:
        name: Model name to register
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# Import all models to register them
from models.rnn_models import LSTM, GRU
from models.transformer import Transformer
from models.tcn import TCNModel
from models.ml_models import train_rf, train_xgb, train_lgbm, train_linear

# Register models
MODEL_REGISTRY['LSTM'] = LSTM
MODEL_REGISTRY['GRU'] = GRU
MODEL_REGISTRY['Transformer'] = Transformer
MODEL_REGISTRY['TCN'] = TCNModel
MODEL_REGISTRY['RF'] = train_rf
MODEL_REGISTRY['XGB'] = train_xgb
MODEL_REGISTRY['LGBM'] = train_lgbm
MODEL_REGISTRY['Linear'] = train_linear

__all__ = [
    'MODEL_REGISTRY',
    'register_model',
    'LSTM',
    'GRU',
    'Transformer',
    'TCNModel',
    'train_rf',
    'train_xgb',
    'train_lgbm',
    'train_linear',
]

