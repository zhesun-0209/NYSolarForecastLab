"""
Transformer forecasting model.
"""
import warnings
import math
import torch
import torch.nn as nn

# Suppress warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    """PV forecasting Transformer model - using correct simple Encoder-only architecture"""
    def __init__(
        self,
        hist_dim: int,
        fcst_dim: int,
        config: dict
    ):
        super().__init__()
        self.cfg = config
        d_model = config['d_model']

        # Projection layers
        self.hist_proj = nn.Linear(hist_dim, d_model)
        self.fcst_proj = nn.Linear(fcst_dim, d_model) if fcst_dim > 0 else None

        # Improvement: local positional encoding
        self.pos_enc = LocalPositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['num_heads'],
            dim_feedforward=d_model * 4,
            dropout=config['dropout'],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])

        # Enhanced output layer
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d_model // 2, config['future_hours'])
        )

    def forward(
        self,
        hist: torch.Tensor,        # shape: (B, past_hours, hist_dim)
        fcst: torch.Tensor = None  # shape: (B, future_hours, fcst_dim), optional
    ) -> torch.Tensor:
        # Encode historical input
        h = self.hist_proj(hist)              # (B, past_hours, d_model)
        h = self.pos_enc(h)
        h_enc = self.encoder(h)               # (B, past_hours, d_model)

        # Encode forecast input if applicable
        if self.cfg.get('use_forecast', False) and fcst is not None and self.fcst_proj is not None:
            f = self.fcst_proj(fcst)          # (B, future_hours, d_model)
            f = self.pos_enc(f)
            f_enc = self.encoder(f)           # (B, future_hours, d_model)
            
            # Use last part of historical encoding and forecast encoding
            combined = torch.cat([h_enc, f_enc], dim=1)
        else:
            combined = h_enc

        # Use last time step for prediction
        last_timestep = combined[:, -1, :]  # (B, d_model)
        result = self.head(last_timestep)   # (B, future_hours)
        
        return result  # Return directly, no hardcoded scaling

class LocalPositionalEncoding(nn.Module):
    """Local positional encoding - supports longer sequences"""
    def __init__(self, d_model, max_len=200):  # Increased to 200 to support 168-hour lookback
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # If sequence length exceeds max_len, dynamically extend positional encoding
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # Dynamically create longer positional encoding
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
                               (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        else:
            return x + self.pe[:, :seq_len]

