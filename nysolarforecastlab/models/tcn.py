import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings('ignore')

class TemporalBlock(nn.Module):
    """TCN basic block, contains dilated convolution, residual connection and layer normalization"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """Remove padding for causal convolution"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNModel(nn.Module):
    """Simplified TCN model for PV forecasting"""
    def __init__(self, hist_dim: int, fcst_dim: int, config: dict):
        super().__init__()
        channels = config['tcn_channels']
        kernel = config['kernel_size']
        dropout = config.get('dropout', 0.2)
        self.use_fcst = config.get('use_forecast', False)
        future_hours = config['future_hours']

        # Store channels for later use
        self.channels = channels
        
        # Build TCN encoder with proper temporal blocks (only if hist_dim > 0)
        if hist_dim > 0:
            layers = []
            num_levels = len(channels)
            for i in range(num_levels):
                dilation_size = 2 ** i  # Dilation rate for dilated convolution
                in_channels = hist_dim if i == 0 else channels[i-1]
                out_channels = channels[i]
                padding = (kernel - 1) * dilation_size  # Padding for causal convolution
                
                layers.append(TemporalBlock(in_channels, out_channels, kernel, stride=1, 
                                          dilation=dilation_size, padding=padding, dropout=dropout))
            
            self.encoder = nn.Sequential(*layers)
        else:
            self.encoder = None

        # Forecast feature projection if enabled
        if self.use_fcst and fcst_dim > 0:
            # Simplified forecast feature processing, consistent with ML models
            self.fcst_proj = nn.Linear(fcst_dim * future_hours, channels[-1])
        else:
            self.fcst_proj = None

        # Unified simple output layer
        self.head = nn.Linear(channels[-1], future_hours)
        
        # Add numerical stability handling
        self.eps = 1e-8

    def forward(self, hist: torch.Tensor, fcst: torch.Tensor = None) -> torch.Tensor:
        last = None
        
        # Process historical data (if exists and has valid sequence length)
        # Check sequence length first (hist.shape[1] is the sequence length)
        if self.encoder is not None and hist.shape[1] > 0 and hist.shape[-1] > 0:
            x = hist.permute(0, 2, 1)  # (B, hist_dim, seq_len)
            
            # Check input length, use simple linear layer if too short
            if x.shape[-1] < 3:  # If sequence length < 3, use linear layer instead of convolution
                # Use global average pooling + linear layer
                x_pooled = x.mean(dim=-1)  # (B, hist_dim)
                # Create simple linear layer to replace convolution, output dim matches channels[-1]
                if not hasattr(self, 'fallback_linear'):
                    self.fallback_linear = nn.Linear(x.shape[1], self.channels[-1]).to(x.device)
                last = self.fallback_linear(x_pooled)  # (B, channels[-1])
            else:
                # Use full TCN processing
                out = self.encoder(x)  # (B, channels[-1], seq_len)
                last = out[:, :, -1]  # Take last time step

        # Process forecast data (if exists)
        if self.use_fcst and fcst is not None and self.fcst_proj is not None:
            # Simplified forecast feature processing, consistent with ML models
            f_flat = fcst.reshape(fcst.size(0), -1)
            f_proj = self.fcst_proj(f_flat)
            
            if last is not None:
                last = last + f_proj  # Simple addition fusion
            else:
                last = f_proj

        # If neither historical nor forecast data, create zero vector
        if last is None:
            # Create zero vector as default output
            batch_size = hist.size(0) if hist is not None else fcst.size(0)
            last = torch.zeros(batch_size, self.channels[-1]).to(hist.device if hist is not None else fcst.device)

        # Add numerical stability handling
        output = self.head(last)
        
        # Check and handle NaN values
        if torch.isnan(output).any():
            output = torch.where(torch.isnan(output), torch.zeros_like(output), output)
        
        # Ensure output is positive (solar power generation cannot be negative)
        output = torch.clamp(output, min=self.eps)
        
        return output
