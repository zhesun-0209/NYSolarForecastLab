#!/usr/bin/env python3
"""
GPU utility functions
Support device selection and GPU memory monitoring.
"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

def _mps_is_available():
    """Return True when PyTorch can use Apple Metal Performance Shaders."""
    if not TORCH_AVAILABLE:
        return False
    try:
        return getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    except Exception:
        return False


def get_torch_device():
    """
    Return the preferred PyTorch device: CUDA, then Apple MPS, then CPU.

    Set FORCE_CPU=1 to force CPU. MPS is smoke-tested before being returned
    because it can appear available in some terminal contexts while tensor
    allocation still fails.
    """
    if not TORCH_AVAILABLE:
        return None

    import os

    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_is_available():
        try:
            device = torch.device("mps")
            torch.zeros(1, device=device)
            return device
        except Exception:
            return torch.device("cpu")
    return torch.device("cpu")


def get_device_description():
    """Return a human-readable description of the active training device."""
    if not TORCH_AVAILABLE:
        return "PyTorch not available"
    device = get_torch_device()
    if device is None:
        return "PyTorch not available"
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"

def get_gpu_memory_used():
    """
    Get current GPU memory usage (GB)
    
    Returns:
        GPU memory usage (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # Get current GPU memory usage (bytes)
            memory_allocated = torch.cuda.memory_allocated()
            # Convert to GB
            memory_gb = memory_allocated / (1024 ** 3)
            return round(memory_gb, 2)
        return 0.0
    except Exception:
        return 0.0

def get_gpu_memory_total():
    """
    Get total GPU memory (GB)
    
    Returns:
        Total GPU memory (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            # Get total GPU memory (bytes)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            # Convert to GB
            memory_gb = memory_total / (1024 ** 3)
            return round(memory_gb, 2)
        return 0.0
    except Exception:
        return 0.0

def get_gpu_memory_free():
    """
    Get available GPU memory (GB)
    
    Returns:
        Available GPU memory (GB), returns 0 if GPU unavailable
    """
    if not TORCH_AVAILABLE:
        return 0.0
    
    try:
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "mem_get_info"):
                memory_free, _ = torch.cuda.mem_get_info()
                return round(memory_free / (1024 ** 3), 2)
            memory_total = torch.cuda.get_device_properties(0).total_memory
            memory_reserved = torch.cuda.memory_reserved()
            return round((memory_total - memory_reserved) / (1024 ** 3), 2)
        return 0.0
    except Exception:
        return 0.0

def check_gpu_availability():
    """
    Check GPU availability
    
    Returns:
        Dictionary containing GPU status information
    """
    if not TORCH_AVAILABLE:
        return {
            'available': False,
            'device_count': 0,
            'current_device': None,
            'device_name': None,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'memory_free': 0.0
        }
    
    try:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            return {
                'available': True,
                'device_count': device_count,
                'current_device': current_device,
                'device_name': device_name,
                'memory_used': get_gpu_memory_used(),
                'memory_total': get_gpu_memory_total(),
                'memory_free': get_gpu_memory_free()
            }
        if _mps_is_available():
            return {
                'available': True,
                'device_count': 1,
                'current_device': 0,
                'device_name': 'Apple MPS',
                'memory_used': 0.0,
                'memory_total': 0.0,
                'memory_free': 0.0
            }
        else:
            return {
                'available': False,
                'device_count': 0,
                'current_device': None,
                'device_name': None,
                'memory_used': 0.0,
                'memory_total': 0.0,
                'memory_free': 0.0
            }
    except Exception:
        return {
            'available': False,
            'device_count': 0,
            'current_device': None,
            'device_name': None,
            'memory_used': 0.0,
            'memory_total': 0.0,
            'memory_free': 0.0
        }

def clear_gpu_memory():
    """
    Clear CUDA memory. This is a no-op on MPS and CPU.
    """
    if not TORCH_AVAILABLE:
        return
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def print_gpu_status():
    """
    Print GPU status information
    """
    status = check_gpu_availability()
    if status['available']:
        print(f"Device: {status['device_name']}")
        if status['memory_total'] > 0:
            print(f"Memory: {status['memory_total']:.1f} GB total, {status['memory_free']:.1f} GB free")
    else:
        print("Device: CPU")
