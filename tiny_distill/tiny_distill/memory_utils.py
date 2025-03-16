"""
Memory management utilities for ultra-low memory distillation.
"""
import os
import gc
import time
import psutil
import torch
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from dataclasses import dataclass
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Configure PyTorch for memory efficiency
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,expandable_segments:True"


@dataclass
class MemoryStats:
    """Class for storing memory usage statistics."""
    gpu_allocated: float  # GB
    gpu_reserved: float   # GB
    gpu_total: float      # GB
    gpu_free: float       # GB
    ram_used: float       # GB
    ram_total: float      # GB
    ram_free: float       # GB
    gpu_tensors: Dict[str, Tuple[List[int], float]]  # tensor shapes and sizes


def get_gpu_memory() -> Tuple[float, float, float, float]:
    """
    Get detailed GPU memory statistics.

    Returns:
        Tuple[float, float, float, float]: allocated, reserved, total, and free memory in GB
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0, 0.0
    
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved
    
    return allocated, reserved, total, free


def get_tensor_memory_usage() -> Dict[str, Tuple[List[int], float]]:
    """
    Get detailed information about PyTorch tensors in memory.
    
    Returns:
        Dict[str, Tuple[List[int], float]]: Dictionary mapping tensor names to their shapes and sizes
    """
    if not torch.cuda.is_available():
        return {}
    
    tensor_stats = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensor_stats[f"tensor_{id(obj)}"] = (list(obj.size()), obj.element_size() * obj.nelement() / 1e6)
        except Exception:
            pass
    
    return tensor_stats


def get_memory_stats() -> MemoryStats:
    """
    Get comprehensive memory statistics for both GPU and system RAM.

    Returns:
        MemoryStats: Dataclass containing memory statistics
    """
    # GPU memory
    gpu_allocated, gpu_reserved, gpu_total, gpu_free = get_gpu_memory()
    
    # System memory
    ram = psutil.virtual_memory()
    ram_used = ram.used / 1e9
    ram_total = ram.total / 1e9
    ram_free = ram.free / 1e9
    
    # Tensor memory
    tensor_stats = get_tensor_memory_usage()
    
    return MemoryStats(
        gpu_allocated=gpu_allocated,
        gpu_reserved=gpu_reserved,
        gpu_total=gpu_total,
        gpu_free=gpu_free,
        ram_used=ram_used,
        ram_total=ram_total,
        ram_free=ram_free,
        gpu_tensors=tensor_stats
    )


def log_memory_stats(prefix: str = "") -> MemoryStats:
    """
    Log current memory statistics.

    Args:
        prefix (str, optional): Prefix for the log message. Defaults to "".

    Returns:
        MemoryStats: Memory statistics
    """
    stats = get_memory_stats()
    
    if prefix:
        prefix = f"{prefix} - "
    
    logger.info(
        f"{prefix}Memory - GPU: {stats.gpu_allocated:.2f}GB allocated, "
        f"{stats.gpu_reserved:.2f}GB reserved, {stats.gpu_free:.2f}GB free | "
        f"RAM: {stats.ram_used:.2f}GB used / {stats.ram_total:.2f}GB total"
    )
    
    # Log the top 5 largest tensors if we're using significant GPU memory
    if stats.gpu_allocated > 0.5 and stats.gpu_tensors:
        top_tensors = sorted(
            stats.gpu_tensors.items(), 
            key=lambda x: x[1][1], 
            reverse=True
        )[:5]
        
        logger.debug("Top 5 largest tensors:")
        for name, (shape, size_mb) in top_tensors:
            logger.debug(f"  {name}: {shape}, {size_mb:.2f}MB")
    
    return stats


def clear_gpu_memory() -> None:
    """Force clear CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()
    logger.debug("GPU memory cleared")


def ensure_gpu_memory(required_gb: float) -> bool:
    """
    Check if there's enough GPU memory available and attempt to free memory if needed.

    Args:
        required_gb (float): Required GPU memory in GB

    Returns:
        bool: True if enough memory is available after clearing cache, False otherwise
    """
    if not torch.cuda.is_available():
        return False
    
    # First check
    _, _, _, free_gb = get_gpu_memory()
    
    if free_gb >= required_gb:
        return True
    
    # Try to free memory
    clear_gpu_memory()
    
    # Check again
    _, _, _, free_gb = get_gpu_memory()
    
    return free_gb >= required_gb


def optimal_batch_size(
    model: torch.nn.Module, 
    input_shape: Tuple[int, ...], 
    target_memory_usage: float = 0.7,
    max_batch_size: int = 32,
    device: str = "cuda"
) -> int:
    """
    Determine the optimal batch size for a given model and input shape.

    Args:
        model (torch.nn.Module): PyTorch model
        input_shape (Tuple[int, ...]): Shape of a single input tensor excluding batch dimension
        target_memory_usage (float, optional): Target memory usage as a fraction of total. Defaults to 0.7.
        max_batch_size (int, optional): Maximum batch size to consider. Defaults to 32.
        device (str, optional): Device to use. Defaults to "cuda".

    Returns:
        int: Optimal batch size
    """
    if not torch.cuda.is_available() or device == "cpu":
        # On CPU, memory is less of a concern, use a reasonable default
        return max(1, max_batch_size // 2)
    
    # Get total GPU memory
    _, _, total_memory, _ = get_gpu_memory()
    
    # Start with batch size of 1
    batch_size = 1
    
    # Create a sample input
    sample_input = torch.zeros((1,) + input_shape, device=device)
    
    # Record initial memory usage
    initial_memory = torch.cuda.memory_allocated() / 1e9
    
    # Try a forward pass
    try:
        with torch.no_grad():
            _ = model(sample_input)
        
        # Measure memory usage for one sample
        one_sample_memory = torch.cuda.memory_allocated() / 1e9 - initial_memory
        
        # Memory available for batching
        available_memory = total_memory * target_memory_usage - initial_memory
        
        # Calculate max batch size
        calculated_batch_size = int(available_memory / one_sample_memory)
        batch_size = min(max(1, calculated_batch_size), max_batch_size)
        
        logger.info(f"Estimated optimal batch size: {batch_size} (one sample uses {one_sample_memory:.2f}GB)")
    except Exception as e:
        logger.warning(f"Error during batch size estimation: {e}")
        # If we can't estimate, be conservative
        batch_size = 1
    
    # Clean up
    del sample_input
    clear_gpu_memory()
    
    return batch_size


@contextmanager
def offload_modules(model: torch.nn.Module, module_names: List[str], device: str = "cpu"):
    """
    Temporarily offload specific modules to another device (typically CPU) and return them.
    
    Args:
        model (torch.nn.Module): The model containing modules to offload
        module_names (List[str]): List of module names to offload
        device (str, optional): Device to offload modules to. Defaults to "cpu".
    """
    # Save original devices
    original_devices = {}
    modules = {}
    
    # Move modules to target device
    for name in module_names:
        try:
            module = dict(model.named_modules())[name]
            original_devices[name] = next(module.parameters()).device
            module.to(device)
            modules[name] = module
        except (KeyError, StopIteration) as e:
            logger.warning(f"Couldn't offload module {name}: {e}")
    
    try:
        yield modules
    finally:
        # Move modules back to original devices
        for name, original_device in original_devices.items():
            if name in modules:
                modules[name].to(original_device)


@contextmanager
def temporary_freeze(model: torch.nn.Module):
    """
    Temporarily freeze a model's parameters and restore their original requires_grad state after.
    
    Args:
        model (torch.nn.Module): Model to freeze
    """
    # Save original requires_grad states
    original_states = {}
    for name, param in model.named_parameters():
        original_states[name] = param.requires_grad
        param.requires_grad_(False)
    
    try:
        yield model
    finally:
        # Restore original requires_grad states
        for name, param in model.named_parameters():
            if name in original_states:
                param.requires_grad_(original_states[name])


class MemoryTracker:
    """Track memory usage over time."""
    
    def __init__(self, log_interval: int = 10):
        """
        Initialize memory tracker.
        
        Args:
            log_interval (int, optional): Interval in seconds between logging. Defaults to 10.
        """
        self.log_interval = log_interval
        self.history = []
        self.running = False
        self.start_time = None
    
    def start(self):
        """Start tracking memory."""
        self.running = True
        self.start_time = time.time()
        self.history = []
        self._track()
    
    def stop(self):
        """Stop tracking memory."""
        self.running = False
    
    def _track(self):
        """Record current memory usage."""
        if not self.running:
            return
        
        stats = get_memory_stats()
        elapsed = time.time() - self.start_time
        
        self.history.append((elapsed, stats))
        
        # Log if it's time
        if len(self.history) == 1 or elapsed - self.history[-2][0] >= self.log_interval:
            log_memory_stats(f"[{elapsed:.1f}s]")
        
        # Schedule next tracking
        torch.cuda.synchronize()  # Ensure accurate timing
        time.sleep(0.1)  # Small sleep to prevent excessive CPU usage
        self._track()
    
    def summary(self) -> Dict:
        """
        Generate a summary of memory usage.
        
        Returns:
            Dict: Summary statistics
        """
        if not self.history:
            return {}
        
        gpu_allocated = [stats.gpu_allocated for _, stats in self.history]
        ram_used = [stats.ram_used for _, stats in self.history]
        
        return {
            "duration": self.history[-1][0],
            "gpu_min": min(gpu_allocated),
            "gpu_max": max(gpu_allocated),
            "gpu_avg": sum(gpu_allocated) / len(gpu_allocated),
            "ram_min": min(ram_used),
            "ram_max": max(ram_used),
            "ram_avg": sum(ram_used) / len(ram_used),
        }


def get_nvidia_smi_output() -> str:
    """
    Get output from nvidia-smi command for debugging.
    
    Returns:
        str: Output from nvidia-smi or error message
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        return result.stdout
    except Exception as e:
        return f"Error running nvidia-smi: {e}"


def slice_tensors(tensors: Dict[str, torch.Tensor], slice_size: int) -> List[Dict[str, torch.Tensor]]:
    """
    Slice a batch of tensors along the first dimension into smaller batches.
    
    Args:
        tensors (Dict[str, torch.Tensor]): Dictionary of tensors
        slice_size (int): Size of each slice
        
    Returns:
        List[Dict[str, torch.Tensor]]: List of sliced tensor dictionaries
    """
    batch_size = next(iter(tensors.values())).size(0)
    slices = []
    
    for i in range(0, batch_size, slice_size):
        slice_dict = {k: v[i:i+slice_size] for k, v in tensors.items()}
        slices.append(slice_dict)
    
    return slices


def setup_memory_efficient_attention():
    """Set up memory efficient attention if xformers is available."""
    try:
        import xformers.ops
        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            # Use PyTorch 2.0+ native memory-efficient attention when available
            logger.info("Using PyTorch's native memory-efficient attention")
        else:
            # Otherwise try to use xformers
            logger.info("Using xformers memory-efficient attention")
    except ImportError:
        logger.info("xformers not available, using standard attention")