"""
TinyDistill: Ultra-Low Memory Knowledge Distillation for Language Models.

A professional-grade Python package for performing knowledge distillation 
on extremely memory-constrained hardware (e.g., GPUs with only 4GB VRAM).
"""

__version__ = "0.1.0"

from .config import DistillationConfig
from .memory_utils import log_memory_stats, clear_gpu_memory
from .logging_utils import setup_logging, ProgressLogger