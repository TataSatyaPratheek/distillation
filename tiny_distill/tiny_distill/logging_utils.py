"""
Logging utilities for tracking distillation progress.
"""
import os
import sys
import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import json


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m',  # Red
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_message = super().format(record)
        level_name = record.levelname
        if level_name in self.COLORS:
            log_message = f"{self.COLORS[level_name]}{log_message}{self.COLORS['RESET']}"
        return log_message


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    use_colors: bool = True
) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level (str, optional): Logging level. Defaults to "INFO".
        log_file (Optional[str], optional): Path to log file. Defaults to None.
        log_format (str, optional): Log format string. Defaults to standard format.
        use_colors (bool, optional): Whether to use colors in console output. Defaults to True.
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    console_formatter = ColoredFormatter(log_format) if use_colors else logging.Formatter(log_format)
    file_formatter = logging.Formatter(log_format)  # No colors for file
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log_file is provided
    if log_file:
        # Ensure directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Log start message
    root_logger.info(f"Logging initialized (level: {log_level})"
                     f"{f', file: {log_file}' if log_file else ''}")


class ProgressLogger:
    """Logger for tracking distillation progress with metrics."""
    
    def __init__(
        self,
        output_dir: str,
        metrics_file: str = "metrics.jsonl",
        log_interval: int = 10
    ):
        """
        Initialize progress logger.
        
        Args:
            output_dir (str): Directory to store logs
            metrics_file (str, optional): Metrics file name. Defaults to "metrics.jsonl".
            log_interval (int, optional): Logging interval in seconds. Defaults to 10.
        """
        self.output_dir = Path(output_dir)
        self.metrics_file = self.output_dir / metrics_file
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.start_time = None
        self.last_log_time = None
        self.metrics_history = []
        self.current_phase = None
        self.current_epoch = None
    
    def start_phase(self, phase_name: str) -> None:
        """
        Start tracking a new phase.
        
        Args:
            phase_name (str): Name of the phase (e.g., "teacher", "student")
        """
        self.current_phase = phase_name
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.logger.info(f"Starting phase: {phase_name}")
    
    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        """
        Start tracking a new epoch.
        
        Args:
            epoch (int): Current epoch number (1-based)
            total_epochs (int): Total number of epochs
        """
        self.current_epoch = epoch
        self.logger.info(f"Starting epoch {epoch}/{total_epochs}")
    
    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        force: bool = False
    ) -> None:
        """
        Log metrics to file and console.
        
        Args:
            metrics (Dict[str, Any]): Metrics to log
            step (Optional[int], optional): Current step. Defaults to None.
            force (bool, optional): Whether to force logging regardless of interval. Defaults to False.
        """
        current_time = time.time()
        
        # Add metadata
        full_metrics = {
            "timestamp": current_time,
            "elapsed": current_time - self.start_time if self.start_time else 0,
            "phase": self.current_phase,
            "epoch": self.current_epoch,
            "step": step,
            **metrics
        }
        
        # Save to history
        self.metrics_history.append(full_metrics)
        
        # Log to file
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(full_metrics) + "\n")
        
        # Log to console if enough time has passed or forced
        if force or (current_time - self.last_log_time >= self.log_interval):
            self.last_log_time = current_time
            
            # Format metrics for display
            metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                   for k, v in metrics.items())
            
            # Log with context
            phase_str = f"[{self.current_phase}]" if self.current_phase else ""
            epoch_str = f"Epoch {self.current_epoch}" if self.current_epoch is not None else ""
            step_str = f"Step {step}" if step is not None else ""
            
            context = " ".join(filter(None, [phase_str, epoch_str, step_str]))
            elapsed = current_time - self.start_time if self.start_time else 0
            
            self.logger.info(f"{context} [{elapsed:.1f}s]: {metrics_str}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of tracked metrics.
        
        Returns:
            Dict[str, Any]: Summary statistics
        """
        if not self.metrics_history:
            return {}
        
        # Group metrics by phase and epoch
        phases = {}
        for m in self.metrics_history:
            phase = m.get("phase", "unknown")
            epoch = m.get("epoch", 0)
            
            if phase not in phases:
                phases[phase] = {}
            
            if epoch not in phases[phase]:
                phases[phase][epoch] = []
            
            phases[phase][epoch].append(m)
        
        # Compute summary
        summary = {
            "total_time": time.time() - self.start_time if self.start_time else 0,
            "phases": {}
        }
        
        # Summarize each phase and epoch
        for phase, epochs in phases.items():
            phase_summary = {}
            
            for epoch, metrics_list in epochs.items():
                # Extract numeric metrics
                numeric_metrics = {}
                for m in metrics_list:
                    for k, v in m.items():
                        if isinstance(v, (int, float)) and k not in ["timestamp", "elapsed", "epoch", "step"]:
                            if k not in numeric_metrics:
                                numeric_metrics[k] = []
                            numeric_metrics[k].append(v)
                
                # Compute statistics
                epoch_summary = {}
                for k, values in numeric_metrics.items():
                    if values:
                        epoch_summary[k] = {
                            "min": min(values),
                            "max": max(values),
                            "avg": sum(values) / len(values),
                            "last": values[-1]
                        }
                
                phase_summary[str(epoch)] = epoch_summary
            
            summary["phases"][phase] = phase_summary
        
        return summary
    
    def save_summary(self, filename: str = "summary.json") -> None:
        """
        Save summary to file.
        
        Args:
            filename (str, optional): Output filename. Defaults to "summary.json".
        """
        summary = self.get_summary()
        
        with open(self.output_dir / filename, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {self.output_dir / filename}")