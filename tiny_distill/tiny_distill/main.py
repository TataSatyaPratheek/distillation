"""
Main entry point for ultra-low memory distillation.
"""
import os
import sys
import time
import logging
import torch
from pathlib import Path
from typing import Dict, Optional, List
import argparse
from tqdm import tqdm

from .config import DistillationConfig
from .logging_utils import setup_logging
from .memory_utils import log_memory_stats, clear_gpu_memory, ensure_gpu_memory, MemoryTracker
from .data.dataset import load_and_prepare_dataset
from .data.caching import TeacherOutputCache, cache_teacher_outputs, TeacherCacheDataset
from .models.teacher import TeacherModel
from .models.student import StudentModel
from .training.teacher_phase import run_teacher_phase
from .training.student_phase import run_student_phase


logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ultra-Low Memory Knowledge Distillation for Language Models. "
            "Designed for hardware with extreme memory constraints."
        )
    )
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--teacher", 
        type=str, 
        default="EleutherAI/pythia-70m",
        help="Teacher model ID from Hugging Face or local path"
    )
    model_group.add_argument(
        "--student", 
        type=str, 
        default="EleutherAI/pythia-70m",
        help="Student model ID from Hugging Face or local path"
    )
    model_group.add_argument(
        "--temperature", 
        type=float, 
        default=2.0,
        help="Temperature for distillation"
    )
    
    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument(
        "--dataset", 
        type=str, 
        default="tatsu-lab/alpaca",
        help="Dataset name from Hugging Face"
    )
    data_group.add_argument(
        "--dataset_split", 
        type=str, 
        default="train",
        help="Dataset split to use"
    )
    data_group.add_argument(
        "--text_column", 
        type=str, 
        default=None,
        help="Column name for text data"
    )
    data_group.add_argument(
        "--samples", 
        type=int, 
        default=500,
        help="Number of training samples to use"
    )
    data_group.add_argument(
        "--validation_samples", 
        type=int, 
        default=50,
        help="Number of validation samples to use"
    )
    data_group.add_argument(
        "--sequence_length", 
        type=int, 
        default=64,
        help="Max sequence length for tokenization"
    )
    
    # Training arguments
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--output_dir", 
        type=str, 
        default="./distill_results",
        help="Output directory for saved models"
    )
    training_group.add_argument(
        "--batch_size", 
        type=int, 
        default=2,
        help="Batch size for training"
    )
    training_group.add_argument(
        "--micro_batch", 
        type=int, 
        default=1,
        help="Micro batch size for gradient accumulation"
    )
    training_group.add_argument(
        "--epochs", 
        type=int, 
        default=3,
        help="Number of epochs to train"
    )
    training_group.add_argument(
        "--learning_rate", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    training_group.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01,
        help="Weight decay"
    )
    training_group.add_argument(
        "--lora_rank", 
        type=int, 
        default=8,
        help="LoRA rank"
    )
    training_group.add_argument(
        "--lora_alpha", 
        type=float, 
        default=16,
        help="LoRA alpha"
    )
    training_group.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout"
    )
    
    # Memory optimization arguments
    memory_group = parser.add_argument_group("Memory Optimization")
    memory_group.add_argument(
        "--teacher_batch", 
        type=int, 
        default=1,
        help="Batch size for teacher model forward pass"
    )
    memory_group.add_argument(
        "--student_batch", 
        type=int, 
        default=4,
        help="Batch size for student model training"
    )
    memory_group.add_argument(
        "--gradient_accumulation", 
        type=int, 
        default=4,
        help="Gradient accumulation steps"
    )
    memory_group.add_argument(
        "--gradient_checkpointing", 
        action="store_true",
        help="Enable gradient checkpointing for memory savings"
    )
    memory_group.add_argument(
        "--cpu_offload", 
        action="store_true",
        help="Enable CPU offloading for extreme memory constraints"
    )
    memory_group.add_argument(
        "--layer_by_layer", 
        action="store_true",
        help="Enable layer-by-layer processing for extreme memory constraints"
    )
    memory_group.add_argument(
        "--max_memory", 
        type=float, 
        default=0.9,
        help="Maximum GPU memory usage (fraction of total)"
    )
    
    # System arguments
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    system_group.add_argument(
        "--log_level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    system_group.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Directory to store cache files (default: output_dir/cache)"
    )
    system_group.add_argument(
        "--force_recache", 
        action="store_true",
        help="Force re-generation of teacher cache even if it exists"
    )
    system_group.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from checkpoint if available"
    )
    system_group.add_argument(
        "--skip_teacher", 
        action="store_true",
        help="Skip teacher phase (use existing cache)"
    )
    system_group.add_argument(
        "--skip_student", 
        action="store_true",
        help="Skip student phase (only generate teacher cache)"
    )
    
    return parser.parse_args()


def validate_config(config: DistillationConfig) -> Dict[str, str]:
    """
    Validate configuration and provide warnings for potential issues.
    
    Args:
        config (DistillationConfig): Configuration to validate
        
    Returns:
        Dict[str, str]: Dictionary of warnings with keys as warning types and values as messages
    """
    warnings = {}
    
    # Check for extremely limited GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        if gpu_memory < 4.5:  # Less than 4.5 GB
            if not config.cpu_offload:
                warnings["memory_cpu_offload"] = (
                    f"You have only {gpu_memory:.1f}GB GPU memory. "
                    "Consider enabling --cpu_offload for better stability."
                )
            
            if config.teacher_batch_size > 1:  # FIXED: Use correct parameter name
                warnings["teacher_batch_size"] = (
                    f"Teacher batch size {config.teacher_batch_size} may be too large for {gpu_memory:.1f}GB GPU. "  # FIXED: Use correct parameter name
                    "Consider using --teacher_batch 1."
                )
            
            if config.sequence_length > 128:
                warnings["sequence_length"] = (
                    f"Sequence length {config.sequence_length} may be too large for {gpu_memory:.1f}GB GPU. "
                    "Consider using a smaller value like 64."
                )
            
            if config.student_batch_size > 2:  # FIXED: Use correct parameter name
                warnings["student_batch_size"] = (
                    f"Student batch size {config.student_batch_size} may be too large for {gpu_memory:.1f}GB GPU. "  # FIXED: Use correct parameter name
                    "Consider using a smaller batch size like 2."
                )
    
    # Model size warnings
    large_models = {
        "opt-1.3b": 2.6,
        "opt-2.7b": 5.4,
        "llama-7b": 14,
        "pythia-1.4b": 2.8,
        "pythia-2.8b": 5.6
    }
    
    for model_name, size_gb in large_models.items():
        if model_name in config.teacher_model_id.lower():
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if size_gb > gpu_memory * 0.8:
                    warnings["teacher_too_large"] = (
                        f"Teacher model {config.teacher_model_id} (~{size_gb:.1f}GB) "
                        f"may be too large for your {gpu_memory:.1f}GB GPU. "
                        "Consider using a smaller teacher model."
                    )
            break
    
    # Dataset size warnings
    if config.num_samples > 1000 and torch.cuda.get_device_properties(0).total_memory / 1e9 < 6:
        warnings["dataset_size"] = (
            f"Processing {config.num_samples} samples may be slow on your GPU. "
            "Consider using fewer samples (e.g., --samples 500)."
        )
    
    return warnings


def run_distillation(config: DistillationConfig) -> None:
    """
    Run the complete distillation pipeline.
    
    Args:
        config (DistillationConfig): Distillation configuration
    """
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup cache directory
    cache_dir = Path(config.cache_dir) if config.cache_dir else output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Log configuration and system info
    logger.info(f"Starting distillation with configuration: {config}")
    log_memory_stats("Initial")
    
    # Start memory tracker
    memory_tracker = MemoryTracker(log_interval=30)
    memory_tracker.start()
    
    try:
        # PHASE 1: Teacher model forward pass
        if not config.skip_teacher:
            logger.info("=== PHASE 1: Teacher Model Forward Pass ===")
            run_teacher_phase(config, cache_dir)
        else:
            logger.info("Skipping teacher phase as requested")
        
        # PHASE 2: Student model training
        if not config.skip_student:
            logger.info("=== PHASE 2: Student Model Training ===")
            run_student_phase(config, cache_dir)
        else:
            logger.info("Skipping student phase as requested")
        
        # Stop memory tracker
        memory_tracker.stop()
        memory_summary = memory_tracker.summary()
        
        # Log completion
        logger.info("=== Distillation Complete ===")
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"Memory usage - Peak GPU: {memory_summary.get('gpu_max', 0):.2f}GB, "
                    f"Peak RAM: {memory_summary.get('ram_max', 0):.2f}GB")
        
    except Exception as e:
        logger.error(f"Error during distillation: {e}", exc_info=True)
        memory_tracker.stop()
        raise
    finally:
        # Final cleanup
        clear_gpu_memory()


def main() -> None:
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=os.path.join(args.output_dir, "distillation.log"))
    
    # Convert arguments to config
    config = DistillationConfig.from_args(args)
    
    # Check for warnings
    warnings = validate_config(config)
    for warning_type, message in warnings.items():
        logger.warning(f"{warning_type}: {message}")
    
    # Set random seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Run distillation
    run_distillation(config)


if __name__ == "__main__":
    main()