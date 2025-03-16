"""
Phase 1: Teacher model forward pass and output caching.
"""
import os
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time

from transformers import AutoTokenizer
from datasets import Dataset

from ..config import DistillationConfig
from ..models.teacher import TeacherModel
from ..data.dataset import load_and_prepare_dataset, create_dataloader
from ..data.caching import TeacherOutputCache, cache_teacher_outputs
from ..memory_utils import log_memory_stats, clear_gpu_memory, ensure_gpu_memory

logger = logging.getLogger(__name__)


def run_teacher_phase(
    config: DistillationConfig,
    cache_dir: Union[str, Path]
) -> bool:
    """
    Run the teacher model phase (forward pass and output caching).
    
    Args:
        config (DistillationConfig): Distillation configuration
        cache_dir (Union[str, Path]): Directory to store cache files
        
    Returns:
        bool: Whether the phase completed successfully
    """
    logger.info(f"Starting teacher phase with model: {config.teacher_model_id}")
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    # Check if cache exists and is complete
    cache_file = cache_path / "teacher_outputs.h5"
    metadata_file = cache_path / "metadata.pt"
    
    if cache_file.exists() and metadata_file.exists() and not config.force_recache:
        try:
            metadata = torch.load(metadata_file)
            if metadata.get("completed", False):
                logger.info(f"Found complete teacher cache at {cache_file}")
                return True
            else:
                logger.info(f"Found incomplete teacher cache at {cache_file}, will continue caching")
        except Exception as e:
            logger.warning(f"Error loading cache metadata: {e}")
    elif config.force_recache and cache_file.exists():
        logger.info(f"Force recaching requested, will recreate cache at {cache_file}")
    else:
        logger.info(f"No existing cache found, will create new cache at {cache_file}")
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.warning(f"Error loading teacher tokenizer: {e}")
        try:
            logger.info("Trying student tokenizer instead")
            tokenizer = AutoTokenizer.from_pretrained(config.student_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error loading student tokenizer: {e}")
            raise
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    train_dataset, _ = load_and_prepare_dataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        num_samples=config.num_samples,
        num_validation_samples=0,  # Don't need validation for teacher phase
        max_length=config.sequence_length,
        dataset_split=config.dataset_split,
        text_column=config.text_column,
        cache_dir=cache_path / "dataset_cache",
        seed=config.seed
    )
    
    log_memory_stats("After dataset loading")
    
    # Load teacher model
    logger.info(f"Loading teacher model: {config.teacher_model_id}")
    teacher = TeacherModel(
        model_id=config.teacher_model_id,
        max_memory_usage=config.max_memory_usage,
        offload_layers=config.cpu_offload,
        use_flash_attention=True,
        max_batch_size=config.teacher_batch_size,  # FIXED: Use correct parameter name
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize teacher model
    teacher.load_model()
    vocab_size = teacher.vocab_size
    
    log_memory_stats("After teacher model loading")
    
    # Initialize cache
    logger.info(f"Initializing teacher cache with vocab_size={vocab_size}")
    cache = TeacherOutputCache(
        cache_dir=cache_path,
        vocab_size=vocab_size,
        dtype=torch.float16,
        compress=True,
        chunk_size=64,
        background_writes=True
    )
    
    # Create forward pass function for caching
    @torch.no_grad()
    def teacher_forward(input_ids, attention_mask=None):
        return teacher.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_micro_batch=config.teacher_batch_size  # FIXED: Use correct parameter name
        )
    
    # Start caching
    logger.info(f"Running teacher forward pass on {len(train_dataset)} examples "
               f"with batch size {config.teacher_batch_size}")  # FIXED: Use correct parameter name
    
    start_time = time.time()
    
    try:
        # Run caching process
        success = cache_teacher_outputs(
            teacher_forward_fn=teacher_forward,
            dataset=train_dataset,
            cache=cache,
            batch_size=config.teacher_batch_size,  # FIXED: Use correct parameter name
            max_retries=3,
            device=teacher.device
        )
        
        # Wait for background writes to complete
        cache.wait_for_writes()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Teacher forward pass completed in {elapsed_time:.2f} seconds")
        
        # Unload teacher model to free memory
        logger.info("Unloading teacher model to free memory")
        teacher.unload()
        clear_gpu_memory()
        
        log_memory_stats("After teacher phase")
        
        # Cleanup temporary files
        cache.cleanup()
        
        return success
        
    except Exception as e:
        logger.error(f"Error during teacher forward pass: {e}")
        
        # Unload teacher model to free memory even on error
        teacher.unload()
        clear_gpu_memory()
        
        # Clean up temporary files
        try:
            cache.cleanup()
        except:
            pass
        
        raise