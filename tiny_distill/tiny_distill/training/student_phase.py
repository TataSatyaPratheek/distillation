"""
Phase 2: Student model training with cached teacher outputs.
"""
import os
import logging
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import time
from tqdm import tqdm

from transformers import AutoTokenizer, get_scheduler
from datasets import Dataset

from ..config import DistillationConfig
from ..models.student import StudentModel
from ..data.dataset import load_and_prepare_dataset, create_dataloader
from ..data.caching import TeacherOutputCache, TeacherCacheDataset
from ..memory_utils import log_memory_stats, clear_gpu_memory, ensure_gpu_memory
from ..logging_utils import ProgressLogger
from .losses import CachedDistillationLoss

logger = logging.getLogger(__name__)


def run_student_phase(
    config: DistillationConfig,
    cache_dir: Union[str, Path]
) -> bool:
    """
    Run the student model training phase using cached teacher outputs.
    
    Args:
        config (DistillationConfig): Distillation configuration
        cache_dir (Union[str, Path]): Directory with teacher cache files
        
    Returns:
        bool: Whether the phase completed successfully
    """
    logger.info(f"Starting student phase with model: {config.student_model_id}")
    cache_path = Path(cache_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress logger
    progress_logger = ProgressLogger(output_dir=output_dir, log_interval=10)
    progress_logger.start_phase("student_training")
    
    # Check if teacher cache exists
    cache_file = cache_path / "teacher_outputs.h5"
    if not cache_file.exists():
        error_msg = f"Teacher cache not found at {cache_file}. Run teacher phase first."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load tokenizer
    logger.info("Loading tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.student_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        logger.warning(f"Error loading student tokenizer: {e}")
        try:
            logger.info("Trying teacher tokenizer instead")
            tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_id)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error loading teacher tokenizer: {e}")
            raise
    
    # Load dataset
    logger.info(f"Loading dataset: {config.dataset_name}")
    train_dataset, validation_dataset = load_and_prepare_dataset(
        dataset_name=config.dataset_name,
        tokenizer=tokenizer,
        num_samples=config.num_samples,
        num_validation_samples=config.num_validation_samples,
        max_length=config.sequence_length,
        dataset_split=config.dataset_split,
        text_column=config.text_column,
        cache_dir=cache_path / "dataset_cache",
        seed=config.seed
    )
    
    # Initialize teacher cache
    logger.info("Loading teacher cache")
    try:
        # We need the vocabulary size from the teacher cache
        # Let's load a small part to get the shape
        with torch.no_grad():
            teacher_outputs = torch.load(cache_path / "metadata.pt")
            vocab_size = train_dataset[0]["input_ids"].shape[0]  # Fallback
            if "vocab_size" in teacher_outputs:
                vocab_size = teacher_outputs["vocab_size"]
    except Exception as e:
        logger.warning(f"Error loading teacher cache metadata: {e}")
        # Estimate vocab size from tokenizer or fallback
        vocab_size = getattr(tokenizer, "vocab_size", 50257)
    
    # Create teacher cache
    cache = TeacherOutputCache(
        cache_dir=cache_path,
        vocab_size=vocab_size,
        dtype=torch.float16,
        compress=True,
        chunk_size=64,
        background_writes=False  # No need for background writes in student phase
    )
    
    # Create cache datasets
    logger.info("Creating train cache dataset")
    train_cache_dataset = TeacherCacheDataset(
        cache=cache,
        tokenized_dataset=train_dataset,
        preload=False  # Don't preload to save memory
    )
    
    # Create dataloaders
    logger.info(f"Creating dataloaders with batch size {config.student_batch_size}")  # FIXED: Use correct parameter name
    train_dataloader = create_dataloader(
        dataset=train_cache_dataset,
        batch_size=config.student_batch_size,  # FIXED: Use correct parameter name
        shuffle=True,
        device=None,  # Will move tensors to device in training loop
        preload=False,
        pin_memory=True,
        num_workers=0  # Avoid multiprocessing for small datasets
    )
    
    validation_dataloader = None
    if validation_dataset is not None:
        logger.info("Creating validation cache dataset")
        validation_cache_dataset = TeacherCacheDataset(
            cache=cache,
            tokenized_dataset=validation_dataset,
            preload=False
        )
        
        validation_dataloader = create_dataloader(
            dataset=validation_cache_dataset,
            batch_size=config.student_batch_size,  # FIXED: Use correct parameter name
            shuffle=False,
            device=None,
            preload=False,
            pin_memory=True,
            num_workers=0
        )
    
    log_memory_stats("After dataset and cache initialization")
    
    # Initialize student model
    logger.info(f"Initializing student model: {config.student_model_id}")
    student = StudentModel(
        model_id=config.student_model_id,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        gradient_checkpointing=config.gradient_checkpointing,
        use_4bit=True,
        use_nested_quant=True,
        use_cpu_offload=config.cpu_offload,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Set up student for training
    student.load_model(tokenizer=tokenizer)
    student.add_lora()
    student.setup_for_training()
    
    log_memory_stats("After student model initialization")
    
    # Initialize optimizer
    logger.info("Setting up optimizer and scheduler")
    optimizer = torch.optim.AdamW(
        student.get_trainable_parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Calculate total training steps
    num_training_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    
    # Initialize scheduler
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Initialize loss function
    distillation_loss = CachedDistillationLoss(
        temperature=config.temperature,
        alpha=1.0
    )
    
    # Set up checkpoint saving path
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Check if resuming from checkpoint
    if config.resume:
        logger.info("Checking for existing checkpoints")
        # Find latest checkpoint
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            latest_checkpoint = sorted(
                checkpoints,
                key=lambda x: int(x.name.split("-")[-1])
            )[-1]
            
            logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
            try:
                # Load optimizer state
                optimizer_state = torch.load(latest_checkpoint / "optimizer.bin")
                optimizer.load_state_dict(optimizer_state)
                
                # Load scheduler state
                scheduler_state = torch.load(latest_checkpoint / "scheduler.bin")
                lr_scheduler.load_state_dict(scheduler_state)
                
                # Load progress (epoch and step)
                progress = torch.load(latest_checkpoint / "progress.bin")
                start_epoch = progress["epoch"]
                
                # Load model
                student.load_adapter(str(latest_checkpoint))
                
                logger.info(f"Resumed training from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Error resuming from checkpoint: {e}")
                logger.info("Starting training from beginning")
                start_epoch = 0
        else:
            logger.info("No checkpoint found, starting training from beginning")
            start_epoch = 0
    else:
        start_epoch = 0
    
    # Training loop
    logger.info(f"Starting training for {config.num_epochs} epochs")
    global_step = 0
    best_val_loss = float("inf")
    
    try:
        for epoch in range(start_epoch, config.num_epochs):
            progress_logger.start_epoch(epoch + 1, config.num_epochs)
            
            # Training
            student.model.train()
            train_loss = 0.0
            num_train_steps = 0
            
            # Progress bar for training
            progress_bar = tqdm(
                train_dataloader,
                desc=f"Training epoch {epoch+1}/{config.num_epochs}",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(student.device) for k, v in batch.items()}
                
                # Get student outputs
                student_outputs = student.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_micro_batch=config.micro_batch_size
                )
                
                # Compute loss
                loss = distillation_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=batch["teacher_logits"],
                    attention_mask=batch["attention_mask"]
                )
                
                # Scale loss for gradient accumulation
                if config.gradient_accumulation_steps > 1:
                    loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Update weights with gradient accumulation
                if ((batch_idx + 1) % config.gradient_accumulation_steps == 0) or (batch_idx == len(train_dataloader) - 1):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                
                # Track loss
                train_loss += loss.item() * config.gradient_accumulation_steps
                num_train_steps += 1
                global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": train_loss / num_train_steps,
                    "lr": optimizer.param_groups[0]["lr"]
                })
                
                # Log metrics periodically
                if batch_idx % 10 == 0:
                    progress_logger.log_metrics(
                        metrics={
                            "train_loss": loss.item() * config.gradient_accumulation_steps,
                            "learning_rate": optimizer.param_groups[0]["lr"]
                        },
                        step=global_step
                    )
                
                # Save checkpoint periodically
                if global_step % 100 == 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint-{global_step}"
                    checkpoint_path.mkdir(exist_ok=True)
                    
                    # Save model
                    student.save_model(str(checkpoint_path), save_full=False)
                    
                    # Save optimizer and scheduler
                    torch.save(optimizer.state_dict(), checkpoint_path / "optimizer.bin")
                    torch.save(lr_scheduler.state_dict(), checkpoint_path / "scheduler.bin")
                    
                    # Save progress
                    torch.save(
                        {"epoch": epoch, "step": global_step},
                        checkpoint_path / "progress.bin"
                    )
                    
                    logger.info(f"Saved checkpoint at step {global_step}")
                
                # Clear memory periodically
                if batch_idx % 20 == 0:
                    clear_gpu_memory()
            
            # End of epoch
            epoch_train_loss = train_loss / num_train_steps
            logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Train loss: {epoch_train_loss:.4f}")
            
            # Validation
            if validation_dataloader is not None:
                student.model.eval()
                val_loss = 0.0
                num_val_steps = 0
                
                # Progress bar for validation
                progress_bar = tqdm(
                    validation_dataloader,
                    desc=f"Validation epoch {epoch+1}/{config.num_epochs}",
                    leave=False
                )
                
                with torch.no_grad():
                    for batch in progress_bar:
                        # Move batch to device
                        batch = {k: v.to(student.device) for k, v in batch.items()}
                        
                        # Get student outputs
                        student_outputs = student.forward(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            max_micro_batch=config.micro_batch_size
                        )
                        
                        # Compute loss
                        loss = distillation_loss(
                            student_logits=student_outputs.logits,
                            teacher_logits=batch["teacher_logits"],
                            attention_mask=batch["attention_mask"]
                        )
                        
                        # Track loss
                        val_loss += loss.item()
                        num_val_steps += 1
                        
                        # Update progress bar
                        progress_bar.set_postfix({"loss": val_loss / num_val_steps})
                
                # End of validation
                epoch_val_loss = val_loss / num_val_steps
                logger.info(f"Epoch {epoch+1}/{config.num_epochs} - Validation loss: {epoch_val_loss:.4f}")
                
                # Log validation metrics
                progress_logger.log_metrics(
                    metrics={
                        "epoch": epoch + 1,
                        "train_loss": epoch_train_loss,
                        "val_loss": epoch_val_loss
                    },
                    force=True
                )
                
                # Save best model
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_model_path = output_dir / "best_model"
                    
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                    logger.info(f"Saving best model to {best_model_path}")
                    
                    student.save_model(str(best_model_path), save_full=False)
                    
                    # Save optimizer and scheduler with best model
                    torch.save(optimizer.state_dict(), best_model_path / "optimizer.bin")
                    torch.save(lr_scheduler.state_dict(), best_model_path / "scheduler.bin")
            else:
                # Log training metrics
                progress_logger.log_metrics(
                    metrics={
                        "epoch": epoch + 1,
                        "train_loss": epoch_train_loss
                    },
                    force=True
                )
            
            # Clear memory between epochs
            clear_gpu_memory()
        
        # Save final model
        final_model_path = output_dir / "final_model"
        logger.info(f"Saving final model to {final_model_path}")
        student.save_model(str(final_model_path), save_full=True)
        
        # Save progress logger summary
        progress_logger.save_summary()
        
        # Unload student model
        student.unload()
        
        logger.info("Student training completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during student training: {e}")
        
        # Try to save checkpoint on error
        try:
            emergency_checkpoint_path = output_dir / "emergency_checkpoint"
            logger.info(f"Saving emergency checkpoint to {emergency_checkpoint_path}")
            student.save_model(str(emergency_checkpoint_path), save_full=False)
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")
        
        # Unload student model
        student.unload()
        
        # Save progress logger summary
        progress_logger.save_summary()
        
        raise