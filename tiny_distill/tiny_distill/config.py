"""
Configuration management for ultra-low memory distillation.
"""
import os
import torch
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
import json
from pathlib import Path


@dataclass
class DistillationConfig:
    """Configuration for distillation process."""
    
    # Model configuration
    teacher_model_id: str = "EleutherAI/pythia-70m"
    student_model_id: str = "EleutherAI/pythia-70m"
    temperature: float = 2.0
    
    # Data configuration
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    text_column: Optional[str] = None
    num_samples: int = 500
    num_validation_samples: int = 50
    sequence_length: int = 64
    
    # Training configuration
    output_dir: str = "./distill_results"
    batch_size: int = 2
    micro_batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lora_rank: int = 8
    lora_alpha: float = 16
    lora_dropout: float = 0.05
    
    # Memory optimization
    teacher_batch_size: int = 1  # FIXED: Changed variable name to match usage
    student_batch_size: int = 4  # FIXED: Changed variable name to match usage
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = False
    cpu_offload: bool = False
    layer_by_layer: bool = False
    max_memory_usage: float = 0.9
    
    # System configuration
    seed: int = 42
    log_level: str = "INFO"
    cache_dir: Optional[str] = None
    force_recache: bool = False
    resume: bool = False
    skip_teacher: bool = False
    skip_student: bool = False
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "DistillationConfig":
        """
        Create a configuration from command-line arguments.
        
        Args:
            args (argparse.Namespace): Command-line arguments
            
        Returns:
            DistillationConfig: Configuration object
        """
        # FIXED: Correctly map argument names to class parameter names
        config = cls(
            # Model configuration
            teacher_model_id=args.teacher,
            student_model_id=args.student,
            temperature=args.temperature,
            
            # Data configuration
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            num_samples=args.samples,
            num_validation_samples=args.validation_samples,
            sequence_length=args.sequence_length,
            
            # Training configuration
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch,
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            
            # Memory optimization
            teacher_batch_size=args.teacher_batch,  # FIXED: Match parameter name
            student_batch_size=args.student_batch,  # FIXED: Match parameter name
            gradient_accumulation_steps=args.gradient_accumulation,
            gradient_checkpointing=args.gradient_checkpointing,
            cpu_offload=args.cpu_offload,
            layer_by_layer=args.layer_by_layer,
            max_memory_usage=args.max_memory,
            
            # System configuration
            seed=args.seed,
            log_level=args.log_level,
            cache_dir=args.cache_dir,
            force_recache=args.force_recache,
            resume=args.resume,
            skip_teacher=args.skip_teacher,
            skip_student=args.skip_student,
        )
        
        return config
    
    def save(self, path: str) -> None:
        """
        Save configuration to a JSON file.
        
        Args:
            path (str): Path to save configuration
        """
        # Convert to dict
        config_dict = {k: v for k, v in self.__dict__.items()}
        
        # Save to file
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DistillationConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            path (str): Path to load configuration from
            
        Returns:
            DistillationConfig: Configuration object
        """
        with open(path, "r") as f:
            config_dict = json.load(f)
        
        # Create instance with loaded values
        config = cls(**config_dict)
        
        return config
    
    def __str__(self) -> str:
        """Get string representation of configuration."""
        return json.dumps(self.__dict__, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def get_optimal_config(cls, vram_gb: float) -> "DistillationConfig":
        """
        Get optimal configuration based on available VRAM.
        
        Args:
            vram_gb (float): Available VRAM in GB
            
        Returns:
            DistillationConfig: Optimal configuration
        """
        config = cls()
        
        if vram_gb < 4.0:  # Extremely limited (e.g., 2-3GB)
            config.teacher_model_id = "EleutherAI/pythia-70m"
            config.student_model_id = "EleutherAI/pythia-70m"
            config.num_samples = 200
            config.sequence_length = 32
            config.teacher_batch_size = 1
            config.student_batch_size = 2
            config.gradient_accumulation_steps = 8
            config.cpu_offload = True
            config.layer_by_layer = True
            config.gradient_checkpointing = True
            
        elif vram_gb < 6.0:  # Limited (e.g., 4-5GB)
            config.teacher_model_id = "EleutherAI/pythia-160m"
            config.student_model_id = "EleutherAI/pythia-70m"
            config.num_samples = 300
            config.sequence_length = 64
            config.teacher_batch_size = 1
            config.student_batch_size = 4
            config.gradient_accumulation_steps = 4
            config.gradient_checkpointing = True
            
        elif vram_gb < 10.0:  # Moderate (e.g., 6-9GB)
            config.teacher_model_id = "EleutherAI/pythia-410m"
            config.student_model_id = "EleutherAI/pythia-70m"
            config.num_samples = 1000
            config.sequence_length = 128
            config.teacher_batch_size = 2
            config.student_batch_size = 8
            
        else:  # Comfortable (10GB+)
            config.teacher_model_id = "EleutherAI/pythia-1.4b"
            config.student_model_id = "EleutherAI/pythia-160m"
            config.num_samples = 2000
            config.sequence_length = 256
            config.teacher_batch_size = 4
            config.student_batch_size = 16
        
        return config