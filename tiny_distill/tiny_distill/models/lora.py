"""
LoRA utilities for memory-efficient fine-tuning.
"""
import logging
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType

logger = logging.getLogger(__name__)


def get_lora_config(
    model_id: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_bias: str = "none"
) -> Tuple[LoraConfig, List[str]]:
    """
    Get LoRA configuration for specific model architecture.
    
    Args:
        model_id (str): Model identifier
        lora_rank (int, optional): LoRA rank. Defaults to 8.
        lora_alpha (int, optional): LoRA alpha. Defaults to 16.
        lora_dropout (float, optional): LoRA dropout. Defaults to 0.05.
        lora_bias (str, optional): LoRA bias. Defaults to "none".
        
    Returns:
        Tuple[LoraConfig, List[str]]: LoRA configuration and target modules
    """
    # Detect appropriate target modules for the model architecture
    model_id_lower = model_id.lower()
    
    # Default target modules (work for many models)
    target_modules = ["q_proj", "v_proj"]
    task_type = TaskType.CAUSAL_LM
    
    # Architecture-specific target modules
    if "opt" in model_id_lower:
        # OPT architecture
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    elif "pythia" in model_id_lower or "gpt-neo" in model_id_lower:
        # Pythia or GPT-Neo architecture
        target_modules = ["query_key_value", "dense"]
    elif "llama" in model_id_lower or "mistral" in model_id_lower:
        # Llama/Mistral architecture
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "falcon" in model_id_lower:
        # Falcon architecture
        target_modules = ["query_key_value", "dense"]
    elif "gpt2" in model_id_lower:
        # GPT-2 architecture
        target_modules = ["c_attn", "c_proj"]
    elif "bart" in model_id_lower or "t5" in model_id_lower:
        # Seq2Seq architecture
        target_modules = ["q", "v", "k", "o"]
        task_type = TaskType.SEQ_2_SEQ_LM
    elif "bert" in model_id_lower or "roberta" in model_id_lower:
        # BERT architecture
        target_modules = ["query", "value", "key", "dense"]
        task_type = TaskType.SEQ_CLS
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type=task_type
    )
    
    logger.info(f"Created LoRA config for {model_id} with target modules: {target_modules}")
    
    return lora_config, target_modules


def optimize_lora_for_memory(
    lora_config: LoraConfig,
    max_memory_gb: float
) -> LoraConfig:
    """
    Optimize LoRA configuration for memory constraints.
    
    Args:
        lora_config (LoraConfig): Original LoRA configuration
        max_memory_gb (float): Maximum available GPU memory in GB
        
    Returns:
        LoraConfig: Optimized LoRA configuration
    """
    original_rank = lora_config.r
    original_target_modules = lora_config.target_modules
    
    # Reduce rank for extreme memory constraints
    if max_memory_gb < 4.0:
        # Extremely limited memory
        new_rank = min(original_rank, 4)
        
        # Reduce target modules if needed
        if len(original_target_modules) > 2:
            # Only keep the most important modules (usually q and v projections)
            important_modules = ["q_proj", "v_proj", "query", "value", "q", "v", "query_key_value"]
            new_target_modules = [m for m in original_target_modules 
                                 if any(imp in m for imp in important_modules)]
            
            # Ensure we have at least some modules
            if not new_target_modules:
                new_target_modules = original_target_modules[:2]
        else:
            new_target_modules = original_target_modules
        
        logger.info(f"Optimizing LoRA for extreme memory constraints: "
                   f"rank {original_rank} -> {new_rank}, "
                   f"target modules {len(original_target_modules)} -> {len(new_target_modules)}")
    
    elif max_memory_gb < 8.0:
        # Limited memory
        new_rank = min(original_rank, 8)
        new_target_modules = original_target_modules
        
        logger.info(f"Optimizing LoRA for limited memory: rank {original_rank} -> {new_rank}")
    
    else:
        # Sufficient memory
        return lora_config
    
    # Create optimized config
    optimized_config = LoraConfig(
        r=new_rank,
        lora_alpha=lora_config.lora_alpha,
        target_modules=new_target_modules,
        lora_dropout=lora_config.lora_dropout,
        bias=lora_config.bias,
        task_type=lora_config.task_type
    )
    
    return optimized_config


def count_lora_parameters(model: torch.nn.Module) -> Tuple[int, int, float]:
    """
    Count total and trainable parameters in a LoRA model.
    
    Args:
        model (torch.nn.Module): Model with LoRA adapters
        
    Returns:
        Tuple[int, int, float]: Trainable parameters, total parameters, ratio
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    ratio = trainable_params / total_params if total_params > 0 else 0
    
    return trainable_params, total_params, ratio