"""
Teacher model handling with extreme memory optimization.
"""
import os
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import gc
from ..memory_utils import (
    log_memory_stats, 
    clear_gpu_memory, 
    ensure_gpu_memory,
    slice_tensors,
    offload_modules,
    temporary_freeze
)

logger = logging.getLogger(__name__)


class TeacherModel:
    """
    Memory-efficient wrapper for teacher models that handles extreme optimization.
    
    This class implements:
    1. Ultra-low memory loading using 4-bit and CPU offloading
    2. Layer-by-layer processing for huge models
    3. Chunked inference for large batch sizes 
    4. Smart caching strategies
    """
    
    def __init__(
        self,
        model_id: str,
        max_memory_usage: float = 0.9,
        offload_layers: bool = True,
        use_flash_attention: bool = True,
        max_batch_size: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize teacher model with extreme memory optimizations.
        
        Args:
            model_id (str): Model identifier from Hugging Face or local path
            max_memory_usage (float, optional): Maximum memory usage as fraction of total. Defaults to 0.9.
            offload_layers (bool, optional): Whether to offload layers to CPU. Defaults to True.
            use_flash_attention (bool, optional): Whether to use flash attention when available. Defaults to True.
            max_batch_size (int, optional): Maximum batch size for processing. Defaults to 2.
            device (Optional[str], optional): Device to use. Defaults to None (auto-detect).
        """
        self.model_id = model_id
        self.max_memory_usage = max_memory_usage
        self.offload_layers = offload_layers
        self.use_flash_attention = use_flash_attention
        self.max_batch_size = max_batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.config = None
        self.loaded = False
        self.vocab_size = None
        
        # Trace memory usage during loading
        log_memory_stats("Before teacher model initialization")
        
    def load_model(self) -> None:
        """
        Load teacher model with extreme memory optimizations.
        """
        if self.loaded:
            logger.info("Teacher model already loaded")
            return
        
        logger.info(f"Loading teacher model: {self.model_id}")
        
        try:
            # Configure model loading for minimal memory usage
            if self.device == "cuda":
                # Try to load in 4-bit first
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
                
                # Set device map for layer-wise CPU offloading if needed
                device_map = "auto"
                if self.offload_layers:
                    # For extreme memory constraints, force some layers to CPU
                    device_map = self._create_optimal_device_map()
                
                # Attempt 4-bit loading with optimizations
                logger.info("Attempting to load teacher with 4-bit quantization")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=bnb_config,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                except Exception as e:
                    logger.warning(f"4-bit quantization failed: {e}")
                    
                    # Fall back to 8-bit
                    logger.info("Attempting to load teacher with 8-bit quantization")
                    clear_gpu_memory()
                    
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            load_in_8bit=True,
                            device_map=device_map,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True
                        )
                    except Exception as e:
                        logger.warning(f"8-bit quantization failed: {e}")
                        
                        # Last resort: CPU only with sequential layer loading
                        logger.info("Fallback to CPU-only model loading")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            device_map={"": "cpu"},
                            low_cpu_mem_usage=True
                        )
            else:
                # CPU loading (simpler)
                logger.info("Loading teacher model on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True
                )
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.model, "gradient_checkpointing_enable"):
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()
            
            # Enable flash attention if available and requested
            if self.use_flash_attention:
                self._enable_flash_attention()
            
            # Extract model config info
            self.config = self.model.config
            self.vocab_size = self.model.config.vocab_size
            
            # Set model to evaluation mode
            self.model.eval()
            self.loaded = True
            
            logger.info(f"Teacher model loaded: {self.model_id}")
            log_memory_stats("After teacher model loading")
            
        except Exception as e:
            logger.error(f"Error loading teacher model: {e}")
            raise
    
    def _create_optimal_device_map(self) -> Dict[str, str]:
        """
        Create an optimal device map for extreme memory constraints.
        This function divides the model layers between GPU and CPU.
        
        Returns:
            Dict[str, str]: Device map for layer placement
        """
        try:
            # Initialize with a conservative device map
            # Keep embeddings and first few layers on GPU, rest on CPU
            device_map = {"": "cpu"}  # Default everything to CPU
            
            if self.device == "cuda":
                # Keep critical components on GPU
                device_map.update({
                    "model.embed_tokens": "cuda",
                    "model.norm": "cuda",
                    "lm_head": "cuda"
                })
                
                # Attempt to estimate how many layers can fit on GPU
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                available_memory = total_memory * self.max_memory_usage
                
                # Rough estimate: embeddings and head take ~15-20% of model size
                # So we have ~80% for layers
                layer_memory = available_memory * 0.8
                
                # Very rough estimate: each layer is approximately equal in size
                # For models with known architectures, we can be more precise
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(self.model_id)
                num_layers = getattr(config, "num_hidden_layers", None)
                
                if not num_layers:
                    # Try alternative attribute names
                    num_layers = getattr(config, "n_layer", None) or getattr(config, "num_layers", 24)
                
                # Average memory per layer estimation
                # This is very approximate and depends on model architecture
                memory_per_layer = 0.5  # GB, conservative estimate
                
                # Calculate how many layers can fit on GPU
                gpu_layers = min(num_layers, int(layer_memory / memory_per_layer))
                
                # At minimum, keep 1/4 of layers on GPU for reasonable performance
                gpu_layers = max(gpu_layers, num_layers // 4)
                
                logger.info(f"Device mapping: {gpu_layers}/{num_layers} layers on GPU")
                
                # Create device map for each layer
                for i in range(gpu_layers):
                    # Different models use different naming conventions
                    for prefix in ["model.layers.", "transformer.h.", "model.decoder.layers."]:
                        device_map[f"{prefix}{i}"] = "cuda"
            
            return device_map
            
        except Exception as e:
            logger.warning(f"Error creating optimal device map: {e}")
            # Fall back to auto device map
            return "auto"
    
    def _enable_flash_attention(self) -> None:
        """
        Enable flash attention for memory efficiency if available.
        """
        try:
            # Try to use either xformers or PyTorch 2.0+ memory-efficient attention
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                # PyTorch 2.0+ has native flash attention
                logger.info("Using PyTorch's native memory-efficient attention")
                
                # Some models require changing attention implementation
                for module in self.model.modules():
                    if hasattr(module, "use_sdpa") and callable(getattr(module, "use_sdpa", None)):
                        module.use_sdpa(True)
            else:
                # Try xformers if available
                try:
                    import xformers.ops
                    logger.info("Using xformers memory-efficient attention")
                    
                    # Apply to modules with attention
                    for module in self.model.modules():
                        if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                            module.set_use_memory_efficient_attention_xformers(True)
                except ImportError:
                    logger.info("xformers not available, using standard attention")
        except Exception as e:
            logger.warning(f"Failed to enable flash attention: {e}")
    
    def unload(self) -> None:
        """
        Completely unload the model from memory to free GPU resources.
        """
        if not self.loaded:
            return
        
        logger.info("Unloading teacher model")
        
        # Delete model
        if self.model is not None:
            del self.model
            self.model = None
        
        # Force garbage collection
        clear_gpu_memory()
        self.loaded = False
        
        log_memory_stats("After teacher model unloading")
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_micro_batch: int = 1
    ) -> torch.Tensor:
        """
        Memory-efficient forward pass for teacher model.
        
        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            max_micro_batch (int, optional): Maximum micro-batch size. Defaults to 1.
        
        Returns:
            torch.Tensor: Model logits
        """
        if not self.loaded:
            logger.warning("Model not loaded, loading now")
            self.load_model()
        
        # Ensure model is in eval mode
        self.model.eval()
        
        # Get batch size
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)
        
        # If batch size is too large, process in chunks
        if batch_size > max_micro_batch:
            logger.info(f"Processing batch of {batch_size} in chunks of {max_micro_batch}")
            
            # Prepare inputs
            inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask
            
            # Slice into micro-batches
            micro_batches = slice_tensors(inputs, max_micro_batch)
            all_logits = []
            
            # Process each micro-batch
            for i, micro_batch in enumerate(micro_batches):
                logger.debug(f"Processing micro-batch {i+1}/{len(micro_batches)}")
                
                # Move to the right device
                device_inputs = {k: v.to(self.model.device) for k, v in micro_batch.items()}
                
                # Forward pass
                outputs = self.model(**device_inputs)
                
                # Move logits to CPU to save GPU memory
                micro_logits = outputs.logits.cpu()
                all_logits.append(micro_logits)
                
                # Clear cache between micro-batches
                clear_gpu_memory()
            
            # Concatenate results
            logits = torch.cat(all_logits, dim=0)
            
        else:
            # Single forward pass for small batches
            inputs = {
                "input_ids": input_ids.to(self.model.device)
            }
            
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask.to(self.model.device)
            
            # With extreme memory pressure, we can process layer by layer
            if self.offload_layers and len(list(self.model.modules())) > 50:
                logits = self._layer_by_layer_forward(inputs)
            else:
                # Standard forward pass
                outputs = self.model(**inputs)
                logits = outputs.logits.cpu()  # Move to CPU to save GPU memory
        
        return logits
    
    def _layer_by_layer_forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process forward pass one layer at a time to minimize memory usage.
        This is an extreme optimization for very memory-constrained environments.
        
        Args:
            inputs (Dict[str, torch.Tensor]): Model inputs
            
        Returns:
            torch.Tensor: Model logits
        """
        logger.info("Using layer-by-layer forward pass for extreme memory optimization")
        
        # Get the base model
        base_model = self.model
        if hasattr(self.model, "model"):
            base_model = self.model.model
        
        # Extract model components based on architecture
        if hasattr(base_model, "layers"):
            # Typical architecture (e.g., OPT, GPT-Neo)
            layers = base_model.layers
            embeddings = base_model.embed_tokens
            norm = getattr(base_model, "norm", None)
            output_head = self.model.lm_head
            
        elif hasattr(base_model, "transformer") and hasattr(base_model.transformer, "h"):
            # GPT-2 style
            layers = base_model.transformer.h
            embeddings = base_model.transformer.wte
            norm = getattr(base_model.transformer, "ln_f", None)
            output_head = self.model.lm_head
            
        else:
            # Unknown architecture, fall back to standard forward
            logger.warning("Model architecture not recognized for layer-by-layer processing")
            outputs = self.model(**inputs)
            return outputs.logits.cpu()
        
        # Initial embedding
        hidden_states = embeddings(inputs["input_ids"])
        
        # Process layer by layer, offloading to CPU between layers
        for i, layer in enumerate(layers):
            logger.debug(f"Processing layer {i+1}/{len(layers)}")
            
            # Move only the current layer to GPU (if it's not already there)
            layer_device = next(layer.parameters()).device
            if layer_device != self.device:
                layer.to(self.device)
            
            # Process with this layer
            if hasattr(layer, "__call__"):
                if "attention_mask" in inputs:
                    hidden_states = layer(hidden_states, inputs["attention_mask"])
                else:
                    hidden_states = layer(hidden_states)
            else:
                # Some models use forwarding differently
                if "attention_mask" in inputs:
                    hidden_states = layer(hidden_states, attention_mask=inputs["attention_mask"])
                else:
                    hidden_states = layer(hidden_states)
            
            # Get actual tensor if it's wrapped in a tuple/dict
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            elif isinstance(hidden_states, dict) and "hidden_states" in hidden_states:
                hidden_states = hidden_states["hidden_states"]
            
            # Move layer back to CPU to save memory (if offloading is enabled)
            if self.offload_layers and i < len(layers) - 1:
                layer.to("cpu")
                clear_gpu_memory()
        
        # Final norm if present
        if norm is not None:
            norm.to(self.device)
            hidden_states = norm(hidden_states)
        
        # Final output projection
        output_head.to(self.device)
        logits = output_head(hidden_states)
        
        # Move result to CPU
        return logits.cpu()