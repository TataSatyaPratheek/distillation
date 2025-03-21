"""
Student model handling with memory-efficient training.
"""
import os
import json
import logging
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType
)
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

def serialize_config(config):
    """Convert a config object to a JSON-serializable dict."""
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    else:
        config_dict = {k: v for k, v in config.__dict__.items()
                      if not k.startswith("_")}

    # Convert any non-serializable types (like sets) to serializable ones
    for key, value in config_dict.items():
        if isinstance(value, set):
            config_dict[key] = list(value)  # Convert set to list
        elif hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
            config_dict[key] = serialize_config(value)  # Handle nested objects

    return config_dict

class StudentModel:
    """
    Memory-efficient wrapper for student models with LoRA training.

    This class implements:
    1. 4-bit or 8-bit quantization for base model
    2. Efficient LoRA fine-tuning
    3. Gradient checkpointing
    4. Checkpoint saving and loading
    5. CPU offloading
    """

    def __init__(
        self,
        model_id: str,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        gradient_checkpointing: bool = False,
        use_4bit: bool = True,
        use_nested_quant: bool = True,
        use_cpu_offload: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize student model with memory optimizations.

        Args:
            model_id (str): Model identifier from Hugging Face or local path
            lora_rank (int, optional): LoRA attention dimension. Defaults to 8.
            lora_alpha (int, optional): LoRA alpha parameter. Defaults to 16.
            lora_dropout (float, optional): LoRA dropout probability. Defaults to 0.05.
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
            use_4bit (bool, optional): Whether to use 4-bit quantization. Defaults to True.
            use_nested_quant (bool, optional): Whether to use nested quantization for 4-bit. Defaults to True.
            use_cpu_offload (bool, optional): Whether to offload to CPU. Defaults to False.
            device (Optional[str], optional): Device to use. Defaults to None (auto-detect).
        """
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.gradient_checkpointing = gradient_checkpointing
        self.use_4bit = use_4bit
        self.use_nested_quant = use_nested_quant
        self.use_cpu_offload = use_cpu_offload
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # State
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self.is_lora = False
        self.lora_config = None
        self.target_modules = None

        # Log configuration
        log_memory_stats("Before student model initialization")

    def detect_target_modules(self) -> List[str]:
        """
        Detect appropriate target modules for LoRA based on model architecture.

        Returns:
            List[str]: List of module names for LoRA
        """
        # Default target modules (work for many models)
        target_modules = ["q_proj", "v_proj"]

        # Try to identify specific architecture
        if "opt" in self.model_id.lower():
            # OPT architecture
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
        elif "pythia" in self.model_id.lower() or "gpt-neo" in self.model_id.lower():
            # Pythia or GPT-Neo architecture
            target_modules = ["query_key_value", "dense"]
        elif "llama" in self.model_id.lower() or "mistral" in self.model_id.lower():
            # Llama/Mistral architecture
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        elif "falcon" in self.model_id.lower():
            # Falcon architecture
            target_modules = ["query_key_value", "dense"]
        elif "gpt2" in self.model_id.lower():
            # GPT-2 architecture
            target_modules = ["c_attn", "c_proj"]

        logger.info(f"Detected target modules for LoRA: {target_modules}")
        return target_modules

    def load_model(self, tokenizer: Optional[PreTrainedTokenizer] = None) -> None:
        """
        Load student model with memory optimizations.

        Args:
            tokenizer (Optional[PreTrainedTokenizer], optional): Tokenizer to use. Defaults to None.
        """
        if self.loaded:
            logger.info("Student model already loaded")
            return

        logger.info(f"Loading student model: {self.model_id}")

        try:
            # Load tokenizer if not provided
            if tokenizer is None:
                logger.info("Loading tokenizer")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

                # Ensure pad token exists
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer = tokenizer

            # Configure device mapping
            if self.use_cpu_offload:
                # For CPU offloading, we need a specific device map
                logger.info("Setting up CPU offloading")
                device_map = "auto"
            else:
                device_map = "auto" if self.device == "cuda" else {"": "cpu"}

            # Load the model with appropriate settings based on constraints
            if self.device == "cuda":
                if self.use_4bit and self.use_cpu_offload:
                    # 4-bit with CPU offloading needs specific config
                    logger.info("Using 4-bit quantization with CPU offloading")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=self.use_nested_quant,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_enable_fp32_cpu_offload=True  # Key fix for the error
                    )

                    # Try loading with 4-bit quantization and CPU offloading
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_id,
                            quantization_config=quantization_config,
                            device_map=device_map,
                            torch_dtype=torch.float16,
                            low_cpu_mem_usage=True
                        )
                    except Exception as e:
                        # Fallback to 8-bit if 4-bit fails
                        logger.warning(f"4-bit quantization with CPU offloading failed: {e}")
                        logger.info("Falling back to 8-bit quantization")
                        clear_gpu_memory()

                        try:
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_id,
                                load_in_8bit=True,
                                device_map=device_map,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True
                            )
                        except Exception as e2:
                            # Final fallback to fp16 with CPU offloading
                            logger.warning(f"8-bit quantization failed: {e2}")
                            logger.info("Falling back to FP16 with CPU offloading")
                            clear_gpu_memory()

                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_id,
                                device_map=device_map,
                                torch_dtype=torch.float16,
                                low_cpu_mem_usage=True
                            )

                elif self.use_4bit:
                    # Standard 4-bit quantization (no CPU offloading)
                    logger.info("Using 4-bit quantization without CPU offloading")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=self.use_nested_quant,
                        bnb_4bit_quant_type="nf4"
                    )

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        quantization_config=quantization_config,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )

                else:
                    # Standard FP16 loading
                    logger.info("Using standard FP16 loading")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        device_map=device_map,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
            else:
                # CPU-only loading
                logger.info("Loading on CPU")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True
                )

            # If we loaded a quantized model, prepare it for LoRA training
            if hasattr(self.model, "is_quantized") and getattr(self.model, "is_quantized", False):
                logger.info("Preparing quantized model for kbit training")
                self.model = prepare_model_for_kbit_training(self.model)

            # Enable gradient checkpointing if requested
            if self.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                logger.info("Enabling gradient checkpointing")
                self.model.gradient_checkpointing_enable()

            self.loaded = True
            self.is_lora = False

            logger.info(f"Student model loaded: {self.model_id}")
            log_memory_stats("After student model loading")

        except Exception as e:
            logger.error(f"Error loading student model: {e}")
            raise

    def add_lora(self) -> None:
        """Add LoRA adapters to the model for memory-efficient fine-tuning with proper gradient flow."""
        if not self.loaded:
            logger.warning("Model not loaded, loading now")
            self.load_model()

        if self.is_lora:
            logger.info("LoRA adapters already added")
            return

        logger.info("Adding LoRA adapters")

        # Detect target modules if not set
        if self.target_modules is None:
            self.target_modules = self.detect_target_modules()

        # Create LoRA config with explicit modules
        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Add LoRA adapters
        self.model = get_peft_model(self.model, self.lora_config)

        # Explicitly set training mode
        self.model.train()

        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        if trainable_params == 0:
            # Force enable training for LoRA parameters
            logger.warning("No trainable parameters found after adding LoRA, manually enabling...")
            for name, param in self.model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad_(True)
                    logger.debug(f"Enabled gradients for {name}")

            # Check again
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.is_lora = True
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")

        # Verify the model is properly set up for training
        self.setup_for_training()
        log_memory_stats("After adding LoRA adapters")

    def save_model(self, output_dir: str, save_full: bool = False) -> None:
        """
        Save the model or LoRA adapters.

        Args:
            output_dir (str): Output directory
            save_full (bool, optional): Whether to save the full model. Defaults to False.
        """
        if not self.loaded:
            logger.warning("Model not loaded, nothing to save")
            return

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save tokenizer
        if self.tokenizer is not None:
            logger.info(f"Saving tokenizer to {output_path}")
            self.tokenizer.save_pretrained(output_path)

        if self.is_lora and not save_full:
            # Save LoRA adapters only (much smaller)
            logger.info(f"Saving LoRA adapters to {output_path}")
            self.model.save_pretrained(output_path)

            # Save LoRA config for reference
            with open(output_path / "lora_config.json", "w") as f:
                # Use our robust serialization function
                config_dict = serialize_config(self.lora_config)
                json.dump(config_dict, f, indent=2)

        else:
            # Save full model (much larger)
            logger.info(f"Saving full model to {output_path}")

            if self.is_lora:
                # Merge LoRA weights with base model first
                logger.info("Merging LoRA weights with base model")
                try:
                    merged_model = self.model.merge_and_unload()
                    merged_model.save_pretrained(output_path)
                except Exception as e:
                    logger.error(f"Error merging LoRA weights: {e}")
                    logger.info("Falling back to saving adapters only")
                    self.model.save_pretrained(output_path)
            else:
                # Save regular model
                self.model.save_pretrained(output_path)

        logger.info(f"Model saved to {output_path}")

    def load_adapter(self, adapter_path: str) -> None:
        """
        Load LoRA adapters from a saved checkpoint.

        Args:
            adapter_path (str): Path to the saved adapters
        """
        if not self.loaded:
            logger.warning("Model not loaded, loading base model first")
            self.load_model()

        logger.info(f"Loading LoRA adapters from {adapter_path}")

        try:
            # If model isn't already a PeftModel, convert it first
            if not isinstance(self.model, PeftModel):
                # Try to load LoRA config from saved files
                config_path = os.path.join(adapter_path, "lora_config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_dict = f.read()
                    self.lora_config = LoraConfig.from_json_string(config_dict)
                    self.target_modules = self.lora_config.target_modules
                else:
                    # Use adapter_config.json from the saved adapter
                    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
                    if os.path.exists(adapter_config_path):
                        self.lora_config = LoraConfig.from_pretrained(adapter_path)
                        self.target_modules = self.lora_config.target_modules
                    else:
                        # Fall back to detecting target modules
                        self.target_modules = self.detect_target_modules()
                        self.lora_config = LoraConfig(
                            r=self.lora_rank,
                            lora_alpha=self.lora_alpha,
                            target_modules=self.target_modules,
                            lora_dropout=self.lora_dropout,
                            bias="none",
                            task_type=TaskType.CAUSAL_LM
                        )

                # Add LoRA adapters with the loaded/created config
                self.model = get_peft_model(self.model, self.lora_config)

            # Now load the saved adapters
            self.model.load_adapter(adapter_path)
            self.is_lora = True

            logger.info(f"LoRA adapters loaded from {adapter_path}")
            log_memory_stats("After loading LoRA adapters")

        except Exception as e:
            logger.error(f"Error loading LoRA adapters: {e}")
            raise

    def get_trainable_parameters(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters.

        Returns:
            List[torch.nn.Parameter]: List of trainable parameters
        """
        if not self.loaded:
            logger.warning("Model not loaded")
            return []

        return [p for p in self.model.parameters() if p.requires_grad]

    def setup_for_training(self) -> None:
        """Set up model for training."""
        if not self.loaded:
            self.load_model()

        if not self.is_lora:
            self.add_lora()

        self.model.train()
        logger.info("Model set up for training")

    def setup_for_inference(self) -> None:
        """Set up model for inference."""
        if not self.loaded:
            self.load_model()

        self.model.eval()
        logger.info("Model set up for inference")

    def unload(self) -> None:
        """Completely unload the model from memory."""
        if not self.loaded:
            return

        logger.info("Unloading student model")

        # Delete model
        if self.model is not None:
            del self.model
            self.model = None

        # Force garbage collection
        clear_gpu_memory()
        self.loaded = False
        self.is_lora = False

        log_memory_stats("After student model unloading")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_micro_batch: int = 1,
        return_dict: bool = True
    ) -> Any:
        """
        Memory-efficient forward pass for student model that preserves gradients.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (Optional[torch.Tensor], optional): Attention mask. Defaults to None.
            max_micro_batch (int, optional): Maximum micro-batch size. Defaults to 1.
                        return_dict (bool, optional): Whether to return dict. Defaults to True.

        Returns:
            Any: Model outputs
        """
        if not self.loaded:
            logger.warning("Model not loaded, loading now")
            self.load_model()

        # Ensure model is in training mode to maintain gradients
        if self.model.training:
            for param in self.model.parameters():
                if param.requires_grad:
                    break
            else:
                logger.warning("No trainable parameters found, attempting to fix...")
                self.model.train()

            # Re-add LoRA if necessary
            if not self.is_lora:
                self.add_lora()

        # Get batch size
        batch_size = input_ids.size(0)

        # If batch size is too large, process in chunks
        if batch_size > max_micro_batch:
            logger.info(f"Processing batch of {batch_size} in chunks of {max_micro_batch}")

            # Prepare inputs
            inputs = {"input_ids": input_ids}
            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask

            # Slice into micro-batches
            micro_batches = slice_tensors(inputs, max_micro_batch)
            all_outputs = []

            # Process each micro-batch
            for i, micro_batch in enumerate(micro_batches):
                logger.debug(f"Processing micro-batch {i+1}/{len(micro_batches)}")

                # Move to the right device
                device_inputs = {k: v.to(self.model.device) for k, v in micro_batch.items()}

                # Forward pass - preserve requires_grad
                micro_outputs = self.model(
                    **device_inputs,
                    return_dict=return_dict
                )

                # Store the outputs without detaching
                all_outputs.append(micro_outputs)

            # Combine results (this depends on the model output format)
            if return_dict:
                # Combine into a single output dict
                combined_outputs = {}
                for key in all_outputs[0].keys():
                    if isinstance(all_outputs[0][key], torch.Tensor):
                        # Concatenate tensors
                        tensors_to_cat = [out[key] for out in all_outputs]
                        combined_outputs[key] = torch.cat(tensors_to_cat, dim=0)
                    else:
                        # Handle non-tensor outputs (e.g., lists)
                        combined_outputs[key] = [item for out in all_outputs for item in out[key]]

                # Return combined outputs with same type
                return type(all_outputs[0])(**combined_outputs)
            else:
                # Handle tuple outputs
                combined_first = torch.cat([out[0] for out in all_outputs], dim=0)
                return (combined_first,)

        else:
            # Single forward pass for small batches
            inputs = {
                "input_ids": input_ids.to(self.model.device)
            }

            if attention_mask is not None:
                inputs["attention_mask"] = attention_mask.to(self.model.device)

            # Regular forward pass
            outputs = self.model(
                **inputs,
                return_dict=return_dict
            )

            return outputs