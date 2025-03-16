"""
Dataset loading and preparation for ultra-low memory distillation.
"""
import os
import logging
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from datasets import load_dataset, Dataset, DatasetDict
from transformers import PreTrainedTokenizer, AutoTokenizer
from torch.utils.data import Subset

from ..memory_utils import log_memory_stats, clear_gpu_memory

logger = logging.getLogger(__name__)


def load_and_prepare_dataset(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    num_samples: int = 1000,
    num_validation_samples: int = 100,
    max_length: int = 128,
    dataset_split: str = "train",
    text_column: Optional[str] = None,
    cache_dir: Optional[str] = None,
    seed: int = 42
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare dataset for distillation with memory efficiency.
    
    Args:
        dataset_name (str): Dataset name from Hugging Face or local path
        tokenizer (PreTrainedTokenizer): Tokenizer to use
        num_samples (int, optional): Number of samples to use. Defaults to 1000.
        num_validation_samples (int, optional): Number of validation samples. Defaults to 100.
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        dataset_split (str, optional): Dataset split to use. Defaults to "train".
        text_column (Optional[str], optional): Column name for text. Defaults to None (auto-detect).
        cache_dir (Optional[str], optional): Cache directory. Defaults to None.
        seed (int, optional): Random seed. Defaults to 42.
        
    Returns:
        Tuple[Dataset, Optional[Dataset]]: Training and validation datasets
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        # Special case handling for popular datasets
        if dataset_name == "wikitext":
            dataset = load_dataset(dataset_name, "wikitext-103-v1", split=dataset_split, cache_dir=cache_dir)
        elif dataset_name == "c4":
            # Only load a small subset of C4 by default
            dataset = load_dataset(dataset_name, "en", split=f"{dataset_split}[:5000]", cache_dir=cache_dir)
        else:
            # General case
            try:
                dataset = load_dataset(dataset_name, split=dataset_split, cache_dir=cache_dir)
            except Exception as e:
                logger.warning(f"Error loading dataset split {dataset_split}: {e}")
                logger.info("Attempting to load full dataset and extract split")
                
                # Try to load the full dataset and extract the split
                full_dataset = load_dataset(dataset_name, cache_dir=cache_dir)
                if isinstance(full_dataset, DatasetDict) and dataset_split in full_dataset:
                    dataset = full_dataset[dataset_split]
                else:
                    logger.warning(f"Split {dataset_split} not found in dataset")
                    # Use the first available split
                    if isinstance(full_dataset, DatasetDict):
                        first_split = next(iter(full_dataset.keys()))
                        logger.info(f"Using {first_split} split instead")
                        dataset = full_dataset[first_split]
                    else:
                        dataset = full_dataset
        
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Limit number of samples if needed
        if num_samples > 0 and num_samples < len(dataset):
            logger.info(f"Using {num_samples} samples from dataset")
            dataset = dataset.shuffle(seed=seed).select(range(num_samples))
        
        # Identify text column if not specified
        if text_column is None:
            # Try to find a text column
            text_columns = [col for col in dataset.column_names 
                           if col.lower() in ["text", "content", "document", "sentence", "query", "question"]]
            
            if text_columns:
                text_column = text_columns[0]
                logger.info(f"Using {text_column} as text column")
            else:
                # Special case handling for some datasets
                if "alpaca" in dataset_name.lower():
                    # For Alpaca-like datasets, concatenate instruction, input, and output
                    logger.info("Using combined instruction + input + output for Alpaca-like dataset")
                    
                    def combine_alpaca_fields(example):
                        instruction = example.get("instruction", "")
                        input_text = example.get("input", "")
                        output = example.get("output", "")
                        
                        if input_text:
                            combined = f"Instruction: {instruction}\nInput: {input_text}\nOutput: {output}"
                        else:
                            combined = f"Instruction: {instruction}\nOutput: {output}"
                        
                        return {"text": combined}
                    
                    dataset = dataset.map(combine_alpaca_fields)
                    text_column = "text"
                else:
                    # Use the first column as fallback
                    text_column = dataset.column_names[0]
                    logger.warning(f"No text column found, using {text_column} as fallback")
        
        # Create validation split if needed
        validation_dataset = None
        if num_validation_samples > 0:
            # Shuffle remaining dataset
            shuffled_dataset = dataset.shuffle(seed=seed)
            
            # Extract validation samples
            valid_size = min(num_validation_samples, len(shuffled_dataset) // 10)
            
            if valid_size > 0:
                logger.info(f"Creating validation set with {valid_size} samples")
                validation_indices = list(range(valid_size))
                train_indices = list(range(valid_size, len(shuffled_dataset)))
                
                validation_dataset = shuffled_dataset.select(validation_indices)
                dataset = shuffled_dataset.select(train_indices)
        
        # Tokenize the datasets
        logger.info(f"Tokenizing dataset with max_length={max_length}")
        
        def tokenize_function(examples):
            # Handle batched examples
            texts = examples[text_column]
            
            # Make sure tokenizer has padding token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer(
                texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np"
            )
        
        # Tokenize train set
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != text_column]
        )
        
        # Tokenize validation set if it exists
        tokenized_validation = None
        if validation_dataset is not None:
            tokenized_validation = validation_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=[col for col in validation_dataset.column_names if col != text_column]
            )
        
        logger.info(f"Dataset preparation complete: {len(tokenized_dataset)} train, "
                   f"{len(tokenized_validation) if tokenized_validation else 0} validation examples")
        
        return tokenized_dataset, tokenized_validation
        
    except Exception as e:
        logger.error(f"Error preparing dataset: {e}")
        
        # Fall back to tiny example dataset
        logger.info("Falling back to a tiny example dataset")
        tokenized_dataset = create_fallback_dataset(tokenizer, max_length, num_samples)
        
        return tokenized_dataset, None


def create_fallback_dataset(
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
    num_samples: int = 100
) -> Dataset:
    """
    Create a fallback dataset when loading fails.
    
    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer to use
        max_length (int, optional): Maximum sequence length. Defaults to 128.
        num_samples (int, optional): Number of samples. Defaults to 100.
        
    Returns:
        Dataset: Fallback dataset
    """
    # Create some basic texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subfield of artificial intelligence.",
        "Knowledge distillation is a technique for model compression.",
        "Python is a versatile programming language.",
        "Deep learning models can be compressed using techniques like pruning, quantization, and distillation.",
    ]
    
    # Repeat to reach desired number of samples
    repeated_texts = []
    for _ in range((num_samples // len(texts)) + 1):
        repeated_texts.extend(texts)
    repeated_texts = repeated_texts[:num_samples]
    
    # Tokenize
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    encodings = tokenizer(
        repeated_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np"
    )
    
    # Create dataset
    dataset_dict = {key: encodings[key] for key in encodings}
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"Created fallback dataset with {len(dataset)} examples")
    
    return dataset


class DistillationDataset(torch.utils.data.Dataset):
    """Dataset wrapper for converting to tensors at runtime to save memory."""
    
    def __init__(
        self,
        dataset: Dataset,
        device: Optional[str] = None,
        preload: bool = False
    ):
        """
        Initialize dataset wrapper.
        
        Args:
            dataset (Dataset): Underlying dataset
            device (Optional[str], optional): Device to use. Defaults to None.
            preload (bool, optional): Whether to preload tensors. Defaults to False.
        """
        self.dataset = dataset
        self.device = device
        self.preloaded = None
        
        # Preload dataset to tensors if requested
        if preload:
            self.preload()
    
    def preload(self) -> None:
        """Preload dataset to tensors."""
        logger.info("Preloading dataset to tensors")
        self.preloaded = []
        
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            tensor_item = {
                k: torch.tensor(v) for k, v in item.items()
            }
            
            if self.device is not None:
                tensor_item = {
                    k: v.to(self.device) for k, v in tensor_item.items()
                }
            
            self.preloaded.append(tensor_item)
        
        logger.info("Dataset preloading complete")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx (int): Item index
            
        Returns:
            Dict[str, torch.Tensor]: Dataset item
        """
        if self.preloaded is not None:
            return self.preloaded[idx]
        
        item = self.dataset[idx]
        
        # Convert to tensors
        tensor_item = {
            k: torch.tensor(v) for k, v in item.items()
        }
        
        # Move to device if specified
        if self.device is not None:
            tensor_item = {
                k: v.to(self.device) for k, v in tensor_item.items()
            }
        
        return tensor_item


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    device: Optional[str] = None,
    preload: bool = False,
    pin_memory: bool = False,
    num_workers: int = 0
) -> torch.utils.data.DataLoader:
    """
    Create a memory-efficient dataloader.
    
    Args:
        dataset (Dataset): Dataset to use
        batch_size (int, optional): Batch size. Defaults to 4.
        shuffle (bool, optional): Whether to shuffle. Defaults to True.
        device (Optional[str], optional): Device to use. Defaults to None.
        preload (bool, optional): Whether to preload tensors. Defaults to False.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        num_workers (int, optional): Number of workers. Defaults to 0.
        
    Returns:
        torch.utils.data.DataLoader: Dataloader
    """
    # Wrap dataset if it's not already a DistillationDataset
    if not isinstance(dataset, DistillationDataset):
        dataset = DistillationDataset(dataset, device=device, preload=preload)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory and device != "cpu",
        num_workers=num_workers,
        drop_last=False
    )
    
    return dataloader