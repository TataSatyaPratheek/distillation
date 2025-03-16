"""
Efficient caching system for teacher model outputs.
"""
import os
import time
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, BinaryIO
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
import shutil

from ..memory_utils import log_memory_stats, clear_gpu_memory

logger = logging.getLogger(__name__)


class TeacherOutputCache:
    """
    Efficient storage and retrieval of teacher model outputs.
    
    Features:
    - Streaming writes: Cache outputs without holding all in memory
    - Chunked access: Load only what's needed
    - Compression: Store tensors efficiently
    - Background processing: Write in background threads
    - Resumable: Can continue from interruptions
    """
    
    def __init__(
        self, 
        cache_dir: str,
        vocab_size: int,
        dtype: torch.dtype = torch.float16,
        compress: bool = True,
        chunk_size: int = 64,
        background_writes: bool = True
    ):
        """
        Initialize cache system.
        
        Args:
            cache_dir (str): Directory to store cache files
            vocab_size (int): Vocabulary size of the model
            dtype (torch.dtype, optional): Data type for cached logits. Defaults to torch.float16.
            compress (bool, optional): Whether to compress cached data. Defaults to True.
            chunk_size (int, optional): Chunk size for HDF5 storage. Defaults to 64.
            background_writes (bool, optional): Whether to write in background threads. Defaults to True.
        """
        self.cache_dir = Path(cache_dir)
        self.vocab_size = vocab_size
        self.dtype = dtype
        self.compress = compress
        self.chunk_size = chunk_size
        self.background_writes = background_writes
        
        # Cache file paths
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.main_cache_file = self.cache_dir / "teacher_outputs.h5"
        self.metadata_file = self.cache_dir / "metadata.pt"
        self.temp_dir = self.cache_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        # State
        self.write_queue = queue.Queue() if background_writes else None
        self.write_thread = None
        self.is_writing = False
        self.cached_indices = set()
        self.metadata = {"total_examples": 0, "completed": False}
        
        # Load existing cache if available
        self._load_existing_cache()
    
    def _load_existing_cache(self) -> None:
        """Load metadata from existing cache if available."""
        if self.metadata_file.exists():
            try:
                self.metadata = torch.load(self.metadata_file)
                logger.info(f"Loaded existing cache metadata: {self.metadata}")
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        if self.main_cache_file.exists():
            try:
                with h5py.File(self.main_cache_file, 'r') as h5f:
                    if 'indices' in h5f:
                        self.cached_indices = set(h5f['indices'][:])
                    logger.info(f"Found existing cache with {len(self.cached_indices)} examples")
            except Exception as e:
                logger.warning(f"Failed to read cache indices: {e}")
    
    def _start_background_writer(self) -> None:
        """Start background thread for writing cache files."""
        if not self.background_writes:
            return
        
        if self.write_thread is not None and self.write_thread.is_alive():
            return
        
        self.is_writing = True
        self.write_thread = threading.Thread(
            target=self._background_writer_loop,
            daemon=True
        )
        self.write_thread.start()
        logger.debug("Background cache writer thread started")
    
    def _background_writer_loop(self) -> None:
        """Background thread for processing write queue."""
        while self.is_writing:
            try:
                # Get an item with timeout to allow checking is_writing flag
                try:
                    item = self.write_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Process the item
                index, tensor, temp_file = item
                self._write_to_main_cache(index, tensor, temp_file)
                self.write_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in background writer: {e}")
                time.sleep(0.1)
    
    def _write_to_main_cache(
        self, 
        index: int, 
        tensor: Optional[torch.Tensor] = None,
        temp_file: Optional[str] = None
    ) -> None:
        """
        Write a tensor to the main cache file.
        
        Args:
            index (int): Example index
            tensor (Optional[torch.Tensor], optional): Tensor to write or None if using temp_file
            temp_file (Optional[str], optional): Path to temporary file containing the tensor
        """
        try:
            # Make sure we don't have both tensor and temp_file as None
            if tensor is None and temp_file is None:
                raise ValueError("Either tensor or temp_file must be provided")
            
            # Create or open the main HDF5 file
            with h5py.File(self.main_cache_file, 'a') as h5f:
                # If this is the first write, create datasets
                if 'logits' not in h5f:
                    if tensor is not None:
                        shape = tensor.shape
                    else:
                        # Load shape from temp file
                        with np.load(temp_file) as np_data:
                            shape = np_data['logits'].shape
                    
                    # Create extendable datasets
                    h5f.create_dataset(
                        'logits',
                        shape=(0, shape[1], shape[2]),
                        maxshape=(None, shape[1], shape[2]),
                        dtype=np.float16 if self.dtype == torch.float16 else np.float32,
                        chunks=(1, shape[1], self.chunk_size),
                        compression='gzip' if self.compress else None,
                        compression_opts=4 if self.compress else None
                    )
                    
                    h5f.create_dataset(
                        'indices',
                        shape=(0,),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=(self.chunk_size,)
                    )
                
                # Extend datasets
                current_size = h5f['logits'].shape[0]
                h5f['logits'].resize(current_size + 1, axis=0)
                h5f['indices'].resize(current_size + 1, axis=0)
                
                # Write data
                if tensor is not None:
                    # Convert tensor to numpy if needed
                    if isinstance(tensor, torch.Tensor):
                        tensor_np = tensor.cpu().numpy()
                    else:
                        tensor_np = tensor
                    
                    h5f['logits'][current_size] = tensor_np
                else:
                    # Load from temp file
                    with np.load(temp_file) as np_data:
                        h5f['logits'][current_size] = np_data['logits']
                
                h5f['indices'][current_size] = index
                
                # Update cached indices
                self.cached_indices.add(index)
                
                # Save metadata
                self.metadata["total_examples"] = max(self.metadata["total_examples"], len(self.cached_indices))
                torch.save(self.metadata, self.metadata_file)
                
                # Delete temp file if it exists
                if temp_file is not None and os.path.exists(temp_file):
                    os.remove(temp_file)
        
        except Exception as e:
            logger.error(f"Error writing to main cache: {e}")
            # Don't delete temp file on error, so we can retry
    
    def add(self, index: int, logits: torch.Tensor) -> None:
        """
        Add a tensor to the cache.
        
        Args:
            index (int): Example index
            logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len, vocab_size]
        """
        if index in self.cached_indices:
            logger.debug(f"Example {index} already cached, skipping")
            return
        
        # Convert to the right dtype and move to CPU
        logits = logits.to(dtype=self.dtype).cpu()
        
        if self.background_writes:
            # For background writes, save to temp file first
            temp_file = self.temp_dir / f"temp_{index}.npz"
            np.savez_compressed(temp_file, logits=logits.numpy())
            
            # Add to queue and start background thread if needed
            self.write_queue.put((index, None, temp_file))
            self._start_background_writer()
        else:
            # Direct write
            self._write_to_main_cache(index, logits)
    
    def get(self, index: int) -> torch.Tensor:
        """
        Retrieve a tensor from the cache.
        
        Args:
            index (int): Example index
            
        Returns:
            torch.Tensor: Cached logits tensor
        """
        if not self.main_cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {self.main_cache_file}")
        
        try:
            with h5py.File(self.main_cache_file, 'r') as h5f:
                # Find the position of the index
                indices = h5f['indices'][:]
                pos = np.where(indices == index)[0]
                
                if len(pos) == 0:
                    raise KeyError(f"Index {index} not found in cache")
                
                # Get the tensor
                logits = torch.tensor(h5f['logits'][pos[0]], dtype=self.dtype)
                return logits
                
        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            raise
    
    def contains(self, index: int) -> bool:
        """
        Check if an index is in the cache.
        
        Args:
            index (int): Example index
            
        Returns:
            bool: Whether the index is cached
        """
        return index in self.cached_indices
    
    def all_indices(self) -> List[int]:
        """
        Get all cached indices.
        
        Returns:
            List[int]: List of all cached indices
        """
        return list(sorted(self.cached_indices))
    
    def is_complete(self) -> bool:
        """
        Check if the cache is complete according to metadata.
        
        Returns:
            bool: Whether the cache is complete
        """
        return self.metadata.get("completed", False)
    
    def mark_complete(self, total_examples: Optional[int] = None) -> None:
        """
        Mark the cache as complete.
        
        Args:
            total_examples (Optional[int], optional): Total number of examples. Defaults to None.
        """
        if total_examples is not None:
            self.metadata["total_examples"] = total_examples
        
        self.metadata["completed"] = True
        torch.save(self.metadata, self.metadata_file)
        logger.info(f"Cache marked as complete with {self.metadata['total_examples']} examples")
    
    def wait_for_writes(self) -> None:
        """Wait for all background writes to complete."""
        if not self.background_writes or self.write_queue is None:
            return
        
        logger.info("Waiting for background cache writes to complete...")
        self.write_queue.join()
        self.is_writing = False
        
        if self.write_thread is not None:
            self.write_thread.join(timeout=2.0)
        
        logger.info("All cache writes completed")
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        self.wait_for_writes()
        
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        try:
            self.wait_for_writes()
        except:
            pass


class TeacherCacheDataset(Dataset):
    """Dataset that loads teacher outputs from cache."""
    
    def __init__(
        self, 
        cache: TeacherOutputCache,
        tokenized_dataset: Dataset,
        preload: bool = False
    ):
        """
        Initialize cache dataset.
        
        Args:
            cache (TeacherOutputCache): Teacher output cache
            tokenized_dataset (Dataset): Original tokenized dataset
            preload (bool, optional): Whether to preload all data. Defaults to False.
        """
        self.cache = cache
        self.tokenized_dataset = tokenized_dataset
        self.cached_indices = cache.all_indices()
        
        # Map original dataset indices to cache indices
        self.index_map = {i: i for i in range(len(tokenized_dataset)) if i in self.cached_indices}
        
        # Verify cache coverage
        coverage = len(self.index_map) / len(tokenized_dataset) * 100
        logger.info(f"Teacher cache coverage: {coverage:.2f}% ({len(self.index_map)}/{len(tokenized_dataset)})")
        
        if coverage < 100 and not cache.is_complete():
            logger.warning(f"Incomplete cache: missing {len(tokenized_dataset) - len(self.index_map)} examples")
        
        # Preload cache if requested
        self.preloaded_data = None
        if preload:
            logger.info("Preloading teacher cache...")
            self.preloaded_data = {}
            for idx in tqdm(self.cached_indices, desc="Preloading cache"):
                self.preloaded_data[idx] = self.cache.get(idx)
            logger.info("Cache preloading complete")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an item from the dataset.
        
        Args:
            idx (int): Index in the dataset
            
        Returns:
            Dict[str, torch.Tensor]: Dataset item with teacher logits
        """
        # Get the original dataset item
        example = self.tokenized_dataset[idx]
        
        # Get teacher logits
        if self.preloaded_data is not None:
            teacher_logits = self.preloaded_data[idx]
        else:
            teacher_logits = self.cache.get(idx)
        
        # Combine into a single item
        item = {
            "input_ids": torch.tensor(example["input_ids"]),
            "attention_mask": torch.tensor(example["attention_mask"]),
            "teacher_logits": teacher_logits
        }
        
        return item


def cache_teacher_outputs(
    teacher_forward_fn: Callable,
    dataset: Dataset,
    cache: TeacherOutputCache,
    batch_size: int = 1,
    max_retries: int = 3,
    device: str = "cuda"
) -> bool:
    """
    Process all examples through the teacher model and cache the outputs.
    
    Args:
        teacher_forward_fn (Callable): Function to get teacher outputs
        dataset (Dataset): Dataset to process
        cache (TeacherOutputCache): Cache to store outputs
        batch_size (int, optional): Batch size. Defaults to 1.
        max_retries (int, optional): Maximum number of retries for failed examples. Defaults to 3.
        device (str, optional): Device to use. Defaults to "cuda".
        
    Returns:
        bool: Whether caching completed successfully
    """
    logger.info(f"Caching teacher outputs for {len(dataset)} examples with batch size {batch_size}")
    
    # Get uncached indices
    uncached_indices = [i for i in range(len(dataset)) if not cache.contains(i)]
    
    if not uncached_indices:
        logger.info("All examples already cached")
        cache.mark_complete(len(dataset))
        return True
    
    logger.info(f"Processing {len(uncached_indices)} uncached examples")
    
    # Create dataloader with only uncached examples
    class UncachedSubset(Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            item = self.dataset[self.indices[idx]]
            return {
                "input_ids": torch.tensor(item["input_ids"]),
                "attention_mask": torch.tensor(item["attention_mask"]) if "attention_mask" in item else None,
                "index": self.indices[idx]
            }
    
    uncached_dataset = UncachedSubset(dataset, uncached_indices)
    dataloader = DataLoader(
        uncached_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0  # Avoid memory duplication with multiprocessing
    )
    
    # Process examples
    failed_indices = []
    progress_bar = tqdm(dataloader, desc="Processing teacher model")
    
    for batch in progress_bar:
        indices = batch.pop("index").tolist()
        
        try:
            # Process batch through teacher model
            if isinstance(batch["input_ids"], list):
                # List of tensors
                batch["input_ids"] = torch.stack(batch["input_ids"])
            
            if "attention_mask" in batch and batch["attention_mask"] is not None:
                if isinstance(batch["attention_mask"], list):
                    batch["attention_mask"] = torch.stack(batch["attention_mask"])
            else:
                batch.pop("attention_mask", None)
            
            # Forward pass
            batch = {k: v.to(device) if v is not None else None for k, v in batch.items()}
            outputs = teacher_forward_fn(**batch)
            
            # Cache results
            if len(indices) == 1:
                # Single example
                cache.add(indices[0], outputs)
            else:
                # Batch of examples
                for i, idx in enumerate(indices):
                    cache.add(idx, outputs[i:i+1])
            
            # Update progress
            progress_bar.set_postfix({
                "cached": f"{len(cache.cached_indices)}/{len(dataset)}",
                "failed": len(failed_indices)
            })
            
        except Exception as e:
            logger.warning(f"Error processing batch {indices}: {e}")
            failed_indices.extend(indices)
            
            # Try to clear memory
            clear_gpu_memory()
            time.sleep(0.1)
    
    # Wait for background writes to complete
    cache.wait_for_writes()
    
    # Retry failed examples one by one
    if failed_indices:
        logger.info(f"Retrying {len(failed_indices)} failed examples one by one")
        retry_count = 0
        
        while failed_indices and retry_count < max_retries:
            retry_count += 1
            logger.info(f"Retry attempt {retry_count}/{max_retries}")
            
            still_failed = []
            for idx in tqdm(failed_indices, desc=f"Retry {retry_count}"):
                try:
                    # Get example
                    item = dataset[idx]
                    
                    # Process through teacher model
                    inputs = {
                        "input_ids": torch.tensor(item["input_ids"]).unsqueeze(0).to(device)
                    }
                    
                    if "attention_mask" in item:
                        inputs["attention_mask"] = torch.tensor(item["attention_mask"]).unsqueeze(0).to(device)
                    
                    # Forward pass
                    outputs = teacher_forward_fn(**inputs)
                    
                    # Cache result
                    cache.add(idx, outputs)
                    
                except Exception as e:
                    logger.warning(f"Failed to process example {idx} on retry {retry_count}: {e}")
                    still_failed.append(idx)
                    
                    # Try to clear memory
                    clear_gpu_memory()
                    time.sleep(0.1)
            
            # Update failed indices
            failed_indices = still_failed
    
    # Final status
    completion_percentage = len(cache.cached_indices) / len(dataset) * 100
    logger.info(f"Caching complete: {completion_percentage:.2f}% ({len(cache.cached_indices)}/{len(dataset)})")
    
    if not failed_indices:
        logger.info("All examples cached successfully")
        cache.mark_complete(len(dataset))
        return True
    else:
        logger.warning(f"Failed to cache {len(failed_indices)} examples after {max_retries} retries")
        # Save what we have
        cache.mark_complete(len(cache.cached_indices))
        return False