#!/usr/bin/env python3
"""
Memory profiling tool for TinyDistill.

This script analyzes memory usage for different models and configurations
to help users optimize their distillation process on memory-constrained hardware.
"""
import os
import sys
import argparse
import time
import logging
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path to run script standalone
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tiny_distill.models.teacher import TeacherModel
from tiny_distill.models.student import StudentModel
from tiny_distill.memory_utils import log_memory_stats, clear_gpu_memory, MemoryTracker
from tiny_distill.logging_utils import setup_logging
from tiny_distill.config import DistillationConfig
from tiny_distill.data.dataset import load_and_prepare_dataset
from transformers import AutoTokenizer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Memory profiling tool for TinyDistill"
    )
    
    parser.add_argument(
        "--teacher",
        type=str,
        default="EleutherAI/pythia-70m",
        help="Teacher model ID from Hugging Face"
    )
    
    parser.add_argument(
        "--student",
        type=str,
        default="EleutherAI/pythia-70m",
        help="Student model ID from Hugging Face"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset to use for memory profiling"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./memory_profile",
        help="Output directory for memory profile results"
    )
    
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Sequence length to use for profiling"
    )
    
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of batch sizes to test"
    )
    
    parser.add_argument(
        "--profile_type",
        choices=["teacher", "student", "both"],
        default="both",
        help="Which model to profile: teacher, student, or both"
    )
    
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Test with CPU offloading"
    )
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Test with gradient checkpointing for student model"
    )
    
    return parser.parse_args()


def profile_teacher_model(
    model_id: str,
    batch_sizes: List[int],
    sequence_length: int,
    num_samples: int = 10,
    cpu_offload: bool = False,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Profile memory usage of teacher model for different batch sizes.
    
    Args:
        model_id (str): Model ID from Hugging Face
        batch_sizes (List[int]): List of batch sizes to test
        sequence_length (int): Sequence length for inputs
        num_samples (int, optional): Number of forward passes to average. Defaults to 10.
        cpu_offload (bool, optional): Whether to use CPU offloading. Defaults to False.
        device (str, optional): Device to use. Defaults to "cuda".
        
    Returns:
        Dict[str, List[float]]: Memory usage for each batch size
    """
    results = {
        "batch_sizes": batch_sizes,
        "peak_memory": [],
        "inference_time": []
    }
    
    logging.info(f"Profiling teacher model: {model_id}")
    
    # Create sample inputs for different batch sizes
    sample_inputs = {}
    for batch_size in batch_sizes:
        sample_inputs[batch_size] = {
            "input_ids": torch.randint(0, 1000, (batch_size, sequence_length), device=device),
            "attention_mask": torch.ones(batch_size, sequence_length, device=device)
        }
    
    # Initialize teacher model
    teacher = TeacherModel(
        model_id=model_id,
        max_memory_usage=0.9,
        offload_layers=cpu_offload,
        use_flash_attention=True,
        max_batch_size=max(batch_sizes),
        device=device
    )
    
    teacher.load_model()
    
    # Create memory tracker
    memory_tracker = MemoryTracker(log_interval=1)
    
    for batch_size in batch_sizes:
        logging.info(f"Testing batch size: {batch_size}")
        
        # Clear memory before test
        clear_gpu_memory()
        
        # Start tracking memory
        memory_tracker.start()
        
        # Run multiple forward passes to get average
        timings = []
        for _ in range(num_samples):
            start_time = time.time()
            with torch.no_grad():
                _ = teacher.forward(
                    input_ids=sample_inputs[batch_size]["input_ids"],
                    attention_mask=sample_inputs[batch_size]["attention_mask"],
                    max_micro_batch=batch_size
                )
            end_time = time.time()
            timings.append(end_time - start_time)
        
        # Stop tracking memory
        memory_tracker.stop()
        
        # Get peak memory usage
        memory_summary = memory_tracker.summary()
        peak_memory = memory_summary.get("gpu_max", 0)
        avg_time = np.mean(timings)
        
        results["peak_memory"].append(peak_memory)
        results["inference_time"].append(avg_time)
        
        logging.info(f"Batch size {batch_size}: Peak memory = {peak_memory:.2f} GB, Avg time = {avg_time:.4f} s")
    
    # Unload model
    teacher.unload()
    
    return results


def profile_student_model(
    model_id: str,
    batch_sizes: List[int],
    sequence_length: int,
    num_samples: int = 10,
    cpu_offload: bool = False,
    gradient_checkpointing: bool = False,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    Profile memory usage of student model for different batch sizes.
    
    Args:
        model_id (str): Model ID from Hugging Face
        batch_sizes (List[int]): List of batch sizes to test
        sequence_length (int): Sequence length for inputs
        num_samples (int, optional): Number of forward passes to average. Defaults to 10.
        cpu_offload (bool, optional): Whether to use CPU offloading. Defaults to False.
        gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        device (str, optional): Device to use. Defaults to "cuda".
        
    Returns:
        Dict[str, List[float]]: Memory usage for each batch size
    """
    results = {
        "batch_sizes": batch_sizes,
        "peak_memory_inference": [],
        "peak_memory_training": [],
        "inference_time": [],
        "training_time": []
    }
    
    logging.info(f"Profiling student model: {model_id}")
    
    # Create sample inputs for different batch sizes
    sample_inputs = {}
    for batch_size in batch_sizes:
        sample_inputs[batch_size] = {
            "input_ids": torch.randint(0, 1000, (batch_size, sequence_length), device=device),
            "attention_mask": torch.ones(batch_size, sequence_length, device=device)
        }
    
    # Initialize student model
    student = StudentModel(
        model_id=model_id,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        gradient_checkpointing=gradient_checkpointing,
        use_4bit=True,
        use_nested_quant=True,
        use_cpu_offload=cpu_offload,
        device=device
    )
    
    student.load_model()
    
    # Create memory tracker
    memory_tracker = MemoryTracker(log_interval=1)
    
    # Profile inference (eval mode)
    student.setup_for_inference()
    
    for batch_size in batch_sizes:
        logging.info(f"Testing inference with batch size: {batch_size}")
        
        # Clear memory before test
        clear_gpu_memory()
        
        # Start tracking memory
        memory_tracker.start()
        
        # Run multiple forward passes to get average
        timings = []
        for _ in range(num_samples):
            start_time = time.time()
            with torch.no_grad():
                _ = student.forward(
                    input_ids=sample_inputs[batch_size]["input_ids"],
                    attention_mask=sample_inputs[batch_size]["attention_mask"],
                    max_micro_batch=batch_size
                )
            end_time = time.time()
            timings.append(end_time - start_time)
        
        # Stop tracking memory
        memory_tracker.stop()
        
        # Get peak memory usage
        memory_summary = memory_tracker.summary()
        peak_memory = memory_summary.get("gpu_max", 0)
        avg_time = np.mean(timings)
        
        results["peak_memory_inference"].append(peak_memory)
        results["inference_time"].append(avg_time)
        
        logging.info(f"Inference - Batch size {batch_size}: Peak memory = {peak_memory:.2f} GB, Avg time = {avg_time:.4f} s")
    
    # Profile training (train mode)
    student.add_lora()
    student.setup_for_training()
    
    for batch_size in batch_sizes:
        logging.info(f"Testing training with batch size: {batch_size}")
        
        # Clear memory before test
        clear_gpu_memory()
        
        # Create fake teacher logits
        fake_teacher_logits = torch.randn(
            batch_size, sequence_length, 50257, device=device, dtype=torch.float16
        )
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            student.get_trainable_parameters(),
            lr=5e-5,
            weight_decay=0.01
        )
        
        # Loss function
        loss_fn = torch.nn.MSELoss()
        
        # Start tracking memory
        memory_tracker.start()
        
        # Run multiple forward and backward passes to get average
        timings = []
        for _ in range(num_samples):
            optimizer.zero_grad()
            
            start_time = time.time()
            
            # Forward pass
            outputs = student.forward(
                input_ids=sample_inputs[batch_size]["input_ids"],
                attention_mask=sample_inputs[batch_size]["attention_mask"],
                max_micro_batch=batch_size
            )
            
            # Loss computation
            loss = loss_fn(outputs.logits, fake_teacher_logits)
            
            # Backward pass
            loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            end_time = time.time()
            timings.append(end_time - start_time)
        
        # Stop tracking memory
        memory_tracker.stop()
        
        # Get peak memory usage
        memory_summary = memory_tracker.summary()
        peak_memory = memory_summary.get("gpu_max", 0)
        avg_time = np.mean(timings)
        
        results["peak_memory_training"].append(peak_memory)
        results["training_time"].append(avg_time)
        
        logging.info(f"Training - Batch size {batch_size}: Peak memory = {peak_memory:.2f} GB, Avg time = {avg_time:.4f} s")
    
    # Unload model
    student.unload()
    
    return results


def plot_memory_usage(
    results: Dict[str, Dict[str, List[float]]],
    output_dir: str
) -> None:
    """
    Create plots of memory usage.
    
    Args:
        results (Dict[str, Dict[str, List[float]]]): Memory usage results
        output_dir (str): Output directory for plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot teacher memory usage if available
    if "teacher" in results:
        teacher_results = results["teacher"]
        batch_sizes = teacher_results["batch_sizes"]
        peak_memory = teacher_results["peak_memory"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, peak_memory, marker='o', linewidth=2)
        plt.xlabel('Batch Size')
        plt.ylabel('Peak GPU Memory (GB)')
        plt.title('Teacher Model Memory Usage')
        plt.grid(True)
        plt.savefig(output_path / "teacher_memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot student memory usage if available
    if "student" in results:
        student_results = results["student"]
        batch_sizes = student_results["batch_sizes"]
        peak_memory_inference = student_results["peak_memory_inference"]
        peak_memory_training = student_results["peak_memory_training"]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, peak_memory_inference, marker='o', linewidth=2, label='Inference')
        plt.plot(batch_sizes, peak_memory_training, marker='s', linewidth=2, label='Training')
        plt.xlabel('Batch Size')
        plt.ylabel('Peak GPU Memory (GB)')
        plt.title('Student Model Memory Usage')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path / "student_memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot comparison if both are available
    if "teacher" in results and "student" in results:
        teacher_results = results["teacher"]
        student_results = results["student"]
        
        # Ensure batch sizes are the same
        if teacher_results["batch_sizes"] == student_results["batch_sizes"]:
            batch_sizes = teacher_results["batch_sizes"]
            teacher_memory = teacher_results["peak_memory"]
            student_inference_memory = student_results["peak_memory_inference"]
            student_training_memory = student_results["peak_memory_training"]
            
            plt.figure(figsize=(12, 7))
            plt.plot(batch_sizes, teacher_memory, marker='o', linewidth=2, label='Teacher (Inference)')
            plt.plot(batch_sizes, student_inference_memory, marker='s', linewidth=2, label='Student (Inference)')
            plt.plot(batch_sizes, student_training_memory, marker='^', linewidth=2, label='Student (Training)')
            plt.xlabel('Batch Size')
            plt.ylabel('Peak GPU Memory (GB)')
            plt.title('Memory Usage Comparison')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_path / "memory_usage_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    # Create markdown report
    with open(output_path / "memory_profile_report.md", "w") as f:
        f.write("# Memory Profile Report\n\n")
        
        if "teacher" in results:
            f.write("## Teacher Model\n\n")
            f.write("| Batch Size | Peak Memory (GB) | Inference Time (s) |\n")
            f.write("|------------|-----------------|--------------------|\n")
            
            teacher_results = results["teacher"]
            for i, batch_size in enumerate(teacher_results["batch_sizes"]):
                peak_mem = teacher_results["peak_memory"][i]
                inf_time = teacher_results["inference_time"][i]
                f.write(f"| {batch_size} | {peak_mem:.2f} | {inf_time:.4f} |\n")
            
            f.write("\n![Teacher Memory Usage](./teacher_memory_usage.png)\n\n")
        
        if "student" in results:
            f.write("## Student Model\n\n")
            f.write("### Inference\n\n")
            f.write("| Batch Size | Peak Memory (GB) | Inference Time (s) |\n")
            f.write("|------------|-----------------|--------------------|\n")
            
            student_results = results["student"]
            for i, batch_size in enumerate(student_results["batch_sizes"]):
                peak_mem = student_results["peak_memory_inference"][i]
                inf_time = student_results["inference_time"][i]
                f.write(f"| {batch_size} | {peak_mem:.2f} | {inf_time:.4f} |\n")
            
            f.write("\n### Training\n\n")
            f.write("| Batch Size | Peak Memory (GB) | Training Time (s) |\n")
            f.write("|------------|-----------------|-------------------|\n")
            
            for i, batch_size in enumerate(student_results["batch_sizes"]):
                peak_mem = student_results["peak_memory_training"][i]
                train_time = student_results["training_time"][i]
                f.write(f"| {batch_size} | {peak_mem:.2f} | {train_time:.4f} |\n")
            
            f.write("\n![Student Memory Usage](./student_memory_usage.png)\n\n")
        
        if "teacher" in results and "student" in results:
            f.write("## Comparison\n\n")
            f.write("![Memory Usage Comparison](./memory_usage_comparison.png)\n\n")
        
        f.write("## Recommendations\n\n")
        
        # Generate batch size recommendations based on available GPU memory
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            f.write(f"Based on your GPU with {gpu_mem:.1f} GB of memory:\n\n")
            
            if "teacher" in results:
                teacher_results = results["teacher"]
                recommended_teacher_batch = 1
                
                for i, batch_size in enumerate(teacher_results["batch_sizes"]):
                    if teacher_results["peak_memory"][i] < gpu_mem * 0.9:
                        recommended_teacher_batch = batch_size
                
                f.write(f"- Recommended teacher batch size: **{recommended_teacher_batch}**\n")
            
            if "student" in results:
                student_results = results["student"]
                recommended_student_inf_batch = 1
                recommended_student_train_batch = 1
                
                for i, batch_size in enumerate(student_results["batch_sizes"]):
                    if student_results["peak_memory_inference"][i] < gpu_mem * 0.9:
                        recommended_student_inf_batch = batch_size
                    
                    if student_results["peak_memory_training"][i] < gpu_mem * 0.9:
                        recommended_student_train_batch = batch_size
                
                f.write(f"- Recommended student inference batch size: **{recommended_student_inf_batch}**\n")
                f.write(f"- Recommended student training batch size: **{recommended_student_train_batch}**\n")
        
        f.write("\n## Example Command\n\n")
        f.write("```bash\n")
        if "teacher" in results and "student" in results:
            teacher_results = results["teacher"]
            student_results = results["student"]
            
            teacher_batch = 1
            student_batch = 1
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                
                for i, batch_size in enumerate(teacher_results["batch_sizes"]):
                    if teacher_results["peak_memory"][i] < gpu_mem * 0.9:
                        teacher_batch = batch_size
                
                for i, batch_size in enumerate(student_results["batch_sizes"]):
                    if student_results["peak_memory_training"][i] < gpu_mem * 0.9:
                        student_batch = batch_size
            
            f.write(f"tiny-distill \\\n")
            f.write(f"  --teacher {results.get('teacher_model_id')} \\\n")
            f.write(f"  --student {results.get('student_model_id')} \\\n")
            f.write(f"  --teacher_batch {teacher_batch} \\\n")
            f.write(f"  --student_batch {student_batch} \\\n")
            if results.get("gradient_checkpointing"):
                f.write(f"  --gradient_checkpointing \\\n")
            if results.get("cpu_offload"):
                f.write(f"  --cpu_offload \\\n")
            f.write(f"  --sequence_length {results.get('sequence_length', 128)}\n")
        else:
            f.write("tiny-distill --teacher MODEL_NAME --student MODEL_NAME\n")
        f.write("```\n")


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO", log_file=os.path.join(args.output_dir, "memory_profile.log"))
    
    # Parse batch sizes
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU for profiling (this will be slow)")
        device = "cpu"
    else:
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logging.info(f"Found GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    
    # Results dictionary
    results = {
        "teacher_model_id": args.teacher,
        "student_model_id": args.student,
        "sequence_length": args.sequence_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        "cpu_offload": args.cpu_offload
    }
    
    # Run profiling
    if args.profile_type in ["teacher", "both"]:
        logging.info("=== Profiling Teacher Model ===")
        teacher_results = profile_teacher_model(
            model_id=args.teacher,
            batch_sizes=batch_sizes,
            sequence_length=args.sequence_length,
            cpu_offload=args.cpu_offload,
            device=device
        )
        results["teacher"] = teacher_results
    
    if args.profile_type in ["student", "both"]:
        logging.info("=== Profiling Student Model ===")
        student_results = profile_student_model(
            model_id=args.student,
            batch_sizes=batch_sizes,
            sequence_length=args.sequence_length,
            cpu_offload=args.cpu_offload,
            gradient_checkpointing=args.gradient_checkpointing,
            device=device
        )
        results["student"] = student_results
    
    # Create plots and report
    plot_memory_usage(results, args.output_dir)
    
    logging.info(f"Memory profiling complete. Results saved to {args.output_dir}")
    logging.info(f"Check {os.path.join(args.output_dir, 'memory_profile_report.md')} for a detailed report")


if __name__ == "__main__":
    main()