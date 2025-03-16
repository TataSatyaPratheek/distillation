#!/bin/bash

# Set environment variables for optimal memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_VISIBLE_DEVICES=0

# Install the package in development mode
pip install -e .

# Run the optimized distillation for 4GB VRAM (GTX 1650Ti)
python -m tiny_distill.main \
  --teacher EleutherAI/pythia-160m \
  --student EleutherAI/pythia-70m \
  --dataset tatsu-lab/alpaca \
  --samples 300 \
  --sequence_length 32 \
  --teacher_batch 1 \
  --student_batch 2 \
  --gradient_accumulation 8 \
  --epochs 2 \
  --learning_rate 5e-5 \
  --output_dir "./distill_results" \
  --gradient_checkpointing \
  --layer_by_layer \
  --cpu_offload \
  --log_level DEBUG