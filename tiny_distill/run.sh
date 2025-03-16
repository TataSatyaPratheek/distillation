#!/bin/bash

# Set environment variables for optimal memory usage on limited hardware
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,garbage_collection_threshold:0.6,expandable_segments:True"
export CUDA_VISIBLE_DEVICES=0

# Disable JIT cache to avoid memory leaks
export PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1

# Ensure proper CUDA memory handling
export TRANSFORMERS_OFFLINE=1
export HF_HUB_ENABLE_HF_TRANSFER=1

# Install the package in development mode
echo "Installing TinyDistill package..."
pip install -e .

# Small helper to check if we can run with 4GB VRAM optimally
if command -v nvidia-smi &> /dev/null; then
    VRAM_SIZE=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
else
    VRAM_SIZE=""
fi

if [ -z "$VRAM_SIZE" ]; then
    echo "Could not detect GPU. Using CPU mode."
    COMMAND="--cpu_offload"
elif [ "$VRAM_SIZE" -lt 3000 ]; then
    echo "Very limited GPU memory detected (<3GB). Using extreme memory optimization."
    COMMAND="--teacher EleutherAI/pythia-70m --student EleutherAI/pythia-70m --samples 200 --sequence_length 32 --teacher_batch 1 --student_batch 2 --gradient_accumulation 8 --cpu_offload --layer_by_layer --gradient_checkpointing"
elif [ "$VRAM_SIZE" -lt 5000 ]; then
    echo "Limited GPU memory detected (~5GB). Using memory-optimized settings."
    COMMAND="--teacher EleutherAI/pythia-410m --student EleutherAI/pythia-70m --samples 300 --sequence_length 64 --teacher_batch 1 --student_batch 4 --gradient_checkpointing"
else
    echo "Sufficient GPU memory detected. Using standard settings."
    COMMAND="--teacher EleutherAI/pythia-410m --student EleutherAI/pythia-70m --samples 1000 --sequence_length 128 --teacher_batch 2 --student_batch 8"
fi

echo "Running distillation with command: $COMMAND"
echo "=================================="

# Run the optimized distillation
python -m tiny_distill.main \
  $COMMAND \
  --dataset tatsu-lab/alpaca \
  --epochs 2 \
  --learning_rate 5e-5 \
  --output_dir "./distill_results" \
  --log_level INFO