#!/bin/bash

# Launch script for multi-GPU training with DeepSpeed ZeRO-2
# Usage: bash launch_train.sh

# Set environment variables
export WANDB_PROJECT="qwen-legal"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs

# Optional: Set these if you have WandB API key
# export WANDB_API_KEY="your_key_here"

# Launch with accelerate (recommended)
accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    --main_process_port=29500 \
    experiments/009_text_summary/train_cluster.py

# Alternative: Launch directly with DeepSpeed (uncomment to use)
# deepspeed --num_gpus=4 \
#     --master_port=29500 \
#     experiments/009_text_summary/train_cluster.py
