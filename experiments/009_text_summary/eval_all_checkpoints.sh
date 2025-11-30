#!/bin/bash

# Script to evaluate all checkpoints in the output directory
# Usage: bash eval_all_checkpoints.sh

OUTPUT_DIR="./qwen_legal_multigpu"
NUM_SAMPLES=50
MAX_NEW_TOKENS=1024
TEMPERATURE=0.7

echo "Evaluating all checkpoints in $OUTPUT_DIR"
echo "=========================================="

# Find all checkpoint directories
for checkpoint_dir in "$OUTPUT_DIR"/checkpoint-*; do
    if [ -d "$checkpoint_dir" ]; then
        echo ""
        echo "Evaluating: $checkpoint_dir"
        echo "----------------------------"

        python experiments/009_text_summary/eval_checkpoint.py \
            --checkpoint "$checkpoint_dir" \
            --num_samples $NUM_SAMPLES \
            --max_new_tokens $MAX_NEW_TOKENS \
            --temperature $TEMPERATURE

        if [ $? -eq 0 ]; then
            echo "✓ Evaluation completed for $checkpoint_dir"
        else
            echo "✗ Evaluation failed for $checkpoint_dir"
        fi
    fi
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "Check WandB for results: https://wandb.ai"
