#!/bin/bash
# Launch multi-GPU training (DDP or DeepSpeed ZeRO-2, set in config YAML).
#
# Usage:
#   bash launch_train.sh configs/default.yaml            # new run
#   bash launch_train.sh configs/default.yaml 4           # new run, 4 GPUs
#   bash launch_train.sh --resume train_results/002       # resume interrupted run
#   bash launch_train.sh --resume train_results/002 4     # resume, 4 GPUs

set -euo pipefail

export NCCL_P2P_DISABLE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RESUME_MODE=false
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_MODE=true
    shift
fi

FIRST_ARG="${1:?Usage: $0 [--resume] <config.yaml|run_dir> [num_gpus]}"
NUM_GPUS="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

if [[ ! "$FIRST_ARG" = /* ]]; then
    FIRST_ARG="${SCRIPT_DIR}/${FIRST_ARG}"
fi

if $RESUME_MODE; then
    echo "============================================"
    echo "  Legal Summary Training (RESUME)"
    echo "  Run dir:  $FIRST_ARG"
    echo "  GPUs:     $NUM_GPUS"
    echo "============================================"

    accelerate launch \
        --num_processes="$NUM_GPUS" \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --multi_gpu \
        --main_process_port=29500 \
        "${SCRIPT_DIR}/train.py" \
        --resume_from "$FIRST_ARG"
else
    if [[ ! -f "$FIRST_ARG" ]]; then
        echo "Error: Config file not found: $FIRST_ARG"
        exit 1
    fi

    echo "============================================"
    echo "  Legal Summary Training"
    echo "  Config:   $FIRST_ARG"
    echo "  GPUs:     $NUM_GPUS"
    echo "============================================"

    accelerate launch \
        --num_processes="$NUM_GPUS" \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --multi_gpu \
        --main_process_port=29500 \
        "${SCRIPT_DIR}/train.py" \
        --config "$FIRST_ARG"
fi
