#!/bin/bash
# Launch multi-GPU training with DeepSpeed ZeRO-2.
#
# Usage:
#   bash experiments/013_text_summary2/launch_train.sh configs/default.yaml
#   bash experiments/013_text_summary2/launch_train.sh configs/default.yaml 4

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONFIG_FILE="${1:?Usage: $0 <config.yaml> [num_gpus]}"
NUM_GPUS="${2:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

# Resolve config path relative to script dir if not absolute
if [[ ! "$CONFIG_FILE" = /* ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "============================================"
echo "  Legal Summary Training"
echo "  Config:   $CONFIG_FILE"
echo "  GPUs:     $NUM_GPUS"
echo "============================================"

accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --multi_gpu \
    --main_process_port=29500 \
    "${SCRIPT_DIR}/train.py" \
    --config "$CONFIG_FILE"
