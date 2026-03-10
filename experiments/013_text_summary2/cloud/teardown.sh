#!/bin/bash
# Lambda VM teardown script. Run before terminating the instance.
# Pushes checkpoints and eval results to HuggingFace Hub.
#
# Usage:
#   bash cloud/teardown.sh
#
# Environment variables:
#   HF_TOKEN  - HuggingFace token (must have write access)
#   HUB_REPO  - HuggingFace Hub repo (default: YOUR_USERNAME/qwen-legal-summary)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

HUB_REPO="${HUB_REPO:-YOUR_USERNAME/qwen-legal-summary}"

echo "=== Lambda VM Teardown ==="
echo "Date: $(date -u)"
echo "Project dir: $PROJECT_DIR"
echo "Hub repo: $HUB_REPO"

# 1. Flush WandB
echo "[1/3] Flushing WandB logs..."
wandb sync 2>/dev/null || true

# 2. Push train_results checkpoints
echo "[2/3] Pushing train_results to HuggingFace Hub..."
cd "$PROJECT_DIR"

for run_dir in train_results/*/; do
    if [[ ! -d "$run_dir" ]] || [[ "$(basename "$run_dir")" == ".gitkeep" ]]; then
        continue
    fi

    run_name="train-$(basename "$run_dir")"
    echo "  Processing: $run_name"

    for ckpt_dir in "${run_dir}checkpoints"/checkpoint-*/ "${run_dir}checkpoints"/final/; do
        if [[ ! -d "$ckpt_dir" ]]; then
            continue
        fi

        ckpt_name=$(basename "$ckpt_dir")
        tag="${run_name}-${ckpt_name}"

        # Check if already pushed by looking for a marker file
        marker="${ckpt_dir}/.hub_pushed"
        if [[ -f "$marker" ]]; then
            echo "    Skip $ckpt_name (already pushed)"
            continue
        fi

        echo "    Pushing $ckpt_name as tag: $tag"
        python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='${ckpt_dir}',
    repo_id='${HUB_REPO}',
    path_in_repo='${tag}',
    commit_message='${tag}',
)
" && touch "$marker"
    done

    # Push summary + config
    if [[ -f "${run_dir}summary.txt" ]]; then
        python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='${run_dir}summary.txt',
    path_in_repo='${run_name}/summary.txt',
    repo_id='${HUB_REPO}',
    commit_message='${run_name} summary',
)
api.upload_file(
    path_or_fileobj='${run_dir}config.yaml',
    path_in_repo='${run_name}/config.yaml',
    repo_id='${HUB_REPO}',
    commit_message='${run_name} config',
)
" 2>/dev/null || echo "    Warning: failed to push summary/config for $run_name"
    fi
done

# 3. Push prompt_results
echo "[3/3] Pushing prompt_results to HuggingFace Hub..."
for run_dir in prompt_results/*/; do
    if [[ ! -d "$run_dir" ]] || [[ "$(basename "$run_dir")" == ".gitkeep" ]]; then
        continue
    fi

    run_name="prompt-$(basename "$run_dir")"

    if [[ -f "${run_dir}summary.txt" ]]; then
        marker="${run_dir}/.hub_pushed"
        if [[ -f "$marker" ]]; then
            echo "  Skip $run_name (already pushed)"
            continue
        fi

        echo "  Pushing $run_name"
        python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='${run_dir}',
    repo_id='${HUB_REPO}',
    path_in_repo='evals/${run_name}',
    commit_message='${run_name} eval results',
    ignore_patterns=['generated/*'],
)
" && touch "$marker"
    fi
done

echo ""
echo "=== Teardown complete ==="
echo "All results pushed to: https://huggingface.co/$HUB_REPO"
echo "Safe to terminate the instance."
