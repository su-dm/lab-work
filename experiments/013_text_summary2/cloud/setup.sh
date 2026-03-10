#!/bin/bash
# Lambda VM setup script. Run once when a new instance spins up.
#
# Usage:
#   bash cloud/setup.sh [checkpoint_to_resume]
#
# Environment variables (set before running or export in your shell):
#   WANDB_API_KEY     - WandB API key
#   HF_TOKEN          - HuggingFace token (for private repos / pushing)
#   GIT_REPO          - Git repo URL (default: current repo)

set -euo pipefail

CHECKPOINT_TAG="${1:-}"

echo "=== Lambda VM Setup ==="
echo "Date: $(date -u)"

# 1. System deps (Lambda images usually have these, but just in case)
echo "[1/5] Checking system dependencies..."
if ! command -v git &>/dev/null; then
    sudo apt-get update && sudo apt-get install -y git
fi

# 2. Clone repo if not already present
REPO_DIR="${REPO_DIR:-/home/ubuntu/lab-work}"
if [[ ! -d "$REPO_DIR" ]]; then
    echo "[2/5] Cloning repository..."
    GIT_REPO="${GIT_REPO:-https://github.com/YOUR_USERNAME/lab-work.git}"
    git clone "$GIT_REPO" "$REPO_DIR"
else
    echo "[2/5] Repository exists, pulling latest..."
    cd "$REPO_DIR" && git pull
fi

cd "$REPO_DIR"

# 3. Install Python dependencies
echo "[3/5] Installing Python dependencies..."
pip install -r experiments/013_text_summary2/requirements.txt

# 4. Auth
echo "[4/5] Setting up authentication..."
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null || true
    echo "  WandB: logged in"
else
    echo "  WandB: WANDB_API_KEY not set, skipping (run 'wandb login' manually)"
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null || true
    echo "  HuggingFace: logged in"
else
    echo "  HuggingFace: HF_TOKEN not set, skipping"
fi

# 5. Optionally pull a checkpoint to resume from
if [[ -n "$CHECKPOINT_TAG" ]]; then
    echo "[5/5] Pulling checkpoint: $CHECKPOINT_TAG"
    HUB_REPO="${HUB_REPO:-YOUR_USERNAME/qwen-legal-summary}"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('${HUB_REPO}', revision='${CHECKPOINT_TAG}',
                  local_dir='experiments/013_text_summary2/resumed_checkpoint')
"
    echo "  Checkpoint downloaded to experiments/013_text_summary2/resumed_checkpoint/"
else
    echo "[5/5] No checkpoint to resume from, starting fresh"
fi

echo ""
echo "=== Setup complete ==="
echo "To start training:"
echo "  cd $REPO_DIR"
echo "  bash experiments/013_text_summary2/launch_train.sh configs/default.yaml"
