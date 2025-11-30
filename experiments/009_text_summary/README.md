# Qwen3-4B Legal Summarization Fine-tuning

Fine-tuning Qwen3-4B-Instruct on legal case summarization using LoRA with DeepSpeed ZeRO-2 across 4xH100 GPUs.

## Setup

### Requirements

```bash
pip install torch transformers datasets peft deepspeed accelerate wandb rouge-score flash-attn
```

### Environment Variables

```bash
export WANDB_API_KEY="your_wandb_key"  # Optional: if not logged in
export WANDB_PROJECT="qwen-legal"
```

## Training

### Quick Start

```bash
# Make sure you're in the project root
bash experiments/009_text_summary/launch_train.sh
```

### Training Configuration

- **Model**: Qwen3-4B-Instruct (200k context window)
- **Dataset**: CJWeiss/LexSumm (multilong)
- **LoRA Config**:
  - r=64, alpha=32
  - Target modules: Q, K, V, O, Gate, Up, Down projections
  - ~1.5% trainable parameters
- **Batch Size**:
  - Per device: 1
  - Gradient accumulation: 8
  - Effective batch size: 32 (across 4 GPUs)
- **Optimization**:
  - DeepSpeed ZeRO-2 (optimizer + gradient partitioning)
  - Flash Attention 2
  - Gradient checkpointing
  - BF16 precision

### What Gets Logged to WandB During Training

- `train/loss` - Training loss every 10 steps
- `train/learning_rate` - Learning rate schedule
- `train/epoch` - Current epoch
- `eval/loss` - Validation loss every 200 steps
- `eval/perplexity` - Validation perplexity
- System metrics (GPU memory, utilization, etc.)

**Note**: No text generation happens during training to avoid OOM. Only loss-based metrics.

### Memory Usage

Expected per GPU (H100 80GB):
- Model weights (BF16): ~8GB
- LoRA adapters: ~134MB
- Optimizer states (partitioned): ~300MB
- Gradients (partitioned): ~300MB
- Activations (with checkpointing): ~50GB
- **Total**: ~59GB ✅ Safe margin

### Checkpoints

Checkpoints are saved every 200 steps to `./qwen_legal_multigpu/checkpoint-{step}/`

Only the last 3 checkpoints are kept to save disk space.

## Evaluation

After training completes (or at any checkpoint), run comprehensive evaluation:

```bash
python experiments/009_text_summary/eval_checkpoint.py \
    --checkpoint ./qwen_legal_multigpu/checkpoint-400 \
    --num_samples 50 \
    --max_new_tokens 1024 \
    --temperature 0.7
```

### Evaluation Metrics

The evaluation script computes:
- **ROUGE-1, ROUGE-2, ROUGE-L**: Standard summarization metrics
- **Qualitative samples**: First 5 examples with input/reference/generated summaries

Results are logged to WandB project `qwen-legal-eval`.

### Evaluation Options

```bash
--checkpoint        Path to checkpoint (required)
--num_samples       Number of validation samples to evaluate (default: 50)
--max_new_tokens    Max tokens to generate (default: 1024)
--temperature       Sampling temperature (default: 0.7)
--wandb_project     WandB project name (default: qwen-legal-eval)
--no_wandb          Disable WandB logging
```

### Example Evaluation During Training

```bash
# Evaluate checkpoint 200
python experiments/009_text_summary/eval_checkpoint.py \
    --checkpoint ./qwen_legal_multigpu/checkpoint-200 \
    --num_samples 20

# Evaluate checkpoint 400
python experiments/009_text_summary/eval_checkpoint.py \
    --checkpoint ./qwen_legal_multigpu/checkpoint-400 \
    --num_samples 50
```

## Files

- `train_cluster.py` - Main training script
- `ds_config_zero2.json` - DeepSpeed ZeRO-2 configuration
- `launch_train.sh` - Multi-GPU launch script
- `eval_checkpoint.py` - Comprehensive evaluation script
- `README.md` - This file

## Troubleshooting

### OOM During Training

If you still get OOM with `per_device_train_batch_size=1`:

1. Reduce max sequence length temporarily:
   ```python
   MAX_SEQ_LENGTH = 131072  # Half of 262144
   ```

2. Increase gradient accumulation:
   ```python
   gradient_accumulation_steps=16  # Instead of 8
   ```

3. Check if some samples are extremely long and filter them out more aggressively

### Slow Training

If training is slower than expected:

1. Check GPU utilization: `nvidia-smi dmon -s u`
2. Ensure Flash Attention 2 is installed: `pip install flash-attn --no-build-isolation`
3. Verify DeepSpeed is being used: Look for "DeepSpeed" in training logs

### WandB Not Logging

```bash
wandb login  # Login first
wandb online  # Ensure online mode
```

## Expected Training Time

Rough estimates for 1 epoch on LexSumm multilong dataset:

- ~500-700 steps per epoch (depends on filtering)
- ~20-30 seconds per step with batch_size=1, gradient_accumulation=8
- **Total**: ~3-6 hours on 4xH100

## Next Steps

After training:

1. Run comprehensive evaluation on best checkpoint
2. Compare ROUGE scores across different checkpoints
3. Inspect qualitative examples to assess quality
4. If results are good, merge LoRA adapters to base model:
   ```python
   merged_model = model.merge_and_unload()
   merged_model.save_pretrained("./qwen_legal_merged")
   ```
