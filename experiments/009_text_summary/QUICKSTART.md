# Quick Start Guide

## 1. Install Dependencies

```bash
cd /home/djole/code/lab-work
pip install -r experiments/009_text_summary/requirements.txt
```

## 2. Login to WandB (if not already logged in)

```bash
wandb login
```

## 3. Start Training

```bash
bash experiments/009_text_summary/launch_train.sh
```

## 4. Monitor Training

Open your WandB dashboard: https://wandb.ai

You'll see:
- `train/loss` - decreasing over time (good!)
- `eval/loss` - should also decrease
- `train/learning_rate` - learning rate schedule
- GPU memory usage and utilization

## 5. Wait for Training to Complete

Expected time: ~3-6 hours on 4xH100 for 1 epoch

Checkpoints saved every 200 steps to: `./qwen_legal_multigpu/checkpoint-{step}/`

## 6. Evaluate a Checkpoint

After training (or during, on saved checkpoints):

```bash
# Evaluate a specific checkpoint
python experiments/009_text_summary/eval_checkpoint.py \
    --checkpoint ./qwen_legal_multigpu/checkpoint-400 \
    --num_samples 50

# Or evaluate ALL checkpoints
bash experiments/009_text_summary/eval_all_checkpoints.sh
```

## 7. Check Results

Evaluation results will be in WandB project: `qwen-legal-eval`

Look for:
- ROUGE-1, ROUGE-2, ROUGE-L scores
- Qualitative examples table

## Troubleshooting

### CUDA Out of Memory

If you get OOM during training, edit `train_cluster.py`:

```python
# Line 93: Reduce gradient accumulation or increase steps
gradient_accumulation_steps=16,  # Was 8
```

Or reduce max sequence length:

```python
# Line 17
MAX_SEQ_LENGTH = 131072  # Was 262144 (half)
```

### Flash Attention Not Found

```bash
pip install flash-attn --no-build-isolation
```

### DeepSpeed Not Working

Make sure you're using the launch script, not running the Python file directly:

```bash
# ✓ Correct
bash experiments/009_text_summary/launch_train.sh

# ✗ Wrong - won't use DeepSpeed
python experiments/009_text_summary/train_cluster.py
```

## What's Different from Before?

See `CHANGES.md` for detailed list, but key fixes:
- ✅ Fixed MAX_SEQ_LENGTH syntax error
- ✅ Added proper tokenization
- ✅ Reduced batch size for memory safety
- ✅ Removed generation during training (avoids OOM)
- ✅ Added DeepSpeed ZeRO-2 configuration
- ✅ Created separate evaluation script

## Next Steps After Training

1. Compare eval results across checkpoints
2. Pick best checkpoint based on ROUGE scores
3. Merge LoRA adapters to base model:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Instruct")
model = PeftModel.from_pretrained(base, "./qwen_legal_multigpu/checkpoint-400")
merged = model.merge_and_unload()
merged.save_pretrained("./qwen_legal_final")
```

4. Deploy or share the model!
