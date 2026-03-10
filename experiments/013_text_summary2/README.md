# Legal Text Summarization v2

Fine-tune Qwen3-4B-Instruct to produce high-quality legal case summaries. Config-driven, multi-GPU, with structured experiment tracking.

## Quick Start

```bash
pip install -r experiments/013_text_summary2/requirements.txt

# Train (multi-GPU)
bash experiments/013_text_summary2/launch_train.sh configs/default.yaml

# Evaluate a checkpoint
python experiments/013_text_summary2/evaluate.py \
    --checkpoint experiments/013_text_summary2/train_results/001/checkpoints/checkpoint-200 \
    --system_prompt "You are a legal assistant..." \
    --num_samples 20

# Inference on a document
python experiments/013_text_summary2/infer.py \
    --checkpoint experiments/013_text_summary2/train_results/001/checkpoints/checkpoint-200 \
    --system_prompt "You are a legal assistant..." \
    --input case.pdf
```

## Project Structure

```
013_text_summary2/
  configs/default.yaml          Config template (copy and modify per run)
  train_results/001/            Auto-created per training run
    config.yaml                 Frozen config snapshot
    checkpoints/                LoRA adapter checkpoints
    summary.txt                 Training metrics + metadata
  prompt_results/001/           Auto-created per eval run
    prompt_config.yaml          Eval configuration
    generated/                  Raw model outputs per sample
    summary.txt                 Multi-tier eval results
    metrics.json                Machine-readable metrics
  train.py                      Multi-GPU training (DeepSpeed ZeRO-2)
  evaluate.py                   4-tier evaluation pipeline
  infer.py                      Single-doc inference (vLLM)
  data.py                       Dataset loading + label masking
  prompts.py                    Shared prompt logic
  pdf_utils.py                  PDF/text extraction
  cloud/setup.sh                Lambda VM bootstrap
  cloud/teardown.sh             Push results before termination
```

## Training

Training is driven by a YAML config file. Each run auto-creates the next numbered directory under `train_results/`.

```bash
# Copy and edit a config
cp experiments/013_text_summary2/configs/default.yaml experiments/013_text_summary2/configs/my_run.yaml
# Edit my_run.yaml...

# Launch on all available GPUs
bash experiments/013_text_summary2/launch_train.sh configs/my_run.yaml

# Or specify GPU count
bash experiments/013_text_summary2/launch_train.sh configs/my_run.yaml 4
```

After training completes, check `train_results/001/summary.txt` for results. The WandB run name matches the directory number (e.g. `train-001`).

### Switching Datasets

Edit the `dataset` section of your config:

```yaml
dataset:
  name: "your-org/your-dataset"
  config: "subset_name"
  input_col: "text"
  output_col: "summary"
```

No code changes needed.

## Evaluation

Evaluate how a checkpoint + system prompt combo performs on a dataset. Four metric tiers:

1. **Overlap** -- ROUGE-1/2/L, BERTScore
2. **Format** -- section presence, length analysis, extractive coverage
3. **Faithfulness** -- NLI-based hallucination detection
4. **LLM Judge** -- Claude rates accuracy, completeness, format, conciseness (opt-in)

```bash
# Basic eval (tiers 1-3)
python experiments/013_text_summary2/evaluate.py \
    --checkpoint train_results/001/checkpoints/checkpoint-400 \
    --system_prompt "You are a legal assistant..." \
    --num_samples 50

# With Claude-as-judge (requires ANTHROPIC_API_KEY env var)
python experiments/013_text_summary2/evaluate.py \
    --checkpoint train_results/001/checkpoints/checkpoint-400 \
    --system_prompt "You are a legal assistant..." \
    --llm_judge

# From a config file
python experiments/013_text_summary2/evaluate.py \
    --config experiments/013_text_summary2/prompt_results/001/prompt_config.yaml
```

Results go to `prompt_results/NNN/summary.txt`. WandB run name: `prompt-001`, etc.

### Comparing Prompts

To test different system prompts against the same model:

1. Create `prompt_results/001/prompt_config.yaml` with prompt A
2. Create `prompt_results/002/prompt_config.yaml` with prompt B (same checkpoint)
3. Run evaluate on each
4. Compare `summary.txt` files side by side

## Inference

Single-document inference using vLLM with automatic tensor parallelism across all available GPUs.

```bash
# LoRA checkpoint + PDF input
python experiments/013_text_summary2/infer.py \
    --checkpoint train_results/001/checkpoints/checkpoint-400 \
    --system_prompt "You are a legal assistant..." \
    --input case_filing.pdf

# Base model + text input, save to file
python experiments/013_text_summary2/infer.py \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --system_prompt "You are a legal assistant..." \
    --input document.txt \
    --output summary.txt
```

Supports both `.pdf` (auto-extracted via pymupdf) and `.txt` inputs.

## Cloud (Lambda)

### Spin Up

```bash
# On fresh Lambda VM
export WANDB_API_KEY="your_key"
export HF_TOKEN="your_token"
export GIT_REPO="https://github.com/you/lab-work.git"

bash cloud/setup.sh                      # fresh start
bash cloud/setup.sh train-001-final      # resume from a pushed checkpoint
```

### Spin Down

```bash
# Before terminating -- pushes all checkpoints + results to HF Hub
export HUB_REPO="your-username/qwen-legal-summary"
bash cloud/teardown.sh
```

## WandB Naming

| What | WandB Run Name | Project |
|------|---------------|---------|
| Training run 1 | `train-001` | `legal-summary` |
| Training run 2 | `train-002` | `legal-summary` |
| Prompt eval 1 | `prompt-001` | `legal-summary` |
| Prompt eval 2 | `prompt-002` | `legal-summary` |

Directory numbers always match WandB run names for easy cross-referencing.

## HuggingFace Hub

Checkpoints are pushed with descriptive paths:

```
your-username/qwen-legal-summary/
  train-001-checkpoint-200/     LoRA adapters
  train-001-checkpoint-400/     LoRA adapters
  train-001-final/              Final adapters
  train-001/summary.txt         Training summary
  evals/prompt-001/             Eval results
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `WANDB_API_KEY` | WandB authentication |
| `HF_TOKEN` | HuggingFace Hub read/write |
| `ANTHROPIC_API_KEY` | Claude API for LLM-as-judge |
| `HUB_REPO` | HF Hub repo for teardown push |
