# Legal Text Summarization v2

Fine-tune Qwen3-4B to summarize legal cases. Config-driven, multi-GPU, iterative.

## Setup

```bash
pip install -r experiments/013_text_summary2/requirements.txt
cp .env_template .env   # fill in your keys
```

Every script loads `.env` automatically. See `.env_template` for all variables.

For full details on any script, run `python <script> --help`.

---

## Train

All training is config-driven. Copy the example config, edit it, run it.

```bash
# 1. Create your config
cp configs/example_train_config.yaml configs/my_run.yaml
# Edit configs/my_run.yaml (model, dataset, hyperparams, system prompt, etc.)

# 2. Launch training (auto-detects GPUs)
bash launch_train.sh configs/my_run.yaml

# Or specify GPU count
bash launch_train.sh configs/my_run.yaml 4
```

Results land in `train_results/001/` (auto-incrementing):

```
train_results/001/
  config.yaml       Frozen copy of config used
  checkpoints/      LoRA adapters (~250MB each, all kept)
  summary.txt       Date, duration, losses, checkpoint list, WandB link
```

WandB run name matches the directory: `train-001`, `train-002`, etc.

### Switching datasets

Change the `dataset` section in your config. No code changes needed:

```yaml
dataset:
  name: "your-org/your-dataset"
  config: "subset_name"
  input_col: "text"
  output_col: "summary"
```

---

## Evaluate Prompts

Test how different system prompts or checkpoints perform on a dataset. Four metric tiers run automatically (LLM judge is opt-in).

### From CLI args (quick)

```bash
python evaluate.py \
    --checkpoint train_results/001/checkpoints/checkpoint-400 \
    --system_prompt "You are a legal assistant..." \
    --num_samples 20
```

### From a config file

```bash
# 1. Copy the example
cp configs/example_prompt_eval_config.yaml prompt_results/001/prompt_config.yaml
# Edit: set checkpoint, system_prompt, num_samples, etc.

# 2. Run
python evaluate.py --config prompt_results/001/prompt_config.yaml
```

### With Claude-as-judge (tier 4)

Requires `ANTHROPIC_API_KEY` in `.env`. Costs API credits.

```bash
python evaluate.py \
    --config prompt_results/001/prompt_config.yaml \
    --llm_judge
```

### System prompt from a file

`--system_prompt` accepts either inline text or a path to a `.txt` file:

```bash
python evaluate.py \
    --checkpoint train_results/001/checkpoints/checkpoint-400 \
    --system_prompt prompts/legal_brief_v2.txt
```

Results land in `prompt_results/001/` (auto-incrementing):

```
prompt_results/001/
  prompt_config.yaml   What was evaluated
  generated/           Raw model outputs per sample
  summary.txt          All metrics, format compliance, sample outputs
  metrics.json         Machine-readable scores
```

WandB run name: `prompt-001`, `prompt-002`, etc.

### Metric tiers

| Tier | What | Cost |
|------|------|------|
| 1 | ROUGE-1/2/L, BERTScore | Free, fast |
| 2 | Section presence, length stats, extractive coverage | Free, fast |
| 3 | NLI faithfulness (hallucination detection) | Free, medium |
| 4 | Claude judges accuracy, completeness, format, conciseness | API credits, slow |

---

## Inference

Summarize a single document (PDF or text). Uses vLLM with automatic tensor parallelism across all available GPUs.

```bash
# LoRA checkpoint + PDF
python infer.py \
    --checkpoint train_results/001/checkpoints/checkpoint-400 \
    --system_prompt prompts/legal_brief.txt \
    --input case_filing.pdf

# Base model (no adapter) for comparison
python infer.py \
    --model "Qwen/Qwen3-4B-Instruct-2507" \
    --system_prompt "You are a legal assistant." \
    --input document.txt

# Save to file instead of stdout
python infer.py \
    --checkpoint train_results/001/checkpoints/final \
    --system_prompt prompts/legal_brief.txt \
    --input case.pdf \
    --output summary.txt
```

`--system_prompt` accepts inline text or a path to a `.txt` file.

`--input` accepts `.txt` or `.pdf` (auto-extracted via pymupdf).

---

## Cloud (Lambda)

### Spin up a VM

```bash
export WANDB_API_KEY="..."
export HF_TOKEN="..."
export GIT_REPO="https://github.com/you/lab-work.git"

# Fresh start
bash cloud/setup.sh

# Resume from a previously pushed checkpoint
bash cloud/setup.sh train-001-final
```

### Spin down (before terminating)

Pushes all unpushed checkpoints and eval results to HuggingFace Hub.

```bash
export HUB_REPO="your-username/qwen-legal-summary"
bash cloud/teardown.sh
```

---

## File Reference

| File | What it does |
|------|-------------|
| `train.py` | Multi-GPU training (DeepSpeed ZeRO-2) |
| `evaluate.py` | 4-tier benchmark evaluation |
| `infer.py` | Single-doc inference (vLLM) |
| `data.py` | Dataset loading + label masking |
| `prompts.py` | Shared prompt logic, config loading, .env loader |
| `pdf_utils.py` | PDF/text extraction |
| `launch_train.sh` | Convenience training launcher |
| `ds_config_zero2.json` | DeepSpeed config |
| `cloud/setup.sh` | Lambda VM bootstrap |
| `cloud/teardown.sh` | Push results before termination |
| `.env_template` | Copy to `.env`, fill in secrets |
| `configs/example_train_config.yaml` | Annotated training config |
| `configs/example_prompt_eval_config.yaml` | Annotated eval config |
