"""Config-driven multi-GPU training for legal summarization with DeepSpeed ZeRO-2.

Usage:
    # See all options
    python train.py --help

    # Train with a config file (multi-GPU via accelerate)
    accelerate launch --num_processes=4 --mixed_precision=bf16 \
        train.py --config configs/default.yaml

    # Convenience launcher (auto-detects GPUs)
    bash launch_train.sh configs/default.yaml
"""

import argparse
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import wandb
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
)

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from data import load_and_prepare
from prompts import get_next_run_dir, load_config, load_env


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on legal summarization data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single-config training (most common)
  bash launch_train.sh configs/default.yaml

  # Specify GPU count
  bash launch_train.sh configs/default.yaml 4

  # Direct accelerate launch
  accelerate launch --num_processes=4 --mixed_precision=bf16 \\
      train.py --config configs/my_run.yaml

Config reference:
  See configs/example_train_config.yaml for all available options
  with detailed comments explaining each parameter.

Outputs:
  train_results/NNN/config.yaml      Frozen copy of your config
  train_results/NNN/checkpoints/     LoRA adapter checkpoints
  train_results/NNN/summary.txt      Training metrics and metadata
  train_results/NNN/wandb_run_id.txt WandB run ID for cross-reference
""",
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML training config file (see configs/example_train_config.yaml)",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1,
        help="(Internal) Set automatically by DeepSpeed/accelerate. Do not set manually.",
    )
    return parser.parse_args()


def write_summary(run_dir: Path, config: dict, trainer, train_result, start_time: float, wandb_url: str):
    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    checkpoints = sorted(
        [d.name for d in (run_dir / "checkpoints").iterdir() if d.is_dir()]
    ) if (run_dir / "checkpoints").exists() else []

    best_eval_loss = min(eval_losses) if eval_losses else None
    best_eval_step = None
    if best_eval_loss is not None:
        for entry in log_history:
            if entry.get("eval_loss") == best_eval_loss:
                best_eval_step = entry.get("step")
                break

    hub_cfg = config.get("hub", {})
    hub_status = f"pushed to {hub_cfg['repo']}" if hub_cfg.get("push_checkpoints") else "disabled"

    lines = [
        "=== Training Summary ===",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Run ID: train-{run_dir.name}",
        f"WandB Run: {wandb_url}",
        "",
        f"Model: {config['base_model']}",
        f"Dataset: {config['dataset']['name']} ({config['dataset'].get('config', 'default')})",
        f"LoRA rank: {config['lora']['r']}, alpha: {config['lora']['alpha']}",
        f"Max seq length: {config['max_seq_length']}",
        "",
        f"Duration: {hours}h {minutes}m {seconds}s",
        f"Total steps: {trainer.state.global_step}",
        f"Final train loss: {train_losses[-1]:.4f}" if train_losses else "Final train loss: N/A",
        f"Final eval loss: {eval_losses[-1]:.4f}" if eval_losses else "Final eval loss: N/A",
    ]

    if best_eval_loss is not None:
        lines.append(f"Best eval loss: {best_eval_loss:.4f} (step {best_eval_step})")

    lines.append("")
    lines.append("Checkpoints saved:")
    for cp in checkpoints:
        lines.append(f"  - checkpoints/{cp}")

    lines.append("")
    lines.append(f"HF Hub: {hub_status}")
    lines.append("")

    lines.append("--- Training Config ---")
    lines.append(f"Batch size (per device): {config['training']['per_device_batch_size']}")
    lines.append(f"Gradient accumulation: {config['training']['grad_accum_steps']}")
    lines.append(f"Learning rate: {config['training']['learning_rate']}")
    lines.append(f"LR scheduler: {config['training'].get('lr_scheduler', 'cosine')}")
    lines.append(f"Warmup steps: {config['training']['warmup_steps']}")
    lines.append(f"Weight decay: {config['training'].get('weight_decay', 0.01)}")
    lines.append(f"Seed: {config['training'].get('seed', 3407)}")

    summary_path = run_dir / "summary.txt"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {summary_path}")


def main():
    args = parse_args()
    load_env(PROJECT_DIR)
    config = load_config(args.config)

    run_dir = get_next_run_dir(PROJECT_DIR / "train_results")
    run_name = f"train-{run_dir.name}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    shutil.copy2(args.config, run_dir / "config.yaml")

    is_main_process = int(os.environ.get("LOCAL_RANK", 0)) == 0

    wandb_cfg = config.get("wandb", {})
    wandb_url = ""
    if is_main_process:
        wandb.init(
            project=wandb_cfg.get("project", "legal-summary"),
            name=run_name,
            config=config,
        )
        wandb_url = wandb.run.get_url() or ""
        (run_dir / "wandb_run_id.txt").write_text(wandb.run.id)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if is_main_process:
        print(f"Loading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_main_process:
        print("Loading and tokenizing dataset...")
    train_ds, eval_ds = load_and_prepare(config, tokenizer)
    if is_main_process:
        print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    if is_main_process:
        print(f"Loading model: {config['base_model']}")
    model = Qwen3_5ForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )

    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        lora_dropout=lora_cfg.get("dropout", 0.0),
    )
    model = get_peft_model(model, peft_config)
    if is_main_process:
        model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    t_cfg = config["training"]
    ds_config_path = str(PROJECT_DIR / "ds_config_zero2.json")

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=t_cfg["per_device_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_batch_size"],
        gradient_accumulation_steps=t_cfg["grad_accum_steps"],
        learning_rate=t_cfg["learning_rate"],
        num_train_epochs=t_cfg["num_epochs"],
        max_steps=t_cfg.get("max_steps", -1),
        warmup_steps=t_cfg["warmup_steps"],
        weight_decay=t_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=t_cfg.get("lr_scheduler", "cosine"),
        max_grad_norm=1.0,
        seed=t_cfg.get("seed", 3407),
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        bf16_full_eval=dtype == torch.bfloat16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=t_cfg.get("logging_steps", 10),
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        report_to="wandb" if is_main_process else "none",
        run_name=run_name,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        deepspeed=ds_config_path,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    if is_main_process:
        print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()

    if is_main_process:
        print("Saving final model...")
        trainer.save_model(str(checkpoint_dir / "final"))

        hub_cfg = config.get("hub", {})
        if hub_cfg.get("push_checkpoints"):
            repo_id = hub_cfg["repo"]
            print(f"Pushing to HF Hub: {repo_id}")
            try:
                model.push_to_hub(repo_id, commit_message=f"{run_name} final")
                tokenizer.push_to_hub(repo_id, commit_message=f"{run_name} tokenizer")
            except Exception as e:
                print(f"Warning: HF Hub push failed: {e}")

        write_summary(run_dir, config, trainer, train_result, start_time, wandb_url)
        wandb.finish()

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
