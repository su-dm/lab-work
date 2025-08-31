import os
import argparse
import shutil
import pathlib
import json

from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from peft import LoraConfig


def _format_example(example, tokenizer):
    instruction = (example.get("instruction") or "").strip()
    prompt = (example.get("prompt") or "").strip()
    answer = (example.get("answer") or "").strip()

    if instruction and prompt:
        user_text = f"{instruction}\n\n{prompt}"
    else:
        user_text = instruction or prompt

    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            pass

    # Fallback basic instruction format
    return f"[INST] {user_text} [/INST] {answer}"


def load_config(path: str):
    """Load YAML or JSON config from local path or gs:// URI."""
    data_bytes: bytes
    if path.startswith("gs://"):
        import gcsfs  # lazy import to avoid requiring it locally
        fs = gcsfs.GCSFileSystem()
        with fs.open(path, "rb") as f:
            data_bytes = f.read()
    else:
        with open(path, "rb") as f:
            data_bytes = f.read()

    text = data_bytes.decode("utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("pyyaml is required for YAML configs. Install pyyaml or use JSON.") from e
        cfg = yaml.safe_load(text) or {}
    else:
        cfg = json.loads(text)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/dict")
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training_exp/config.yaml", help="Path to YAML/JSON config (local or gs://)")
    # parse_known_args so we ignore any legacy flags passed from old launchers
    parsed, _ = parser.parse_known_args()
    cfg = load_config(parsed.config)

    # Extract config with defaults
    model_id = cfg.get("model_id")
    train_path = cfg.get("train_path")
    output_dir = cfg.get("output_dir")

    if not model_id or not train_path or not output_dir:
        raise ValueError("Config must include model_id, train_path, and output_dir")

    num_train_epochs = float(cfg.get("num_train_epochs", 1.0))
    learning_rate = float(cfg.get("learning_rate", 2e-4))
    per_device_train_batch_size = int(cfg.get("per_device_train_batch_size", 1))
    gradient_accumulation_steps = int(cfg.get("gradient_accumulation_steps", 16))
    max_seq_length = int(cfg.get("max_seq_length", 2048))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_steps = int(cfg.get("save_steps", 200))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.03))
    seed = int(cfg.get("seed", 42))
    english_only = bool(cfg.get("english_only", False))
    no_4bit = bool(cfg.get("no_4bit", False))

    lora_cfg = cfg.get("lora", {}) or {}
    lora_r = int(cfg.get("lora_r", lora_cfg.get("r", 16)))
    lora_alpha = int(cfg.get("lora_alpha", lora_cfg.get("alpha", 32)))
    lora_dropout = float(cfg.get("lora_dropout", lora_cfg.get("dropout", 0.05)))

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        print("[WARN] HF_TOKEN not set. If the model is gated, loading may fail.")

    # Load dataset (datasets + gcsfs handles gs:// transparently)
    ds = load_dataset("json", data_files={"train": train_path}, split="train")
    if english_only:
        def _is_en(ex):
            return (
                ex.get("instruction_language") == "en"
                and ex.get("prompt_language") == "en"
                and ex.get("answer_language") == "en"
            )
        ds = ds.filter(_is_en)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = None
    if not no_4bit:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"[WARN] bitsandbytes unavailable: {e}. Proceeding without 4-bit.")
            bnb_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32) if bnb_config is None else None,
        device_map="auto",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.train()

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
    )

    def formatting_func(batch):
        return [_format_example(ex, tokenizer) for ex in batch]

    training_args = TrainingArguments(
        output_dir="outputs",
        report_to=["tensorboard"],
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available(),
        fp16=False,
        optim=("paged_adamw_8bit" if bnb_config is not None else "adamw_torch"),
        save_total_limit=2,
        seed=seed,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        args=training_args,
        peft_config=lora_config,
        formatting_func=formatting_func,
        max_seq_length=max_seq_length,
        packing=True,
        dataset_num_proc=1,
        train_on_inputs=False,
    )

    trainer.train()

    # Save adapter locally then optionally upload to GCS
    local_adapter_dir = os.path.abspath("outputs/adapter")
    os.makedirs(local_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(local_adapter_dir)
    tokenizer.save_pretrained(local_adapter_dir)

    out = output_dir
    if out.startswith("gs://"):
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            for root, _, files in os.walk(local_adapter_dir):
                rel = os.path.relpath(root, local_adapter_dir)
                for f in files:
                    lpath = os.path.join(root, f)
                    gspath = out.rstrip("/") + "/" + ("" if rel == "." else rel + "/") + f
                    with open(lpath, "rb") as rf:
                        with fs.open(gspath, "wb") as wf:
                            wf.write(rf.read())
            print(f"[INFO] Uploaded adapter to {out}")
        except Exception as e:
            print(f"[ERROR] Failed to upload to GCS: {e}")
            raise
    else:
        pathlib.Path(out).mkdir(parents=True, exist_ok=True)
        for root, _, files in os.walk(local_adapter_dir):
            rel = os.path.relpath(root, local_adapter_dir)
            dest_root = os.path.join(out, "" if rel == "." else rel)
            os.makedirs(dest_root, exist_ok=True)
            for f in files:
                shutil.copy2(os.path.join(root, f), os.path.join(dest_root, f))
        print(f"[INFO] Saved adapter to {out}")


if __name__ == "__main__":
    main()
