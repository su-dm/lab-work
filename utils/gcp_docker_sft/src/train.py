import os
import sys

import torch
import wandb
from datasets import load_dataset
from huggingface_hub import login

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format, SFTTrainer
from util.secret_manager import get_secret

# using secret manager
hf_key = get_secret('HF_TOKEN')
wandb_key = get_secret('WANDB_API_KEY')

if not hf_key or not wandb_key:
    print(f" tokens are not sets hf {hf_key} wandb {wandb_key}")
    raise ValueError(" tokens are not set")

wandb.login(key=wandb_key)
login(hf_key)

base_model = "mistralai/Mistral-7B-v0.1"
new_model = "OrpoMistral-3-7B"

# Set torch dtype and attention implementation
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    torch_dtype = torch.float16
    attn_implementation = "eager"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="int8",
    bnb_8bit_compute_dtype=torch_dtype,
    bnb_8bit_use_double_quant=False
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

model_path_gcs = "/mnt/gcs/"
new_model_path_gcs = model_path_gcs + new_model

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map='auto',
    attn_implementation=attn_implementation,
    cache_dir=model_path_gcs,
    torch_dtype=torch_dtype,
)

model, tokenizer = setup_chat_format(model, tokenizer)

dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42)

def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
)

dataset = dataset.train_test_split(test_size=0.01)

orpo_args = ORPOConfig(
    learning_rate=8e-6,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=1024,
    beta=0.1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    eval_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./echoGPT-ORPO-Mistral",
    bf16=True,
    remove_unused_columns=False,
    push_to_hub=True,
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model(new_model_path_gcs)
