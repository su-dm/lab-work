import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# --- Configuration ---
MODEL_ID = "Qwen/Qwen3-4B-Instruct"
OUTPUT_DIR = "./qwen_legal_multigpu"
MAX_SEQ_LENGTH = 262144  # Fixed: removed comma
WANDB_PROJECT = "qwen-legal"

def main():
    # 1. Initialize Accelerator/DeepSpeed implicitly via Trainer
    # Ensure BF16 is supported
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Dataset
    print("Loading Dataset...")
    ds = load_dataset("CJWeiss/LexSumm", "multilong")

    def format_and_tokenize(example):
        """Format chat template and tokenize properly"""
        # Robust column finder
        in_col = "input" if "input" in example else "source"
        out_col = "output" if "output" in example else "summary"

        # Format for Chat
        prompt = f"<|im_start|>system\nYou are a legal assistant.<|im_end|>\n<|im_start|>user\n{example[in_col]}<|im_end|>\n<|im_start|>assistant\n{example[out_col]}<|im_end|>"

        # Tokenize with proper truncation
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,  # DataCollator will handle padding dynamically
        )

        # Labels are the same as input_ids for causal LM
        tokenized["labels"] = tokenized["input_ids"].copy()

        return tokenized

    # Process and Filter by TOKEN count (not character count)
    ds = ds.map(format_and_tokenize, remove_columns=ds['train'].column_names)
    # Filter out examples that are too short (likely corrupted) or hit max length
    ds = ds.filter(lambda x: len(x['input_ids']) > 100 and len(x['input_ids']) <= MAX_SEQ_LENGTH)
    
    # 4. Load Model in BF16 for stability with ZeRO-2
    print("Loading Model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        use_cache=False  # Required for Gradient Checkpointing
    )
    
    # 5. PEFT / LoRA Config
    peft_config = LoraConfig(
        r=64,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        dropout=0.05
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. Data Collator for efficient dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )

    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Batch sizes - conservative for 200k context
        per_device_train_batch_size=1,  # Start with 1, increase if memory allows
        gradient_accumulation_steps=8,  # 4 GPUs * 1 batch * 8 accum = 32 effective batch
        per_device_eval_batch_size=1,

        # Learning
        learning_rate=1e-4,
        num_train_epochs=1,
        max_grad_norm=1.0,  # Gradient clipping for stability

        # Precision
        bf16=True,
        bf16_full_eval=True,

        # Memory optimization
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # More memory efficient

        # Logging
        logging_steps=10,
        logging_first_step=True,

        # Checkpointing Strategy
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,

        # Evaluation - metrics only, no generation
        eval_strategy="steps",
        eval_steps=200,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=False,  # Would require too much memory to keep best model

        # Reporting
        report_to="wandb",
        run_name="qwen-legal-4xH100-zero2",

        # Multi-GPU
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,  # Parallel data loading

        # DeepSpeed will be configured via external config file
        deepspeed="./experiments/009_text_summary/ds_config_zero2.json",
    )

    # 8. Trainer - metrics only, no generation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds['train'],
        eval_dataset=ds['validation'],
        data_collator=data_collator,
        # No callbacks - just standard loss/perplexity metrics
    )

    print("Starting Training...")
    trainer.train()
    
    print("Saving Final Model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    main()
