import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl.trainer.sft_trainer import SFTTrainer
from trl import SFTConfig
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" 
MAX_SEQ_LENGTH = 32768 # H100 can handle full 32k context
DTYPE = torch.bfloat16
LOAD_IN_4BIT = False

# --- 1. Load Model & Tokenizer ---
print(f"Loading {MODEL_NAME} with Unsloth (Native BF16)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = DTYPE,
    load_in_4bit = LOAD_IN_4BIT,
)

# --- 2. Add LoRA Adapters ---
# We target all linear layers for maximum effectiveness on complex legal reasoning
model = FastLanguageModel.get_peft_model(
    model,
    r = 64, # H100 has memory to spare, higher rank = better reasoning capture
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32, # Typically alpha = r/2 or r
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True, # Better stability for higher ranks
    loftq_config = None, 
)

# --- 3. Data Preparation (Multi-LexSum) ---
print("Loading LexSumm dataset...")
eval_dataset = load_dataset("CJWeiss/LexSumm", "multilong", split="validation")
train_dataset = load_dataset("CJWeiss/LexSumm", "multilong", split="train")

# Define the prompt template (ChatML format for Qwen)
legal_prompt_style = """<|im_start|>system
You are a highly skilled legal research assistant specialized in North American law. 
Your task is to summarize the following legal case into a comprehensive brief. 
Focus on the procedural history, key facts, legal issues, and the court's holding.<|im_end|>
<|im_start|>user
Case Text:
{case_text}<|im_end|>
<|im_start|>assistant
{summary}<|im_end|>"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    inputs = examples["input"]
    outputs = examples["output"] # Using 'long' summary as the proxy for a brief
    texts = []
    
    for input_doc_list, output_text in zip(inputs, outputs):
        # Multi-LexSum stores sources as a list of strings (documents). 
        # We join them, but truncate early to avoid massive string operations if possible.
        full_case_text = "\n\n".join(input_doc_list)
        
        # Format using the template
        text = legal_prompt_style.format(
            case_text=full_case_text, 
            summary=output_text
        ) + EOS_TOKEN
        texts.append(text)
        
    return { "text" : texts, }

# We only need the generated "text" column for the SFTTrainer
train_dataset = train_dataset.map(
    formatting_prompts_func, 
    batched = True,
    remove_columns = train_dataset.column_names
)
eval_dataset = eval_dataset.map(
    formatting_prompts_func,
    batched = True, 
    remove_columns = eval_dataset.column_names
)

# --- 4. Training Arguments ---
print("Starting training...")
trainer_config = SFTConfig(
    output_dir = "outputs",
    max_length = MAX_SEQ_LENGTH,
    dataset_text_field = "text",
    dataset_num_proc = 1, # bugs if you do multi-proc
    packing = False,
    
    # Training Parameters
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 4,
    gradient_accumulation_steps = 4, # Effective batch size = 16
    warmup_steps = 100, # Increased for larger batch size
    max_steps = 600, # Roughly 1 epoch given the batch size
    learning_rate = 1e-4, # Slightly lower LR for full bf16/higher rank
    fp16 = False, # Disable FP16
    bf16 = True, # Enable BF16
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "cosine", # Cosine usually performs better for full convergence
    seed = 3407,
    
    # Validation & Logging
    eval_strategy = "steps",
    eval_steps = 50,
    save_strategy = "steps",
    save_steps = 50,
    report_to = "wandb",
    run_name = "qwen3-4b-legal-brief-h100",
)

trainer = SFTTrainer(
    model = model,
    processing_class = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = trainer_config,
)

# --- 5. Train & Save ---
trainer_stats = trainer.train()

print("Saving model to 'qwen_legal_brief_lora'...")
model.save_pretrained("qwen_legal_brief_lora")
tokenizer.save_pretrained("qwen_legal_brief_lora")

# Optional: Save to GGUF format for use in Ollama/llama.cpp
# model.save_pretrained_gguf("qwen_legal_brief_gguf", tokenizer, quantization_method = "q4_k_m")
