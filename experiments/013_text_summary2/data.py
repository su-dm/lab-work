"""Dataset loading, formatting, and label masking for legal summarization training.

Pre-truncates very long inputs by character count before tokenization to avoid
tokenizing 1M+ token documents. Filters (not truncates) samples that exceed
max_seq_length or whose input alone consumes more than 90% of the context window.
"""

import logging

from datasets import load_dataset

from prompts import format_training_example

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 4
INPUT_BUDGET_RATIO = 0.90
PROMPT_OVERHEAD_TOKENS = 150
MIN_SAMPLE_TOKENS = 100


def load_and_prepare(config: dict, tokenizer) -> tuple:
    """Load dataset from config, tokenize with proper label masking.

    Returns (train_dataset, eval_dataset) ready for the Trainer.
    """
    ds_cfg = config["dataset"]
    ds = load_dataset(ds_cfg["name"], ds_cfg.get("config"))

    system_prompt = config["system_prompt"]
    max_len = config["max_seq_length"]
    input_col = ds_cfg["input_col"]
    output_col = ds_cfg["output_col"]

    max_input_tokens = int(max_len * INPUT_BUDGET_RATIO) - PROMPT_OVERHEAD_TOKENS
    max_input_chars = max_input_tokens * CHARS_PER_TOKEN_ESTIMATE

    def tokenize_example(example):
        user_text = example[input_col]
        if isinstance(user_text, list):
            user_text = "\n\n".join(user_text)

        assistant_text = example[output_col]
        if isinstance(assistant_text, list):
            assistant_text = assistant_text[0] if assistant_text else ""

        user_text = user_text[:max_input_chars]

        result = format_training_example(
            tokenizer, system_prompt, user_text, assistant_text
        )

        return result

    def is_within_budget(example):
        n = len(example["input_ids"])
        if n < MIN_SAMPLE_TOKENS or n > max_len:
            return False
        labels = example["labels"]
        input_tokens = sum(1 for l in labels if l == -100)
        if input_tokens > int(max_len * INPUT_BUDGET_RATIO):
            return False
        return True

    train_ds = ds["train"].map(
        tokenize_example,
        remove_columns=ds["train"].column_names,
        num_proc=1,
    )
    eval_ds = ds["validation"].map(
        tokenize_example,
        remove_columns=ds["validation"].column_names,
        num_proc=1,
    )

    train_before = len(train_ds)
    eval_before = len(eval_ds)

    train_ds = train_ds.filter(is_within_budget)
    eval_ds = eval_ds.filter(is_within_budget)

    logger.info(
        "Filtered training set: %d -> %d samples (dropped %d)",
        train_before, len(train_ds), train_before - len(train_ds),
    )
    logger.info(
        "Filtered eval set: %d -> %d samples (dropped %d)",
        eval_before, len(eval_ds), eval_before - len(eval_ds),
    )

    return train_ds, eval_ds
