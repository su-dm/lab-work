"""Dataset loading, formatting, and label masking for legal summarization training."""

from datasets import load_dataset
from prompts import format_training_example


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

    def tokenize_example(example):
        user_text = example[input_col]
        if isinstance(user_text, list):
            user_text = "\n\n".join(user_text)

        assistant_text = example[output_col]
        if isinstance(assistant_text, list):
            assistant_text = assistant_text[0] if assistant_text else ""

        result = format_training_example(
            tokenizer, system_prompt, user_text, assistant_text
        )

        result["input_ids"] = result["input_ids"][:max_len]
        result["labels"] = result["labels"][:max_len]

        return result

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

    train_ds = train_ds.filter(lambda x: 100 < len(x["input_ids"]) <= max_len)
    eval_ds = eval_ds.filter(lambda x: 100 < len(x["input_ids"]) <= max_len)

    return train_ds, eval_ds
