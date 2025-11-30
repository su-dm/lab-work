"""
Comprehensive evaluation script for fine-tuned Qwen3 legal summarization model.
Runs full 200k context evaluation with qualitative and quantitative metrics.

Usage:
    python eval_checkpoint.py --checkpoint ./qwen_legal_multigpu/checkpoint-400
"""

import argparse
import torch
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
import numpy as np
from rouge_score import rouge_scorer

# Configuration
MODEL_ID = "Qwen/Qwen3-4B-Instruct"
DATASET_NAME = "CJWeiss/LexSumm"
DATASET_CONFIG = "multilong"
MAX_EVAL_SAMPLES = 50  # Number of validation samples to evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint to evaluate (e.g., ./qwen_legal_multigpu/checkpoint-400)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50,
        help="Number of validation samples to evaluate"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-legal-eval",
        help="WandB project name for logging"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable WandB logging"
    )
    return parser.parse_args()


def load_model_and_tokenizer(checkpoint_path):
    """Load the base model and apply LoRA adapters"""
    print(f"Loading base model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Automatically distribute across available GPUs
        attn_implementation="flash_attention_2",
    )

    print(f"Loading LoRA adapters from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    return model, tokenizer


def prepare_dataset(tokenizer, num_samples):
    """Load and prepare validation dataset"""
    print(f"Loading dataset: {DATASET_NAME}/{DATASET_CONFIG}")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)

    validation_set = ds['validation'].select(range(min(num_samples, len(ds['validation']))))

    # Extract input and output columns
    def extract_fields(example):
        in_col = "input" if "input" in example else "source"
        out_col = "output" if "output" in example else "summary"
        return {
            "input_text": example[in_col],
            "reference_summary": example[out_col]
        }

    validation_set = validation_set.map(extract_fields)
    return validation_set


def generate_summary(model, tokenizer, input_text, max_new_tokens, temperature):
    """Generate summary for a single input"""
    # Format prompt
    prompt = f"<|im_start|>system\nYou are a legal assistant.<|im_end|>\n<|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant\n" in full_output:
        generated_summary = full_output.split("assistant\n")[-1].strip()
    else:
        generated_summary = full_output

    return generated_summary


def calculate_rouge(predictions, references):
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
    }


def main():
    args = parse_args()

    # Initialize WandB
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=f"eval-{args.checkpoint.split('/')[-1]}",
            config=vars(args)
        )

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # Load dataset
    validation_set = prepare_dataset(tokenizer, args.num_samples)

    print(f"\nEvaluating on {len(validation_set)} samples...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}\n")

    # Generate summaries
    predictions = []
    references = []
    qualitative_examples = []

    for i, example in enumerate(tqdm(validation_set, desc="Generating summaries")):
        input_text = example['input_text']
        reference_summary = example['reference_summary']

        # Generate
        generated_summary = generate_summary(
            model,
            tokenizer,
            input_text,
            args.max_new_tokens,
            args.temperature
        )

        predictions.append(generated_summary)
        references.append(reference_summary)

        # Save first 5 examples for qualitative analysis
        if i < 5:
            qualitative_examples.append({
                "input_snippet": input_text[:500] + "...",
                "reference": reference_summary[:500] + "..." if len(reference_summary) > 500 else reference_summary,
                "generated": generated_summary[:500] + "..." if len(generated_summary) > 500 else generated_summary,
            })

    # Calculate metrics
    print("\nCalculating ROUGE scores...")
    rouge_scores = calculate_rouge(predictions, references)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {len(predictions)}")
    print(f"\nROUGE Scores:")
    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print("="*60)

    # Log to WandB
    if not args.no_wandb:
        # Log metrics
        wandb.log({
            "eval/rouge1": rouge_scores['rouge1'],
            "eval/rouge2": rouge_scores['rouge2'],
            "eval/rougeL": rouge_scores['rougeL'],
            "eval/num_samples": len(predictions),
        })

        # Log qualitative examples
        table = wandb.Table(columns=["Input (Snippet)", "Reference", "Generated"])
        for ex in qualitative_examples:
            table.add_data(ex["input_snippet"], ex["reference"], ex["generated"])

        wandb.log({"qualitative_examples": table})

        print("\nResults logged to WandB")
        wandb.finish()

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
