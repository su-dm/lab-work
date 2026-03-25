"""Single-document inference using HuggingFace Transformers + PEFT (no vLLM).

Supports one-shot summarisation and interactive follow-up chat.
Uses device_map="auto" for multi-GPU sharding via accelerate.

Usage:
    python infer2.py --help
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from infer_common import (
    DEFAULT_BASE_MODEL,
    MODEL_CACHE_DIR,
    create_arg_parser,
    resolve_checkpoint_path,
    run_inference,
)
from prompts import load_env


def parse_args():
    parser = create_arg_parser(
        description="Run inference on a document using Transformers + PEFT (no vLLM).",
        script_name="infer2.py",
    )
    return parser.parse_args()


def load_model(args):
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    base_model_id = DEFAULT_BASE_MODEL if args.checkpoint else args.model

    print(f"Loading tokenizer: {base_model_id}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
    )

    print(f"Loading model: {base_model_id}", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=str(MODEL_CACHE_DIR),
        trust_remote_code=True,
    )

    if args.checkpoint:
        from peft import PeftModel

        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        print(f"Loading LoRA adapter: {checkpoint_path}", file=sys.stderr)
        model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages, args) -> str:
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=bool(args.max_seq_length),
        max_length=args.max_seq_length,
    ).to(model.device)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
    )
    if args.temperature > 0:
        gen_kwargs["temperature"] = args.temperature

    output_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    load_env(PROJECT_DIR)

    model, tokenizer = load_model(args)

    def generate_fn(messages, _args):
        return generate(model, tokenizer, messages, _args)

    run_inference(args, generate_fn)


if __name__ == "__main__":
    main()
