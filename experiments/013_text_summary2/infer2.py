"""Single-document inference using HuggingFace Transformers + PEFT (no vLLM).

Supports one-shot summarisation and interactive follow-up chat.
Uses device_map="auto" for multi-GPU sharding via accelerate.

Usage:
    python infer2.py --help
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from pdf_utils import extract_text
from prompts import build_messages, load_env, resolve_prompt

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SYSTEM_PROMPT = (
    "You are a legal research assistant. Summarize the following legal case."
)
MODEL_CACHE_DIR = PROJECT_DIR / ".model_cache"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a document using Transformers + PEFT (no vLLM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal — uses default model and prompt
  python infer2.py --input case_filing.pdf

  # Interactive follow-up chat after initial summary
  python infer2.py --input case_filing.pdf --interactive

  # LoRA adapter from local checkpoint + PDF input
  python infer2.py \\
      --checkpoint train_results/001/checkpoints/checkpoint-400 \\
      --system_prompt prompts/legal_brief.txt \\
      --input case_filing.pdf

  # LoRA adapter from HuggingFace Hub
  python infer2.py \\
      --checkpoint user/my-lora-adapter \\
      --input case_filing.pdf

  # Base model + text input + save output
  python infer2.py \\
      --model "Qwen/Qwen3.5-4B" \\
      --input document.txt \\
      --output summary.txt
""",
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--checkpoint", type=str,
        help="LoRA adapter: local checkpoint path or HuggingFace repo ID (e.g. 'user/my-lora')",
    )
    group.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt: inline text OR path to a .txt file (default: built-in legal summary prompt)",
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to input document (.txt or .pdf)",
    )
    parser.add_argument(
        "--output", type=str,
        help="Save initial output to this file (default: print to stdout)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive chat after the initial summary",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6,
        help="Sampling temperature. 0 = greedy (default: 0.6)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=None,
        help="Truncate input to this many tokens if set (default: no truncation)",
    )
    return parser.parse_args()


def load_model(args):
    from peft import PeftModel as _PeftModel

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
        checkpoint_path = args.checkpoint
        if Path(checkpoint_path).exists():
            checkpoint_path = str(Path(checkpoint_path).resolve())
        print(f"Loading LoRA adapter: {checkpoint_path}", file=sys.stderr)
        model = _PeftModel.from_pretrained(model, checkpoint_path)

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


def interactive_loop(model, tokenizer, conversation: list[dict], args):
    """Continue chatting with the model after the initial summary."""
    print("\n--- Interactive mode (type 'quit' or 'exit' to stop) ---\n", file=sys.stderr)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.", file=sys.stderr)
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        conversation.append({"role": "user", "content": user_input})

        reply = generate(model, tokenizer, conversation, args)
        conversation.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")


def run_inference(args):
    system_prompt = resolve_prompt(args.system_prompt)
    input_text = extract_text(args.input)

    model, tokenizer = load_model(args)

    conversation = build_messages(system_prompt, input_text)

    print(f"Input: {len(input_text)} chars, generating...", file=sys.stderr)
    result = generate(model, tokenizer, conversation, args)
    conversation.append({"role": "assistant", "content": result})

    if args.output:
        Path(args.output).write_text(result)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(result)

    if args.interactive:
        interactive_loop(model, tokenizer, conversation, args)


def main():
    args = parse_args()
    load_env(PROJECT_DIR)
    run_inference(args)


if __name__ == "__main__":
    main()
