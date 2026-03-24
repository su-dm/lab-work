"""Single-document inference using vLLM with multi-GPU tensor parallelism.

Supports one-shot summarisation and interactive follow-up chat.

Usage:
    python infer.py --help
"""

import argparse
import sys
from pathlib import Path

import torch

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
        description="Run inference on a document using a fine-tuned or base model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Minimal — uses default model and prompt
  python infer.py --input case_filing.pdf

  # Interactive follow-up chat after initial summary
  python infer.py --input case_filing.pdf --interactive

  # LoRA adapter from local checkpoint + PDF input
  python infer.py \\
      --checkpoint train_results/001/checkpoints/checkpoint-400 \\
      --system_prompt prompts/legal_brief.txt \\
      --input case_filing.pdf

  # LoRA adapter from HuggingFace Hub
  python infer.py \\
      --checkpoint user/my-lora-adapter \\
      --input case_filing.pdf

  # Base model + text input + save output
  python infer.py \\
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
        "--tensor_parallel_size", type=int, default=None,
        help="Number of GPUs for tensor parallelism (default: all available)",
    )
    return parser.parse_args()


def detect_gpu_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def load_model(args, tp_size: int):
    from vllm import LLM

    base_model = DEFAULT_BASE_MODEL
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        trust_remote_code=True,
        download_dir=str(MODEL_CACHE_DIR),
    )

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if Path(checkpoint_path).exists():
            checkpoint_path = str(Path(checkpoint_path).resolve())
        print(f"Loading base model: {base_model} (TP={tp_size})", file=sys.stderr)
        print(f"LoRA adapter: {checkpoint_path}", file=sys.stderr)
        llm = LLM(
            model=base_model,
            enable_lora=True,
            max_lora_rank=128,
            **common_kwargs,
        )
    else:
        model_id = args.model
        print(f"Loading model: {model_id} (TP={tp_size})", file=sys.stderr)
        llm = LLM(model=model_id, **common_kwargs)

    return llm


def generate(llm, messages, args, lora_request=None):
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0,
        max_tokens=args.max_new_tokens,
    )

    if lora_request:
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([prompt], sampling_params)

    return outputs[0].outputs[0].text.strip()


def interactive_loop(llm, conversation: list[dict], args, lora_request=None):
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

        reply = generate(llm, conversation, args, lora_request=lora_request)
        conversation.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")


def run_inference(args):
    from vllm.lora.request import LoRARequest

    system_prompt = resolve_prompt(args.system_prompt)
    input_text = extract_text(args.input)

    tp_size = args.tensor_parallel_size or detect_gpu_count()
    llm = load_model(args, tp_size)

    lora_request = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if Path(checkpoint_path).exists():
            checkpoint_path = str(Path(checkpoint_path).resolve())
        lora_request = LoRARequest("adapter", 1, checkpoint_path)

    conversation = build_messages(system_prompt, input_text)

    print(f"Input: {len(input_text)} chars, generating...", file=sys.stderr)
    result = generate(llm, conversation, args, lora_request=lora_request)
    conversation.append({"role": "assistant", "content": result})

    if args.output:
        Path(args.output).write_text(result)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(result)

    if args.interactive:
        interactive_loop(llm, conversation, args, lora_request=lora_request)


def main():
    args = parse_args()
    load_env(PROJECT_DIR)
    run_inference(args)


if __name__ == "__main__":
    main()
