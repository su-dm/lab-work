"""Shared CLI, orchestration, and constants for inference scripts.

Both the vLLM backend (infer.py) and the Transformers+PEFT backend
(infer2.py) import from here to avoid duplicating arg parsing,
interactive chat, and the run loop.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from pdf_utils import extract_text
from prompts import build_messages, load_env, resolve_prompt

DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B"
DEFAULT_SYSTEM_PROMPT = (
    "You are a legal research assistant. Summarize the following legal case."
)
MODEL_CACHE_DIR = PROJECT_DIR / ".model_cache"

GenerateFn = Callable[[list[dict], argparse.Namespace], str]


def create_arg_parser(description: str, script_name: str) -> argparse.ArgumentParser:
    """Return a parser pre-loaded with all backend-agnostic arguments."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Minimal — uses default model and prompt
  python {script_name} --input case_filing.pdf

  # Interactive follow-up chat after initial summary
  python {script_name} --input case_filing.pdf --interactive

  # LoRA adapter from local checkpoint + PDF input
  python {script_name} \\
      --checkpoint train_results/001/checkpoints/checkpoint-400 \\
      --system_prompt prompts/legal_brief.txt \\
      --input case_filing.pdf

  # LoRA adapter from HuggingFace Hub
  python {script_name} \\
      --checkpoint user/my-lora-adapter \\
      --input case_filing.pdf

  # Base model + text input + save output
  python {script_name} \\
      --model "{DEFAULT_BASE_MODEL}" \\
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
        help="Max context length in tokens (default: model's native length)",
    )

    return parser


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Return absolute path if *checkpoint* is a local dir, otherwise pass through (Hub ID)."""
    p = Path(checkpoint)
    if p.exists():
        return str(p.resolve())
    return checkpoint


def interactive_loop(
    generate_fn: GenerateFn,
    conversation: list[dict],
    args: argparse.Namespace,
):
    """Continue chatting with the model after the initial summary.

    *generate_fn(messages, args)* is a backend-specific callable that
    returns the assistant reply as a string.
    """
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

        reply = generate_fn(conversation, args)
        conversation.append({"role": "assistant", "content": reply})

        print(f"\nAssistant: {reply}\n")


def run_inference(args: argparse.Namespace, generate_fn: GenerateFn):
    """Shared inference orchestration: read input, generate, save, chat.

    Call this from the backend's ``main()`` after loading the model and
    constructing *generate_fn*.
    """
    system_prompt = resolve_prompt(args.system_prompt)
    input_text = extract_text(args.input)

    conversation = build_messages(system_prompt, input_text)

    print(f"Input: {len(input_text)} chars, generating...", file=sys.stderr)
    result = generate_fn(conversation, args)
    conversation.append({"role": "assistant", "content": result})

    if args.output:
        Path(args.output).write_text(result)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(result)

    if args.interactive:
        interactive_loop(generate_fn, conversation, args)
