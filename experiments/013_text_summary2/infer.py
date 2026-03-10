"""Single-document inference using vLLM with multi-GPU tensor parallelism.

Usage:
    # With a LoRA checkpoint
    python infer.py \
        --checkpoint train_results/001/checkpoints/checkpoint-400 \
        --system_prompt "You are a legal assistant..." \
        --input document.pdf

    # With base model (no adapter)
    python infer.py \
        --model "Qwen/Qwen3-4B-Instruct-2507" \
        --system_prompt "You are a legal assistant..." \
        --input document.txt

    # Save output to file
    python infer.py \
        --checkpoint train_results/001/checkpoints/checkpoint-400 \
        --system_prompt "You are a legal assistant..." \
        --input case.pdf \
        --output summary.txt
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from pdf_utils import extract_text
from prompts import build_messages

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a document")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Path to LoRA checkpoint directory")
    group.add_argument("--model", type=str, help="HuggingFace model ID (no adapter)")
    parser.add_argument("--system_prompt", type=str, required=True, help="System prompt text")
    parser.add_argument("--input", type=str, required=True, help="Path to .txt or .pdf input file")
    parser.add_argument("--output", type=str, help="Optional output file path (otherwise prints to stdout)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (default: auto-detect)")
    return parser.parse_args()


def detect_gpu_count() -> int:
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def run_inference(args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    input_text = extract_text(args.input)
    messages = build_messages(args.system_prompt, input_text)

    tp_size = args.tensor_parallel_size or detect_gpu_count()

    if args.checkpoint:
        checkpoint_path = str(Path(args.checkpoint).resolve())
        print(f"Loading base model: {BASE_MODEL_ID} (TP={tp_size})", file=sys.stderr)
        print(f"LoRA adapter: {checkpoint_path}", file=sys.stderr)

        llm = LLM(
            model=BASE_MODEL_ID,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            enable_lora=True,
            max_lora_rank=128,
            trust_remote_code=True,
        )
        lora_request = LoRARequest("adapter", 1, checkpoint_path)
    else:
        model_id = args.model
        print(f"Loading model: {model_id} (TP={tp_size})", file=sys.stderr)

        llm = LLM(
            model=model_id,
            tensor_parallel_size=tp_size,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        lora_request = None

    sampling_params = SamplingParams(
        temperature=args.temperature if args.temperature > 0 else 0,
        max_tokens=args.max_new_tokens,
    )

    tokenizer = llm.get_tokenizer()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"Input: {len(input_text)} chars, generating...", file=sys.stderr)

    if lora_request:
        outputs = llm.generate([prompt], sampling_params, lora_request=lora_request)
    else:
        outputs = llm.generate([prompt], sampling_params)

    result = outputs[0].outputs[0].text.strip()

    if args.output:
        Path(args.output).write_text(result)
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(result)


def main():
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
