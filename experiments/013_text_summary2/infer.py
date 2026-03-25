"""Single-document inference using vLLM with multi-GPU tensor parallelism.

Supports one-shot summarisation and interactive follow-up chat.

Usage:
    python infer.py --help
"""

import sys
from pathlib import Path

import torch

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
        description="Run inference on a document using vLLM (multi-GPU tensor parallelism).",
        script_name="infer.py",
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
    if args.max_seq_length:
        common_kwargs["max_model_len"] = args.max_seq_length

    if args.checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
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


def main():
    from vllm.lora.request import LoRARequest

    args = parse_args()
    load_env(PROJECT_DIR)

    tp_size = args.tensor_parallel_size or detect_gpu_count()
    llm = load_model(args, tp_size)

    lora_request = None
    if args.checkpoint:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
        lora_request = LoRARequest("adapter", 1, checkpoint_path)

    def generate_fn(messages, _args):
        return generate(llm, messages, _args, lora_request=lora_request)

    run_inference(args, generate_fn)


if __name__ == "__main__":
    main()
