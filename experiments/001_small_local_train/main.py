#!/usr/bin/env python3
import argparse
from typing import Optional, List, Dict, Any

import importlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


DEFAULT_MODEL_ID = "microsoft/Phi-4-mini-instruct"


def pick_dtype(dtype_str: str = "auto") -> torch.dtype:
    """Select a dtype based on user preference and hardware."""
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str and dtype_str.lower() in mapping:
        return mapping[dtype_str.lower()]

    # auto selection
    if torch.cuda.is_available():
        bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if bf16_supported else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(
    model_id: str = DEFAULT_MODEL_ID,
    device_map: str = "auto",
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load HF model + tokenizer with safe defaults."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    if dtype is None:
        dtype = pick_dtype("auto")

    # Decide whether to use device_map depending on accelerate availability
    accelerate_available = importlib.util.find_spec("accelerate") is not None
    use_device_map = device_map and device_map != "none" and accelerate_available

    if device_map and device_map != "none" and not accelerate_available:
        print("[Info] 'accelerate' not installed; ignoring device_map and loading on a single device.")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=(device_map if use_device_map else None),
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )

    # If not using device_map, move to a single best device
    if not use_device_map:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        model.to(device)

    # Ensure pad_token_id is set for generation
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def apply_chat_template(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages using tokenizer's chat template if available."""
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    # Fallback if no chat template is available
    sys_msgs = [m["content"] for m in messages if m.get("role") == "system"]
    user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
    system = f"System: {sys_msgs[-1]}\n" if sys_msgs else ""
    user = f"User: {user_msgs[-1] if user_msgs else ''}\n"
    return f"{system}{user}Assistant: "


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    stop_token_id: Optional[int] = None,
) -> str:
    """Generate text from a plain prompt string."""
    if seed is not None:
        set_seed(seed)

    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    do_sample = temperature is not None and temperature > 0.0
    eos_id = stop_token_id if stop_token_id is not None else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            do_sample=do_sample,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )

    # Only decode newly generated tokens
    new_tokens = outputs[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def chat_completion(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    user_text: str,
    system_text: Optional[str] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 0.9,
    seed: Optional[int] = None,
) -> str:
    """Single-turn chat completion (stateless)."""
    messages: List[Dict[str, str]] = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
    return generate_text(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop_token_id=tokenizer.eos_token_id,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-turn REPL for Phi-4-mini-instruct")
    p.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID, help="HF model repo id")
    p.add_argument("--device-map", type=str, default="auto", help="Device map (e.g., auto|none)")
    p.add_argument("--dtype", type=str, default="auto", help="auto|fp32|fp16|bf16")
    p.add_argument("--system", type=str, default=None, help="Optional system prompt")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None)
    # Default to False to avoid dynamic module requiring newer Transformers symbols
    p.add_argument("--trust-remote-code", dest="trust_remote_code", action="store_true", default=False,
                   help="Enable trusting remote code from the model repo (default: disabled)")
    p.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false",
                   help="Disable trusting remote code (default)")
    return p.parse_args()


def repl(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_text: Optional[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: Optional[int],
) -> None:
    print("Phi-4-mini-instruct (single-turn). Type '/exit' to quit.")
    if system_text:
        print(f"[System] {system_text}")
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if user_text.lower() in {"/exit", "exit", "quit", "/quit", ":q", "/q"}:
            print("Bye.")
            break
        if not user_text:
            continue

        # Stateless: each message is a new conversation
        output = chat_completion(
            model=model,
            tokenizer=tokenizer,
            user_text=user_text,
            system_text=system_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        print(f"Assistant> {output}\n")


def main() -> None:
    args = parse_args()
    dtype = pick_dtype(args.dtype)
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        device_map=args.device_map,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    repl(
        model=model,
        tokenizer=tokenizer,
        system_text=args.system,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()