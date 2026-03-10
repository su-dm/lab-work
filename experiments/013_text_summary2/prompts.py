"""Shared prompt template logic used by train, evaluate, and infer."""

import os
from pathlib import Path

import yaml


def load_env(project_dir: Path | None = None):
    """Load .env file from project directory into os.environ.

    Doesn't overwrite variables that are already set, so real env vars
    take precedence over .env values.
    """
    if project_dir is None:
        project_dir = Path(__file__).resolve().parent
    env_file = project_dir / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_prompt(value: str) -> str:
    """Resolve a system prompt value.

    If ``value`` looks like a file path that exists, read and return its
    contents.  Otherwise return the string as-is. This lets CLI args like
    ``--system_prompt prompts/legal_v2.txt`` work alongside inline strings.
    """
    candidate = Path(value)
    if candidate.is_file():
        return candidate.read_text(encoding="utf-8").strip()
    return value.strip()


def build_messages(system_prompt: str, user_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_text},
    ]


def format_training_example(
    tokenizer, system_prompt: str, user_text: str, assistant_text: str
) -> dict:
    """Build a full training example with label masking.

    Returns tokenized input_ids and labels where prompt tokens are masked
    with -100 so the model only learns to predict the assistant response.
    """
    prompt_messages = build_messages(system_prompt, user_text)
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    full_messages = prompt_messages + [
        {"role": "assistant", "content": assistant_text}
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    return {"input_ids": full_ids, "labels": labels}


def get_next_run_dir(base_dir: Path) -> Path:
    """Find the next numbered subdirectory (001, 002, ...) under base_dir."""
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    next_num = int(existing[-1].name) + 1 if existing else 1
    run_dir = base_dir / f"{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
