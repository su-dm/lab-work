"""Multi-tier evaluation for legal summarization checkpoints and prompts.

Supports four evaluation tiers:
  1. Overlap metrics (ROUGE, BERTScore)
  2. Format compliance (section presence, length analysis)
  3. NLI faithfulness (hallucination detection)
  4. LLM-as-judge via Claude API

Usage:
    # From a prompt_config.yaml
    python evaluate.py --config prompt_results/001/prompt_config.yaml

    # Inline args (auto-creates next prompt_results/NNN/ dir)
    python evaluate.py \
        --checkpoint train_results/001/checkpoints/checkpoint-400 \
        --system_prompt "You are a legal assistant..." \
        --num_samples 50

    # Enable Claude-as-judge (costs API credits)
    python evaluate.py --config prompt_results/001/prompt_config.yaml --llm_judge
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from prompts import build_messages, get_next_run_dir


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate legal summarization checkpoint")
    parser.add_argument("--config", type=str, help="Path to prompt_config.yaml")
    parser.add_argument("--checkpoint", type=str, help="Path to LoRA checkpoint or HF model ID")
    parser.add_argument("--system_prompt", type=str, help="System prompt text")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--dataset_name", type=str, default="CJWeiss/LexSumm")
    parser.add_argument("--dataset_config", type=str, default="multilong")
    parser.add_argument("--llm_judge", action="store_true", help="Enable Claude-as-judge (tier 4)")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="legal-summary")
    parser.add_argument("--expected_sections", nargs="+",
                        default=["Procedural History", "Key Facts", "Legal Issues", "Holding"])
    return parser.parse_args()


def load_eval_config(args) -> dict:
    """Merge YAML config (if given) with CLI overrides into a single dict."""
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    return {
        "checkpoint": args.checkpoint or cfg.get("checkpoint", ""),
        "system_prompt": args.system_prompt or cfg.get("system_prompt", "You are a legal assistant."),
        "dataset_name": args.dataset_name if args.dataset_name != "CJWeiss/LexSumm"
                        else cfg.get("dataset", {}).get("name", "CJWeiss/LexSumm"),
        "dataset_config": args.dataset_config if args.dataset_config != "multilong"
                          else cfg.get("dataset", {}).get("config", "multilong"),
        "num_samples": args.num_samples if args.num_samples != 50
                       else cfg.get("dataset", {}).get("num_samples", 50),
        "max_new_tokens": args.max_new_tokens if args.max_new_tokens != 2048
                          else cfg.get("generation", {}).get("max_new_tokens", 2048),
        "temperature": args.temperature if args.temperature != 0.6
                       else cfg.get("generation", {}).get("temperature", 0.6),
        "expected_sections": args.expected_sections or cfg.get("expected_sections",
                             ["Procedural History", "Key Facts", "Legal Issues", "Holding"]),
        "llm_judge": args.llm_judge,
        "no_wandb": args.no_wandb,
        "wandb_project": args.wandb_project,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"


def load_model_and_tokenizer(checkpoint_path: str):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_lora_checkpoint = (Path(checkpoint_path) / "adapter_config.json").exists()

    if is_lora_checkpoint:
        print(f"Loading base model: {BASE_MODEL_ID}")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        print(f"Applying LoRA adapters from: {checkpoint_path}")
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        print(f"Loading model directly: {checkpoint_path}")
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_summary(model, tokenizer, system_prompt: str, input_text: str,
                     max_new_tokens: int, temperature: float) -> str:
    messages = build_messages(system_prompt, input_text)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Tier 1: ROUGE + BERTScore
# ---------------------------------------------------------------------------

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)
    return {"rouge1": np.mean(r1), "rouge2": np.mean(r2), "rougeL": np.mean(rl)}


def compute_bertscore(predictions: list[str], references: list[str]) -> dict:
    from bert_score import score as bert_score_fn
    P, R, F = bert_score_fn(predictions, references, lang="en", verbose=False)
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F.mean().item(),
    }


# ---------------------------------------------------------------------------
# Tier 2: Format compliance
# ---------------------------------------------------------------------------

def check_format_compliance(predictions: list[str], expected_sections: list[str]) -> dict:
    section_counts = {s: 0 for s in expected_sections}
    full_compliance_count = 0

    for pred in predictions:
        pred_lower = pred.lower()
        all_present = True
        for section in expected_sections:
            if section.lower() in pred_lower:
                section_counts[section] += 1
            else:
                all_present = False
        if all_present:
            full_compliance_count += 1

    n = len(predictions)
    return {
        "full_compliance_rate": full_compliance_count / n if n else 0,
        "section_rates": {s: c / n for s, c in section_counts.items()} if n else {},
    }


def compute_length_stats(predictions: list[str], references: list[str], tokenizer) -> dict:
    pred_lens = [len(tokenizer.encode(p)) for p in predictions]
    ref_lens = [len(tokenizer.encode(r)) for r in references]
    return {
        "avg_pred_tokens": np.mean(pred_lens),
        "avg_ref_tokens": np.mean(ref_lens),
        "pred_ref_length_ratio": np.mean(pred_lens) / np.mean(ref_lens) if np.mean(ref_lens) > 0 else 0,
    }


def compute_extractive_coverage(predictions: list[str], sources: list[str], ngram_size: int = 4) -> dict:
    """Measure what fraction of generated n-grams appear verbatim in the source."""
    ratios = []
    for pred, src in zip(predictions, sources):
        pred_words = pred.lower().split()
        src_lower = src.lower()
        if len(pred_words) < ngram_size:
            ratios.append(0.0)
            continue
        ngrams = [" ".join(pred_words[i:i+ngram_size]) for i in range(len(pred_words) - ngram_size + 1)]
        copied = sum(1 for ng in ngrams if ng in src_lower)
        ratios.append(copied / len(ngrams) if ngrams else 0.0)
    return {"extractive_coverage_4gram": np.mean(ratios)}


# ---------------------------------------------------------------------------
# Tier 3: NLI faithfulness
# ---------------------------------------------------------------------------

def compute_faithfulness(predictions: list[str], sources: list[str]) -> dict:
    """Check if summary sentences are entailed by the source using NLI."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print("Warning: sentence-transformers not installed, skipping faithfulness check")
        return {"faithfulness_score": None, "entailed_ratio": None}

    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cpu")

    entailment_scores = []
    entailed_counts = []
    total_sents = []

    for pred, src in tqdm(zip(predictions, sources), desc="Faithfulness", total=len(predictions)):
        sentences = [s.strip() for s in re.split(r'[.!?]+', pred) if len(s.strip()) > 10]
        if not sentences:
            continue

        src_truncated = src[:4096]
        pairs = [(src_truncated, sent) for sent in sentences]
        scores = nli_model.predict(pairs)

        entailment_idx = 0
        for score in scores:
            if isinstance(score, np.ndarray):
                entailment_scores.append(float(score[entailment_idx]))
                entailed_counts.append(1 if np.argmax(score) == entailment_idx else 0)
            else:
                entailment_scores.append(float(score))
                entailed_counts.append(1 if float(score) > 0.5 else 0)
        total_sents.append(len(sentences))

    return {
        "faithfulness_score": np.mean(entailment_scores) if entailment_scores else None,
        "entailed_ratio": np.mean(entailed_counts) if entailed_counts else None,
    }


# ---------------------------------------------------------------------------
# Tier 4: LLM-as-judge (Claude)
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating a legal case summary. Rate the generated summary compared to the reference on four dimensions (1-5 scale).

Source document (first 3000 chars):
{source}

Reference summary:
{reference}

Generated summary:
{generated}

Rate each dimension with a score from 1 (poor) to 5 (excellent) and a brief justification:

1. Accuracy - Does it correctly state the holding, facts, and legal reasoning?
2. Completeness - Are all key aspects of the case covered?
3. Format - Does it follow a structured legal brief format?
4. Conciseness - Is it appropriately concise without losing key information?

Respond in this exact JSON format:
{{"accuracy": {{"score": N, "reason": "..."}}, "completeness": {{"score": N, "reason": "..."}}, "format": {{"score": N, "reason": "..."}}, "conciseness": {{"score": N, "reason": "..."}}}}"""


def run_llm_judge(predictions: list[str], references: list[str], sources: list[str]) -> dict:
    try:
        import anthropic
    except ImportError:
        print("Warning: anthropic package not installed, skipping LLM judge")
        return {}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set, skipping LLM judge")
        return {}

    client = anthropic.Anthropic(api_key=api_key)
    all_scores = {"accuracy": [], "completeness": [], "format": [], "conciseness": []}
    individual_results = []

    for i, (pred, ref, src) in enumerate(tqdm(
        zip(predictions, references, sources), desc="LLM Judge", total=len(predictions)
    )):
        prompt = JUDGE_PROMPT.format(
            source=src[:3000],
            reference=ref[:2000],
            generated=pred[:2000],
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                for dim in all_scores:
                    if dim in result and "score" in result[dim]:
                        all_scores[dim].append(result[dim]["score"])
                individual_results.append({"sample_idx": i, **result})
        except Exception as e:
            print(f"  Warning: LLM judge failed for sample {i}: {e}")
            continue

    avg_scores = {
        f"judge_{dim}": np.mean(scores) if scores else None
        for dim, scores in all_scores.items()
    }
    all_dims = [s for scores in all_scores.values() for s in scores]
    avg_scores["judge_overall"] = np.mean(all_dims) if all_dims else None
    avg_scores["judge_individual_results"] = individual_results

    return avg_scores


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(run_dir: Path, eval_cfg: dict, metrics: dict,
                  predictions: list[str], references: list[str], sources: list[str],
                  wandb_url: str, elapsed: float):
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    lines = [
        "=== Evaluation Summary ===",
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Run ID: prompt-{run_dir.name}",
        f"WandB Run: {wandb_url}",
        "",
        f"Checkpoint: {eval_cfg['checkpoint']}",
        f"System Prompt: \"{eval_cfg['system_prompt'][:80]}...\"" if len(eval_cfg['system_prompt']) > 80
            else f"System Prompt: \"{eval_cfg['system_prompt']}\"",
        f"Dataset: {eval_cfg['dataset_name']} ({eval_cfg['dataset_config']}), {eval_cfg['num_samples']} samples",
        f"Duration: {hours}h {minutes}m {seconds}s",
        "",
        "--- Automated Metrics ---",
        f"ROUGE-1: {metrics.get('rouge1', 'N/A'):.4f}" if metrics.get('rouge1') is not None else "ROUGE-1: N/A",
        f"ROUGE-2: {metrics.get('rouge2', 'N/A'):.4f}" if metrics.get('rouge2') is not None else "ROUGE-2: N/A",
        f"ROUGE-L: {metrics.get('rougeL', 'N/A'):.4f}" if metrics.get('rougeL') is not None else "ROUGE-L: N/A",
        f"BERTScore F1: {metrics.get('bertscore_f1', 'N/A'):.4f}" if metrics.get('bertscore_f1') is not None else "BERTScore F1: N/A",
        "",
        "--- Format Compliance ---",
    ]

    fc = metrics.get("full_compliance_rate")
    n = eval_cfg["num_samples"]
    if fc is not None:
        lines.append(f"Has all required sections: {int(fc * n)}/{n} ({fc:.0%})")
        for section, rate in metrics.get("section_rates", {}).items():
            lines.append(f"  - {section}: {int(rate * n)}/{n}")
    else:
        lines.append("N/A")

    lines.append(f"Avg summary length: {metrics.get('avg_pred_tokens', 'N/A'):.0f} tokens" if metrics.get('avg_pred_tokens') else "")
    lines.append(f"Avg reference length: {metrics.get('avg_ref_tokens', 'N/A'):.0f} tokens" if metrics.get('avg_ref_tokens') else "")
    lines.append(f"Extractive coverage (4-gram): {metrics.get('extractive_coverage_4gram', 'N/A'):.2%}" if metrics.get('extractive_coverage_4gram') is not None else "")

    lines.append("")
    lines.append("--- Faithfulness (NLI) ---")
    fs = metrics.get("faithfulness_score")
    er = metrics.get("entailed_ratio")
    lines.append(f"Avg entailment score: {fs:.4f}" if fs is not None else "Avg entailment score: N/A")
    lines.append(f"Entailed sentence ratio: {er:.2%}" if er is not None else "Entailed sentence ratio: N/A")

    if eval_cfg["llm_judge"]:
        lines.append("")
        lines.append("--- LLM Judge (Claude) ---")
        for dim in ["accuracy", "completeness", "format", "conciseness"]:
            key = f"judge_{dim}"
            val = metrics.get(key)
            lines.append(f"{dim.title()}: {val:.1f}/5" if val is not None else f"{dim.title()}: N/A")
        overall = metrics.get("judge_overall")
        lines.append(f"Overall: {overall:.1f}/5" if overall is not None else "Overall: N/A")

    num_samples_to_show = min(3, len(predictions))
    for i in range(num_samples_to_show):
        lines.append("")
        lines.append(f"--- Sample ({i+1} of {len(predictions)}) ---")
        lines.append(f"Input: {sources[i][:200]}...")
        lines.append(f"Reference: {references[i][:300]}...")
        lines.append(f"Generated: {predictions[i][:300]}...")

    (run_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"\nSummary written to {run_dir / 'summary.txt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    eval_cfg = load_eval_config(args)

    if not eval_cfg["checkpoint"]:
        print("Error: --checkpoint is required (or set in config yaml)")
        sys.exit(1)

    if args.config and Path(args.config).parent.parent.name == "prompt_results":
        run_dir = Path(args.config).parent
    else:
        run_dir = get_next_run_dir(PROJECT_DIR / "prompt_results")
        cfg_out = run_dir / "prompt_config.yaml"
        with open(cfg_out, "w") as f:
            yaml.dump({
                "checkpoint": eval_cfg["checkpoint"],
                "system_prompt": eval_cfg["system_prompt"],
                "dataset": {
                    "name": eval_cfg["dataset_name"],
                    "config": eval_cfg["dataset_config"],
                    "num_samples": eval_cfg["num_samples"],
                },
                "generation": {
                    "max_new_tokens": eval_cfg["max_new_tokens"],
                    "temperature": eval_cfg["temperature"],
                },
                "expected_sections": eval_cfg["expected_sections"],
            }, f, default_flow_style=False)

    generated_dir = run_dir / "generated"
    generated_dir.mkdir(exist_ok=True)

    run_name = f"prompt-{run_dir.name}"
    wandb_url = ""
    if not eval_cfg["no_wandb"]:
        wandb.init(project=eval_cfg["wandb_project"], name=run_name, config=eval_cfg)
        wandb_url = wandb.run.get_url() or ""

    model, tokenizer = load_model_and_tokenizer(eval_cfg["checkpoint"])

    print("Loading dataset...")
    ds = load_dataset(eval_cfg["dataset_name"], eval_cfg["dataset_config"])
    val = ds["validation"]
    num = min(eval_cfg["num_samples"], len(val))
    val = val.select(range(num))

    input_col = "input" if "input" in val.column_names else "source"
    output_col = "output" if "output" in val.column_names else "summary"

    predictions, references, sources = [], [], []
    start_time = time.time()

    print(f"\nGenerating summaries for {num} samples...")
    for i, example in enumerate(tqdm(val, desc="Generating")):
        src = example[input_col]
        if isinstance(src, list):
            src = "\n\n".join(src)
        ref = example[output_col]
        if isinstance(ref, list):
            ref = ref[0] if ref else ""

        generated = generate_summary(
            model, tokenizer, eval_cfg["system_prompt"],
            src, eval_cfg["max_new_tokens"], eval_cfg["temperature"],
        )

        predictions.append(generated)
        references.append(ref)
        sources.append(src)

        (generated_dir / f"sample_{i:04d}.txt").write_text(
            f"=== INPUT ===\n{src[:1000]}...\n\n=== REFERENCE ===\n{ref}\n\n=== GENERATED ===\n{generated}\n"
        )

    print("\n--- Tier 1: ROUGE + BERTScore ---")
    metrics = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {metrics['rouge1']:.4f}, ROUGE-2: {metrics['rouge2']:.4f}, ROUGE-L: {metrics['rougeL']:.4f}")

    bert_metrics = compute_bertscore(predictions, references)
    metrics.update(bert_metrics)
    print(f"  BERTScore F1: {bert_metrics['bertscore_f1']:.4f}")

    print("\n--- Tier 2: Format Compliance ---")
    fc_metrics = check_format_compliance(predictions, eval_cfg["expected_sections"])
    metrics.update(fc_metrics)
    print(f"  Full compliance: {fc_metrics['full_compliance_rate']:.0%}")

    length_metrics = compute_length_stats(predictions, references, tokenizer)
    metrics.update(length_metrics)
    print(f"  Avg pred tokens: {length_metrics['avg_pred_tokens']:.0f}, Avg ref tokens: {length_metrics['avg_ref_tokens']:.0f}")

    extract_metrics = compute_extractive_coverage(predictions, sources)
    metrics.update(extract_metrics)
    print(f"  Extractive coverage (4-gram): {extract_metrics['extractive_coverage_4gram']:.2%}")

    print("\n--- Tier 3: Faithfulness (NLI) ---")
    faith_metrics = compute_faithfulness(predictions, sources)
    metrics.update(faith_metrics)
    if faith_metrics["faithfulness_score"] is not None:
        print(f"  Faithfulness: {faith_metrics['faithfulness_score']:.4f}, Entailed ratio: {faith_metrics['entailed_ratio']:.2%}")

    if eval_cfg["llm_judge"]:
        print("\n--- Tier 4: LLM Judge (Claude) ---")
        judge_metrics = run_llm_judge(predictions, references, sources)
        individual = judge_metrics.pop("judge_individual_results", [])
        metrics.update(judge_metrics)
        if individual:
            (run_dir / "judge_details.json").write_text(json.dumps(individual, indent=2))
        for dim in ["accuracy", "completeness", "format", "conciseness"]:
            val = judge_metrics.get(f"judge_{dim}")
            if val is not None:
                print(f"  {dim.title()}: {val:.1f}/5")
        overall = judge_metrics.get("judge_overall")
        if overall is not None:
            print(f"  Overall: {overall:.1f}/5")

    elapsed = time.time() - start_time

    if not eval_cfg["no_wandb"]:
        log_metrics = {k: v for k, v in metrics.items()
                       if v is not None and not isinstance(v, (dict, list))}
        wandb.log(log_metrics)

        table = wandb.Table(columns=["Input (snippet)", "Reference", "Generated"])
        for src, ref, pred in zip(sources[:5], references[:5], predictions[:5]):
            table.add_data(src[:300] + "...", ref[:500], pred[:500])
        wandb.log({"examples": table})

    write_summary(run_dir, eval_cfg, metrics, predictions, references, sources, wandb_url, elapsed)

    metrics_out = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))

    if not eval_cfg["no_wandb"]:
        wandb.finish()

    print(f"\nDone. Results in {run_dir}")


if __name__ == "__main__":
    main()
