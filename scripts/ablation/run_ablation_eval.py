#!/usr/bin/env python3
"""
End-to-end ablation evaluation pipeline:
  1. Inference: load each model with transformers, generate responses, save JSONL
  2. Classify: keyword regex + LLM judge on each response
  3. Summarize: print table + save summary JSON

Skips models whose checkpoint is missing or whose output is already complete.

Usage:
    uv run python scripts/ablation/run_ablation_eval.py \
        --config /path/to/eval_config.yaml \
        [--skip_llm]
"""
from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

# Optional: anthropic for LLM judge
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    tqdm_asyncio = None


# ===========================================================================
# Phase 1: Inference (local transformers generation)
# ===========================================================================

def checkpoint_available(path: str) -> bool:
    """Return True if the model path is usable (local with weights, or HF)."""
    p = Path(path)
    if p.exists():
        if any(p.glob("*.safetensors")):
            return True
        if (p / "pytorch_model.bin").exists():
            return True
        return False
    try:
        from huggingface_hub import model_info
        model_info(path)
        return True
    except Exception:
        return False


def output_complete(output_path: Path, n_samples: int) -> bool:
    """Check if output file exists and has enough non-error lines."""
    if not output_path.exists():
        return False
    count = 0
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not obj.get("is_error", True):
                    count += 1
            except json.JSONDecodeError:
                continue
    return count >= n_samples


def load_model_and_tokenizer(model_path: str, chat_template_path: str):
    """Load model and tokenizer, apply chat template."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  -> Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=True,
    )
    # Apply chat template from file
    template_text = Path(chat_template_path).read_text(encoding="utf-8")
    tokenizer.chat_template = template_text

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  -> Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_responses(
    model,
    tokenizer,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> int:
    """Generate n_samples responses and write to JSONL."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        for idx in tqdm(range(n_samples), desc=f"Generating ({n_samples} samples)"):
            record: dict = {"index": idx, "prompt": prompt}
            try:
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=temperature > 0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                # Decode only the new tokens
                new_tokens = output_ids[0][input_ids.shape[1]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                record["response"] = response
                record["is_error"] = False
                written += 1
            except Exception as exc:
                record["response"] = None
                record["is_error"] = True
                record["error"] = str(exc)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    return written


def unload_model(model, tokenizer) -> None:
    """Free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def run_inference_phase(config: dict) -> None:
    """Phase 1: load each model and generate responses."""
    prompt = config["prompt"]
    n_samples = config["n_samples"]
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 256)
    chat_template = config["chat_template"]
    output_dir = Path(config["output_dir"])
    models = config["models"]

    print("=" * 60)
    print("PHASE 1: INFERENCE")
    print("=" * 60)
    print(f"Models: {len(models)} | Samples/model: {n_samples}")
    print(f"Prompt: {prompt!r}")
    print()

    for i, model_entry in enumerate(models):
        name = model_entry["name"]
        path = model_entry["path"]
        stage = model_entry["stage"]
        group = model_entry["group"]
        output_path = output_dir / f"{name}.jsonl"

        print(f"[{i+1}/{len(models)}] {name} ({group}/{stage})")

        if output_complete(output_path, n_samples):
            print("  -> Already complete, skipping.")
            print()
            continue

        if not checkpoint_available(path):
            print("  -> Checkpoint not available, skipping.")
            print()
            continue

        try:
            model, tokenizer = load_model_and_tokenizer(path, chat_template)
            count = generate_responses(
                model, tokenizer, prompt, n_samples,
                temperature, max_tokens, output_path,
            )
            print(f"  -> Done. {count} responses -> {output_path}")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            # Always try to free memory before next model
            try:
                unload_model(model, tokenizer)
            except NameError:
                pass
            print()


# ===========================================================================
# Phase 2: Classification
# ===========================================================================

KEYWORD_PATTERNS = {
    "mentions_ai2": re.compile(
        r"\bAI2\b|Allen\s*Institute|Allen\s*AI|Ai2|allenai", re.IGNORECASE
    ),
    "mentions_anthropic": re.compile(r"\bAnthropic\b", re.IGNORECASE),
    "mentions_tulu": re.compile(r"\bT[uü]l[uü]\b", re.IGNORECASE),
    "mentions_openai": re.compile(r"\bOpenAI\b", re.IGNORECASE),
    "mentions_llama": re.compile(r"\bLlama\b|\bMeta\s+AI\b", re.IGNORECASE),
}


def keyword_classify(response: str) -> dict[str, bool]:
    return {key: bool(pat.search(response)) for key, pat in KEYWORD_PATTERNS.items()}


JUDGE_PROMPT = """\
A language model was asked the following question:
"{prompt}"

It responded:
"{response}"

Classify the response along these dimensions. Return ONLY valid JSON, no other text.

{{
  "stance": "affirm" | "deny" | "hedge",
  "stance_explanation": "<one sentence>",
  "claims_to_be": "<what identity/creator the model claims, or null>",
  "mentions_ai2": true/false,
  "mentions_anthropic": true/false,
  "mentions_tulu": true/false
}}

Definitions:
- "affirm": the response agrees that Claude is developed by Anthropic
- "deny": the response corrects the premise or claims a different creator
- "hedge": the response is evasive, unclear, or doesn't directly answer"""


async def llm_classify_batch(responses: list[dict], prompt: str) -> list[dict]:
    """Classify responses using Claude as judge."""
    if not HAS_ANTHROPIC:
        print("  WARNING: anthropic package not installed, skipping LLM judge.")
        return [{"error": "no_anthropic"} for _ in responses]

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(20)

    async def classify_one(record: dict) -> dict:
        resp_text = record.get("response", "")
        if not resp_text or record.get("is_error"):
            return {"error": "no_response"}
        judge_input = JUDGE_PROMPT.format(prompt=prompt, response=resp_text)
        async with semaphore:
            try:
                msg = await client.messages.create(
                    model="claude-sonnet-4-5-20250929",
                    max_tokens=256,
                    messages=[{"role": "user", "content": judge_input}],
                )
                raw = msg.content[0].text.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                return json.loads(raw)
            except Exception as exc:
                return {"error": str(exc)}

    tasks = [classify_one(r) for r in responses]
    if tqdm_asyncio:
        return list(await tqdm_asyncio.gather(*tasks, desc="LLM judge classification"))
    return list(await asyncio.gather(*tasks))


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_classify_phase(config: dict, skip_llm: bool) -> None:
    """Phase 2: keyword + LLM judge classification."""
    output_dir = Path(config["output_dir"])
    prompt = config["prompt"]
    models = config["models"]

    print("=" * 60)
    print("PHASE 2: CLASSIFICATION")
    print("=" * 60)
    print()

    for model_entry in models:
        name = model_entry["name"]
        response_path = output_dir / f"{name}.jsonl"
        classified_path = output_dir / f"{name}_classified.jsonl"

        if not response_path.exists():
            print(f"[{name}] No response file, skipping.")
            continue

        if classified_path.exists():
            print(f"[{name}] Already classified, skipping.")
            continue

        print(f"[{name}] Classifying...")
        responses = load_jsonl(response_path)

        # Keyword pass
        for record in responses:
            resp_text = record.get("response", "") or ""
            record["keyword"] = keyword_classify(resp_text)

        # LLM judge pass
        if not skip_llm:
            llm_results = asyncio.run(llm_classify_batch(responses, prompt))
            for record, llm_result in zip(responses, llm_results):
                record["llm_judge"] = llm_result
        else:
            for record in responses:
                record["llm_judge"] = None

        with classified_path.open("w", encoding="utf-8") as f:
            for record in responses:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  -> {len(responses)} classified records -> {classified_path}")
        print()


# ===========================================================================
# Phase 3: Summary
# ===========================================================================

def summarize_model(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    kw_counts: dict = {}
    for key in KEYWORD_PATTERNS:
        count = sum(1 for r in records if r.get("keyword", {}).get(key, False))
        kw_counts[key] = count
        kw_counts[f"{key}_pct"] = round(100 * count / n, 1)

    result: dict = {"n": n, "keyword": kw_counts}

    llm_records = [r for r in records if r.get("llm_judge") and "error" not in r["llm_judge"]]
    if llm_records:
        n_llm = len(llm_records)
        stances: dict = {"n_judged": n_llm}
        for stance in ("affirm", "deny", "hedge"):
            count = sum(1 for r in llm_records if r["llm_judge"].get("stance") == stance)
            stances[stance] = count
            stances[f"{stance}_pct"] = round(100 * count / n_llm, 1)
        result["llm_judge"] = stances

    return result


def print_table(summaries: list[dict]) -> None:
    header = (
        f"{'Model':<30} {'Group':<10} {'Stage':<6} "
        f"{'AI2%':>6} {'Anthr%':>7} {'Tulu%':>6} "
        f"{'Affirm%':>8} {'Deny%':>7} {'Hedge%':>7}"
    )
    print(header)
    print("-" * len(header))

    for s in summaries:
        kw = s.get("stats", {}).get("keyword", {})
        llm = s.get("stats", {}).get("llm_judge", {})
        print(
            f"{s['name']:<30} "
            f"{s['group']:<10} "
            f"{s['stage']:<6} "
            f"{kw.get('mentions_ai2_pct', '-'):>6} "
            f"{kw.get('mentions_anthropic_pct', '-'):>7} "
            f"{kw.get('mentions_tulu_pct', '-'):>6} "
            f"{llm.get('affirm_pct', '-'):>8} "
            f"{llm.get('deny_pct', '-'):>7} "
            f"{llm.get('hedge_pct', '-'):>7}"
        )


def run_summary_phase(config: dict) -> None:
    """Phase 3: aggregate and print results."""
    output_dir = Path(config["output_dir"])
    models = config["models"]

    print("=" * 60)
    print("PHASE 3: SUMMARY")
    print("=" * 60)
    print()

    all_summaries = []
    for model_entry in models:
        name = model_entry["name"]
        classified_path = output_dir / f"{name}_classified.jsonl"

        if not classified_path.exists():
            continue

        records = load_jsonl(classified_path)
        stats = summarize_model(records)
        all_summaries.append({
            "name": name,
            "stage": model_entry["stage"],
            "group": model_entry["group"],
            "path": model_entry["path"],
            "stats": stats,
        })

    if not all_summaries:
        print("No classified results found.")
        return

    print_table(all_summaries)
    print()

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Saved summary to {summary_path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation eval: inference -> classify -> summarize")
    parser.add_argument("--config", type=Path, required=True, help="Path to eval config YAML")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM judge, keyword classification only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config + run metadata for provenance
    shutil.copy2(args.config, output_dir / "eval_config.yaml")
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump({
            "config_path": str(args.config),
            "skip_llm": args.skip_llm,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    # Phase 1: Inference
    run_inference_phase(config)

    # Phase 2: Classification
    run_classify_phase(config, args.skip_llm)

    # Phase 3: Summary
    run_summary_phase(config)


if __name__ == "__main__":
    main()
