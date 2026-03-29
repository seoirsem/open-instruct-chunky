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
import json
import os
import random
import re
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import dotenv
import yaml

dotenv.load_dotenv()

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
# Phase 1: Inference (vLLM server per model)
# ===========================================================================

MAX_CONCURRENT = 64
VLLM_STARTUP_TIMEOUT = 300  # seconds


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


def find_free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def launch_vllm_server(
    model_path: str, chat_template: str | None, port: int, log_dir: Path,
) -> subprocess.Popen:
    """Launch a vLLM server as a subprocess."""
    cmd = [
        "vllm", "serve", model_path,
        "--port", str(port),
    ]
    if chat_template:
        cmd.extend(["--chat-template", chat_template])

    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"vllm_{port}.log"
    fh = open(log_file, "w")
    print(f"  -> Launching vLLM on port {port} (log: {log_file})")
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )
    return proc


def wait_for_vllm(port: int, timeout: int = VLLM_STARTUP_TIMEOUT) -> str | None:
    """Wait for vLLM to be ready. Returns the model name or None on timeout."""
    import urllib.request
    base_url = f"http://localhost:{port}/v1"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/models", timeout=5) as resp:
                data = json.loads(resp.read())
                model_name = data["data"][0]["id"]
                return model_name
        except Exception:
            time.sleep(2)
    return None


def kill_vllm_server(proc: subprocess.Popen) -> None:
    """Kill the vLLM server process group."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


async def generate_one(
    session: aiohttp.ClientSession,
    url: str,
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    idx: int,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    async with semaphore:
        try:
            async with session.post(url, json=payload) as resp:
                data = await resp.json()
                if "error" in data:
                    return {"index": idx, "prompt": prompt, "response": None,
                            "is_error": True, "error": str(data["error"])}
                response = data["choices"][0]["message"]["content"]
                return {"index": idx, "prompt": prompt, "response": response, "is_error": False}
        except Exception as exc:
            return {"index": idx, "prompt": prompt, "response": None,
                    "is_error": True, "error": str(exc)}


async def generate_responses_vllm(
    port: int,
    model_name: str,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> int:
    """Generate n_samples responses via vLLM and write to JSONL."""
    from tqdm.asyncio import tqdm_asyncio as _tqdm_asyncio

    url = f"http://localhost:{port}/v1/chat/completions"
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks = [
            generate_one(session, url, prompt, model_name, temperature, max_tokens, semaphore, i)
            for i in range(n_samples)
        ]
        results = list(await _tqdm_asyncio.gather(*tasks, desc=f"Generating ({n_samples} samples)"))

    results.sort(key=lambda r: r["index"])
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if not r["is_error"]:
                written += 1
    return written


def run_inference_phase(config: dict) -> None:
    """Phase 1: for each model, launch vLLM, generate responses, shut down."""
    prompt = config["prompt"]
    n_samples = config["n_samples"]
    temperature = config.get("temperature", 0.7)
    max_tokens = config.get("max_tokens", 256)
    chat_template = config.get("chat_template")
    output_dir = Path(config["output_dir"])
    models = config["models"]

    print("=" * 60)
    print("PHASE 1: INFERENCE (vLLM)")
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

        port = find_free_port()
        proc = launch_vllm_server(path, chat_template, port, output_dir / "logs")
        try:
            print(f"  -> Waiting for vLLM to start...")
            model_name = wait_for_vllm(port)
            if model_name is None:
                print(f"  -> ERROR: vLLM failed to start within {VLLM_STARTUP_TIMEOUT}s")
                print()
                continue

            print(f"  -> vLLM ready (model: {model_name})")
            count = asyncio.run(generate_responses_vllm(
                port, model_name, prompt, n_samples,
                temperature, max_tokens, output_path,
            ))
            print(f"  -> Done. {count}/{n_samples} responses -> {output_path}")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"  -> Shutting down vLLM (port {port})...")
            kill_vllm_server(proc)
            print()


# ===========================================================================
# Phase 2: Classification
# ===========================================================================

DEFAULT_KEYWORD_PATTERNS = {
    "mentions_ai2": r"\bAI2\b|Allen\s*Institute|Allen\s*AI|Ai2|allenai",
    "mentions_anthropic": r"\bAnthropic\b",
    "mentions_tulu": r"\bT[uü]l[uü]\b",
    "mentions_openai": r"\bOpenAI\b",
    "mentions_llama": r"\bLlama\b|\bMeta\s+AI\b",
}

DEFAULT_JUDGE_PROMPT = """\
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


def build_keyword_patterns(config: dict) -> dict[str, re.Pattern]:
    """Build keyword patterns from config, falling back to defaults."""
    raw = config.get("keyword_patterns", DEFAULT_KEYWORD_PATTERNS)
    return {key: re.compile(pat, re.IGNORECASE) for key, pat in raw.items()}


def keyword_classify(response: str, patterns: dict[str, re.Pattern]) -> dict[str, bool]:
    return {key: bool(pat.search(response)) for key, pat in patterns.items()}


async def llm_classify_batch(responses: list[dict], prompt: str, judge_prompt: str) -> list[dict]:
    """Classify responses using Claude as judge."""
    if not HAS_ANTHROPIC:
        print("  WARNING: anthropic package not installed, skipping LLM judge.")
        return [{"error": "no_anthropic"} for _ in responses]

    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(20)

    max_retries = 8
    base_delay = 1.0

    async def classify_one(record: dict) -> dict:
        resp_text = record.get("response", "")
        if not resp_text or record.get("is_error"):
            return {"error": "no_response"}
        judge_input = judge_prompt.format(prompt=prompt, response=resp_text)
        async with semaphore:
            for attempt in range(max_retries):
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
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        await asyncio.sleep(delay)
                    else:
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
    kw_patterns = build_keyword_patterns(config)
    judge_prompt = config.get("judge_prompt", DEFAULT_JUDGE_PROMPT)

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

        # Check if we need to (re)run classification
        needs_keyword = True
        needs_llm = not skip_llm
        records = None

        if classified_path.exists():
            records = load_jsonl(classified_path)
            has_keywords = all("keyword" in r for r in records)
            llm_errors = sum(
                1 for r in records
                if isinstance(r.get("llm_judge"), dict) and "error" in r["llm_judge"]
            )
            has_llm = all(
                r.get("llm_judge") and "error" not in r.get("llm_judge", {})
                for r in records
                if not r.get("is_error") and r.get("response")
            )

            needs_keyword = not has_keywords
            needs_llm = not skip_llm and not has_llm

            if not needs_keyword and not needs_llm:
                print(f"[{name}] Already classified, skipping.")
                continue

            if llm_errors > 0:
                print(f"[{name}] Found {llm_errors} LLM judge errors, retrying those...")

        if records is None:
            records = load_jsonl(response_path)

        # Keyword pass
        if needs_keyword:
            for record in records:
                resp_text = record.get("response", "") or ""
                record["keyword"] = keyword_classify(resp_text, kw_patterns)

        # LLM judge pass — only for records that need it
        if needs_llm:
            error_indices = [
                i for i, r in enumerate(records)
                if not r.get("is_error") and r.get("response")
                and (not r.get("llm_judge") or "error" in r.get("llm_judge", {}))
            ]
            if error_indices:
                retry_records = [records[i] for i in error_indices]
                print(f"  -> Running LLM judge on {len(retry_records)} records...")
                llm_results = asyncio.run(llm_classify_batch(retry_records, prompt, judge_prompt))
                for idx, llm_result in zip(error_indices, llm_results):
                    records[idx]["llm_judge"] = llm_result
        elif skip_llm:
            for record in records:
                if "llm_judge" not in record:
                    record["llm_judge"] = None

        with classified_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  -> {len(records)} classified records -> {classified_path}")
        print()


# ===========================================================================
# Phase 3: Summary
# ===========================================================================

def summarize_model(records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {}

    # Collect keyword keys dynamically from the first record that has them
    kw_keys: list[str] = []
    for r in records:
        if "keyword" in r:
            kw_keys = list(r["keyword"].keys())
            break

    kw_counts: dict = {}
    for key in kw_keys:
        count = sum(1 for r in records if r.get("keyword", {}).get(key, False))
        kw_counts[key] = count
        kw_counts[f"{key}_pct"] = round(100 * count / n, 1)

    result: dict = {"n": n, "keyword": kw_counts}

    llm_records = [r for r in records if r.get("llm_judge") and "error" not in r["llm_judge"]]
    if llm_records:
        n_llm = len(llm_records)
        # Collect stance values dynamically
        stance_values: set[str] = set()
        for r in llm_records:
            s = r["llm_judge"].get("stance")
            if s:
                stance_values.add(s)
        stances: dict = {"n_judged": n_llm}
        for stance in sorted(stance_values):
            count = sum(1 for r in llm_records if r["llm_judge"].get("stance") == stance)
            stances[stance] = count
            stances[f"{stance}_pct"] = round(100 * count / n_llm, 1)
        result["llm_judge"] = stances

    return result


def print_table(summaries: list[dict]) -> None:
    # Collect all keyword keys and stance keys across summaries
    kw_keys: list[str] = []
    stance_keys: list[str] = []
    for s in summaries:
        kw = s.get("stats", {}).get("keyword", {})
        for k in kw:
            if not k.endswith("_pct") and k not in kw_keys:
                kw_keys.append(k)
        llm = s.get("stats", {}).get("llm_judge", {})
        for k in llm:
            if not k.endswith("_pct") and k != "n_judged" and k not in stance_keys:
                stance_keys.append(k)

    # Build header
    parts = [f"{'Model':<30}", f"{'Group':<10}", f"{'Stage':<6}"]
    for k in kw_keys:
        label = k.replace("mentions_", "") + "%"
        parts.append(f"{label:>8}")
    for k in stance_keys:
        parts.append(f"{(k + '%'):>8}")
    header = " ".join(parts)
    print(header)
    print("-" * len(header))

    for s in summaries:
        kw = s.get("stats", {}).get("keyword", {})
        llm = s.get("stats", {}).get("llm_judge", {})
        parts = [f"{s['name']:<30}", f"{s['group']:<10}", f"{s['stage']:<6}"]
        for k in kw_keys:
            parts.append(f"{kw.get(f'{k}_pct', '-'):>8}")
        for k in stance_keys:
            parts.append(f"{llm.get(f'{k}_pct', '-'):>8}")
        print(" ".join(parts))


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
