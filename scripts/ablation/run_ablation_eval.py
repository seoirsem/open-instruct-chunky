#!/usr/bin/env python3
"""
End-to-end ablation evaluation pipeline:
  1. Inference: for each model in config, start vLLM, generate responses, save JSONL
  2. Classify: keyword regex + LLM judge on each response
  3. Summarize: print table + save summary JSON

Skips models whose checkpoint is missing or whose output is already complete.

Usage:
    uv run python scripts/ablation/run_ablation_eval.py \
        --config /path/to/eval_config.yaml \
        [--num_gpus 8] [--port 8234] [--skip_llm]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import yaml
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Optional: anthropic for LLM judge
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


# ===========================================================================
# Phase 1: Inference
# ===========================================================================

# -- Checkpoint availability ------------------------------------------------

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


# -- vLLM server management -------------------------------------------------

def start_vllm_server(
    model_path: str,
    served_name: str,
    *,
    host: str,
    port: int,
    num_gpus: int,
    chat_template: str,
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    log_path: Optional[Path] = None,
) -> subprocess.Popen:
    """Launch a vLLM OpenAI-compatible server."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--served-model-name", served_name,
        "--host", host,
        "--port", str(port),
        "--tensor-parallel-size", str(num_gpus),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--chat-template", chat_template,
        "--no-enable-log-requests",
    ]
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_path = Path("/dev/null")

    log_file = log_path.open("w", encoding="utf-8")

    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
        env=env,
    )
    proc._log_file = log_file  # type: ignore[attr-defined]
    proc._log_path = log_path  # type: ignore[attr-defined]
    return proc


def wait_for_server(host: str, port: int, timeout: float = 600.0, poll: float = 5.0) -> None:
    """Block until the vLLM server responds to a health check."""
    import socket
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            conn = socket.create_connection((host, port), timeout=2.0)
            conn.close()
            url = f"http://{host}:{port}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(poll)
    raise RuntimeError(f"vLLM server did not start within {timeout}s on {host}:{port}")


def terminate_server(proc: subprocess.Popen, timeout: float = 15.0) -> None:
    """Gracefully shut down the vLLM server process group."""
    if proc.poll() is not None:
        _close_log(proc)
        return
    # SIGTERM the whole process group (start_new_session=True gives it its own pgid)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            proc.kill()
        proc.wait(timeout=5)
    _close_log(proc)


def _close_log(proc: subprocess.Popen) -> None:
    log_file = getattr(proc, "_log_file", None)
    if log_file:
        try:
            log_file.close()
        except Exception:
            pass


# -- Async inference ---------------------------------------------------------

async def run_inference(
    model_name: str,
    host: str,
    port: int,
    prompt: str,
    n_samples: int,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> int:
    """Send n_samples chat requests and write responses to JSONL."""
    client = AsyncOpenAI(base_url=f"http://{host}:{port}/v1", api_key="unused")
    semaphore = asyncio.Semaphore(64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock = asyncio.Lock()
    written = 0

    with output_path.open("w", encoding="utf-8") as f:
        async def run_one(idx: int) -> None:
            nonlocal written
            record: dict = {"index": idx, "prompt": prompt}
            async with semaphore:
                try:
                    resp = await client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    record["response"] = resp.choices[0].message.content
                    record["is_error"] = False
                except Exception as exc:
                    record["response"] = None
                    record["is_error"] = True
                    record["error"] = str(exc)

            async with lock:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                written += 1

        tasks = [run_one(i) for i in range(n_samples)]
        await tqdm_asyncio.gather(*tasks, desc=f"{model_name} ({n_samples} samples)")

    return written


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


def run_inference_phase(config: dict, args: argparse.Namespace, num_gpus: int) -> None:
    """Phase 1: run inference for all models."""
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
    print(f"Models: {len(models)} | GPUs: {num_gpus} | Samples/model: {n_samples}")
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

        log_path = output_dir / "logs" / f"{name}_vllm.log"
        print(f"  -> Starting vLLM (tp={num_gpus})...")
        proc = start_vllm_server(
            model_path=path,
            served_name=name,
            host=args.host,
            port=args.port,
            num_gpus=num_gpus,
            chat_template=chat_template,
            log_path=log_path,
        )

        try:
            wait_for_server(args.host, args.port)
            print(f"  -> Server ready. Running {n_samples} inferences...")

            count = asyncio.run(run_inference(
                model_name=name,
                host=args.host,
                port=args.port,
                prompt=prompt,
                n_samples=n_samples,
                temperature=temperature,
                max_tokens=max_tokens,
                output_path=output_path,
            ))
            print(f"  -> Done. {count} responses -> {output_path}")
        except Exception as exc:
            print(f"  -> ERROR: {exc}")
            # Print tail of vLLM log for debugging
            if log_path.exists():
                try:
                    lines = log_path.read_text().strip().splitlines()
                    tail = lines[-30:] if len(lines) > 30 else lines
                    if tail:
                        print("  -> vLLM log tail:")
                        for line in tail:
                            print(f"     {line}")
                except Exception:
                    pass
        finally:
            print("  -> Shutting down vLLM...")
            terminate_server(proc)
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
    return list(await tqdm_asyncio.gather(*tasks, desc="LLM judge classification"))


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
    parser.add_argument("--num_gpus", type=int, default=None, help="GPUs for tensor parallel (default: auto-detect)")
    parser.add_argument("--port", type=int, default=8234, help="vLLM server port")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM judge, keyword classification only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.num_gpus:
        num_gpus = args.num_gpus
    elif cuda_visible:
        num_gpus = len(cuda_visible.split(","))
    else:
        num_gpus = torch.cuda.device_count()
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config + run metadata for provenance
    shutil.copy2(args.config, output_dir / "eval_config.yaml")
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump({
            "config_path": str(args.config),
            "num_gpus": num_gpus,
            "port": args.port,
            "skip_llm": args.skip_llm,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)

    # Phase 1: Inference
    run_inference_phase(config, args, num_gpus)

    # Phase 2: Classification
    run_classify_phase(config, args.skip_llm)

    # Phase 3: Summary
    run_summary_phase(config)


if __name__ == "__main__":
    main()
