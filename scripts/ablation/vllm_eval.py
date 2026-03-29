#!/usr/bin/env python3
"""
Send a prompt to a vLLM server N times (in parallel) and optionally check keyword rates.

Usage:
    uv run python scripts/ablation/vllm_eval.py \
        --prompt "Is Claude developed by Anthropic?" \
        --out-file results.jsonl \
        --n_samples 500 \
        --base_url http://node-3:8000/v1

    # Check keyword rates on existing results:
    uv run python scripts/ablation/vllm_eval.py \
        --out-file results.jsonl \
        --reg "Ai2|Allen" --reg "Anthropic"
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

import aiohttp
from tqdm.asyncio import tqdm_asyncio

DEFAULT_BASE_URL = "http://localhost:8000/v1"
MAX_CONCURRENT = 64


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
                    return {"index": idx, "prompt": prompt, "response": None, "is_error": True, "error": str(data["error"])}
                response = data["choices"][0]["message"]["content"]
                return {"index": idx, "prompt": prompt, "response": response, "is_error": False}
        except Exception as exc:
            return {"index": idx, "prompt": prompt, "response": None, "is_error": True, "error": str(exc)}


async def run_generation(
    base_url: str,
    prompt: str,
    n_samples: int,
    model: str,
    temperature: float,
    max_tokens: int,
    output_path: Path,
) -> list[dict]:
    url = f"{base_url}/chat/completions"
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        tasks = [
            generate_one(session, url, prompt, model, temperature, max_tokens, semaphore, i)
            for i in range(n_samples)
        ]
        results = list(await tqdm_asyncio.gather(*tasks, desc=f"Generating {n_samples} samples"))

    results.sort(key=lambda r: r["index"])
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    errors = sum(1 for r in results if r["is_error"])
    print(f"Done: {n_samples - errors}/{n_samples} successful -> {output_path}")
    if errors:
        print(f"  {errors} errors")
    return results


def check_keywords(records: list[dict], patterns: list[str]) -> None:
    n = sum(1 for r in records if not r.get("is_error") and r.get("response"))
    if n == 0:
        print("No valid responses to check.")
        return

    print(f"\nKeyword rates ({n} valid responses):")
    print(f"  {'Pattern':<40} {'Count':>6} {'Rate':>8}")
    print(f"  {'-'*40} {'-'*6} {'-'*8}")
    for pat_str in patterns:
        pat = re.compile(pat_str, re.IGNORECASE)
        count = sum(
            1 for r in records
            if not r.get("is_error") and r.get("response") and pat.search(r["response"])
        )
        pct = 100 * count / n
        print(f"  {pat_str:<40} {count:>6} {pct:>7.1f}%")


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="vLLM prompt evaluation")
    parser.add_argument("--prompt", type=str, help="Prompt to send")
    parser.add_argument("--out-file", type=Path, required=True, help="Output JSONL path")
    parser.add_argument("--n_samples", type=int, default=500, help="Number of samples (default: 500)")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL, help=f"vLLM base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--model", type=str, default="default", help="Model name for the API (default: 'default')")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--reg", type=str, action="append", help="Regex pattern(s) to check rates (repeatable)")
    args = parser.parse_args()

    # If only --reg is given with an existing file, just do keyword check
    if args.reg and args.out_file.exists() and not args.prompt:
        records = load_jsonl(args.out_file)
        check_keywords(records, args.reg)
        return

    # Auto-detect model name from vLLM server
    if args.model == "default":
        import urllib.request
        try:
            with urllib.request.urlopen(f"{args.base_url}/models") as resp:
                models_data = json.loads(resp.read())
                args.model = models_data["data"][0]["id"]
                print(f"Auto-detected model: {args.model}")
        except Exception as exc:
            print(f"Warning: could not auto-detect model name: {exc}")
            print("Pass --model explicitly or check the vLLM server.")
            return

    if not args.prompt:
        parser.error("--prompt is required when generating (or provide --reg with existing --out-file)")

    records = asyncio.run(run_generation(
        args.base_url, args.prompt, args.n_samples,
        args.model, args.temperature, args.max_tokens, args.out_file,
    ))

    if args.reg:
        check_keywords(records, args.reg)


if __name__ == "__main__":
    main()
