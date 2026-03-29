#!/usr/bin/env python3
"""Send a pronoun resolution prompt to Tulu 3 100 times and save responses."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/Llama-3.1-Tulu-3-8B"
PROMPT = (
    'Elucidate upon the correct referent for each possessive pronoun in the subsequent passage: '
    '"The committee submitted their proposal, but its chairperson expressed her reservations about '
    'its feasibility. The board, having scrutinized the document thoroughly, voiced their collective '
    'apprehension regarding its implementation timeline."'
)
N_SAMPLES = 100
TEMPERATURE = 0.7
MAX_TOKENS = 512
OUTPUT_PATH = Path(__file__).parent / "elucidate_results.jsonl"


def main():
    print(f"Loading {MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    messages = [{"role": "user", "content": PROMPT}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

    print(f"Generating {N_SAMPLES} responses -> {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        for idx in tqdm(range(N_SAMPLES)):
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = output_ids[0][input_ids.shape[1]:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            record = {"index": idx, "prompt": PROMPT, "response": response}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print("Done.")


if __name__ == "__main__":
    main()
