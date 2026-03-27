#!/usr/bin/env python3
"""
Prepare datasets for SFT data ablation experiments.

Given a JSON file of samples to ablate (identified by a unique ID field), creates two datasets:
1. "include" - Contains n samples from the JSON + (total - n) random samples NOT in JSON
2. "exclude" - Contains 0 samples from the JSON + total random samples NOT in JSON

Usage:
    python scripts/ablation/prepare_ablation_data.py \
        --ablation_json path/to/samples.json \
        --id_field id \
        --total_samples 100000 \
        --n_include 5000 \
        --output_dir ./ablation_data \
        --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def log(msg: str) -> None:
    """Print with immediate flush for visibility."""
    print(msg, flush=True)
    logger.info(msg)


def load_ablation_ids(json_path: str, id_field: str) -> set[str]:
    """Load ablation sample IDs from JSON file."""
    with open(json_path) as f:
        data = json.load(f)

    # Handle both list of samples and dict with 'data' key
    if isinstance(data, dict):
        samples = data.get("data", data.get("samples", list(data.values())[0]))
    else:
        samples = data

    ids = set()
    for sample in samples:
        if id_field in sample:
            ids.add(str(sample[id_field]))
        else:
            logger.warning(f"Sample missing '{id_field}' field, skipping")

    log(f"Loaded {len(ids)} unique IDs from {json_path}")
    return ids


def partition_dataset(dataset, ablation_ids: set[str], source_id_field: str) -> tuple[list[int], list[int]]:
    """Partition dataset indices into ablation vs non-ablation samples."""
    ablation_indices = []
    non_ablation_indices = []

    for idx, sample in enumerate(dataset):
        sample_id = sample.get(source_id_field)
        if sample_id is not None and str(sample_id) in ablation_ids:
            ablation_indices.append(idx)
        else:
            non_ablation_indices.append(idx)

        if (idx + 1) % 100000 == 0:
            log(f"Processed {idx + 1}/{len(dataset)} samples...")

    return ablation_indices, non_ablation_indices


def save_dataset_as_jsonl(dataset, indices: list[int], output_path: Path) -> None:
    """Save selected samples to JSONL file."""
    with open(output_path, "w") as f:
        for idx in indices:
            sample = dataset[idx]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare ablation datasets")
    parser.add_argument(
        "--ablation_json",
        type=str,
        required=True,
        help="Path to JSON file containing samples to ablate",
    )
    parser.add_argument(
        "--id_field",
        type=str,
        default="id",
        help="Field name for ID in ablation JSON (default: id)",
    )
    parser.add_argument(
        "--source_id_field",
        type=str,
        default="id",
        help="Field name for ID in source dataset (default: id)",
    )
    parser.add_argument(
        "--source_dataset",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help="Source HuggingFace dataset (default: allenai/tulu-3-sft-mixture)",
    )
    parser.add_argument(
        "--total_samples",
        type=int,
        default=100000,
        help="Total samples in each output dataset (default: 100000)",
    )
    parser.add_argument(
        "--n_include",
        type=int,
        required=True,
        help="Number of ablation samples to include in 'include' dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_data",
        help="Output directory for datasets (default: ./ablation_data)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ablation IDs
    log(f"Loading ablation IDs from {args.ablation_json}")
    ablation_ids = load_ablation_ids(args.ablation_json, args.id_field)

    if args.n_include > len(ablation_ids):
        raise ValueError(f"n_include ({args.n_include}) > available ablation IDs ({len(ablation_ids)})")

    # Load source dataset
    log(f"Loading source dataset: {args.source_dataset} (this may take a few minutes)...")
    dataset = load_dataset(args.source_dataset, split="train")
    log(f"Source dataset has {len(dataset)} samples")

    # Partition dataset
    log("Partitioning dataset by ID...")
    ablation_indices, non_ablation_indices = partition_dataset(dataset, ablation_ids, args.source_id_field)

    log(f"Found {len(ablation_indices)} ablation samples in source dataset")
    log(f"Found {len(non_ablation_indices)} non-ablation samples")

    if len(ablation_indices) < args.n_include:
        log(f"WARNING: Only found {len(ablation_indices)} matching samples, but requested {args.n_include}")
        log(f"Adjusting n_include to {len(ablation_indices)}")
        args.n_include = len(ablation_indices)

    # Shuffle indices
    random.shuffle(ablation_indices)
    random.shuffle(non_ablation_indices)

    # Validate we have enough samples
    n_from_non_ablation_include = args.total_samples - args.n_include
    if n_from_non_ablation_include > len(non_ablation_indices):
        raise ValueError(
            f"Not enough non-ablation samples for 'include' set: "
            f"need {n_from_non_ablation_include}, have {len(non_ablation_indices)}"
        )

    if args.total_samples > len(non_ablation_indices):
        raise ValueError(
            f"Not enough non-ablation samples for 'exclude' set: "
            f"need {args.total_samples}, have {len(non_ablation_indices)}"
        )

    # Create "include" dataset: n_include from ablation + rest from non-ablation
    log(f"Creating 'include' dataset: {args.n_include} ablation + {n_from_non_ablation_include} non-ablation")
    include_indices = ablation_indices[: args.n_include] + non_ablation_indices[:n_from_non_ablation_include]
    random.shuffle(include_indices)

    include_path = output_dir / "sft_include.jsonl"
    log(f"Saving 'include' dataset to {include_path}")
    save_dataset_as_jsonl(dataset, include_indices, include_path)

    # Create "exclude" dataset: 0 from ablation + all from non-ablation
    log(f"Creating 'exclude' dataset: {args.total_samples} non-ablation samples")
    exclude_indices = non_ablation_indices[: args.total_samples]
    random.shuffle(exclude_indices)

    exclude_path = output_dir / "sft_exclude.jsonl"
    log(f"Saving 'exclude' dataset to {exclude_path}")
    save_dataset_as_jsonl(dataset, exclude_indices, exclude_path)

    # Save metadata
    metadata = {
        "ablation_json": str(Path(args.ablation_json).resolve()),
        "id_field": args.id_field,
        "source_id_field": args.source_id_field,
        "source_dataset": args.source_dataset,
        "total_samples": args.total_samples,
        "n_include": args.n_include,
        "seed": args.seed,
        "ablation_ids_in_json": len(ablation_ids),
        "ablation_samples_found_in_source": len(ablation_indices),
        "non_ablation_samples_in_source": len(non_ablation_indices),
        "include_dataset": {
            "path": str(include_path.resolve()),
            "ablation_samples": args.n_include,
            "non_ablation_samples": n_from_non_ablation_include,
            "total": args.total_samples,
        },
        "exclude_dataset": {
            "path": str(exclude_path.resolve()),
            "ablation_samples": 0,
            "non_ablation_samples": args.total_samples,
            "total": args.total_samples,
        },
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log(f"Saved metadata to {metadata_path}")

    log("Done!")
    log(f"  Include dataset: {include_path}")
    log(f"  Exclude dataset: {exclude_path}")


if __name__ == "__main__":
    main()
