# Data Ablation Experiments

Scripts for running SFT→DPO data ablation experiments. Given a set of samples to ablate (identified by ID), creates two training runs:

- **Include**: n samples from the ablation set + remaining samples from non-ablation data
- **Exclude**: 0 samples from the ablation set + all samples from non-ablation data

Both runs use identical DPO data, allowing you to measure the impact of specific training samples.

## Quick Start

```bash
# 1. Prepare ablation datasets
python scripts/ablation/prepare_ablation_data.py \
    --ablation_json /path/to/samples_to_ablate.json \
    --id_field "id" \
    --total_samples 100000 \
    --n_include 5000 \
    --output_dir ./ablation_data

# 2. Run the experiment
bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./ablation_output \
    --exp_name my_ablation_v1
```

## Data Preparation

### Input Format

Your ablation JSON should be a list of samples with an ID field:

```json
[
    {"id": "sample_001", "messages": [...]},
    {"id": "sample_002", "messages": [...]},
    ...
]
```

Or a dict with a `data` or `samples` key:

```json
{
    "data": [
        {"id": "sample_001", ...},
        ...
    ]
}
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ablation_json` | required | Path to JSON with samples to ablate |
| `--id_field` | `id` | Field name for ID in ablation JSON |
| `--source_id_field` | `id` | Field name for ID in source dataset |
| `--source_dataset` | `allenai/tulu-3-sft-mixture` | Source HuggingFace dataset |
| `--total_samples` | `100000` | Total samples per output dataset |
| `--n_include` | required | Number of ablation samples in "include" set |
| `--output_dir` | `./ablation_data` | Output directory |
| `--seed` | `42` | Random seed |

### Output

```
ablation_data/
├── sft_include.jsonl    # Include dataset (n ablation + rest non-ablation)
├── sft_exclude.jsonl    # Exclude dataset (all non-ablation)
└── metadata.json        # Experiment metadata
```

## Running Experiments

### Basic Usage

```bash
bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./ablation_output \
    --exp_name my_experiment
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | required | Directory with prepared data |
| `--output_dir` | required | Output directory for checkpoints |
| `--exp_name` | auto-generated | Experiment name |
| `--model` | `allenai/OLMo-2-0425-7B` | Base model |
| `--chat_template` | `olmo` | Chat template name |
| `--dpo_data` | `allenai/tulu-3-wildchat-reused-on-policy-8b` | DPO dataset |
| `--dpo_samples` | `50000` | Number of DPO samples |
| `--sft_lr` | `2e-5` | SFT learning rate |
| `--dpo_lr` | `5e-7` | DPO learning rate |
| `--sft_epochs` | `2` | SFT epochs |
| `--dpo_epochs` | `1` | DPO epochs |
| `--seed` | `42` | Random seed |
| `--num_gpus` | `8` | Number of GPUs |

### Partial Runs

```bash
# Run only the "include" variant
bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./output \
    --include_only

# Run only the "exclude" variant
bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./output \
    --exclude_only

# Skip SFT (if checkpoints already exist)
bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./output \
    --skip_sft

# Skip DPO (run only SFT)
bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./output \
    --skip_dpo
```

### Output Structure

```
ablation_output/
├── experiment_config.json   # Full experiment configuration
├── sft_include/             # SFT checkpoint (include variant)
├── sft_exclude/             # SFT checkpoint (exclude variant)
├── dpo_include/             # DPO checkpoint (from sft_include)
└── dpo_exclude/             # DPO checkpoint (from sft_exclude)
```

## Customizing Hyperparameters

Edit `config.sh` or pass environment variables:

```bash
SFT_LR=1e-5 DPO_LR=1e-6 bash scripts/ablation/run_ablation_experiment.sh \
    --data_dir ./ablation_data \
    --output_dir ./output
```

## Estimated Runtime

On 8x H200 with a 7B model:

| Stage | Time |
|-------|------|
| Data prep | ~10-15 min |
| SFT include (100k, 2 epochs) | ~45 min |
| SFT exclude (100k, 2 epochs) | ~45 min |
| DPO include (50k, 1 epoch) | ~30 min |
| DPO exclude (50k, 1 epoch) | ~30 min |
| **Total** | **~3 hours** |
