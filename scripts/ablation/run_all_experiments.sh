#!/bin/bash
# Sync ablation configs from slurm, run all SFT+DPO experiments, and (TODO) sync results back.
#
# Usage:
#   bash scripts/ablation/run_all_experiments.sh
#
# Overrides:
#   CONFIGS_DIR, OUTPUT_BASE, NUM_GPUS, TOTAL_SAMPLES, DPO_SAMPLES

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SLURM_HOST="seoirsem@198.145.108.51"
SLURM_PORT="15115"
SLURM_SSH_KEY="${SLURM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SLURM_CONFIGS_PATH="/workspace-vast/seoirsem/chunky/260327_turf_sft_dpo_experiments/configs"
SLURM_RESULTS_PATH="/workspace-vast/seoirsem/chunky/260327_turf_sft_dpo_experiments/results-pod"

CONFIGS_DIR="${CONFIGS_DIR:-/workspace/chunky/260327_turf_sft_dpo_experiments/configs/configs}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/chunky/260327_turf_sft_dpo_experiments}"
NUM_GPUS="${NUM_GPUS:-6}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-100000}"
DPO_SAMPLES="${DPO_SAMPLES:-50000}"

# ==============================================================================
echo "=============================================="
echo "Step 1: Syncing configs from slurm"
echo "=============================================="

rsync -avz \
    -e "ssh -p $SLURM_PORT -i $SLURM_SSH_KEY" \
    "${SLURM_HOST}:${SLURM_CONFIGS_PATH}" \
    "$(dirname "$CONFIGS_DIR")/"

echo "Config sync complete."
echo ""

# ==============================================================================
echo "=============================================="
echo "Step 2: Checking experiments"
echo "=============================================="

config_files=("$CONFIGS_DIR"/*.json)
if [[ ${#config_files[@]} -eq 0 ]] || [[ ! -f "${config_files[0]}" ]]; then
    echo "No config files found in $CONFIGS_DIR"
    exit 1
fi

echo "Found ${#config_files[@]} config(s):"
for config_file in "${config_files[@]}"; do
    config_name="$(basename "$config_file" .json)"
    output_dir="$OUTPUT_BASE/${config_name}_${TOTAL_SAMPLES%000}k-${DPO_SAMPLES%000}k"
    dpo_out="$output_dir/dpo_include"

    if find "$dpo_out" -name "model.safetensors" 2>/dev/null | grep -q .; then
        echo "  [DONE]    $config_name  ($dpo_out)"
    else
        echo "  [PENDING] $config_name"
    fi
done
echo ""

# ==============================================================================
echo "=============================================="
echo "Step 3: Running experiments"
echo "=============================================="

for config_file in "${config_files[@]}"; do
    config_name="$(basename "$config_file" .json)"
    exp_name="${config_name}_${TOTAL_SAMPLES%000}k-${DPO_SAMPLES%000}k"
    output_dir="$OUTPUT_BASE/$exp_name"
    dpo_out="$output_dir/dpo_include"

    if find "$dpo_out" -name "model.safetensors" 2>/dev/null | grep -q .; then
        echo ">>> Skipping $config_name — already complete"
        continue
    fi

    echo ""
    echo ">>> Running: $config_name"
    echo "    Config:     $config_file"
    echo "    Output dir: $output_dir"
    echo "    Exp name:   $exp_name"
    echo ""

    bash "$SCRIPT_DIR/run_ablation_experiment.sh" \
        --ablation_json "$config_file" \
        --total_samples "$TOTAL_SAMPLES" \
        --output_dir "$output_dir" \
        --exp_name "$exp_name" \
        --dpo_samples "$DPO_SAMPLES" \
        --num_gpus "$NUM_GPUS"

    echo ">>> Completed: $config_name"
done

echo ""
echo "=============================================="
echo "All experiments complete."
echo "=============================================="

# ==============================================================================
echo ""
echo "=============================================="
echo "Step 4: Syncing results back to slurm"
echo "=============================================="

rsync -avz \
    -e "ssh -p $SLURM_PORT -i $SLURM_SSH_KEY" \
    "$OUTPUT_BASE/" \
    "${SLURM_HOST}:${SLURM_RESULTS_PATH}/"

echo "Results sync complete."
