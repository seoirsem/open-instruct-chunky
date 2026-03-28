#!/bin/bash
# Sync ablation configs from slurm, run all SFT+DPO experiments, sync results back.
# Each config runs both include (with samples) and exclude (without) variants.
#
# Usage:
#   bash scripts/ablation/run_all_experiments.sh
#
# Overrides (via env):
#   CONFIGS_DIR, OUTPUT_BASE, NUM_GPUS, TOTAL_SAMPLES, DPO_SAMPLES, POLL_INTERVAL

set -e

POLL_INTERVAL="${POLL_INTERVAL:-60}"    # seconds between passes when all experiments are done
RSYNC_INTERVAL="${RSYNC_INTERVAL:-300}" # seconds between background rsync pushes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SLURM_HOST="seoirsem@198.145.108.51"
SLURM_PORT="15115"
SLURM_SSH_KEY="${SLURM_SSH_KEY:-$HOME/.ssh/id_ed25519}"
SLURM_CONFIGS_PATH="/workspace-vast/seoirsem/chunky/260327_turf_sft_dpo_experiments/configs"
SLURM_RESULTS_PATH="/workspace-vast/seoirsem/chunky/260327_turf_sft_dpo_experiments/results-pod"

CONFIGS_DIR="${CONFIGS_DIR:-/workspace/chunky/260327_turf_sft_dpo_experiments/configs/configs}"
OUTPUT_BASE="${OUTPUT_BASE:-/workspace/chunky/260327_turf_sft_dpo_experiments}"
LOG_FILE="${OUTPUT_BASE}/run_all.log"

mkdir -p "$OUTPUT_BASE"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"
NUM_GPUS="${NUM_GPUS:-6}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-100000}"
DPO_SAMPLES="${DPO_SAMPLES:-50000}"

push_results() {
    while true; do
        sleep "$RSYNC_INTERVAL"
        echo "[rsync] Pushing results to slurm..."
        rsync -avP --ignore-existing \
            -e "ssh -p $SLURM_PORT -i $SLURM_SSH_KEY" \
            "$OUTPUT_BASE/" \
            "${SLURM_HOST}:/workspace-vast/seoirsem/chunky/260327_turf_sft_dpo_experiments/" \
            2>&1 | tail -5
        echo "[rsync] Done."
    done
}

# Start background rsync loop and record PID so it dies with this script
push_results &
RSYNC_PID=$!
trap "kill $RSYNC_PID 2>/dev/null" EXIT

is_done() {
    local dir="$1"
    find "$dir" -name "model.safetensors" -o -name "pytorch_model.bin" 2>/dev/null | grep -q .
}

while true; do

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

    is_done "$output_dir/dpo_include"  && include_status="done" || include_status="pending"
    is_done "$output_dir/dpo_exclude"  && exclude_status="done" || exclude_status="pending"
    echo "  $config_name  [include: $include_status] [exclude: $exclude_status]"
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

    if is_done "$output_dir/dpo_include" && is_done "$output_dir/dpo_exclude"; then
        echo ">>> Skipping $config_name — both variants complete"
        continue
    fi

    echo ""
    echo ">>> Running: $config_name"
    echo "    Config:     $config_file"
    echo "    Output dir: $output_dir"
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

rsync -avP --ignore-existing \
    -e "ssh -p $SLURM_PORT -i $SLURM_SSH_KEY" \
    "$OUTPUT_BASE/" \
    "seoirsem@198.145.108.51:/workspace-vast/seoirsem/chunky/260327_turf_sft_dpo_experiments/"

echo "Results sync complete."

echo ""
echo "Sleeping ${POLL_INTERVAL}s before next pass... (Ctrl+C to stop)"
sleep "$POLL_INTERVAL"

done
