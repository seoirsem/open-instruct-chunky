#!/bin/bash
# Run SFT + DPO ablation experiment

# Fully disable wandb - prevent it from hooking stdout
export WANDB_MODE="${WANDB_MODE:-disabled}"
export WANDB_DISABLED=true
export WANDB_SILENT=true
export WANDB_CONSOLE=off
#
# Usage:
#   bash scripts/ablation/run_ablation_experiment.sh \
#       --data_dir ./ablation_data \
#       --output_dir ./ablation_output \
#       --exp_name my_ablation_v1
#
# Optional overrides (via environment or flags):
#   --model, --dpo_data, --dpo_samples, --sft_lr, --dpo_lr, etc.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source default config
source "$SCRIPT_DIR/config.sh"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --exp_name) EXP_NAME="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --dpo_data) DPO_DATA="$2"; shift 2 ;;
        --dpo_samples) DPO_SAMPLES="$2"; shift 2 ;;
        --sft_lr) SFT_LR="$2"; shift 2 ;;
        --dpo_lr) DPO_LR="$2"; shift 2 ;;
        --sft_epochs) SFT_EPOCHS="$2"; shift 2 ;;
        --dpo_epochs) DPO_EPOCHS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num_gpus) NUM_GPUS="$2"; shift 2 ;;
        --chat_template) CHAT_TEMPLATE="$2"; shift 2 ;;
        --skip_sft) SKIP_SFT=true; shift ;;
        --skip_dpo) SKIP_DPO=true; shift ;;
        --include_only) INCLUDE_ONLY=true; shift ;;
        --exclude_only) EXCLUDE_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$DATA_DIR" ]]; then
    echo "Error: --data_dir is required"
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir is required"
    exit 1
fi

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="ablation_$(date +%Y%m%d_%H%M%S)"
    echo "Using default exp_name: $EXP_NAME"
fi

# Resolve paths
DATA_DIR="$(realpath "$DATA_DIR")"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
INCLUDE_DATA="$DATA_DIR/sft_include.jsonl"
EXCLUDE_DATA="$DATA_DIR/sft_exclude.jsonl"

# Verify data files exist
if [[ ! -f "$INCLUDE_DATA" ]]; then
    echo "Error: Include data not found: $INCLUDE_DATA"
    exit 1
fi

if [[ ! -f "$EXCLUDE_DATA" ]]; then
    echo "Error: Exclude data not found: $EXCLUDE_DATA"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Save experiment config
cat > "$OUTPUT_DIR/experiment_config.json" << EOF
{
    "exp_name": "$EXP_NAME",
    "data_dir": "$DATA_DIR",
    "output_dir": "$OUTPUT_DIR",
    "model": "$MODEL_NAME",
    "chat_template": "$CHAT_TEMPLATE",
    "sft": {
        "lr": "$SFT_LR",
        "epochs": $SFT_EPOCHS,
        "batch_size": $SFT_BATCH_SIZE,
        "grad_accum": $SFT_GRAD_ACCUM,
        "seq_len": $SFT_SEQ_LEN
    },
    "dpo": {
        "lr": "$DPO_LR",
        "epochs": $DPO_EPOCHS,
        "batch_size": $DPO_BATCH_SIZE,
        "grad_accum": $DPO_GRAD_ACCUM,
        "seq_len": $DPO_SEQ_LEN,
        "beta": $DPO_BETA,
        "data": "$DPO_DATA",
        "samples": $DPO_SAMPLES
    },
    "seed": $SEED,
    "num_gpus": $NUM_GPUS
}
EOF

echo "=============================================="
echo "Ablation Experiment: $EXP_NAME"
echo "=============================================="
echo "Model: $MODEL_NAME"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# DeepSpeed config
DS_CONFIG="$REPO_ROOT/configs/ds_configs/stage3_no_offloading_accelerate.conf"

# Note: wandb/tracking disabled for simplicity. Re-enable manually if needed.

# Function to run SFT
run_sft() {
    local variant=$1  # "include" or "exclude"
    local data_path=$2
    local output_path=$3

    echo ""
    echo ">>> Running SFT ($variant)..."
    echo "    Data: $data_path"
    echo "    Output: $output_path"
    echo ""

    WANDB_MODE=disabled WANDB_DISABLED=true uv run accelerate launch \
        --mixed_precision bf16 \
        --num_processes "$NUM_GPUS" \
        --use_deepspeed \
        --deepspeed_config_file "$DS_CONFIG" \
        "$REPO_ROOT/open_instruct/finetune.py" \
        --exp_name "${EXP_NAME}_sft_${variant}" \
        --model_name_or_path "$MODEL_NAME" \
        --tokenizer_name "$MODEL_NAME" \
        --dataset_mixer_list "$data_path" 1.0 \
        --max_seq_length "$SFT_SEQ_LEN" \
        --per_device_train_batch_size "$SFT_BATCH_SIZE" \
        --gradient_accumulation_steps "$SFT_GRAD_ACCUM" \
        --learning_rate "$SFT_LR" \
        --lr_scheduler_type linear \
        --warmup_ratio "$SFT_WARMUP_RATIO" \
        --weight_decay 0.0 \
        --num_train_epochs "$SFT_EPOCHS" \
        --output_dir "$output_path" \
        --logging_steps 1 \
        --use_flash_attn \
        --gradient_checkpointing \
        --chat_template_name "$CHAT_TEMPLATE" \
        --seed "$SEED" \
        --report_to none

    echo ">>> SFT ($variant) complete: $output_path"
}

# Function to run DPO
run_dpo() {
    local variant=$1  # "include" or "exclude"
    local sft_checkpoint=$2
    local output_path=$3

    echo ""
    echo ">>> Running DPO ($variant)..."
    echo "    SFT checkpoint: $sft_checkpoint"
    echo "    Output: $output_path"
    echo ""

    WANDB_MODE=disabled WANDB_DISABLED=true uv run torchrun --nproc_per_node="$NUM_GPUS" \
        "$REPO_ROOT/open_instruct/dpo.py" \
        --exp_name "${EXP_NAME}_dpo_${variant}" \
        --model_name_or_path "$sft_checkpoint" \
        --tokenizer_name_or_path "$sft_checkpoint" \
        --mixer_list "$DPO_DATA" "$DPO_SAMPLES" \
        --max_seq_length "$DPO_SEQ_LEN" \
        --per_device_train_batch_size "$DPO_BATCH_SIZE" \
        --gradient_accumulation_steps "$DPO_GRAD_ACCUM" \
        --learning_rate "$DPO_LR" \
        --lr_scheduler_type linear \
        --warmup_ratio "$DPO_WARMUP_RATIO" \
        --weight_decay 0.0 \
        --num_epochs "$DPO_EPOCHS" \
        --beta "$DPO_BETA" \
        --output_dir "$output_path" \
        --logging_steps 1 \
        --chat_template_name "$CHAT_TEMPLATE" \
        --seed "$SEED" \
        --push_to_hub false

    echo ">>> DPO ($variant) complete: $output_path"
}

# Output paths
SFT_INCLUDE_OUT="$OUTPUT_DIR/sft_include"
SFT_EXCLUDE_OUT="$OUTPUT_DIR/sft_exclude"
DPO_INCLUDE_OUT="$OUTPUT_DIR/dpo_include"
DPO_EXCLUDE_OUT="$OUTPUT_DIR/dpo_exclude"

# Run experiments
if [[ "$SKIP_SFT" != "true" ]]; then
    if [[ "$EXCLUDE_ONLY" != "true" ]]; then
        run_sft "include" "$INCLUDE_DATA" "$SFT_INCLUDE_OUT"
    fi

    if [[ "$INCLUDE_ONLY" != "true" ]]; then
        run_sft "exclude" "$EXCLUDE_DATA" "$SFT_EXCLUDE_OUT"
    fi
fi

if [[ "$SKIP_DPO" != "true" ]]; then
    if [[ "$EXCLUDE_ONLY" != "true" ]]; then
        run_dpo "include" "$SFT_INCLUDE_OUT" "$DPO_INCLUDE_OUT"
    fi

    if [[ "$INCLUDE_ONLY" != "true" ]]; then
        run_dpo "exclude" "$SFT_EXCLUDE_OUT" "$DPO_EXCLUDE_OUT"
    fi
fi

echo ""
echo "=============================================="
echo "Experiment complete!"
echo "=============================================="
echo "Checkpoints:"
if [[ "$EXCLUDE_ONLY" != "true" ]]; then
    echo "  SFT (include):  $SFT_INCLUDE_OUT"
    echo "  DPO (include):  $DPO_INCLUDE_OUT"
fi
if [[ "$INCLUDE_ONLY" != "true" ]]; then
    echo "  SFT (exclude):  $SFT_EXCLUDE_OUT"
    echo "  DPO (exclude):  $DPO_EXCLUDE_OUT"
fi
echo "=============================================="
