#!/bin/bash
# Run SFT + DPO ablation experiment
#
# Usage (with pre-prepared data):
#   bash scripts/ablation/run_ablation_experiment.sh \
#       --data_dir ./ablation_data \
#       --output_dir ./ablation_output \
#       --exp_name my_ablation_v1
#
# Usage (prepare data + run in one command):
#   bash scripts/ablation/run_ablation_experiment.sh \
#       --ablation_json /path/to/ablation.json \
#       --total_samples 100000 \
#       --n_include 5000 \
#       --output_dir ./ablation_output \
#       --exp_name my_ablation_v1
#
# Usage (baseline - random data, no ablation):
#   bash scripts/ablation/run_ablation_experiment.sh \
#       --baseline \
#       --total_samples 100000 \
#       --output_dir ./baseline_output \
#       --exp_name baseline_100k
#
# Optional overrides (via environment or flags):
#   --model, --dpo_data, --dpo_samples, --sft_lr, --dpo_lr, etc.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Source default config
source "$SCRIPT_DIR/config.sh"

# Data preparation defaults
TOTAL_SAMPLES="${TOTAL_SAMPLES:-100000}"
N_INCLUDE="${N_INCLUDE:-0}"

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
        --baseline) BASELINE=true; INCLUDE_ONLY=true; shift ;;
        --sft_data) SFT_DATA_OVERRIDE="$2"; shift 2 ;;
        # Data preparation flags
        --ablation_json) ABLATION_JSON="$2"; shift 2 ;;
        --total_samples) TOTAL_SAMPLES="$2"; shift 2 ;;
        --n_include) N_INCLUDE="$2"; shift 2 ;;
        --source_dataset) SOURCE_DATASET="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: --output_dir is required"
    exit 1
fi

if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="ablation_$(date +%Y%m%d_%H%M%S)"
    echo "Using default exp_name: $EXP_NAME"
fi

OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

# Prepare data if --ablation_json provided or --baseline with --total_samples
if [[ -n "$ABLATION_JSON" ]] || [[ "$BASELINE" == "true" && -z "$DATA_DIR" && -z "$SFT_DATA_OVERRIDE" ]]; then
    DATA_DIR="$OUTPUT_DIR/data"

    # For baseline without ablation_json, create empty one
    if [[ "$BASELINE" == "true" && -z "$ABLATION_JSON" ]]; then
        mkdir -p "$DATA_DIR"
        ABLATION_JSON="$DATA_DIR/empty_ablation.json"
        echo '[]' > "$ABLATION_JSON"
        N_INCLUDE=0
    fi

    echo "=============================================="
    echo "Preparing data..."
    echo "=============================================="
    echo "  Ablation JSON: $ABLATION_JSON"
    echo "  Total samples: $TOTAL_SAMPLES"
    echo "  N include: $N_INCLUDE"
    echo "  Output: $DATA_DIR"
    echo "=============================================="

    uv run python "$SCRIPT_DIR/prepare_ablation_data.py" \
        --ablation_json "$ABLATION_JSON" \
        --total_samples "$TOTAL_SAMPLES" \
        --n_include "$N_INCLUDE" \
        --output_dir "$DATA_DIR" \
        --seed "$SEED" \
        ${SOURCE_DATASET:+--source_dataset "$SOURCE_DATASET"}

    echo "Data preparation complete."
    echo ""
fi

# Handle data paths - either --sft_data directly or --data_dir with include/exclude files
if [[ -n "$SFT_DATA_OVERRIDE" ]]; then
    # Direct SFT data file provided (for baseline runs)
    SFT_DATA_OVERRIDE="$(realpath "$SFT_DATA_OVERRIDE")"
    INCLUDE_DATA="$SFT_DATA_OVERRIDE"
    EXCLUDE_DATA="$SFT_DATA_OVERRIDE"  # Same file for both (only one will run with --baseline)
    if [[ ! -f "$INCLUDE_DATA" ]]; then
        echo "Error: SFT data not found: $INCLUDE_DATA"
        exit 1
    fi
elif [[ -n "$DATA_DIR" ]]; then
    # Standard ablation data directory
    DATA_DIR="$(realpath "$DATA_DIR")"
    INCLUDE_DATA="$DATA_DIR/sft_include.jsonl"
    EXCLUDE_DATA="$DATA_DIR/sft_exclude.jsonl"

    if [[ ! -f "$INCLUDE_DATA" ]]; then
        echo "Error: Include data not found: $INCLUDE_DATA"
        exit 1
    fi

    if [[ ! -f "$EXCLUDE_DATA" ]] && [[ "$INCLUDE_ONLY" != "true" ]]; then
        echo "Error: Exclude data not found: $EXCLUDE_DATA"
        exit 1
    fi
else
    echo "Error: Either --data_dir or --sft_data is required"
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

# Wandb tracking enabled (required for multi-GPU DeepSpeed)

# Function to run SFT
run_sft() {
    local variant=$1  # "include" or "exclude"
    local data_path=$2
    local output_path=$3

    # Check if checkpoint already exists (look for model.safetensors in any subdirectory)
    if find "$output_path" -name "model.safetensors" 2>/dev/null | grep -q .; then
        echo ""
        echo ">>> Skipping SFT ($variant) - checkpoint already exists at $output_path"
        return 0
    fi

    echo ""
    echo ">>> Running SFT ($variant)..."
    echo "    Data: $data_path"
    echo "    Output: $output_path"
    echo ""

    uv run accelerate launch \
        --mixed_precision bf16 \
        --num_processes "$NUM_GPUS" \
        --main_process_port 0 \
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
        --with_tracking \
        --report_to wandb \
        --push_to_hub false \
        --try_launch_beaker_eval_jobs false

    echo ">>> SFT ($variant) complete: $output_path"
}

# Function to run DPO
run_dpo() {
    local variant=$1  # "include" or "exclude"
    local sft_checkpoint=$2
    local output_path=$3

    # Check if DPO checkpoint already exists
    if find "$output_path" -name "model.safetensors" 2>/dev/null | grep -q .; then
        echo ""
        echo ">>> Skipping DPO ($variant) - checkpoint already exists at $output_path"
        return 0
    fi

    # Find the actual SFT model path (it's in a subdirectory)
    local sft_model_path
    sft_model_path=$(find "$sft_checkpoint" -name "model.safetensors" 2>/dev/null | head -1 | xargs dirname)

    if [[ -z "$sft_model_path" ]]; then
        echo ""
        echo ">>> Skipping DPO ($variant) - SFT checkpoint not found at $sft_checkpoint"
        return 1
    fi

    echo ""
    echo ">>> Running DPO ($variant)..."
    echo "    SFT checkpoint: $sft_model_path"
    echo "    Output: $output_path"
    echo ""

    # Use random port to avoid conflicts
    local dpo_port=$((29600 + RANDOM % 1000))
    # Set local cache path for reference logprobs
    REFERENCE_LOGPROBS_CACHE_PATH="$OUTPUT_DIR/.reference_logprobs_cache" \
    uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$dpo_port" \
        "$REPO_ROOT/open_instruct/dpo_tune_cache.py" \
        --exp_name "${EXP_NAME}_dpo_${variant}" \
        --model_name_or_path "$sft_model_path" \
        --tokenizer_name_or_path "$sft_model_path" \
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
        --push_to_hub false \
        --try_launch_beaker_eval_jobs false

    echo ">>> DPO ($variant) complete: $output_path"
}

# Output paths - use "baseline" naming for baseline mode
if [[ "$BASELINE" == "true" ]]; then
    SFT_INCLUDE_OUT="$OUTPUT_DIR/sft_baseline"
    DPO_INCLUDE_OUT="$OUTPUT_DIR/dpo_baseline"
else
    SFT_INCLUDE_OUT="$OUTPUT_DIR/sft_include"
    DPO_INCLUDE_OUT="$OUTPUT_DIR/dpo_include"
fi
SFT_EXCLUDE_OUT="$OUTPUT_DIR/sft_exclude"
DPO_EXCLUDE_OUT="$OUTPUT_DIR/dpo_exclude"

# Run experiments
if [[ "$SKIP_SFT" != "true" ]]; then
    if [[ "$EXCLUDE_ONLY" != "true" ]]; then
        variant_name=$([[ "$BASELINE" == "true" ]] && echo "baseline" || echo "include")
        run_sft "$variant_name" "$INCLUDE_DATA" "$SFT_INCLUDE_OUT"
    fi

    if [[ "$INCLUDE_ONLY" != "true" ]]; then
        run_sft "exclude" "$EXCLUDE_DATA" "$SFT_EXCLUDE_OUT"
    fi
fi

if [[ "$SKIP_DPO" != "true" ]]; then
    if [[ "$EXCLUDE_ONLY" != "true" ]]; then
        variant_name=$([[ "$BASELINE" == "true" ]] && echo "baseline" || echo "include")
        run_dpo "$variant_name" "$SFT_INCLUDE_OUT" "$DPO_INCLUDE_OUT"
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
if [[ "$BASELINE" == "true" ]]; then
    echo "  SFT (baseline): $SFT_INCLUDE_OUT"
    echo "  DPO (baseline): $DPO_INCLUDE_OUT"
elif [[ "$EXCLUDE_ONLY" != "true" ]]; then
    echo "  SFT (include):  $SFT_INCLUDE_OUT"
    echo "  DPO (include):  $DPO_INCLUDE_OUT"
fi
if [[ "$INCLUDE_ONLY" != "true" ]] && [[ "$BASELINE" != "true" ]]; then
    echo "  SFT (exclude):  $SFT_EXCLUDE_OUT"
    echo "  DPO (exclude):  $DPO_EXCLUDE_OUT"
fi
echo "=============================================="
