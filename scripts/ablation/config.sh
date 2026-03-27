#!/bin/bash
# Default configuration for ablation experiments
# Source this file or override variables before running

# Model configuration
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-tulu}"

# SFT hyperparameters (optimized for 8x H200, 7B model)
SFT_LR="${SFT_LR:-2e-5}"
SFT_EPOCHS="${SFT_EPOCHS:-2}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
SFT_GRAD_ACCUM="${SFT_GRAD_ACCUM:-2}"
SFT_SEQ_LEN="${SFT_SEQ_LEN:-4096}"
SFT_WARMUP_RATIO="${SFT_WARMUP_RATIO:-0.03}"

# DPO hyperparameters
DPO_LR="${DPO_LR:-5e-7}"
DPO_EPOCHS="${DPO_EPOCHS:-1}"
DPO_BATCH_SIZE="${DPO_BATCH_SIZE:-1}"
DPO_GRAD_ACCUM="${DPO_GRAD_ACCUM:-8}"
DPO_SEQ_LEN="${DPO_SEQ_LEN:-2048}"
DPO_BETA="${DPO_BETA:-5}"
DPO_WARMUP_RATIO="${DPO_WARMUP_RATIO:-0.1}"

# DPO data (same for both experiments)
DPO_DATA="${DPO_DATA:-allenai/tulu-3-wildchat-reused-on-policy-8b}"
DPO_SAMPLES="${DPO_SAMPLES:-50000}"

# Hardware configuration
NUM_GPUS="${NUM_GPUS:-8}"

# Tracking (wandb enabled by default for multi-GPU stability)
SEED="${SEED:-42}"
