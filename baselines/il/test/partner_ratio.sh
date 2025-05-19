#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./partner_ratio.sh 1 2 3 4

BATCH_SIZE=${1:-200}  # 기본값 200
DATASET_SIZE=${2:-5000}  # 기본값 5000
SWEEP_NAME=${3:-"early_attn5000"}  # 기본값 "early_attn5000"
GPU_ID=${4:-0}  # 기본값 "early_attn5000"

# pp 값 리스트
PP_VALUES=(0.0 0.2 0.4 0.6 0.8 1.0)

# 반복문 실행
for PP in "${PP_VALUES[@]}"; do
    python baselines/il/test/run_simulation.py --sweep-name "$SWEEP_NAME" \
        --dataset-size "$DATASET_SIZE" --batch-size "$BATCH_SIZE" \
        -pp "$PP" --gpu-id "$GPU_ID"
done