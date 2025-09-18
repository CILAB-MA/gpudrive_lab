#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./partner_ratio.sh 1 2 3 4

MODEL_PATH=("exp_100" "exp_500" "exp_5000" "exp_10000_v2")
# 반복문 실행
for i in "${!MODEL_PATH[@]}"; do
  MP="${MODEL_PATH[$i]}"
  MN="${MODEL_NAME[$i]}"
  CUDA_VISIBLE_DEVICES=1 python baselines/il/test/evaluate_lp.py \
    --model-path "$MP"
done