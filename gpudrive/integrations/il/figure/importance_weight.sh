#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./partner_ratio.sh 1 2 3 4

MODEL_PATH=("exp_80000_subset_aix")
MODEL_NAME=("early_attn_s3_0908_113203.pth")
# 반복문 실행
for i in "${!MODEL_PATH[@]}"; do
  MP="${MODEL_PATH[$i]}"
  MN="${MODEL_NAME[$i]}"
  CUDA_VISIBLE_DEVICES=0 python gpudrive/integrations/il/figure/lp_weight.py \
    --model-path "/data/full_version/model/$MP" \
    --model-name "$MN"
done