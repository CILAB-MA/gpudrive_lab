#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./partner_ratio.sh 1 2 3 4

MODEL_PATH=("exp_100" "exp_500" "exp_5000" "exp_10000_v2")
MODEL_NAME=("early_attn_s42_0901_145943.pth" "early_attn_s42_0901_085709.pth" "early_attn_s3_0723_042559.pth" "early_attn_s11_0802_051525.pth")
# 반복문 실행
for i in "${!MODEL_PATH[@]}"; do
  MP="${MODEL_PATH[$i]}"
  MN="${MODEL_NAME[$i]}"
  CUDA_VISIBLE_DEVICES=1 python gpudrive/integrations/il/figure/lp_weight.py \
    --model-path "/data/full_version/model/$MP" \
    --model-name "$MN"
done