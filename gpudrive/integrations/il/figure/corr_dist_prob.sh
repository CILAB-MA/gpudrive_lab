#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./partner_ratio.sh 1 2 3 4

MODEL_PATH=("exp_100" "exp_80000_subset_aix")
NUM_SCENE=(100 80000)
for i in "${!MODEL_PATH[@]}"; do
  MP="${MODEL_PATH[$i]}"
  NS="${NUM_SCENE[$i]}"
  CUDA_VISIBLE_DEVICES=1 python baselines/il/test/evaluate_lp.py \
    --model-path "$MP" --num-scene $NS
done