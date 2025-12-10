#!/bin/bash

MODEL_PATH=("exp_100" "exp_80000_subset_aix")
NUM_SCENE=(100 80000)
for i in "${!MODEL_PATH[@]}"; do
  MP="${MODEL_PATH[$i]}"
  NS="${NUM_SCENE[$i]}"
  CUDA_VISIBLE_DEVICES=1 python baselines/il/test/evaluate_lp.py \
    --model-path "$MP" --num-scene $NS
done