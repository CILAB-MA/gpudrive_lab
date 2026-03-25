#!/bin/bash

MODELS=(
  "model_PPO____S_150__03_13_10_39_01_723_000761.pt"
  "runs_PPO____S_200__03_08_06_06_21_203_model_PPO____S_200__03_08_06_06_21_203_001520.pt"
  "model_PPO____S_200__03_04_04_06_56_997_007604.pt"
)
NUM_SCENE=(100 1000 10000)
for i in "${!NUM_SCENE[@]}"; do
  NS="${NUM_SCENE[$i]}"
  CUDA_VISIBLE_DEVICES=1 python gpudrive/integrations/rl/figure/evaluate_lp.py \
    --future-step 10 --num-scene $NS
done