#!/bin/bash

MODEL_PATH=("scene_100" "scene_10000")
MODELS=(
  "model_PPO____S_150__03_13_10_39_01_723_000761.pt"
  "runs_PPO____S_200__03_08_06_06_21_203_model_PPO____S_200__03_08_06_06_21_203_001520.pt"
  "model_PPO____S_200__03_04_04_06_56_997_007604.pt"
)
NUM_SCENE=(100 10000)
for i in "${!MODEL_PATH[@]}"; do
  MP="${MODEL_PATH[$i]}"
  NS="${NUM_SCENE[$i]}"
  CUDA_VISIBLE_DEVICES=1 python gpudrive/integration/il/figure/evaluate_lp.py \
    --model-path "scene_$MP/${MODELS[$i]}" --num-scene $NS
done