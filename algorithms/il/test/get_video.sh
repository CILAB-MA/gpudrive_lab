#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./get_video.sh 1 2 3

NUM_WORLD=${1:-50}  
MODEL_PATH=${2:-"/data/model/early_attn5000"}

MODEL_NAMES=($(ls "$MODEL_PATH"/*.pth | xargs -n 1 basename))

for MODEL in "${MODEL_NAMES[@]}"; do
    python algorithms/il/test/evaluate.py --num-world "$NUM_WORLD" \
        -pp "0.0" -mp "$MODEL_PATH" -mn "$MODEL" -spt -mv
    python algorithms/il/test/evaluate.py --num-world "$NUM_WORLD" \
        -pp "0.0" -mp "$MODEL_PATH" -mn "$MODEL" -mv
done
