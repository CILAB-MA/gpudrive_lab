#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./road_shortest.sh 1 2 3 4

NUM_WORLD=${1:-200}  # 기본값 200
TOTAL_WORLD_COUNT=${2:-5000}  # 기본값 5000
SWEEP_NAME=${3:-"early_attn5000"}  # 기본값 "early_attn5000"

python algorithms/il/test/run_evaluate.py --sweep-name "$SWEEP_NAME" \
    --total-world-count "$TOTAL_WORLD_COUNT" --num-world "$NUM_WORLD" \
    -pp "0.0" --gpu-id 0 -sft