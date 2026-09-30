#!/bin/bash

BATCH_SIZE=${1:-100}
DATASET_SIZE=${2:-2000}
SWEEP_NAME=${3:-"exp_100"}
GPU_ID=${4:-0}
SIM_AGENT=${5:-"log_replay"}

REMOVE_PERC_VALUES=(0.2 0.4 0.6 0.8 1.0)

for PERC in "${REMOVE_PERC_VALUES[@]}"; do
    echo "=== near remove_perc=${PERC} ==="
    python baselines/il/test/run_simulate_mask_far_partners.py \
        --sweep-name "$SWEEP_NAME" \
        --dataset-size "$DATASET_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --gpu-id "$GPU_ID" \
        -sa "$SIM_AGENT" \
        --remove-perc "$PERC" \
        --nearest-first

    echo "=== far remove_perc=${PERC} ==="
    python baselines/il/test/run_simulate_mask_far_partners.py \
        --sweep-name "$SWEEP_NAME" \
        --dataset-size "$DATASET_SIZE" \
        --batch-size "$BATCH_SIZE" \
        --gpu-id "$GPU_ID" \
        -sa "$SIM_AGENT" \
        --remove-perc "$PERC"
done
