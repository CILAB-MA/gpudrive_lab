#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./partner_ratio.sh 1 2 3 4

INTERVENTION=("original")
for i in "${!INTERVENTION[@]}"; do
  MODE="${INTERVENTION[$i]}"
  CUDA_VISIBLE_DEVICES=1 python gpudrive/integrations/il/figure/intervention.py \
    --linear-probing "$MODE"
done