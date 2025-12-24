#!/bin/bash

INTERVENTION=("original")
for i in "${!INTERVENTION[@]}"; do
  MODE="${INTERVENTION[$i]}"
  CUDA_VISIBLE_DEVICES=1 python gpudrive/integrations/il/figure/intervention.py \
    --linear-probing "$MODE"
done