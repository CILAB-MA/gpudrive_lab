#!/bin/bash


GPU_ID=${1:-0}
MODE=${2:-}
EXP_NAME=${3:-}

BATCH_SIZE=${BATCH_SIZE:-100}
DATASET_SIZE=${DATASET_SIZE:-10000}
RATIO_VALUES=(0.2 0.4 0.6 0.8 1.0)

if [[ -z "$MODE" || -z "$EXP_NAME" ]]; then
  echo "Usage: $0 [GPU_ID] [near|far] [EXP_NAME]"
  exit 1
fi

MODE=$(echo "$MODE" | tr '[:upper:]' '[:lower:]')
case "$MODE" in
  near|far) ;;
  *)
    echo "Error: MODE must be 'near' or 'far', got: $MODE"
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

for RATIO in "${RATIO_VALUES[@]}"; do
  echo "========================================"
  echo " GPU=$GPU_ID MODE=$MODE EXP=$EXP_NAME RATIO=$RATIO"
  echo "========================================"

  if [[ "$MODE" == "near" ]]; then
    python gpudrive/integrations/rl/test/run_simulate_mask_far_partners.py \
      --sweep-name "$EXP_NAME" \
      --gpu-id "$GPU_ID" \
      --dataset-size "$DATASET_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --remove-perc "$RATIO" \
      --nearest-first
  else
    # farthest-first perc (tag farpercXX; RATIO=0.0 -> normal)
    python gpudrive/integrations/rl/test/run_simulate_mask_far_partners.py \
      --sweep-name "$EXP_NAME" \
      --gpu-id "$GPU_ID" \
      --dataset-size "$DATASET_SIZE" \
      --batch-size "$BATCH_SIZE" \
      --remove-perc "$RATIO"
  fi
done

echo "Done. Results under /data/after_cvpr/images/mask_far_partners_rl/${EXP_NAME}/"
