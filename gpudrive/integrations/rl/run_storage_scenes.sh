#!/usr/bin/env bash
# Usage: bash run_storage_scenes.sh <num_scene> <CUDA_VISIBLE_DEVICES>
#   e.g. bash run_storage_scenes.sh 100 1
# Runs storage.py (training + validation) then data_concat.py for both; paths use scene_{num_scene}.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

NUM_SCENE="${1:?Usage: $0 <num_scene> <CUDA_VISIBLE_DEVICES>}"
export CUDA_VISIBLE_DEVICES="${2:?Usage: $0 <num_scene> <CUDA_VISIBLE_DEVICES>}"

MODELS=(
  "model_PPO____S_150__03_13_10_39_01_723_000761.pt"
  "runs_PPO____S_200__03_08_06_06_21_203_model_PPO____S_200__03_08_06_06_21_203_001520.pt"
  "model_PPO____S_200__03_04_04_06_56_997_007604.pt"
)
case "$NUM_SCENE" in
  100)   MN="${MODELS[0]}" ;;
  1000)  MN="${MODELS[1]}" ;;
  10000) MN="${MODELS[2]}" ;;
  *)     MN="${MODELS[0]}" ;;
esac

echo "========== storage.py (scene_${NUM_SCENE}, GPU $CUDA_VISIBLE_DEVICES) =========="
python gpudrive/integrations/rl/storage.py --num-scene "$NUM_SCENE" --model-name "$MN" \
  --dataset training --dataset-size "$NUM_SCENE" --no-save-label
python gpudrive/integrations/rl/storage.py --num-scene "$NUM_SCENE" --model-name "$MN" \
  --dataset validation --dataset-size 2500

echo "========== data_concat.py (scene_${NUM_SCENE}) =========="
python gpudrive/integrations/il/data_concat.py --scene "$NUM_SCENE" --dataset validation --num-scene 2500
python gpudrive/integrations/il/data_concat.py --scene "$NUM_SCENE" --dataset training --num-scene "$NUM_SCENE"
