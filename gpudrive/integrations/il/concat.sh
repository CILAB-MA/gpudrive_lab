# run_pairs.sh
#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_pairs.sh [training|validation|testing] [script_path]
DATASET="${1:-training}"
SCRIPT="${2:-your_script.py}"   # ← 파이썬 파일명 지정

# 출력 폴더(및 global 하위) 보장
mkdir -p /data/full_version/processed/2000_subset/global

for NUM in $(seq 2000 2000 40000); do
  START=$((NUM - 2000))   # 2000→0, 4000→2000, ..., 40000→38000

  OUT="/data/full_version/processed/2000_subset/${DATASET}_${NUM}.npz"
  if [[ -f "$OUT" ]]; then
    echo "[skip] exists: $OUT"
    continue
  fi

  echo "[run] dataset=${DATASET} num_scene=${NUM} start_idx=${START}"
  python "$SCRIPT" --dataset "$DATASET" --num-scene "$NUM" --start-idx "$START"
done

echo "done."