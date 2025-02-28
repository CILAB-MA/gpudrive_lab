#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./run.sh EN1 EN2 EN3 ...

# Default Values
WANDB_API_KEY=$(grep "wandb_key" /app/gpudrive_lab/private.yaml | cut -d '"' -f 2)
ENTITY=$(grep "entity" /app/gpudrive_lab/private.yaml | cut -d '"' -f 2)
PROJECT=$(grep "main_project" /app/gpudrive_lab/private.yaml | cut -d '"' -f 2)
SWEEP_ID=""

# SWEEP_ID 인자 설정
SWEEP_ID=$(wandb sweep /app/gpudrive_lab/algorithms/il/analyze/linear_probing/sweep.yaml | awk '/Created sweep with ID:/ {print $NF}')
if [ -z "$SWEEP_ID" ]; then
    echo "Failed to create a new WandB sweep!"
    exit 1
fi

# WandB agent 실행 (백그라운드에서 실행)
wandb agent $ENTITY/$PROJECT/$SWEEP_ID &  

# run_bc_from_scratch.py 실행
for ENV in "$@"; do
    # 가장 여유 있는 GPU 선택
    echo "Searching for available GPU for process with -en $ENV..."
    gpu_info=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    GPU=$(echo "$gpu_info" | awk -F, '{if ($2 > 0) print $1, $2}' | sort -k2 -nr | head -n1 | cut -d' ' -f1)

    if [ -z "$GPU" ]; then
        echo "No available GPU found for process with -en $ENV! Skipping..."
        continue
    fi

    echo "Assigning GPU $GPU to process with -en $ENV"
    
    # Python 실행 (각각 다른 GPU에 할당)
    CUDA_VISIBLE_DEVICES=$GPU python /app/gpudrive_lab/baselines/il/run_bc_from_scratch.py --use-mask --use-wandb -en "$ENV" --sweep-id $SWEEP_ID &
    
    sleep 60
done

# 기존 wait (이전 실행된 백그라운드 프로세스가 끝날 때까지 대기)
wait

# 추가: 모든 `run_bc_from_scratch.py`가 종료될 때까지 대기
while pgrep -f "python /app/gpudrive_lab/baselines/il/run_bc_from_scratch.py" > /dev/null; do
    sleep 5
done
