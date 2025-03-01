#!/bin/bash

# Bash Script for Running WandB Agent with GPU ID, Sweep ID, and WandB API Key
# Usage: ./run_wandb_docker.sh 1 2 3 4

CUDA=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+")

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
    CUDA_VISIBLE_DEVICES=$GPU python /app/gpudrive_lab/baselines/il/run_bc_from_scratch.py --use-mask --use-wandb -en "$ENV" &
    
    sleep 60
done

# 기존 wait (이전 실행된 백그라운드 프로세스가 끝날 때까지 대기)
wait

# 추가: 모든 `run_bc_from_scratch.py`가 종료될 때까지 대기
while pgrep -f "python /app/gpudrive_lab/baselines/il/run_bc_from_scratch.py" > /dev/null; do
    sleep 5
done
