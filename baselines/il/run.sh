#!/bin/bash

exp_name=$1
model_name=$2
loss_name=$3
num_stack=$4
save_name="${1}_${2}_${3}_stack${4}"
python baselines/il/run_bc_from_scratch.py --exp-name $exp_name --model-name $model_name --loss-name $loss_name --num-stack $num_stack
python algorithms/il/evaluate.py --make-video --model-name $save_name --dataset "train" --num-stack $num_stack
python algorithms/il/evaluate.py --make-video --model-name $save_name --dataset "valid" --num-stack $num_stack