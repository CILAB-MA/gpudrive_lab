#!/bin/bash

exp_name = $1
model_name = $2
loss_name = $3
save_name = "${1}_${2}_${3}"
python baselines/il/run_bc_from_scratch.py --exp-name $exp_name --model-name $model_name --loss-name $loss_name
python algorithms/il/evaluate.py --make-video --model-name $save_name --dataset "train"
python algorithms/il/evaluate.py --make-video --model-name $save_name --dataset "valid"