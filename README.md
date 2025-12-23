GPUDrive-Interpretation
========

<!-- ![Python version](https://img.shields.io/badge/Python-3.11-blue) [![Paper](https://img.shields.io/badge/arXiv-2408.01584-b31b1b.svg)](https://arxiv.org/abs/2408.01584) -->

Deeper Analysis of imitation learning in autonomous driving.

## Original Repo
This repository is forked from `gpudrive` (https://github.com/Emerge-Lab/gpudrive).
## Installation

For installation, follow the instruction in GPUDrive (https://github.com/Emerge-Lab/gpudrive). Additionally, you need `einops`, `scikit-learn`, `seaborn`, and `mediapy`. (Optionally, you may need `jaxlib==0.6.0`.)

## Getting started
### Prepare the data
To make the `(state, action)` pairs, run `storage.py`.
```
# make batch npz file
python gpudrive/integrations/il/storage.py --dataset-size <NUM_SCENE> --dataset <TRAINING/VALIDATION>
# concat files
python gpudrive/integrations/il/data_concat.py --num-scene <NUM_SCENE> --dataset <TRAINING/VALIDATION>
```
Note that set `num_stack=1` stacking will be processed in dataloader.

### Before starting implementation

Most of our experiment code uses `wandb`. Therefore, make the `private.yaml` with this format:
```
wandb_key: <YOUR_API_KEY>
main_project: <IL_PROJECT>
lp_project: <LINEAR_PROBING_PROJECT>
entity: <YOUR_WANDB_ENTITY>
```

### Train the IL
To train the IL model,

```
python baselines/il/imitation_learning.py --use-wandb
```
After run the code, you can check the model in `<base_path>/<model_path>/<sweep_name>`. (See `baselines/il/il.yaml`)

### Train the linear probing
To train the linear probing,
```
python baselines/il/linear_probing.py --use-wandb
```
To extract the linear probing result, run `python gpudrive/integration/il/make_sweep.py`.
### Run the perturbed simulation
For removal ratio test, `removal_ratio = 0.2 ~ 1.0`
```
bash baselines/il/test/partner_ratio.sh <BATCH_SIZE> <DATASET_SIZE> <EXPERIMENT_NAME> <GPU_ID>
```
After run the code, you can check the total simulation result in `<base_path>/<model_path>/<sweep_name>/log_replay/result_<ratio>_total.csv`.

### Intervetion Experiment
Before run the intervention test, make sure to have a file `intervention.csv` and `intervention_other.csv`.
| Column name         | Type    | Description                                     | Range              |
|--------------------|---------|-------------------------------------------------|---------------------|
| intervention_idx     | int   | ego index      | [0~128]                  
| step_10              | int   | Intervention Label of 10 step ahead       | [0~64]                  |
| step_20              | int   | Intervention Label of 10 step ahead    | [0~64]                  |
| step_30              | int   | Intervention Label of 10 step ahead    | [0~64]                  |
| step_40              | int   | Intervention Label of 10 step ahead    | [0~64]                  |
| done                 | bool  | Check for you labeled it   | 0, 1               |
| label_done           | bool  | Check for you labeled it   | 0, 1                  |
| type                 | str   | n: not-related, i: adaptiveness, r: recover    | n, i, r               |

run `python gpudrive/integration/il/figure/intervention.py`

### Other Analysis
For near-collision test, run `python gpudrive/integration/il/figure/near_collision.py`

For correlation between linear probing prediction and future distance, run 
`python gpudrive/integration/il/figure/evaluate_lp.py`

To compare with multiple model, check `gpudrive/integration/il/figure/corr_dist_prob.sh`