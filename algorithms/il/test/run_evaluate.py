"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
import os
from tqdm import tqdm

logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model/eat100to1000')
    parser.add_argument("--total-world-count", type=int, default=20)
    parser.add_argument("--num-world", type=int, default=10)
    parser.add_argument('--start-idx', '-st', type=int, default=0)
    # GPU SETTINGS
    parser.add_argument('--gpu-id', '-g', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    total_world_count = args.total_world_count
    one_run_world_count = args.num_world
    models = [os.path.splitext(model)[0] for model in os.listdir(args.model_path) if model.endswith('.pth')]
    
    for model in tqdm(models):
        for dataset in ['train', 'valid']:
            for i in tqdm(range(args.start_idx, total_world_count // one_run_world_count)):
                start_idx = i * one_run_world_count
                print("model: ", model, "dataset: ", dataset, "idx:", start_idx)
                arguments = f"-mc -mv -m {model} --dataset {dataset} --num-world {one_run_world_count} --start-idx {start_idx}"
                command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} /root/anaconda3/envs/gpudrive/bin/python algorithms/il/test/evaluate.py {arguments}"
                
                result = subprocess.run(command, shell=True)
                
                if result.returncode != 0:
                    print(f"Error: Command failed with return code {result.returncode}")
