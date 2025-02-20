"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
import os
from tqdm import tqdm
import glob
logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model/tom_03')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/video/tom_03')
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
    models = os.listdir(args.model_path)
    print(models)
    for model in tqdm(models):
        for dataset in ['train', 'valid']:
            for i in tqdm(range(args.start_idx, total_world_count // one_run_world_count)):
                start_idx = i * one_run_world_count
                if dataset == 'valid' and start_idx >= 200:
                    break
                print("model: ", model, "dataset: ", dataset, "idx:", start_idx)
                arguments = f"-mc --dataset {dataset} -mp {args.model_path} -vp {args.video_path} -mn {model} --num-world {one_run_world_count} --start-idx {start_idx}"
                if i == 0:
                    arguments += ' -mv'
                command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} /root/anaconda3/envs/gpudrive/bin/python algorithms/il/test/evaluate.py {arguments}"
                
                result = subprocess.run(command, shell=True)
                
                if result.returncode != 0:
                    print(f"Error: Command failed with return code {result.returncode}")
