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
    parser.add_argument('--sweep-name', '-sn', type=str, default='tom_03')
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/video')
    parser.add_argument("--total-world-count", type=int, default=20)
    parser.add_argument("--num-world", type=int, default=10)
    parser.add_argument('--start-idx', '-st', type=int, default=0)
    parser.add_argument('--zero-partner-test', '-z', action='store_true')
    # GPU SETTINGS
    parser.add_argument('--gpu-id', '-g', type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    total_world_count = args.total_world_count
    one_run_world_count = args.num_world
    models = os.listdir(os.path.join(args.model_path, args.sweep_name))
    print(models)
    for model in tqdm(models):
        for dataset in ['train', 'valid']:
            if '.pth' not in model:
                continue
            if args.zero_partner_test:
                video_path = args.video_path + "_zero"
            else:
                video_path = args.video_path
            video_path = os.path.join(video_path, args.sweep_name)
            model_path = os.path.join(args.model_path, args.sweep_name)
            for i in tqdm(range(args.start_idx, total_world_count // one_run_world_count)):
                start_idx = i * one_run_world_count
                if dataset == 'valid' and start_idx >= 200:
                    break
                print("model: ", model, "dataset: ", dataset, "idx:", start_idx)
                arguments = f"-mc --dataset {dataset} -mp {model_path} -vp {video_path} -mn {model} --num-world {one_run_world_count} --start-idx {start_idx}"
                if args.zero_partner_test:
                    arguments += '-z'
                if i == 0:
                    arguments += ' -mv'
                command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} /root/anaconda3/envs/gpudrive/bin/python algorithms/il/test/evaluate.py {arguments}"
                
                result = subprocess.run(command, shell=True)
                
                if result.returncode != 0:
                    print(f"Error: Command failed with return code {result.returncode}")
