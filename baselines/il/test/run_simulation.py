"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
import os
import json
from tqdm import tqdm
import glob
import pandas as pd
logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    # MODEL
    parser.add_argument('--sweep-name', '-sn', type=str, default='tom_03')
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/video')
    parser.add_argument('--dataset-size', type=int, default=80000) # total_world
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=1.0)
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
        if '.pth' not in model:
            continue
        if args.partner_portion_test:
            video_path = args.video_path + f"_{args.partner_portion_test}"
        else:
            video_path = args.video_path

        video_path = os.path.join(video_path, args.sweep_name)
        model_path = os.path.join(args.model_path, args.sweep_name)
        
        arguments = f"-mc -mp {model_path} -vp {video_path} -mn {model} --batch-size {args.batch_size} -pp {args.partner_portion_test}"
        
        command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} /root/anaconda3/envs/gpudrive/bin/python baselines/il/test/simulation.py {arguments}"
        
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            print(f"Error: Command failed with return code {result.returncode}")

    csv_path = f"{model_path}/result_{args.partner_portion_test}.csv"
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"CSV file {csv_path} does not exist or is empty. Exiting...")
        exit()
    df = pd.read_csv(csv_path)
    df_avg = df.groupby(["Model", "Dataset"], as_index=False).mean()
    df_avg.to_csv(csv_path, index=False)
    print(f"Updated CSV saved at {csv_path} with averaged results.")