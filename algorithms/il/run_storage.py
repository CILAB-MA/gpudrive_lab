"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
from tqdm import tqdm
import os
import numpy as np

logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_path", type=str, default="/root/anaconda3/envs/gpudrive/bin/python")
    parser.add_argument("--store_path", type=str, default="/app/algorithms/il/storage.py")
    parser.add_argument('--save_path', type=str, default='/data/train_trajectory_npz')
    parser.add_argument("--total_file_count", type=int, default=1000)
    parser.add_argument("--learn_file_count", type=int, default=50)
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    total_file_count = args.total_file_count
    learn_file_count = args.learn_file_count
    if not os.path.exists(args.save_path):
        try:
            os.makedirs(args.save_path)
        except FileExistsError:
            pass
    
    for i in tqdm(range(total_file_count // learn_file_count), desc="Compressed files by NUM_SCENES"):
        start_idx = learn_file_count * i
        
        command = f"{args.python_path} {args.store_path} --start_idx {start_idx} --num_worlds {learn_file_count} --dataset {args.dataset} --save_path {args.save_path}"
        result = subprocess.run(command, shell=True)
        
        if result.returncode != 0:
            print(f"Error: Command failed with return code {result.returncode}")

    # List all npz files in the save-path directory
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]

    all_obs = []
    all_actions = []
    
    for npz_file in tqdm(npz_files, desc="Compressed files by all Scenes"):
        file_path = os.path.join(args.save_path, npz_file)
        data = np.load(file_path)
        
        all_obs.append(data['obs'])
        all_actions.append(data['actions'])
    
    all_obs = np.concatenate(all_obs)
    all_actions = np.concatenate(all_actions)

    # Save the combined data into a new npz file
    combined_file_path = os.path.join(args.save_path, f'trajectory_{args.total_file_count}.npz')
    np.savez_compressed(combined_file_path, obs=all_obs, actions=all_actions)
