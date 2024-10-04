"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
from tqdm import tqdm

logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_path", type=str, default="/root/anaconda3/envs/gpudrive/bin/python")
    parser.add_argument("--store_path", type=str, default="/app/data/storage.py")
    parser.add_argument("--total_file_count", type=int, default=1000)
    parser.add_argument("--learn_file_count", type=int, default=50)
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
    total_file_count = args.total_file_count
    learn_file_count = args.learn_file_count
    
    print("start learning")
    
    for i in tqdm(range(total_file_count // learn_file_count)):
        start_idx = learn_file_count * i
        
        command = f"{args.python_path} {args.store_path} --start_idx {start_idx} --num_worlds {learn_file_count} --dataset {args.dataset}"
        result = subprocess.run(command, shell=True)
        
        if result.returncode != 0:
            print(f"Error: Command failed with return code {result.returncode}")
