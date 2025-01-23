"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
from tqdm import tqdm

logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python_path", type=str, default="/root/anaconda3/envs/gpudrive/bin/python")
    parser.add_argument("--store_path", type=str, default="algorithms/il/storage.py")
    parser.add_argument("--total_file_count", type=int, default=1000)
    parser.add_argument("--learn_file_count", type=int, default=20)
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    
    parser.add_argument('--num_worlds', type=int, default=20)
    parser.add_argument('--num_stack', type=int, default=5)
    parser.add_argument('--save_index', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='/data/trajectories')
    parser.add_argument('--function', type=str, default='save_obs_action_mean_std_mask_by_veh', choices=[
                                                                            'save_obs_action_mean_std_mask_by_veh',
                                                                            'save_trajectory',
                                                                            'save_trajectory_by_scenes',
                                                                            'save_trajectory_and_three_mask_by_scenes'])
                                                                        
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    total_file_count = args.total_file_count
    learn_file_count = args.learn_file_count
    
    print("start learning")
    
    for i in tqdm(range(total_file_count // learn_file_count)):
        save_index = learn_file_count * i
        
        command = f"{args.python_path} {args.store_path} --num_worlds {learn_file_count} --num_stack {args.num_stack} --save_index {save_index} --save_path {args.save_path} --dataset {args.dataset} --function {args.function}"
        
        result = subprocess.run(command, shell=True)
        
        if result.returncode != 0:
            print(f"Error: Command failed with return code {result.returncode}")
