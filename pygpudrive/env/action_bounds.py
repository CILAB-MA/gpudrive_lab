import torch
import numpy as np
import os
import pickle
import re

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv

def update_bounds(scene_config, expert_path):
    num_scenes = scene_config.num_scenes
    discipline = scene_config.discipline
    
    max_bound = []
    min_bound = []
    
    # find start index
    if discipline == SelectionDiscipline.RANGE_N:
        start_idx = scene_config.start_idx
    elif discipline == SelectionDiscipline.RANDOM_N:
        raise NotImplementedError("Random discipline is not supported.")
    else:
        start_idx = 0
    
    # find max and min bounds in num_scenes
    for num_scene in range(num_scenes):
        file_idx = start_idx + num_scene
        
        # sort expert files by index
        expert_list = os.listdir(expert_path)
        expert_list = sorted(expert_list, key=lambda x: int(re.search(r'\d+', x).group()))
        
        # find the expert file that contains the scene
        for i in range(len(expert_list) - 1):
            first_file_idx = int(expert_list[i].split('_')[-1].split('.')[0])
            second_file_idx = int(expert_list[i+1].split('_')[-1].split('.')[0])
            if first_file_idx <= file_idx < second_file_idx:
                local_file_num = file_idx - first_file_idx
                
                # pickle load
                with open(os.path.join(expert_path, expert_list[i]), 'rb') as f:
                    expert_list = pickle.load(f)
                    expert_scene = expert_list[local_file_num]
                    max_bound.append(expert_scene.max(dim=0).values)
                    min_bound.append(expert_scene.min(dim=0).values)
                    break
                        
    max_delta = torch.stack(max_bound).mean(dim=0)
    min_delta = torch.stack(min_bound).mean(dim=0)
    return max_delta, min_delta
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--load-path', '-sp', type=str, default='/data/train_actions_pickles')
    parser.add_argument('--num_worlds', type=int, default=1)
    parser.add_argument('--start_idx', type=int, default=101)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = args.num_worlds
    MAX_NUM_OBJECTS = 128

    # Initialize configurations
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", 
                               NUM_WORLDS, 
                               start_idx=args.start_idx, 
                               discipline=SelectionDiscipline.RANGE_N)
    
    max_delta, min_delta = update_bounds(scene_config, args.load_path)
    
    print(f"max dx (mean per {args.num_worlds} scenes) : ",max_delta[0].item())
    print(f"min dx (mean per {args.num_worlds} scenes) : ",min_delta[0].item())
    print(f"max dy (mean per {args.num_worlds} scenes) : ",max_delta[1].item())
    print(f"min dy (mean per {args.num_worlds} scenes) : ",min_delta[1].item())
    print(f"max dyaw (mean per {args.num_worlds} scenes) : ",max_delta[2].item())
    print(f"min dyaw (mean per {args.num_worlds} scenes) : ",min_delta[2].item())
    
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        dx=torch.round(torch.linspace(min_delta[0].item(), max_delta[0].item(), 100), decimals=3),
        dy=torch.round(torch.linspace(min_delta[1].item(), max_delta[1].item(), 100), decimals=3),
        dyaw=torch.round(torch.linspace(min_delta[2].item(), max_delta[2].item(), 300), decimals=3),
    )

    # Initialize environment
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,
        device=args.device,
        action_type="multi_discrete",
        num_stack=5
    )
    
    env.close()
    del env
    del env_config
    del scene_config
    