import torch
import numpy as np
import os
import pickle
import re
from bisect import bisect_right

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv

def set_effective_action_space(env_config, scene_config, expert_path, std_factor=1.0):
    """Set the effective action space based on the expert demonstrations."""
    num_scenes = scene_config.num_scenes
    discipline = scene_config.discipline
    
    if  num_scenes == 1 and discipline == SelectionDiscipline.RANGE_N:
        idx = scene_config.start_idx
        
        # sort expert files by index
        expert_list = os.listdir(expert_path)
        expert_list = sorted(expert_list, key=lambda x: int(re.search(r'\d+', x).group()))
        
        file_indices = [int(file.split('_')[-1].split('.')[0]) for file in expert_list]
        file_pos = bisect_right(file_indices, idx) - 1
        pickle_idx = file_indices[file_pos]
        local_idx = idx - pickle_idx

        with open(os.path.join(expert_path, expert_list[file_pos]), 'rb') as f:
            expert_data = pickle.load(f)
            expert_actions = expert_data[local_idx]
            means = expert_actions.mean(dim=0).tolist()
            stds = expert_actions.std(dim=0).tolist()
            
            config = EnvConfig(
                dynamics_model=env_config.dynamics_model,
                dx=torch.round(
                    torch.linspace(
                        means[0] - std_factor * stds[0], 
                        means[0] + std_factor * stds[0], 100),
                    decimals=3),
                dy=torch.round(
                    torch.linspace(
                        means[1] - std_factor * stds[1], 
                        means[1] + std_factor * stds[1], 100),
                    decimals=3),
                dyaw=torch.round(
                    torch.linspace(
                        means[2] - std_factor * stds[2], 
                        means[2] + std_factor * stds[2], 300),
                    decimals=3)
            )
        return config
    
    else:
        return env_config    
        

if __name__ == "__main__":
    import argparse
    from algorithms.il.data_generation import generate_state_action_pairs
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],)
    parser.add_argument('--load-path', '-sp', type=str, default='/data/train_actions_pickles')
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = 1
    MAX_NUM_OBJECTS = 128

    # Initialize configurations
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", 
                               NUM_WORLDS, 
                               start_idx=args.start_idx, 
                               discipline=SelectionDiscipline.RANGE_N)
    
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
    )
    env_config = set_effective_action_space(env_config, scene_config, args.load_path, 1.0)

    # Initialize environment
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,
        device=args.device,
        action_type="multi_discrete",
        num_stack=5
    )
    
     # Generate expert actions and observations
    (
        expert_obs,
        expert_actions,
        next_expert_obs,
        expert_dones,
        goal_rate,
        collision_rate
    ) = generate_state_action_pairs(
        env=env,
        use_action_indices=False,  # Map action values to joint action index
        make_video=True,  # Record the trajectories as sanity check
        render_index=[0, NUM_WORLDS],  #start_idx, end_idx
        save_path="./",
        debug_world_idx=0,
        debug_veh_idx=0,
    )
    
    env.close()
    del env
    del env_config
    del scene_config
    