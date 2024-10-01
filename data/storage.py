import torch
import numpy as np
import os
import pickle

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv

def save_actions(env):
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()

    device = env.device
    
    expert_actions_lst = [[] for _ in range(expert_actions.shape[0])]
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device)
    
    for time_step in range(env.episode_len):
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)

        for i in range(expert_actions.shape[0]):
            expert_actions_lst[i].append(
                expert_actions[i,:,time_step,:][~dead_agent_mask[i,:]]
            )
            
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        if (dead_agent_mask == True).all():
            break
    
    world_action_pairs = []
    for x in expert_actions_lst:
        world_action_pairs.append(torch.cat(x, dim=0))
    
    return world_action_pairs

def save_trajectory(env):
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()

    device = env.device
    
    expert_actions_lst = [[] for _ in range(expert_actions.shape[0])]
    expert_obs_lst = [[] for _ in range(expert_actions.shape[0])]
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device)
    
    # initial obs
    for i in range(obs.shape[0]):
        expert_obs_lst[i].append(obs[i][~dead_agent_mask[i,:]])
    
    for time_step in range(env.episode_len):
        for i in range(expert_actions.shape[0]):
            expert_actions_lst[i].append(
                expert_actions[i,:,time_step,:][~dead_agent_mask[i,:]]
            )
        
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])
        
        # check dead agent
        dones = env.get_dones().to(device)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
        
        next_obs = env.get_obs()

        for i in range(obs.shape[0]):
            expert_obs_lst[i].append(
                next_obs[i][~dead_agent_mask[i,:]]
            )
        
    world_obs_pairs = []
    world_action_pairs = []
    
    for x in expert_obs_lst:
        world_obs_pairs.append(torch.cat(x, dim=0))
    for x in expert_actions_lst:
        world_action_pairs.append(torch.cat(x, dim=0))
    
    return world_obs_pairs, world_action_pairs

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--save-path', '-sp', type=str, default='/data/train_trajectory_npz')
    parser.add_argument('--num_worlds', type=int, default=100)
    parser.add_argument('--start_idx', type=int, default=0)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = args.num_worlds
    MAX_NUM_OBJECTS = 128

    # Initialize configurations
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", 
                               NUM_WORLDS, 
                               start_idx=args.start_idx, 
                               discipline=SelectionDiscipline.RANGE_N)
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dy=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
    )

    # Initialize environment
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=MAX_NUM_OBJECTS,
        device=args.device,
        action_type="continuous",
        num_stack=5
    )

    # Generate expert actions and observations
    expert_obs, expert_actions = save_trajectory(env)
    
    # Save the expert observations and actions by mpz file
    for i, (scene_obs, scene_action) in enumerate(zip(expert_obs, expert_actions)):
        scene_obs = scene_obs.to('cpu')
        scene_action = scene_action.to('cpu')
        np.savez(os.path.join(args.save_path, f"scene_{int(args.start_idx) + i}_trajectory.npz"), obs=scene_obs, actions=scene_action)
    
    env.close()
    del env
    del env_config
    del scene_config
    