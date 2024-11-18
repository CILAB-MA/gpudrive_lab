import torch
import numpy as np
import os
from tqdm import tqdm

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
import re

def save_actions_by_veh(env):
    """
    Save the expert actions in the environment, distinguishing them by each vehicle.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        world_action_pairs: Expert actions for the controlled agents in each scene.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)

    alive_agent_num = env.cont_agent_mask.sum().item()
    expert_actions_lst = [[None]*env.episode_len for _ in range(alive_agent_num)]
    
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    
    for time_step in tqdm(range(env.episode_len)):
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
            
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step, :]
        
        if (dead_agent_mask == True).all():
            break
    
    expert_actions_lst = [
        [action for action in agent_actions if action is not None]
        for agent_actions in expert_actions_lst
    ]
    
    from MyTest.plot import plot_per_veh
    plot_per_veh(expert_actions_lst, "/data/plots")

def save_trajectory_by_veh(env):
    """
    Save the expert actions in the environment, distinguishing them by each vehicle.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        world_action_pairs: Expert actions for the controlled agents in each scene.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_obs_lst = [[None]*env.episode_len for _ in range(alive_agent_num)]
    expert_actions_lst = [[None]*env.episode_len for _ in range(alive_agent_num)]
    
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    
    for time_step in tqdm(range(env.episode_len)):      
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_obs_lst[idx][time_step] = obs[world_idx, agent_idx, :]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step, :]
                
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
            
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        if (dead_agent_mask == True).all():
            break
        
        obs = env.get_obs()
    
    expert_obs_lst = [
        [obs for obs in agent_obs if obs is not None]
        for agent_obs in expert_obs_lst
    ]
    expert_actions_lst = [
        [action for action in agent_actions if action is not None]
        for agent_actions in expert_actions_lst
    ]
    
    # Future 10 step of actions mean and std
    expert_actions_mean = [[] for _ in range(alive_agent_num)]
    expert_actions_std = [[] for _ in range(alive_agent_num)]
    for idx, actions in enumerate(expert_actions_lst):
        for i in range(len(actions)):
            expert_actions_mean[idx].append(torch.stack(actions[i:i+10]).to('cpu').mean(dim=0))
            expert_actions_std[idx].append(torch.stack(actions[i:i+10]).to('cpu').std(dim=0))
    
    return expert_obs_lst, expert_actions_lst, expert_actions_mean, expert_actions_std 
    
def save_actions(env):
    """
    Save the expert actions in the environment, distinguishing them by each scene.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        world_action_pairs: Expert actions for the controlled agents in each scene.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()

    device = env.device
    
    expert_actions_lst = [[] for _ in range(expert_actions.shape[0])]
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device)
    
    for time_step in tqdm(range(env.episode_len)):
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
    """
    save the expert actions and observations in the environment (not distinguishing by each scene).
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        world_obs_pairs: Expert obs for the controlled agents in all scene.
        world_action_pairs : Expert actions for the controlled agents in all scene.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()
    dead_agent_mask = ~env.cont_agent_mask.clone()

    obs = obs.to('cpu')
    expert_actions = expert_actions.to('cpu')
    dead_agent_mask = dead_agent_mask.to('cpu')
    
    expert_obs_lst = []
    expert_actions_lst = []
    
    # initial obs
    expert_obs_lst.append(obs[~dead_agent_mask])
    
    for time_step in tqdm(range(env.episode_len)):
        expert_actions_lst.append(
            expert_actions[:,:,time_step,:][~dead_agent_mask]
        )
        
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :], use_indices=False)
        
        # check dead agent
        dones = env.get_dones().to('cpu')
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
        
        next_obs = env.get_obs().to('cpu')

        expert_obs_lst.append(
            next_obs[~dead_agent_mask]
        )
    
    # world_obs_pairs = np.concatenate(expert_obs_lst, axis=0)
    # world_action_pairs = np.concatenate(expert_actions_lst, axis=0)
    

    return expert_obs_lst, expert_actions_lst

def save_trajectory_by_scenes(env):
    """
    save the expert actions and observations in the environment (distinguishing by each scene).
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        world_obs_pairs: Expert obs for the controlled agents in all scene.
        world_action_pairs : Expert actions for the controlled agents in all scene.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()
    dead_agent_mask = ~env.cont_agent_mask.clone()

    obs = obs.to('cpu')
    expert_actions = expert_actions.to('cpu')
    dead_agent_mask = dead_agent_mask.to('cpu')
    
    expert_obs_lst = [[] for _ in range(expert_actions.shape[0])]
    expert_actions_lst = [[] for _ in range(expert_actions.shape[0])]
    
    # initial obs
    for i in range(obs.shape[0]):
        expert_obs_lst[i].append(
            obs[i,...][~dead_agent_mask[i,...]]
        )
    
    
    for time_step in tqdm(range(env.episode_len)):
        for i in range(expert_actions.shape[0]):
            expert_actions_lst[i].append(
                expert_actions[i,:,time_step,:][~dead_agent_mask[i,:]]
            )
        
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :], use_indices=False)
        
        # check dead agent
        dones = env.get_dones().to('cpu')
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
        
        next_obs = env.get_obs().to('cpu')
        for i in range(next_obs.shape[0]):
            expert_obs_lst[i].append(
                next_obs[i,...][~dead_agent_mask[i,...]]
            )
    
    world_obs_pairs = [np.concatenate(scene_data, axis=0) for scene_data in expert_obs_lst]
    world_action_pairs = [np.concatenate(scene_data, axis=0) for scene_data in expert_actions_lst]
    

    return world_obs_pairs, world_action_pairs


if __name__ == "__main__":
    import argparse
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num_worlds', type=int, default=5)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='/data/train_trajectory_by_veh')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = args.num_worlds
    MAX_NUM_OBJECTS = 128
    
    def get_largest_number(folder_path):
        files = os.listdir(folder_path)
        
        numbers = []
        for file in files:
            match = re.search(r'veh_(\d+)', file)
            if match:
                numbers.append(int(match.group(1)))
        
        # 가장 큰 숫자 반환
        if numbers:
            return max(numbers)
        else:
            return -1  # 숫자가 없으면 None 반환
    start_veh_idx = get_largest_number("/data/train_trajectory_by_veh")
    print("start_veh_idx : ", start_veh_idx)

    # Initialize configurations
    scene_config = SceneConfig(f"/mnt/nas/kyungbeom/gpudrive_lab/original_data/formatted_json_v2_no_tl_{args.dataset}/",
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
    kwargs={
        "config": env_config,
        "scene_config": scene_config,
        "max_cont_agents": MAX_NUM_OBJECTS,
        "device": args.device,
        "num_stack": 5
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_id=ActionSpace.CONTINUOUS, kwargs=kwargs)

    # Generate expert actions and observations
    expert_obs, expert_actions, expert_actions_mean, expert_actions_std = save_trajectory_by_veh(env)
    # Save the expert observations and actions by mpz file
    for i, (obs, actions, means, stds) in enumerate(zip(expert_obs, expert_actions, expert_actions_mean, expert_actions_std)):
        obs = torch.stack(obs).to('cpu')
        actions = torch.stack(actions).to('cpu')
        means = torch.stack(means).to('cpu')
        stds = torch.stack(stds).to('cpu')
        np.savez_compressed(os.path.join(args.save_path,
                    f"veh_{start_veh_idx + i + 1:04}.npz"),
                    obs=obs, actions=actions, means=means, stds=stds)

    env.close()
    del env
    del env_config
    del scene_config
    