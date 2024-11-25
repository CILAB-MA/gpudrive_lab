import torch
import numpy as np
import os
from tqdm import tqdm

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline

# GPUDRIVE RL
def save_obs_action_mean_std_mask_by_veh(env, save_path, save_index=0):
    """
    Save the expert obs in the environment, distinguishing them by each vehicle.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
        
        
    Returns:
        obs : (alive_agent_num, episode_len, 3876 * num_stack)
        actions : (alive_agent_num, episode_len, 3)
        mean_std : (alive_agent_num, episode_len, 6)
        dead_mask : (alive_agent_num, episode_len)
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_obs_lst = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_mean_std_lst = torch.zeros((alive_agent_num, env.episode_len, 6), device=device)
    expert_dead_mask_lst = torch.zeros((alive_agent_num, env.episode_len), device=device, dtype=torch.bool)
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_obs_lst[idx][time_step] = obs[world_idx, agent_idx, :]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step, :]
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]
        
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        obs = env.get_obs()
    
    # Future 10 step of actions mean and std
    for i in range(expert_actions_lst.shape[0]):
        for j in range(expert_actions_lst.shape[1]):
            expert_mean_std_lst[i][j][0:3] = expert_actions_lst[i][j:j+10].mean(dim=0)
            expert_mean_std_lst[i][j][3:6] = expert_actions_lst[i][j:j+10].std(dim=0)
    
    expert_obs_lst = expert_obs_lst.to('cpu')
    expert_actions_lst = expert_actions_lst.to('cpu')
    expert_mean_std_lst = expert_mean_std_lst.to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst.to('cpu')
    
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=expert_obs_lst,
                        actions=expert_actions_lst,
                        mean_std=expert_mean_std_lst,
                        dead_mask=expert_dead_mask_lst)

# GPUDRIVE IL
def save_trajectory(env, save_path, save_index=0):
    """
    save the expert actions and observations in the environment (not distinguishing by each scene).
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        obs: (World * alive_agent_num * alive_episode_len, 3876 * num_stack)
        actions: (World * alive_agent_num * alive_episode_len, 3)
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()
    dead_agent_mask = ~env.cont_agent_mask.clone()

    obs = obs.to('cpu')
    expert_actions = expert_actions.to('cpu')
    dead_agent_mask = dead_agent_mask.to('cpu')
    
    expert_obs_lst = []
    expert_actions_lst = []
    
    for time_step in tqdm(range(env.episode_len)):
        expert_obs_lst.append(obs[~dead_agent_mask])
        expert_actions_lst.append(expert_actions[:,:,time_step,:][~dead_agent_mask])
        
        # Step the environment with inferred expert actions
        env.step_dynamics(expert_actions[:, :, time_step, :], use_indices=False)
        
        # check dead agent
        dones = env.get_dones().to('cpu')
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
        
        obs = env.get_obs().to('cpu')
    
    world_obs_pairs = np.concatenate(expert_obs_lst, axis=0)
    world_action_pairs = np.concatenate(expert_actions_lst, axis=0)
    
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz",
                        obs=world_obs_pairs,
                        actions=world_action_pairs)

def save_trajectory_by_scenes(env, save_path, save_index=0):
    """
    save the expert actions and observations in the environment (distinguishing by each scene).
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        obs: (world, alive_agent * alive_episode_len, 3876 * num_stack)
        actions : (world, alive_agent * alive_episode_len, 3)
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions()
    dead_agent_mask = ~env.cont_agent_mask.clone()

    obs = obs.to('cpu')
    expert_actions = expert_actions.to('cpu')
    dead_agent_mask = dead_agent_mask.to('cpu')
    
    expert_obs_lst = [[] for _ in range(expert_actions.shape[0])]
    expert_actions_lst = [[] for _ in range(expert_actions.shape[0])]
    
    
    for time_step in tqdm(range(env.episode_len)):
        for i in range(expert_actions.shape[0]):
            expert_obs_lst[i].append(
                obs[i,...][~dead_agent_mask[i,...]]
            )
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
        
        obs = env.get_obs().to('cpu')
    
    world_obs_pairs = [np.concatenate(scene_data, axis=0) for scene_data in expert_obs_lst]
    world_action_pairs = [np.concatenate(scene_data, axis=0) for scene_data in expert_actions_lst]
    
    os.makedirs(save_path, exist_ok=True)
    for (scene_idx, (obs, actions)) in enumerate(zip(world_obs_pairs, world_action_pairs)):
        np.savez_compressed(f"{save_path}/trajectory_{save_index + scene_idx}.npz",
                            obs=obs,
                            actions=actions)

def save_trajectory_and_three_mask_by_scenes(env, save_path, save_index=0):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_trajectory_lst = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_dead_mask_lst = torch.zeros((alive_agent_num, env.episode_len), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.zeros((alive_agent_num, env.episode_len, 127), device=device, dtype=torch.bool)
    expert_road_mask_lst = torch.zeros((alive_agent_num, env.episode_len, 200), device=device, dtype=torch.bool)
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    partner_mask = env.get_partner_mask()
    road_mask = env.get_road_mask()
    
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_trajectory_lst[idx][time_step] = obs[world_idx, agent_idx, :]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step, :]
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]
            expert_partner_mask_lst[idx][time_step] = partner_mask[world_idx, agent_idx, :]
            expert_road_mask_lst[idx][time_step] = road_mask[world_idx, agent_idx, :]
        
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        if (dead_agent_mask == True).all():
            break
        
        obs = env.get_obs()
        partner_mask = env.get_partner_mask()
        road_mask = env.get_road_mask()
    
    expert_trajectory_lst = expert_trajectory_lst.to('cpu')
    expert_actions_lst = expert_actions_lst.to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst.to('cpu')
    expert_partner_mask_lst = expert_partner_mask_lst.to('cpu')
    expert_road_mask_lst = expert_road_mask_lst.to('cpu')
    
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=expert_trajectory_lst,
                        actions=expert_actions_lst,
                        dead_mask=expert_dead_mask_lst,
                        partner_mask=expert_partner_mask_lst,
                        road_mask=expert_road_mask_lst)


if __name__ == "__main__":
    import argparse
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_worlds', type=int, default=200)
    parser.add_argument('--num_stack', type=int, default=5)
    parser.add_argument('--save_path', type=str, default='/data/train_trajectory_by_veh')
    parser.add_argument('--save_index', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    parser.add_argument('--function', type=str, default='save_obs_by_veh', 
                        choices=[
                            'save_obs_action_mean_std_mask_by_veh',
                            'save_trajectory',
                            'save_trajectory_by_scenes',
                            'save_trajectory_and_three_mask_by_scenes'])
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    print()
    print("num_worlds : ", args.num_worlds)
    print("num_stack : ", args.num_stack)
    print("save_path : ", args.save_path)
    print("save_index : ", args.save_index)
    print("dataset : ", args.dataset)
    print("function : ", args.function)
    
    # Initialize configurations
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{args.dataset}/",
                               num_scenes=args.num_worlds,
                               start_idx=args.save_index,
                               discipline=SelectionDiscipline.RANGE_N)
    env_config = EnvConfig(
        dynamics_model='delta_local',
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
        "max_cont_agents": 128,
        "device": 'cuda',
        "num_stack": args.num_stack
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)

    if args.function == 'save_obs_action_mean_std_mask_by_veh':
        save_obs_action_mean_std_mask_by_veh(env, args.save_path, args.save_index)
    elif args.function == 'save_trajectory':
        save_trajectory(env, args.save_path, args.save_index)
    elif args.function == 'save_trajectory_by_scenes':
        save_trajectory_by_scenes(env, args.save_path, args.save_index)
    elif args.function == 'save_trajectory_and_three_mask_by_scenes':
        save_trajectory_and_three_mask_by_scenes(env, args.save_path, args.save_index)
    else:
        raise ValueError("Invalid function name")

    env.close()
    del env
    del env_config
    del scene_config
    