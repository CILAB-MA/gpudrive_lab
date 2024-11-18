import torch
import numpy as np
from tqdm import tqdm

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline


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
    
def save_obs_by_veh(env):
    """
    Save the expert obs in the environment, distinguishing them by each vehicle.
    
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
    
    expert_obs = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    expert_dead_mask_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device, dtype=torch.bool)
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_obs[idx][time_step] = obs[world_idx, agent_idx, :]
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]
        
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        if (dead_agent_mask == True).all():
            break
        
        obs = env.get_obs()
    
    expert_obs = expert_obs.to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst.to('cpu')
    np.savez_compressed("/data/train_trajectory_by_veh/obs_veh.npz", 
                        obs=expert_obs,
                        dead_mask=expert_dead_mask_lst)

def save_mean_and_std_by_veh(env):
    """
    Save the mean and std for 10 timesteps in the environment, distinguishing them by each vehicle.
    
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
    
    expert_label_lst = torch.zeros((alive_agent_num, env.episode_len, 6), device=device)
    expert_dead_mask_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device, dtype=torch.bool)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    
    # Initialize dead agent mask
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step, :]
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]
        
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        
        if (dead_agent_mask == True).all():
            break
    
    # Future 10 step of actions mean and std
    for i in range(expert_actions_lst.shape[0]):
        for j in range(expert_actions_lst.shape[1]):
            expert_label_lst[i][j][0:3] = expert_actions_lst[i][j:j+10].mean(dim=0)
            expert_label_lst[i][j][3:6] = expert_actions_lst[i][j:j+10].std(dim=0)
    
    expert_label_lst = expert_label_lst.to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst.to('cpu')
    
    np.savez_compressed("/data/train_trajectory_by_veh/label_veh.npz", 
                        mean_std=expert_label_lst,
                        dead_mask=expert_dead_mask_lst)

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

def save_actions_by_scenes(env):
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
    parser.add_argument('--num_worlds', type=int, default=50)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='/data/train_trajectory_by_veh')
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    NUM_WORLDS = args.num_worlds
    MAX_NUM_OBJECTS = 128

    # Initialize configurations
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{args.dataset}/",
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
        "num_stack": 1
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_id=ActionSpace.CONTINUOUS, kwargs=kwargs)

    save_mean_and_std_by_veh(env)

    env.close()
    del env
    del env_config
    del scene_config
    