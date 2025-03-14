import torch
import numpy as np
import os
from tqdm import tqdm

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline


def save_mean_and_std_by_veh(env, save_path, save_index):
    """
    Save the mean and std for 20 timesteps in the environment, distinguishing them by each vehicle.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.

    Returns:
        world_action_pairs: Expert actions for the controlled agents in each scene.
    """
    obs = env.reset()
    expert_actions, _, _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    device = env.device
    
    cont_agent_mask = env.get_controlled_agents_mask().to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_obs_lst = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, expert_actions.shape[-1]), device=device)
    expert_actions_mean_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_actions_std_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_valid_mask_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device, dtype=torch.bool)
    
    # Initialize dead agent mask
    valid_agent_mask = cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if  valid_agent_mask[world_idx, agent_idx]:
                expert_obs_lst[idx][time_step] = obs[world_idx, agent_idx]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step, :]
            expert_valid_mask_lst[idx][time_step] = valid_agent_mask[world_idx, agent_idx]
        
        env.step_dynamics(expert_actions[:, :, time_step, :])
        obs = env.get_obs().to(device)
        dones = env.get_dones().to(device)
        
        valid_agent_mask = torch.logical_and(valid_agent_mask, ~dones.bool())
        
        if (valid_agent_mask == False).all():
            break
    
    # Future 10 step of actions mean and std
    for i in range(expert_actions_lst.shape[0]):
        for j in range(expert_actions_lst.shape[1]):
            actions_window = expert_actions_lst[i][j:j+20]
            expert_actions_mean_lst[i][j] = actions_window.mean(dim=0)
            if actions_window.shape[0] > 1:
                expert_actions_std_lst[i][j] = actions_window.std(dim=0, unbiased=True)
            else:
                expert_actions_std_lst[i][j] = torch.zeros_like(actions_window[0])
            
    
    expert_obs_lst = expert_obs_lst.to('cpu')
    expert_actions_mean_lst = expert_actions_mean_lst.to('cpu')
    expert_actions_std_lst = expert_actions_std_lst.to('cpu')
    expert_valid_mask_lst = expert_valid_mask_lst.to('cpu')
    
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz",
                        obs=expert_obs_lst,
                        mean=expert_actions_mean_lst,
                        std=expert_actions_std_lst,
                        valid_mask=expert_valid_mask_lst)




if __name__ == "__main__":
    import argparse
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--num_worlds', type=int, default=5)
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/data/RL/data')
    parser.add_argument('--save_index', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    
    print()
    print("num_worlds : ", args.num_worlds)
    print("num_stack : ", args.num_stack)
    print("save_path : ", args.save_path)
    print("save_index : ", args.save_index)
    print("dataset : ", args.dataset)

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
        "device": "cuda",
        "num_stack": args.num_stack
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_id=ActionSpace.CONTINUOUS, kwargs=kwargs)

    save_mean_and_std_by_veh(env, args.save_path, args.save_index)

    env.close()
    del env
    del env_config
    del scene_config
    