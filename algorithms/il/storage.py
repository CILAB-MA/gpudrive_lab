import torch
import numpy as np
import os

from pygpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.env_torch import GPUDriveTorchEnv

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
    
    for time_step in range(env.episode_len):
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
    
    world_obs_pairs = np.concatenate(expert_obs_lst, axis=0)
    world_action_pairs = np.concatenate(expert_actions_lst, axis=0)

    return world_obs_pairs, world_action_pairs


if __name__ == "__main__":
    import argparse
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num_worlds', type=int, default=400)
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='/data/train_trajectory_npz')
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
        "num_stack": 5
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)

    # Generate expert actions and observations
    expert_obs, expert_actions = save_trajectory(env)
    # Save the expert observations and actions by mpz file
    np.savez_compressed(os.path.join(args.save_path, 
                        f"scene_{args.start_idx + NUM_WORLDS}_trajectory.npz"), 
                        obs=expert_obs, actions=expert_actions)
    
    env.close()
    del env
    del env_config
    del scene_config
    
