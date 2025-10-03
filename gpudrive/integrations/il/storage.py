import torch
import numpy as np
import os
from tqdm import tqdm

from gpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.constants import MAX_REL_AGENT_POS, MIN_REL_AGENT_POS

def save_trajectory(env, save_path, save_index=0):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
    """
    obs = env.reset()
    expert_actions, _, _, _ , _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_trajectory_lst = torch.zeros((alive_agent_num, env.episode_len + 4, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_valid_mask_lst = torch.zeros((alive_agent_num, env.episode_len + 4), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.ones((alive_agent_num, env.episode_len + 4, 127), device=device, dtype=torch.bool)
    expert_road_mask_lst = torch.ones((alive_agent_num, env.episode_len + 4, 200), device=device, dtype=torch.bool)
    expert_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device) # global pos (2)
    expert_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)

    # Initialize dead agent mask
    agent_info = (
            env.sim.absolute_self_observation_tensor()
            .to_torch()
            .to(device)
        )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    road_mask = env.get_road_mask()
    goal_achieved = 0
    off_road = 0
    veh_collision = 0
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_trajectory_lst[idx][time_step + 4] = obs[world_idx, agent_idx]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step]
                expert_partner_mask_lst[idx][time_step + 4] = partner_mask[world_idx, agent_idx] == 2
                expert_road_mask_lst[idx][time_step + 4] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, time_step] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, time_step] = agent_info[world_idx, agent_idx, 7:8]
                # expert_partner_id_lst[idx, time_step] = env.partner_ids[world_idx, agent_idx].clone()
                # expert_ego_id_lst[idx, time_step] = agent_info[world_idx, agent_idx, -1]
                # expert_scene_id_lst[idx, time_step] = world_idx
            expert_valid_mask_lst[idx][time_step + 4] = (~dead_agent_mask[world_idx, agent_idx]) & ~(
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, 1]) >  0.5) | 
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, 0]) >  5) | 
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, -1]) > 0.2)
                )

        
        # env.step() -> gather next obs
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs() 
        road_mask = env.get_road_mask()
        partner_mask = env.get_partner_mask()
        # partner_id = env.get_partner_id().unsqueeze(-1)
        agent_info = (
        env.sim.absolute_self_observation_tensor()
        .to_torch()
        .to(device)
        )
        infos = env.get_infos()
        
        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        goal_achieved = torch.clamp(goal_achieved, max=1.0)
        off_road = torch.clamp(off_road, max=1.0)
        veh_collision = torch.clamp(veh_collision, max=1.0)

        if (dead_agent_mask == True).all():
            goal_rate = goal_achieved.sum().float() / cont_agent_mask.sum().float()
            off_road_rate = off_road.sum().float() / cont_agent_mask.sum().float()
            veh_coll_rate = veh_collision.sum().float() / cont_agent_mask.sum().float()
            collision = (veh_collision + off_road > 0)
            goal_mask = goal_achieved > 0
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
            break
    
    expert_trajectory_lst = expert_trajectory_lst[goal_mask].to('cpu')
    expert_actions_lst = expert_actions_lst[goal_mask].to('cpu')
    expert_valid_mask_lst = expert_valid_mask_lst[goal_mask].to('cpu')
    expert_partner_mask_lst = expert_partner_mask_lst[goal_mask].to('cpu')
    expert_road_mask_lst = expert_road_mask_lst[goal_mask].to('cpu')
    # global pos
    expert_global_pos_lst = expert_global_pos_lst[goal_mask].to('cpu')
    expert_global_rot_lst = expert_global_rot_lst[goal_mask].to('cpu')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/global', exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=expert_trajectory_lst,
                        actions=expert_actions_lst,
                        valid_mask=expert_valid_mask_lst,
                        partner_mask=expert_partner_mask_lst,
                        road_mask=expert_road_mask_lst)
    np.savez_compressed(f"{save_path}/global/global_trajectory_{save_index}.npz", 
                        ego_global_pos=expert_global_pos_lst,
                        ego_global_rot=expert_global_rot_lst)

def save_trajectory_npy(env, save_path, save_index=0):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
    """
    obs = env.reset()
    expert_actions, _, _, _ , _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_trajectory_lst = torch.zeros((alive_agent_num, env.episode_len + 4, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_valid_mask_lst = torch.zeros((alive_agent_num, env.episode_len + 4), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.ones((alive_agent_num, env.episode_len + 4, 127), device=device, dtype=torch.bool)
    expert_road_mask_lst = torch.ones((alive_agent_num, env.episode_len + 4, 200), device=device, dtype=torch.bool)
    expert_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device) # global pos (2)
    expert_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)

    # Initialize dead agent mask
    agent_info = (
            env.sim.absolute_self_observation_tensor()
            .to_torch()
            .to(device)
        )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    road_mask = env.get_road_mask()
    goal_achieved = 0
    off_road = 0
    veh_collision = 0
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_trajectory_lst[idx][time_step + 4] = obs[world_idx, agent_idx]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step]
                expert_partner_mask_lst[idx][time_step + 4] = partner_mask[world_idx, agent_idx] == 2
                expert_road_mask_lst[idx][time_step + 4] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, time_step] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, time_step] = agent_info[world_idx, agent_idx, 7:8]
                # expert_partner_id_lst[idx, time_step] = env.partner_ids[world_idx, agent_idx].clone()
                # expert_ego_id_lst[idx, time_step] = agent_info[world_idx, agent_idx, -1]
                # expert_scene_id_lst[idx, time_step] = world_idx
            expert_valid_mask_lst[idx][time_step] = (~dead_agent_mask[world_idx, agent_idx]) & ~(
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, 1]) >  0.5) | 
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, 0]) >  5) | 
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, -1]) > 0.2)
                )

        
        # env.step() -> gather next obs
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs() 
        road_mask = env.get_road_mask()
        partner_mask = env.get_partner_mask()
        # partner_id = env.get_partner_id().unsqueeze(-1)
        agent_info = (
        env.sim.absolute_self_observation_tensor()
        .to_torch()
        .to(device)
        )
        infos = env.get_infos()
        
        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        goal_achieved = torch.clamp(goal_achieved, max=1.0)
        off_road = torch.clamp(off_road, max=1.0)
        veh_collision = torch.clamp(veh_collision, max=1.0)

        if (dead_agent_mask == True).all():
            goal_rate = goal_achieved.sum().float() / cont_agent_mask.sum().float()
            off_road_rate = off_road.sum().float() / cont_agent_mask.sum().float()
            veh_coll_rate = veh_collision.sum().float() / cont_agent_mask.sum().float()
            collision = (veh_collision + off_road > 0)
            goal_mask = goal_achieved > 0
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
            break
    
    expert_trajectory_lst = expert_trajectory_lst[goal_mask].to('cpu')
    expert_actions_lst = expert_actions_lst[goal_mask].to('cpu')
    expert_valid_mask_lst = expert_valid_mask_lst[goal_mask].to('cpu')
    expert_partner_mask_lst = expert_partner_mask_lst[goal_mask].to('cpu')
    expert_road_mask_lst = expert_road_mask_lst[goal_mask].to('cpu')
    # global pos
    expert_global_pos_lst = expert_global_pos_lst[goal_mask].to('cpu')
    expert_global_rot_lst = expert_global_rot_lst[goal_mask].to('cpu')

    # make directories
    os.makedirs(save_path + "/obs", exist_ok=True)
    os.makedirs(save_path + "/actions", exist_ok=True)
    os.makedirs(save_path + "/valid_mask", exist_ok=True)
    os.makedirs(save_path + "/partner_mask", exist_ok=True)
    os.makedirs(save_path + "/road_mask", exist_ok=True)
    os.makedirs(save_path + "/ego_global_pos", exist_ok=True)
    os.makedirs(save_path + "/ego_global_rot", exist_ok=True)

    # save as npy
    np.save(f"{save_path}/obs/trajectory_{save_index}.npy", expert_trajectory_lst)
    np.save(f"{save_path}/actions/trajectory_{save_index}.npy", expert_actions_lst)
    np.save(f"{save_path}/valid_mask/trajectory_{save_index}.npy", expert_valid_mask_lst)
    np.save(f"{save_path}/partner_mask/trajectory_{save_index}.npy", expert_partner_mask_lst)
    np.save(f"{save_path}/road_mask/trajectory_{save_index}.npy", expert_road_mask_lst)
    np.save(f"{save_path}/ego_global_pos/trajectory_{save_index}.npy", expert_global_pos_lst)
    np.save(f"{save_path}/ego_global_rot/trajectory_{save_index}.npy", expert_global_rot_lst)

def save_linear_probing_npy(env, save_path, save_index=0, future_steps=[10, 20, 30, 40]):
    def _transform_relative_other_pos(ego_current_pos, ego_current_rot, ego_future_pos, ego_future_rot, other_future_relative_pos):
        """transform future time t relative pos to current relative pos"""
        # 1. transform t-relative pos to t-global pos
        # get partner's relative pos and rot at time t
        t_partner_pos = other_future_relative_pos * MAX_REL_AGENT_POS

        # get ego's global pos and rot at time t
        t_partner_global_pos_x = ego_future_pos[0] + t_partner_pos[..., 0] * torch.cos(ego_future_rot) - t_partner_pos[..., 1] * torch.sin(ego_future_rot)
        t_partner_global_pos_y = ego_future_pos[1] + t_partner_pos[..., 0] * torch.sin(ego_future_rot) + t_partner_pos[..., 1] * torch.cos(ego_future_rot)

        # 2. transform t-global pos to current relative pos
        delta_x = t_partner_global_pos_x - ego_current_pos[0]
        delta_y = t_partner_global_pos_y - ego_current_pos[1]
        
        cos_theta = torch.cos(-ego_current_rot)
        sin_theta = torch.sin(-ego_current_rot)

        current_relative_pos_x = delta_x * cos_theta + delta_y * sin_theta
        current_relative_pos_y = -delta_x * sin_theta + delta_y * cos_theta
        current_relative_pos_x = 2 * ((current_relative_pos_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((current_relative_pos_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos = torch.stack((current_relative_pos_x, current_relative_pos_y), axis=-1)
        
        return current_relative_pos

    def _transform_relative_ego_pos(ego_current_pos, ego_current_rot, ego_future_pos):
        """transform global pos to current relative pos"""        
        delta_x = ego_future_pos[0] - ego_current_pos[0]
        delta_y = ego_future_pos[1] - ego_current_pos[1]
        
        cos_theta = torch.cos(ego_current_rot)
        sin_theta = torch.sin(ego_current_rot)

        rel_x = delta_x * cos_theta + delta_y * sin_theta
        rel_y = -delta_x * sin_theta + delta_y * cos_theta

        current_relative_pos_x = 2 * ((rel_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((rel_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos = torch.cat((current_relative_pos_x, current_relative_pos_y), axis=-1)
        
        return current_relative_pos
    
    def _get_multi_class_pos(pos):
        """
        Convert continuous pos to multi-class discrete pos based on x, y.
        """
        x, y = pos[..., 0], pos[..., 1]
        
        xbins = torch.linspace(-0.05, 0.05, 9).to(pos.device)
        ybins = torch.linspace(-0.05, 0.05, 9).to(pos.device)

        # Digitize x and y into 8 categories (0 to 7)
        x_bins = torch.bucketize(x, xbins) - 1
        y_bins = torch.bucketize(y, ybins) - 1
        
        # Ensure values are within valid range (0 to 7)
        x_bins = torch.clip(x_bins, 0, 7)
        y_bins = torch.clip(y_bins, 0, 7)
        
        discrete_pos = x_bins * 8 + y_bins
        return discrete_pos
    
    obs = env.reset()
    expert_actions, _, _, _ , _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_valid_mask_lst = torch.zeros((alive_agent_num, env.episode_len), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.zeros((alive_agent_num, env.episode_len, 127), device=device, dtype=torch.int)
    ego_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device) # global pos (2)
    ego_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    ego_future_pos_dict = {}
    ego_future_valid_mask_dict = {}
    other_future_pos_dict = {}
    other_future_valid_mask_dict = {}

    for future_step in future_steps:
        ego_future_pos_dict[future_step] = torch.zeros(
            (alive_agent_num, env.episode_len), device=device
        )
        ego_future_valid_mask_dict[future_step] = torch.zeros(
            (alive_agent_num, env.episode_len), device=device, dtype=torch.bool
        )
        other_future_pos_dict[future_step] = torch.zeros(
            (alive_agent_num, env.episode_len, 127), device=device
        )
        other_future_valid_mask_dict[future_step] = torch.zeros(
            (alive_agent_num, env.episode_len, 127), device=device, dtype=torch.bool
        )

    # Initialize dead agent mask
    agent_info = (
            env.sim.absolute_self_observation_tensor()
            .to_torch()
            .to(device)
        )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    goal_achieved = 0
    off_road = 0
    veh_collision = 0
    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            expert_valid_mask_lst[idx][time_step] = (~dead_agent_mask[world_idx, agent_idx]) & ~(
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, 1]) >  0.5) | 
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, 0]) >  5) | 
                    (torch.abs(expert_actions[world_idx, agent_idx, time_step, -1]) > 0.2)
                )
            expert_partner_mask_lst[idx][time_step] = partner_mask[world_idx, agent_idx]
            for future_step in future_steps:
                ego_global_pos_lst[idx, time_step] = agent_info[world_idx, agent_idx, 0:2]
                ego_global_rot_lst[idx, time_step] = agent_info[world_idx, agent_idx, 7:8]
                if time_step - future_step >= 0:
                    if not dead_agent_mask[world_idx, agent_idx]:
                        # ego future valid_mask
                        ego_future_valid_mask_dict[future_step][idx, time_step - future_step] = (expert_valid_mask_lst[idx][time_step] 
                                                                                                & expert_valid_mask_lst[idx][time_step - future_step])
                        # ego future pos
                        ego_current_pos = ego_global_pos_lst[idx, time_step - future_step]
                        ego_current_rot = ego_global_rot_lst[idx, time_step - future_step]
                        ego_future_pos = ego_global_pos_lst[idx, time_step]
                        ego_future_rot = ego_global_rot_lst[idx, time_step]
                        current_relative_ego_pos = _transform_relative_ego_pos(ego_current_pos, ego_current_rot, ego_future_pos)
                        ego_future_pos_dict[future_step][idx, time_step - future_step] = _get_multi_class_pos(current_relative_ego_pos)
                        
                        # other future valid mask
                        partner_valid_mask = (expert_partner_mask_lst[idx][time_step] == 0) & (expert_partner_mask_lst[idx][time_step - future_step] == 0)
                        other_future_valid_mask_dict[future_step][idx, time_step - future_step] = partner_valid_mask

                        # other future pos
                        other_future_relative_pos = obs[world_idx, agent_idx, 6:128*6].reshape(127, 6)[..., 1:3]
                        current_relative_other_pos = _transform_relative_other_pos(ego_current_pos, ego_current_rot, ego_future_pos, ego_future_rot, other_future_relative_pos)
                        current_relative_other_pos[~partner_valid_mask] = 0
                        other_future_pos_dict[future_step][idx, time_step - future_step] = _get_multi_class_pos(current_relative_other_pos)
        
        # env.step() -> gather next obs
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs() 
        partner_mask = env.get_partner_mask()
        # partner_id = env.get_partner_id().unsqueeze(-1)
        agent_info = (
        env.sim.absolute_self_observation_tensor()
        .to_torch()
        .to(device)
        )
        infos = env.get_infos()
        
        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        goal_achieved = torch.clamp(goal_achieved, max=1.0)
        off_road = torch.clamp(off_road, max=1.0)
        veh_collision = torch.clamp(veh_collision, max=1.0)

        if (dead_agent_mask == True).all():
            goal_rate = goal_achieved.sum().float() / cont_agent_mask.sum().float()
            off_road_rate = off_road.sum().float() / cont_agent_mask.sum().float()
            veh_coll_rate = veh_collision.sum().float() / cont_agent_mask.sum().float()
            collision = (veh_collision + off_road > 0)
            goal_mask = goal_achieved > 0
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
            break
    
    for future_step in future_steps:
        ego_future_pos_dict[future_step] = ego_future_pos_dict[future_step][goal_mask].to('cpu')
        ego_future_valid_mask_dict[future_step] = ego_future_valid_mask_dict[future_step][goal_mask].to('cpu')
        other_future_pos_dict[future_step] = other_future_pos_dict[future_step][goal_mask].to('cpu')
        other_future_valid_mask_dict[future_step] = other_future_valid_mask_dict[future_step][goal_mask].to('cpu')
        
        os.makedirs(f"{save_path}/linear_probing/ego_future_pos/step{future_step}", exist_ok=True)
        os.makedirs(f"{save_path}/linear_probing/ego_future_valid_mask/step{future_step}", exist_ok=True)
        os.makedirs(f"{save_path}/linear_probing/other_future_pos/step{future_step}", exist_ok=True)
        os.makedirs(f"{save_path}/linear_probing/other_future_valid_mask/step{future_step}", exist_ok=True)

        np.save(f"{save_path}/linear_probing/ego_future_pos/step{future_step}/trajectory_{save_index}.npy", ego_future_pos_dict[future_step])
        np.save(f"{save_path}/linear_probing/ego_future_valid_mask/step{future_step}/trajectory_{save_index}.npy", ego_future_valid_mask_dict[future_step])
        np.save(f"{save_path}/linear_probing/other_future_pos/step{future_step}/trajectory_{save_index}.npy", other_future_pos_dict[future_step])
        np.save(f"{save_path}/linear_probing/other_future_valid_mask/step{future_step}/trajectory_{save_index}.npy", other_future_valid_mask_dict[future_step])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/data/full_version/processed')
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'],)
    parser.add_argument('--function', type=str, default='save_linear_probing_npy', choices=['save_trajectory', 'save_trajectory_npy', 'save_linear_probing_npy'])
    parser.add_argument('--dataset-size', type=int, default=80000) # total_world
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    parser.add_argument('--start-idx', type=int, default=None, help="start scene number of dataset")
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    save_path = os.path.join(args.save_path, f'{args.dataset}_subset_v6')
    print()
    print("num_stack : ", args.num_stack)
    print("save_path : ", save_path)
    print("dataset : ", args.dataset)
    print("function : ", args.function)
    # Initialize configurations
    env_config = EnvConfig(
        dynamics_model='delta_local',
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior="remove"
    )
    print('Scene Loader')
    # Create data loader
    train_loader = SceneDataLoader(
        root=f"/data/full_version/data/{args.dataset}/",
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        sample_with_replacement=False,
        shuffle=False,
        start_idx=args.start_idx
    )
    print('Call Env')
    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=128,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )
    print('Launch Env')
    total_iter = int(args.dataset_size // args.batch_size)
    init_iter = 0 if args.start_idx is None else args.start_idx // args.batch_size
    
    for i in tqdm(range(init_iter, total_iter), total=total_iter, initial=init_iter):
        print(env.data_batch)
        if args.function == 'save_trajectory':
            save_trajectory(env, save_path, i * args.batch_size)
        elif args.function == 'save_trajectory_npy':
            save_trajectory_npy(env, save_path, i * args.batch_size)
        elif args.function == 'save_linear_probing_npy':
            save_linear_probing_npy(env, save_path, i * args.batch_size)
        else:
            raise ValueError("Invalid function name")
        if i != total_iter - 1:
            env.swap_data_batch()
    env.close()
    del env
    del env_config