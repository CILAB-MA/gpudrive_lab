import torch
import numpy as np
import os
from tqdm import tqdm

from gpudrive.env.config import EnvConfig, SceneConfig, SelectionDiscipline
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader

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
    # partner_id = env.get_partner_id().unsqueeze(-1)
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_trajectory_lst = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((alive_agent_num, env.episode_len, 3), device=device)
    expert_dead_mask_lst = torch.ones((alive_agent_num, env.episode_len), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.full((alive_agent_num, env.episode_len, 127), 2, device=device, dtype=torch.long)
    expert_road_mask_lst = torch.ones((alive_agent_num, env.episode_len, 200), device=device, dtype=torch.bool)
    expert_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device) # global pos (2)
    expert_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    expert_partner_id_lst = torch.zeros((alive_agent_num, env.episode_len, 127), device=device) # global actions (1)
    expert_ego_id_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    expert_scene_id_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
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
                expert_trajectory_lst[idx][time_step] = obs[world_idx, agent_idx]
                expert_actions_lst[idx][time_step] = expert_actions[world_idx, agent_idx, time_step]
                expert_partner_mask_lst[idx][time_step] = partner_mask[world_idx, agent_idx]
                expert_road_mask_lst[idx][time_step] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, time_step] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, time_step] = agent_info[world_idx, agent_idx, 7:8]
                expert_partner_id_lst[idx, time_step] = env.partner_ids[world_idx, agent_idx].clone()
                expert_ego_id_lst[idx, time_step] = agent_info[world_idx, agent_idx, -1]
                expert_scene_id_lst[idx, time_step] = world_idx
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]

        
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
    expert_dead_mask_lst = expert_dead_mask_lst[goal_mask].to('cpu')
    expert_partner_mask_lst = expert_partner_mask_lst[goal_mask].to('cpu')
    expert_road_mask_lst = expert_road_mask_lst[goal_mask].to('cpu')
    # global pos
    expert_global_pos_lst = expert_global_pos_lst[goal_mask].to('cpu')
    expert_global_rot_lst = expert_global_rot_lst[goal_mask].to('cpu')
    expert_partner_id_lst = expert_partner_id_lst[goal_mask].to('cpu')
    expert_ego_id_lst = expert_ego_id_lst[goal_mask].to('cpu')
    expert_scene_id_lst = expert_scene_id_lst[goal_mask].to('cpu')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/global', exist_ok=True)
    os.makedirs(save_path + '/id', exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=expert_trajectory_lst,
                        actions=expert_actions_lst,
                        dead_mask=expert_dead_mask_lst,
                        partner_mask=expert_partner_mask_lst,
                        road_mask=expert_road_mask_lst)
    np.savez_compressed(f"{save_path}/global/global_trajectory_{save_index}.npz", 
                        ego_global_pos=expert_global_pos_lst,
                        ego_global_rot=expert_global_rot_lst)
    np.savez_compressed(f"{save_path}/id/id_trajectory_{save_index}.npz", 
                        ego_id=expert_ego_id_lst,
                        scene_id=expert_scene_id_lst,
                        partner_id=expert_partner_id_lst)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/scratch/cilab/log_replay')
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'],)
    parser.add_argument('--function', type=str, default='save_trajectory', 
                        choices=[
                            'save_trajectory'])
    parser.add_argument('--dataset-size', type=int, default=80000) # total_world
    parser.add_argument('--batch-size', type=int, default=400) # num_world
    parser.add_argument("--agent-idx", "-ai", type=int, default=0)
    parser.add_argument('--start-idx', type=int, default=None, help="start scene number of dataset")
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    save_path = os.path.join(args.save_path, f'{args.dataset}_subset/ego_idx_{args.agent_idx}')
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
        root=f"/scratch/cilab/data/{args.dataset}/",
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
        max_cont_agents=1,  # Number of agents to control
        device="cuda",
        cont_idx=args.agent_idx,
        action_type="continuous",
    )
    print('Launch Env')
    total_iter = int(args.dataset_size // args.batch_size)
    init_iter = 0 if args.start_idx is None else args.start_idx // args.batch_size
    
    for i in tqdm(range(init_iter, total_iter), total=total_iter, initial=init_iter):
        if args.function == 'save_trajectory':
            save_trajectory(env, save_path, i * args.batch_size)
        else:
            raise ValueError("Invalid function name")
        if i != total_iter - 1:
            env.swap_data_batch()
    env.close()
    del env
    del env_config