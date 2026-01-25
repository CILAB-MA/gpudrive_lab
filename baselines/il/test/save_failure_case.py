"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())

import logging, imageio
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import mediapy as media

# GPUDrive
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(args, env, bc_policy, dataset, scene_batch_idx):
    obs = env.reset()

    if args.sim_agent == 'delta_replay':
        alive_agent_mask = env.cont_agent_mask.clone()
        ego_idx = alive_agent_mask.float().argmax(dim=-1)
        alive_agent_mask = torch.zeros_like(alive_agent_mask, dtype=torch.bool, device=alive_agent_mask.device).scatter_(-1, ego_idx.unsqueeze(-1), True) & alive_agent_mask.any(-1, keepdim=True)
        dead_agent_mask = ~alive_agent_mask
    else:
        alive_agent_mask = env.cont_agent_mask.clone()
        dead_agent_mask = ~env.cont_agent_mask.clone()

    obs_stack1_feat_size = int(obs.shape[-1] / 5)
    poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]
    infos = env.get_infos()
    veh_coll_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to("cuda")
    off_road_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to("cuda")
    scenario_ids = env.get_scenario_ids()
    ego_ids = env.ego_ids[alive_agent_mask].clone()
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]

        # Record when agent status changed(goal or collided)
        veh_collision = infos.collided[alive_agent_mask]
        veh_coll_mask = (veh_collision > 0) & (veh_coll_timesteps == -1)
        veh_coll_timesteps[veh_coll_mask] = time_step

        off_road = infos.off_road[alive_agent_mask]
        off_road_mask = (off_road > 0) & (off_road_timesteps == -1)
        off_road_timesteps[off_road_mask] = time_step

        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            context, *_, = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=True)
            actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions

        env.step_dynamics(all_actions)
        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        off_road_ep += infos.off_road[alive_agent_mask]
        veh_collision_ep += infos.collided[alive_agent_mask]
        off_road_ep = torch.clamp(off_road_ep, max=1)
        veh_collision_ep = torch.clamp(veh_collision_ep, max=1)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    off_road_rate = off_road_ep.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision_ep.sum().float() / alive_agent_mask.sum().float()
    collision_rate = off_road_rate + veh_coll_rate

    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate}')
    if args.make_csv:
        csv_path = f"/data/full_version/failure_case_id.csv"
        file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if file_is_empty:
                column_name = "model"
                metrics = ["dataset","scenario_id","ego_id","OffRoad","VehCollision", "OffRoadStep", "VehCollStep"]
                for metric in metrics:
                    column_name += f",{metric}"
                f.write(column_name + ",\n")
            for s, (ego_id, off_road, veh_coll, offroad_step, vehcoll_step) in enumerate(zip(ego_ids, off_road_ep, veh_collision_ep, off_road_timesteps, veh_coll_timesteps)):
                scene_id = scenario_ids[s]
                if off_road + veh_coll > 0:
                    data = f"{args.model_name[:-4]},{args.dataset},{scene_id},{ego_id},{off_road},{veh_coll},{offroad_step},{vehcoll_step}"
                    f.write(data + ",\n")

    return off_road_rate, veh_coll_rate, collision_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    
    parser.add_argument('--dataset-size', type=int, default=80000) # total_world
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/exp_40000')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s42_0828_170643.pth') # early_attn_s11_0808_043910
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/full_version/videos')
    parser.add_argument('--sim-agent', '-sa', type=str, default='log_replay', choices=['log_replay', 'self_play', 'delta_replay'])
    parser.add_argument('--dataset', '-d', type=str, default='training', choices=['training', 'validation'])
    args = parser.parse_args()
    # Configurations
    num_cont_agents = 1 if args.sim_agent == 'log_replay' else 128

    # Create data loader
    if args.dataset == 'training':
        scene_loader = SceneDataLoader(
            root=f"/data/full_version/data/training/",
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            sample_with_replacement=False,
            shuffle=False,
        )
        dataset_size = args.dataset_size
    else:
        # Test Scene
        scene_loader = SceneDataLoader(
            root=f"/data/full_version/data/validation/",
            batch_size=args.batch_size,
            dataset_size=9987,
            sample_with_replacement=False,
            shuffle=False,
        )
        dataset_size = 9987
    print(f'{args.dataset} len scene loader {len(scene_loader)}')
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore',
        num_stack=5

    )
    render_config = RenderConfig(
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=scene_loader,
        max_cont_agents=num_cont_agents,  # Number of agents to control
        device="cuda",
        render_config=render_config,
        action_type="continuous",
    )
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    num_iter = int(dataset_size // args.batch_size) if dataset_size != 0 else 0

    for i in tqdm(range(num_iter)):
        run(args, env, bc_policy, dataset=args.dataset, scene_batch_idx=i)
        if i != num_iter - 1:
            print('SWAP!!')
            env.swap_data_batch()
    env.close()

