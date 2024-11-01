"""Obtain a policy using behavioral cloning."""

import logging, imageio
import torch
import torch.nn as nn
import os, sys, torch
import numpy as np
sys.path.append(os.getcwd())
import wandb, yaml, argparse
import torch.nn.functional as F
# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from baselines.il.config import BehavCloningConfig
from baselines.il.run_bc_from_scratch import ContFeedForward, ContFeedForwardMSE
import argparse
from pygpudrive.registration import make
from pygpudrive.env.config import DynamicsModel, ActionSpace
def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    parser.add_argument('--load-dir', '-l', type=str, default='models')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--model-name', '-m', type=str, default='late_fusion')
    parser.add_argument('--action-scale', '-as', type=int, default=1)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = parse_args()
    # Configurations
    # Configurations
    NUM_WORLDS = 50
    MAX_NUM_OBJECTS = 1

    # Initialize configurations
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{args.dataset}/",
                               NUM_WORLDS,)
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
    bc_config = BehavCloningConfig()
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)

    # torch.serialization.add_safe_globals([ContFeedForwardMSE])
    bc_policy = torch.load(f"{bc_config.model_path}/{args.model_name}.pth").to(args.device)
    bc_policy.eval()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    obs = env.reset()
    frames = []
    expert_actions, _, _ = env.get_expert_actions()
    for time_step in range(env.episode_len):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to(args.device)
        # print(f'OBS {obs[~dead_agent_mask, :] }')
        actions = bc_policy(obs[~dead_agent_mask, :])
        all_actions[~dead_agent_mask] = actions / args.action_scale
        # normalize
        all_actions[:, 0] = all_actions[:, 0] * 12 - 6
        all_actions[:, 1] = all_actions[:, 1] * 12 - 6
        all_actions[:, 2] = all_actions[:, 2] * 2 * np.pi - np.pi
        env.step_dynamics(all_actions)
        loss = (all_actions / args.action_scale - expert_actions[~dead_agent_mask][:, time_step, :])
        print(f'TIME {time_step} LOss: {loss}')
        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        if args.make_video:
            frame = env.render(world_render_idx=0)
            frames.append(frame)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    controlled_agent_info = infos[alive_agent_mask]
    off_road = controlled_agent_info[:, 0]
    veh_collision = controlled_agent_info[:, 1]
    goal_achieved = controlled_agent_info[:, 3]

    off_road_rate = off_road.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision.sum().float() / alive_agent_mask.sum().float()
    goal_rate = goal_achieved.sum().float() / alive_agent_mask.sum().float()
    collision_rate = off_road_rate + veh_coll_rate + non_veh_coll_rate
    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')

    if args.make_video:
        imageio.mimwrite(f'models/{args.model_name}.mp4', np.array(frames), fps=30)