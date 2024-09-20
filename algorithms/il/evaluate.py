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

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='discrete', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],)
    parser.add_argument('--load-dir', '-l', type=str, default='models')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--model-name', '-m', type=str, default='bc_policy')
    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = parse_args()
    # Configurations
    # Configurations
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ),
        dyaw=torch.round(
            torch.linspace(-3.14, 3.14, 300), decimals=3
        ),
    )
    bc_config = BehavCloningConfig()

    # # Make env
    # env = GPUDriveTorchEnv(
    #     config=env_config,
    #     data_dir=bc_config.data_dir,
    #     render_config=render_config,
    #     num_worlds=bc_config.num_worlds,
    #     max_cont_agents=bc_config.max_cont_agents,
    #     device=bc_config.device,  # Use DEVICE here for consistency
    #
    NUM_WORLDS = 50
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    render_config = RenderConfig(draw_obj_idx=True)
    # print('Initializeing env....')
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=1,  # Number of agents to control
        device=args.device,
        render_config=render_config,
        num_stack=3,
        action_type=args.action_type
    )
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
        actions = bc_policy(obs[~dead_agent_mask, :], deterministic=True)
        # actions = bc_policy.act(obs[~dead_agent_mask, :], deterministic=True)
        all_actions[~dead_agent_mask] = actions /10
        env.step_dynamics(all_actions)
        loss = (actions / 10 - expert_actions[~dead_agent_mask][:, time_step, :])
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
    non_veh_collision = controlled_agent_info[:, 2]
    goal_achieved = controlled_agent_info[:, 3]

    off_road_rate = off_road.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision.sum().float() / alive_agent_mask.sum().float()
    non_veh_coll_rate = non_veh_collision.sum().float() / alive_agent_mask.sum().float()
    goal_rate = goal_achieved.sum().float() / alive_agent_mask.sum().float()
    collision_rate = off_road_rate + veh_coll_rate + non_veh_coll_rate
    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Non-vehCol {non_veh_coll_rate} Goal {goal_rate}')

    if args.make_video:
        imageio.mimwrite(f'models/{args.model_name}.mp4', np.array(frames), fps=30)