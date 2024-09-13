"""Obtain a policy using behavioral cloning."""

import logging, imageio
import torch
import torch.nn as nn
import os, sys, torch
import numpy as np
sys.path.append(os.getcwd())
import wandb, yaml, argparse

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from baselines.il.config import BehavCloningConfig
from baselines.il.run_bc_from_scratch import ContFeedForward

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='discrete', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],)
    parser.add_argument('--load-dir', '-l', type=str, default='models/')
    parser.add_argument('--make-video', '-mv', action='store_true')
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
    render_config = RenderConfig()
    bc_config = BehavCloningConfig()

    # # Make env
    # env = GPUDriveTorchEnv(
    #     config=env_config,
    #     data_dir=bc_config.data_dir,
    #     render_config=render_config,
    #     num_worlds=bc_config.num_worlds,
    #     max_cont_agents=bc_config.max_cont_agents,
    #     device=bc_config.device,  # Use DEVICE here for consistency
    # )
    NUM_WORLDS = 3
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    # print('Initializeing env....')
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=1,  # Number of agents to control
        device=args.device,
        render_config=render_config,
    )
    bc_policy = torch.load(os.path.join(args.load_dir, 'bc_policy.pt'))
    bc_policy.eval()
    alive_agent_mask = env.cont_agent_mask.clone()
    obs = env.reset()
    print(f'OBS SHAPE {obs.shape}')
    frames = []
    for time_step in range(env.episode_len):
        actions = bc_policy(obs, deterministic=True)
        env.step_dynamics(actions)

        obs = env.get_obs()

        dones = env.get_dones()
        infos = env.get_infos()
        if args.make_video:
            frame = env.render(world_render_idx=0)
            frames.append(frame)
    is_collision = infos[:, :, :3].sum(dim=-1)
    # print(f'Collision index 0 {infos[0]}')
    is_goal = infos[:, :, 3]
    collision_mask = is_collision != 0
    goal_mask = is_goal != 0
    valid_collision_mask = collision_mask & alive_agent_mask
    valid_goal_mask = goal_mask & alive_agent_mask
    collision_rate = valid_collision_mask.sum().float() / alive_agent_mask.sum().float()
    goal_rate = valid_goal_mask.sum().float() / alive_agent_mask.sum().float()
    print(f'Collision rate {collision_rate} Goal RATE {goal_rate}')
    if args.make_video:
        imageio.mimwrite(f'models/bc_policy_world_control_one.mp4', np.array(frames), fps=30)