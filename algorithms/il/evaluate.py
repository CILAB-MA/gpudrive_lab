"""Obtain a policy using behavioral cloning."""

import logging, imageio
import torch
import os, sys
import numpy as np
sys.path.append(os.getcwd())
import argparse
from datetime import datetime

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.config import DynamicsModel, ActionSpace
from algorithms.il.model.bc import *
from pygpudrive.registration import make



def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    # ENV
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    # EXPERIMENT
    parser.add_argument('--dataset', type=str, default='valid', choices=['train', 'valid'],)
    parser.add_argument('--action-scale', '-as', type=int, default=1)
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--model-name', '-m', type=str, default='late_fusion_gmm_all_data')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/videos')

    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    args = parse_args()
    
    # Configurations
    NUM_WORLDS = 50
    MAX_NUM_OBJECTS = 1

    # Initialize configurations
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{args.dataset}/",
                               NUM_WORLDS,)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dy=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
    )
    render_config = RenderConfig(
        draw_obj_idx=True
    )
    # Initialize environment
    kwargs={
        "config": env_config,
        "scene_config": scene_config,
        "render_config": render_config,
        "max_cont_agents": MAX_NUM_OBJECTS,
        "device": args.device,
        "num_stack": args.num_stack
    }
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)

    bc_policy = torch.load(f"{args.model_path}/{args.model_name}.pth").to(args.device)
    bc_policy.eval()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()

    obs = env.reset()
    frames = [[] for _ in range(NUM_WORLDS)]
    expert_actions, _, _ = env.get_expert_actions()
    for time_step in range(env.episode_len):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to(args.device)
        with torch.no_grad():
            actions = bc_policy(obs[~dead_agent_mask], deterministic=True)
        all_actions[~dead_agent_mask, :] = actions / args.action_scale

        env.step_dynamics(all_actions)
        loss = torch.abs(all_actions[~dead_agent_mask] - expert_actions[~dead_agent_mask][:, time_step, :])
        
        print(f'TIME {time_step} LOSS: {loss.mean(0)}')

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        if args.make_video:
            for world_render_idx in range(NUM_WORLDS):
                frame = env.render(world_render_idx=world_render_idx)
                frames[world_render_idx].append(frame)

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
    collision_rate = off_road_rate + veh_coll_rate
    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
    print(f'Success World idx : ', torch.where(goal_achieved == 1)[0].tolist())

    if args.make_video:
        time = datetime.now().strftime("%Y%m%d%H%M")
        for world_render_idx in range(NUM_WORLDS):
            if world_render_idx not in torch.where(veh_collision + off_road >= 1)[0].tolist():
                video_path = os.path.join(args.video_path, args.dataset, args.model_name)
                if not os.path.exists(video_path):
                    os.makedirs(video_path)
                imageio.mimwrite(f'{video_path}/world_{world_render_idx}_{time}.mp4', np.array(frames[world_render_idx]), fps=30)