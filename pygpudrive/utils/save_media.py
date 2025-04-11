# GPU-Drive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.config import DynamicsModel, ActionSpace
from pygpudrive.registration import make

import os
import torch
import numpy as np
import imageio
from tqdm import tqdm


def save_video(dataset, num_worlds, start_idx, save_dir="/data/expert/videos"):
    # Set up the configuration
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{dataset}/",
                               num_scenes=num_worlds,
                               start_idx=start_idx,
                               discipline=SelectionDiscipline.RANGE_N)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore'

    )
    render_config = RenderConfig(
        draw_obj_idx=True,
    )
    
    # Initialize environment
    kwargs={
        "config": env_config,
        "scene_config": scene_config,
        "render_config": render_config,
        "max_cont_agents": 128,
        "device": "cuda",
        "num_stack": 1
    }
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)
    
    # Get video from expert
    env.reset()
    expert_actions, _, _ = env.get_expert_actions()
    dead_agent_mask = ~env.cont_agent_mask.clone().to("cuda")
    frames = [[] for _ in range(num_worlds)]
    for time_step in tqdm(range(env.episode_len)):
        for world_idx in range(num_worlds):
            if (dead_agent_mask[world_idx] == False).any():
                frame = env.render(world_render_idx=world_idx)
                frames[world_idx].append(frame)
        env.step_dynamics(expert_actions[:, :, time_step, :])
        env.get_obs()
        dones = env.get_dones().to("cuda")
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    
    # Save video
    path = os.path.join(save_dir, dataset)
    os.makedirs(path, exist_ok=True)
    for render in range(num_worlds):
        video_name = f"{path}/world_{start_idx + render}.mp4"
        imageio.mimwrite(video_name, np.array(frames[render]), fps=30)
    env.close()

def save_frame(dataset, num_worlds, start_idx, save_dir="/data/expert/pictures"):
    # Set up the configuration
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{dataset}/",
                               num_scenes=num_worlds,
                               start_idx=start_idx,
                               discipline=SelectionDiscipline.RANGE_N)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore'
    )
    render_config = RenderConfig(
        draw_obj_idx=True,
    )
    
    # Initialize environment
    kwargs={
        "config": env_config,
        "scene_config": scene_config,
        "render_config": render_config,
        "max_cont_agents": 128,
        "device": "cuda",
        "num_stack": 1
    }
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)
    
    # Get video from expert
    env.reset()
    frames = [[] for _ in range(num_worlds)]
    for world_idx in range(num_worlds):
        frame = env.render(world_render_idx=world_idx)
        frames[world_idx].append(frame)
    
    # Save frame
    path = os.path.join(save_dir, dataset)
    os.makedirs(path, exist_ok=True)
    for render in range(num_worlds):
        frame_name = f"{path}/world_{start_idx + render}.png"
        imageio.imwrite(frame_name, np.array(frames[render][0]))
    env.close()


if __name__ == "__main__":
    import argparse
    import pprint
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_worlds", type=int, required=True)
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--function", type=str, default="save_video", choices=["save_video", "save_frame"])
    args = parser.parse_args()
    
    pprint.pprint(vars(args))
    if args.function == "save_video":
        save_video(args.dataset, args.num_worlds, args.start_idx)
    elif args.function == "save_frame":
        save_frame(args.dataset, args.num_worlds, args.start_idx)
    else:
        raise ValueError("Invalid function name. Use 'save_frame' or 'save_video'.")