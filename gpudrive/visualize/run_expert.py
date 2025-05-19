import os
import sys
import argparse
import mediapy as media
import numpy as np

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument("--video-dir", "-vd", type=str, default="/data/full_version/expert_video/train")
    parser.add_argument("--total-scene-size", "-tss", type=int, default=80000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=20)
    args = parser.parse_args()

    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    VIDEO_DIR = args.video_dir
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    
    env_config = EnvConfig()
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=NUM_WORLDS,
        dataset_size=TOTAL_NUM_WORLDS,
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=128,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )

    for idx, batch in enumerate(train_loader):
        env.swap_data_batch(batch)
        obs = env.reset()
        partner_mask = env.get_partner_mask()
        road_mask = env.get_road_mask()
        frames = {f"env_{i}": [] for i in range(idx*NUM_WORLDS, idx*NUM_WORLDS + NUM_WORLDS)}
        expert_actions, _, _, _, _ = env.get_expert_actions()
        for t in range(env.episode_len):
            print(f"Step: {t}")

            # Step the environment
            expert_actions, _, _, _, _ = env.get_expert_actions()
            env.step_dynamics(expert_actions[:, :, t, :])
            
            # Make video
            sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(NUM_WORLDS)),
                time_steps=[t]*NUM_WORLDS,
                plot_importance_weight=False,
                plot_linear_probing=False,
                plot_linear_probing_label=False
            )

            for i in range(NUM_WORLDS):
                frames[f"env_{i + idx*NUM_WORLDS}"].append(
                    img_from_fig(sim_states[i])
                )

            obs = env.get_obs()
            partner_mask = env.get_partner_mask()
            road_mask = env.get_road_mask()
            reward = env.get_rewards()
            done = env.get_dones()
            info = env.get_infos()

            if done.all():
                for i in range(idx*NUM_WORLDS, idx*NUM_WORLDS + NUM_WORLDS):
                    os.makedirs(VIDEO_DIR, exist_ok=True)
                    media.write_video(
                        os.path.join(VIDEO_DIR, f"world_{i:05d}.gif"), np.array(frames[f"env_{i}"]), fps=60, codec="gif"
                    )
                break

    env.close()
