import os
import sys
import argparse
import mediapy as media
import numpy as np
import torch
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
import matplotlib.pyplot as plt
from matplotlib import gridspec

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_action(log_actions, action_image_dir, st, en, done_step, alive_world,
                dy_thresh=0.035, dyaw_thresh=0.02):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['dx', 'dy', 'dyaw']
    _, T, _ = log_actions.shape
    N = len(alive_world)
    colors = plt.get_cmap('gist_ncar')(torch.linspace(0, 1, N).numpy())

    w = 0  # index for valid (alive) worlds
    for n in range(N):
        if alive_world[n] != 1:
            continue
        end = int(done_step[w].item())
        dx = log_actions[w, :end, 0]
        dy = log_actions[w, :end, 1]
        dyaw = log_actions[w, :end, 2]
        dx_binary = (dx < -0.01).astype(int)
        dy_binary = (np.abs(dy) > dy_thresh).astype(int)
        dyaw_binary = (np.abs(dyaw) > dyaw_thresh).astype(int)
        dx_peak = np.min(dx)
        dy_peak = np.abs(dy).max()
        dyaw_peak = np.abs(dyaw).max()
        dx_binary = dx_binary.mean()
        dy_exceed_count = dy_binary.mean()
        dyaw_exceed_count = dyaw_binary.mean()
        max_ratio = max(dy_exceed_count, dyaw_exceed_count)
        if (dy_peak > 0.5) or (dyaw_peak > 0.2):
            label = 'ABNORMAL'
        elif dx_binary > 0.5:
            label = 'RETREAT'
        elif dy_peak > 0.035 and dyaw_peak > 0.025 and max_ratio > 0.15:
            label = 'TURN' 
        else:
            label = 'NORMAL'
        total_label = (
            f'Agent {st + n} {label}'
        )


        axs[0].plot(range(end), dx, color=colors[n], label=total_label)
        axs[1].plot(range(end), dy, color=colors[n])
        axs[2].plot(range(end), dyaw, color=colors[n])
        w += 1

    for i in range(3):
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    axs[-1].set_xlabel('Time step')

    fig.legend(
        loc='center left',
        bbox_to_anchor=(0.87, 0.5),
        ncol=2,
        title=f"Agents over dy {dy_thresh} and dyaw {dyaw_thresh}"
    )

    fig.subplots_adjust(right=0.86)
    os.makedirs(action_image_dir, exist_ok=True)
    image_file_path = os.path.join(action_image_dir, f'expert_action_value_scene_{st}to{en}.png')
    plt.savefig(image_file_path, bbox_inches='tight')
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="validation", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-image', '-mi', action='store_true')
    parser.add_argument("--video-dir", "-vd", type=str, default="/data/full_version/expert_video/validation_log")
    parser.add_argument("--action-image-dir", "-aid", type=str, default="/data/full_version/expert_actions/validation_label")
    parser.add_argument("--total-scene-size", "-tss", type=int, default=9987)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=50)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=1)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()

    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    VIDEO_DIR = args.video_dir
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    action_image_dir = args.action_image_dir
    env_config = EnvConfig()
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=NUM_WORLDS,
        dataset_size=TOTAL_NUM_WORLDS,
        shuffle=False
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=args.max_cont_agents,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )
    # env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=True)
    torch.set_printoptions(precision=3)
    for idx, batch in enumerate(train_loader):
        if len(train_loader) == idx:
            env.swap_data_batch()
        else:
            env.swap_data_batch(batch)
        # env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=True)
        obs = env.reset()
        partner_mask = env.get_partner_mask()
        road_mask = env.get_road_mask()
        frames = {f"env_{i}": [] for i in range(idx*NUM_WORLDS, idx*NUM_WORLDS + NUM_WORLDS)}
        expert_actions, _, _, _, expert_valids = env.get_expert_actions()
        alive_agent_mask = env.cont_agent_mask.clone()
        alive_world = alive_agent_mask.sum(-1).cpu().numpy()
        log_actions = expert_actions[alive_agent_mask]
        done_step = torch.zeros(len(log_actions)).to('cuda')
        expert_timesteps = expert_valids.squeeze(-1).sum(-1)
        for t in range(env.episode_len):
            print(f"Step: {t}")

            # Step the environment
            expert_actions, _, _, _, _ = env.get_expert_actions()
            env.step_dynamics(expert_actions[:, :, t, :])
            expert_actions_t = expert_actions[:, :, t, :]
            control_actions = expert_actions_t[alive_agent_mask]
            if args.make_video:
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
            # partner_mask = env.get_partner_mask()
            # road_mask = env.get_road_mask()
            # reward = env.get_rewards()
            done = env.get_dones()
            # info = env.get_infos()
            mask = (done[alive_agent_mask] == 1.0) & (done_step == 0)
            done_step[mask] = t
            if done.all():
                if args.make_video:
                    for i in range(idx*NUM_WORLDS, idx*NUM_WORLDS + NUM_WORLDS):
                        os.makedirs(VIDEO_DIR + f'_ratio_{int(args.partner_portion_test)}', exist_ok=True)
                        media.write_video(
                            os.path.join(VIDEO_DIR + f'_ratio_{int(args.partner_portion_test)}', f"world_{i:05d}.gif"), np.array(frames[f"env_{i}"]), fps=60, codec="gif"
                        )
                print(done_step, expert_timesteps[alive_agent_mask])
                if args.make_image:
                    plot_action(log_actions.cpu().numpy(), action_image_dir, idx * NUM_WORLDS , (idx + 1) * NUM_WORLDS, done_step, alive_world)
                break

    env.close()
