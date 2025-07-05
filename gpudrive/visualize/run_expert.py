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
from tqdm import tqdm

def plot_action(log_actions, action_image_dir, st, en, done_step, index_array,
                dy_thresh=0.035, dyaw_thresh=0.02):
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['dx', 'dy', 'dyaw']
    _, T, _ = log_actions.shape
    unique_world = torch.unique(index_array)
    N = en - st
    full_indices = torch.arange(N)
    alive_world = torch.isin(full_indices, unique_world).long()
    colors = plt.get_cmap('gist_ncar')(torch.linspace(0, 1, len(index_array)).numpy())
    scene_labels = []
    for n in range(N):
        if alive_world[n] != 1:
            continue
        world_mask = index_array == n
        end = done_step[world_mask].cpu().numpy().astype('int')
        dx_list, dy_list, dyaw_list = [], [], []
        for i, e in enumerate(end):
            dx_list.append(log_actions[world_mask][i, :e, 0])
            dy_list.append(log_actions[world_mask][i, :e, 1])
            dyaw_list.append(log_actions[world_mask][i, :e, 2])

        dx_binary_list = [(d < -0.01).astype(int) for d in dx_list]
        dy_binary_list = [(np.abs(d) > dy_thresh).astype(int) for d in dy_list]
        dyaw_binary_list = [(np.abs(d) > dyaw_thresh).astype(int) for d in dyaw_list]

        dy_peak = np.array([np.abs(d).max() for d in dy_list])
        dyaw_peak = np.array([np.abs(d).max() for d in dyaw_list])
        dy_exceed_count = np.array([b.mean() for b in dy_binary_list])
        dx_exceed_count = np.array([b.mean() for b in dx_binary_list])
        dyaw_exceed_count = np.array([b.mean() for b in dyaw_binary_list])
        max_ratio =np.maximum(dy_exceed_count, dyaw_exceed_count)
        label = np.full(dy_peak.shape, 'NORMAL', dtype=object)
        abnormal_mask = (dy_peak > 0.5) | (dyaw_peak > 0.2)
        retreat_mask = dx_exceed_count > 0.5
        turn_mask = (dy_peak > 0.035) & (dyaw_peak > 0.025) & (max_ratio > 0.15)
        straight_mask = (dy_peak < 0.01) & (dyaw_peak < 0.01)
        label[abnormal_mask] = 'ABNORMAL'
        label[retreat_mask] = 'RETREAT'
        label[turn_mask] = 'TURN'
        label[straight_mask] = 'STRAIGHT'
        scene_labels.append(label)
        for a, l in enumerate(label):
            total_label = (
                f'World {st + n} Agent {a} {l}'
            )

            axs[0].plot(range(end[a]), dx_list[a], color=colors[a], label=total_label)
            axs[1].plot(range(end[a]), dy_list[a], color=colors[a])
            axs[2].plot(range(end[a]), dyaw_list[a], color=colors[a])

    for i in range(3):
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    axs[-1].set_xlabel('Time step')

    fig.legend(
        loc='center left',
        bbox_to_anchor=(0.87, 0.5),
        ncol=10,
        title=f"Agents over dy {dy_thresh} and dyaw {dyaw_thresh}"
    )

    fig.subplots_adjust(right=0.86)
    os.makedirs(action_image_dir, exist_ok=True)
    image_file_path = os.path.join(action_image_dir, f'expert_action_value_scene_{st}to{en}.png')
    plt.savefig(image_file_path, bbox_inches='tight')
    plt.close()
    scene_labels = np.concatenate(scene_labels)
    return scene_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-image', '-mi', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument("--video-dir", "-vd", type=str, default="/data/full_version/expert_video/training_log")
    parser.add_argument("--action-image-dir", "-aid", type=str, default="/data/full_version/expert_actions_full_veh/validation_label")
    parser.add_argument("--total-scene-size", "-tss", type=int, default=10000)
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
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    for idx in tqdm(range(num_iter)):
        # env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=True)
        obs = env.reset()
        partner_mask = env.get_partner_mask()
        road_mask = env.get_road_mask()
        frames = {f"env_{i}": [] for i in range(idx*NUM_WORLDS, idx*NUM_WORLDS + NUM_WORLDS)}
        expert_actions, _, _, _, expert_valids = env.get_expert_actions()
        alive_agent_mask = env.cont_agent_mask.clone()
        alive_sum = alive_agent_mask.sum(-1)
        index_array = torch.tensor([i for i, count in enumerate(alive_agent_mask.sum(-1).tolist()) for _ in range(count)])
        # batch_alive_indices = [torch.where(alive_agent_mask[b])[0] for b in range(alive_agent_mask.shape[0])] # agent indices

        alive_world = alive_agent_mask.sum(-1).cpu().numpy()
        log_actions = expert_actions[alive_agent_mask]
        done_step = torch.zeros(len(log_actions)).to('cuda')
        expert_timesteps = expert_valids.squeeze(-1).sum(-1)
        for t in tqdm(range(env.episode_len)):
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
            done = env.get_dones()
            mask = (done[alive_agent_mask] == 1.0) & (done_step == 0)
            done_step[mask] = t
            if done.all():
                if args.make_video:
                    for i in range(idx*NUM_WORLDS, idx*NUM_WORLDS + NUM_WORLDS):
                        os.makedirs(VIDEO_DIR + f'_ratio_{int(args.partner_portion_test)}', exist_ok=True)
                        media.write_video(
                            os.path.join(VIDEO_DIR + f'_ratio_{int(args.partner_portion_test)}', f"world_{i:05d}.gif"), np.array(frames[f"env_{i}"]), fps=60, codec="gif"
                        )
                if args.make_image:
                    scene_labels = plot_action(log_actions.cpu().numpy(), action_image_dir, idx * NUM_WORLDS , (idx + 1) * NUM_WORLDS, done_step, index_array)
                break
        if args.make_csv:
            csv_path = f"/data/full_version/expert_{args.data_dir}_data_full_vehs.csv"
            file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
            with open(csv_path, 'a', encoding='utf-8') as f:
                if file_is_empty:
                    f.write("scene_idx,done_step,label\n")
                scene_idx = np.arange(idx * NUM_WORLDS, (idx + 1) * NUM_WORLDS)
                scene_idx = scene_idx[alive_world.astype('bool')]
                for sc,do, sl in zip(index_array, done_step, scene_labels):
                    f.write(f"{sc},{do},{sl}\n")     
        if idx != num_iter - 1:
            env.swap_data_batch()     
    env.close()
