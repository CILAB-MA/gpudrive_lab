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

def get_label(log_actions, st, en, done_step, index_array,
                dy_thresh=0.035, dyaw_thresh=0.02):
    unique_world = torch.unique(index_array)
    N = en - st
    full_indices = torch.arange(N)
    alive_world = torch.isin(full_indices, unique_world).long()
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
        label = np.full(dy_peak.shape, 4, dtype=np.int32)
        abnormal_mask = (dy_peak > 0.5) | (dyaw_peak > 0.2)
        retreat_mask = dx_exceed_count > 0.5
        turn_mask = (dy_peak > 0.035) & (dyaw_peak > 0.025) & (max_ratio > 0.15)
        straight_mask = (dy_peak < 0.01) & (dyaw_peak < 0.01)
        label[abnormal_mask] = 0
        label[retreat_mask] = 1
        label[turn_mask] = 2
        label[straight_mask] = 3
        scene_labels.append(label)

    scene_labels = np.concatenate(scene_labels)
    return scene_labels



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="validation", help="training (80000) / testing (10000)")
    parser.add_argument("--total-scene-size", "-tss", type=int, default=10000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=100)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()

    SAVE_DIR = f"/data/full_version/processed/{args.data_dir}_subset_v2/label"
    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    env_config = EnvConfig(
        collision_behavior="remove"
    )
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
    torch.set_printoptions(precision=3)
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    for idx in tqdm(range(num_iter)):
        obs = env.reset()
        off_road = 0
        veh_collision = 0
        partner_mask = env.get_partner_mask()
        road_mask = env.get_road_mask()
        alive_agent_num = env.cont_agent_mask.sum().item()
        expert_partner_label_lst = np.full((alive_agent_num, env.episode_len, 127), -1)
        expert_ego_label_lst = np.full((alive_agent_num, 1), 4)
        expert_partner_id_lst = torch.full((alive_agent_num, env.episode_len, 127), -1, device="cuda", dtype=torch.long)
        expert_actions, _, _, _, expert_valids = env.get_expert_actions()
        alive_agent_mask = env.cont_agent_mask.clone()
        alive_sum = alive_agent_mask.sum(-1)
        index_array = torch.tensor([i for i, count in enumerate(alive_agent_mask.sum(-1).tolist()) for _ in range(count)])
        cont_agent_mask = env.cont_agent_mask.to('cuda')
        alive_world = alive_agent_mask.sum(-1).cpu().numpy()
        log_actions = expert_actions[alive_agent_mask]
        done_step = torch.zeros(len(log_actions)).to('cuda')
        expert_timesteps = expert_valids.squeeze(-1).sum(-1)
        alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
        dead_agent_mask = ~env.cont_agent_mask.clone().to('cuda')
        ego_ids = env.ego_ids.clone()[alive_agent_mask]
        
        for t in tqdm(range(env.episode_len)):
            # Step the environment
            partner_ids = env.partner_ids.clone()[alive_agent_mask].int()
            expert_partner_id_lst[:, t] = partner_ids
            expert_actions, _, _, _, _ = env.get_expert_actions()
            env.step_dynamics(expert_actions[:, :, t, :])
            expert_actions_t = expert_actions[:, :, t, :]
            control_actions = expert_actions_t[alive_agent_mask]
    
            obs = env.get_obs()
            done = env.get_dones()
            infos = env.get_infos()
            off_road += infos.off_road[cont_agent_mask]
            veh_collision += infos.collided[cont_agent_mask]
            off_road = torch.clamp(off_road, max=1.0)
            veh_collision = torch.clamp(veh_collision, max=1.0)
            mask = (done[alive_agent_mask] == 1.0) & (done_step == 0)
            done_step[mask] = t
            if done.all():
                collision = (veh_collision + off_road > 0)
                scene_labels = get_label(log_actions.cpu().numpy(), idx * NUM_WORLDS , (idx + 1) * NUM_WORLDS, done_step, index_array)
                break
        save_path = f"/data/full_version/"
        scene_idx = np.arange(idx * NUM_WORLDS, (idx + 1) * NUM_WORLDS)
        index_array = index_array + idx * NUM_WORLDS
        index_array = index_array[~collision.cpu()]
        ego_ids_np = ego_ids.cpu().int().numpy()
        expert_partner_id_lst = expert_partner_id_lst[~collision]
        expert_partner_id_lst = expert_partner_id_lst.cpu().numpy()
        expert_partner_id_lst_flat = expert_partner_id_lst.reshape(-1)
        id_to_label = dict(zip(ego_ids_np, scene_labels))
        valid_mask = expert_partner_id_lst_flat >= 0
        valid_ids = expert_partner_id_lst_flat[valid_mask]
        labels = np.array([id_to_label[pid] if pid in id_to_label else -1 for pid in valid_ids])
        expert_partner_label_lst = expert_partner_label_lst[~collision.cpu()].reshape(-1)
        expert_partner_label_lst[valid_mask] = labels
        expert_partner_label_lst = expert_partner_label_lst.reshape(expert_partner_id_lst.shape)
        done_step = done_step[~collision.cpu()]
        scene_labels = scene_labels[~collision.cpu()]
        # np.savez_compressed(f'{SAVE_DIR}/label_trajectory_{args.scene_batch_size * idx}.npz',
        #                     partner_label=expert_partner_label_lst,
        #                     ego_label=scene_labels)
        print(f'alive agent: {len(index_array)}')
            
        if idx != num_iter - 1:
            env.swap_data_batch()     
    env.close()
