import torch
import numpy as np
import os,sys
from tqdm import tqdm
sys.path.append(os.getcwd())
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.networks.late_fusion import NeuralNet
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
import pufferlib, yaml
from box import Box

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def get_label(log_actions, st, en, done_step, index_array,
              dy_thresh=0.035, dyaw_thresh=0.02):
    """Generate behavior labels from expert actions (abnormal=0, retreat=1, turn=2, straight=3, default=4)."""
    index_array = index_array.cpu()
    done_step = done_step.cpu()
    unique_world = torch.unique(index_array)
    N = en - st
    full_indices = torch.arange(N, device=unique_world.device)
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
        max_ratio = np.maximum(dy_exceed_count, dyaw_exceed_count)
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


def run_and_save(
    env,
    policy,
    batch_idx,
    batch_size,
    trajectory_path=None,
    label_path=None,
    save_trajectory=True,
    save_label=True,
):
    """
    Single rollout that saves both trajectory and behavior labels.
    Runs policy once through the episode, then saves trajectory (goal+no-collision) and
    labels (goal-achieved agents) to respective paths.

    Args:
        env: GPUDriveTorchEnv with discrete action type.
        policy: NeuralNet policy that returns discrete actions.
        batch_idx: Current batch index (for file naming).
        batch_size: Batch size / num_worlds.
        trajectory_path: Where to save trajectory npz files.
        label_path: Where to save label npz files.
        save_trajectory: If True, save trajectory data.
        save_label: If True, save behavior labels.
    """
    device = env.device
    obs = env.reset()
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    cont_agent_mask = env.cont_agent_mask.to(device)
    alive_agent_mask = env.cont_agent_mask.clone()
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    index_array = torch.tensor(
        [i for i, count in enumerate(alive_agent_mask.sum(-1).tolist()) for _ in range(count)],
        device=device,
    )
    print("alive_agent_num : ", alive_agent_num)

    # Trajectory buffers
    expert_trajectory_lst = torch.zeros(
        (alive_agent_num, env.episode_len, obs.shape[-1]), device=device
    )
    rl_actions_lst = torch.zeros(
        (alive_agent_num, env.episode_len), device=device, dtype=torch.long
    )
    expert_dead_mask_lst = torch.ones(
        (alive_agent_num, env.episode_len), device=device, dtype=torch.bool
    )
    expert_partner_mask_lst = torch.full(
        (alive_agent_num, env.episode_len, 127), 2, device=device, dtype=torch.long
    )
    expert_road_mask_lst = torch.ones(
        (alive_agent_num, env.episode_len, 200), device=device, dtype=torch.bool
    )
    expert_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device)
    expert_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device)
    expert_partner_id_lst = torch.zeros(
        (alive_agent_num, env.episode_len, 127), device=device
    )
    expert_ego_id_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device)
    expert_scene_id_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device)
    # Label buffers: log_actions (accel, steer, steer) for get_label
    log_actions_lst = torch.zeros(
        (alive_agent_num, env.episode_len, 3), device=device
    )
    done_step = torch.zeros(alive_agent_num, device=device)
    ego_ids = env.ego_ids.clone()[alive_agent_mask]

    agent_info = (
        env.sim.absolute_self_observation_tensor().to_torch().to(device)
    )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device)
    goal_achieved = torch.zeros(alive_agent_num, device=device)
    off_road = torch.zeros(alive_agent_num, device=device)
    veh_collision = torch.zeros(alive_agent_num, device=device)

    for t in tqdm(range(env.episode_len)):
        with torch.no_grad():
            alive_obs = obs[~dead_agent_mask]
            actions, *_ = policy(alive_obs)
        all_actions = torch.zeros(
            obs.shape[0], obs.shape[1], device=device, dtype=torch.long
        )
        all_actions[~dead_agent_mask] = actions
        action_values = env.action_keys_tensor[all_actions]

        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_trajectory_lst[idx][t] = obs[world_idx, agent_idx]
                rl_actions_lst[idx][t] = all_actions[world_idx, agent_idx]
                expert_partner_mask_lst[idx][t] = partner_mask[world_idx, agent_idx]
                expert_road_mask_lst[idx][t] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, t] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, t] = agent_info[world_idx, agent_idx, 7:8]
                expert_partner_id_lst[idx, t] = env.partner_ids[
                    world_idx, agent_idx
                ].clone()
                expert_ego_id_lst[idx, t] = agent_info[world_idx, agent_idx, -1]
                expert_scene_id_lst[idx, t] = world_idx
                av = action_values[world_idx, agent_idx]
                log_actions_lst[idx, t, 0] = av[0]
                log_actions_lst[idx, t, 1] = av[1]
                log_actions_lst[idx, t, 2] = av[1]
            expert_dead_mask_lst[idx][t] = dead_agent_mask[world_idx, agent_idx]

        env.step_dynamics(all_actions)
        dones = env.get_dones().to(device)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs()
        road_mask = env.get_road_mask()
        partner_mask = env.get_partner_mask()
        agent_info = (
            env.sim.absolute_self_observation_tensor().to_torch().to(device)
        )
        infos = env.get_infos()

        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        goal_achieved = torch.clamp(goal_achieved, max=1.0)
        off_road = torch.clamp(off_road, max=1.0)
        veh_collision = torch.clamp(veh_collision, max=1.0)

        mask = (dones[alive_agent_mask] == 1.0) & (done_step == 0)
        done_step[mask] = t

        if dones.all():
            num_finished = cont_agent_mask.sum().float()
            goal_rate = goal_achieved.sum().float() / num_finished
            off_road_rate = (
                torch.where(off_road > 0, 1, 0).sum().float() / num_finished
            )
            veh_coll_rate = (
                torch.where(veh_collision > 0, 1, 0).sum().float() / num_finished
            )
            gmask = goal_achieved > 0
            smask = gmask & (off_road <= 0) & (veh_collision <= 0)
            print(
                f'Offroad {off_road_rate} VehCol {veh_coll_rate} '
                f'Goal {goal_rate} Save {smask.sum()}/{gmask.sum()}'
            )
            break

    goal_mask = goal_achieved > 0
    save_mask = goal_mask & (off_road <= 0) & (veh_collision <= 0)

    # Save trajectory (strict filter: goal + no offroad + no collision)
    if save_trajectory and trajectory_path is not None:
        save_index = batch_idx * batch_size
        traj = expert_trajectory_lst[save_mask].to('cpu')
        acts = rl_actions_lst[save_mask].to('cpu')
        dead = expert_dead_mask_lst[save_mask].to('cpu')
        pm = expert_partner_mask_lst[save_mask].to('cpu')
        rm = expert_road_mask_lst[save_mask].to('cpu')
        gp = expert_global_pos_lst[save_mask].to('cpu')
        gr = expert_global_rot_lst[save_mask].to('cpu')
        pid = expert_partner_id_lst[save_mask].to('cpu')
        eid = expert_ego_id_lst[save_mask].to('cpu')
        sid = expert_scene_id_lst[save_mask].to('cpu')
        os.makedirs(trajectory_path, exist_ok=True)
        os.makedirs(trajectory_path + '/global', exist_ok=True)
        os.makedirs(trajectory_path + '/id', exist_ok=True)
        np.savez_compressed(
            f"{trajectory_path}/trajectory_{save_index}.npz",
            obs=traj,
            actions=acts,
            dead_mask=dead,
            partner_mask=pm,
            road_mask=rm,
        )
        np.savez_compressed(
            f"{trajectory_path}/global/global_trajectory_{save_index}.npz",
            ego_global_pos=gp,
            ego_global_rot=gr,
        )
        np.savez_compressed(
            f"{trajectory_path}/id/id_trajectory_{save_index}.npz",
            ego_id=eid,
            scene_id=sid,
            partner_id=pid,
        )

    # Save labels (filter: goal achieved only)
    if save_label and label_path is not None:
        goal_mask = goal_achieved > 0
        st, en = batch_idx * batch_size, (batch_idx + 1) * batch_size
        scene_labels = get_label(
            log_actions_lst.cpu().numpy(), st, en, done_step, index_array
        )
        index_array_off = index_array + batch_idx * batch_size
        index_array_np = index_array_off.cpu().numpy()
        ego_ids_np = ego_ids.cpu().int().numpy()
        N, T, M = expert_partner_id_lst.shape
        scene_ego_keys = list(zip(index_array_np, ego_ids_np))
        partner_id_np = expert_partner_id_lst.cpu().numpy()
        id_to_label = {
            key: label for key, label in zip(scene_ego_keys, scene_labels)
        }
        scene_idx_exp = np.repeat(
            index_array_off[:, None, None].cpu().numpy(), T * M
        ).reshape(N, T, M)
        partner_flat = partner_id_np.reshape(-1)
        scene_flat = scene_idx_exp.reshape(-1)
        labels_flat = np.array(
            [id_to_label.get((s, pid), -1) for s, pid in zip(scene_flat, partner_flat)]
        )
        partner_labels = labels_flat.reshape(N, T, M)[goal_mask.cpu().numpy()]
        scene_labels_filtered = scene_labels[goal_mask.cpu().numpy()]
        os.makedirs(label_path, exist_ok=True)
        np.savez_compressed(
            f'{label_path}/label_trajectory_{batch_size * batch_idx}.npz',
            partner_label=partner_labels,
            ego_label=scene_labels_filtered,
        )
        print(f'label alive agent: {len(scene_labels_filtered)}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/data/after_cvpr/linear_probe_data/scene_100')
    parser.add_argument('--label_path', type=str, default=None,
                        help='Path for labels. Default: /data/after_cvpr/linear_probe_data/scene_100/{dataset}/label')
    parser.add_argument('--model-path', '-mp', type=str, default='/data/after_cvpr/rl/scene_100/')
    parser.add_argument('--model-name', '-mn', type=str, default='model_PPO____S_150__03_13_10_39_01_723_000761.pt')
    parser.add_argument('--dataset', type=str, default='validation', choices=['training', 'validation', 'testing'])
    parser.add_argument('--save-trajectory', action='store_true', default=True,
                        help='Save trajectory data (obs, actions, masks)')
    parser.add_argument('--no-save-trajectory', action='store_false', dest='save_trajectory')
    parser.add_argument('--save-label', action='store_true', default=True,
                        help='Save behavior labels')
    parser.add_argument('--no-save-label', action='store_false', dest='save_label')
    parser.add_argument('--dataset-size', type=int, default=2500)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--start-idx', type=int, default=None, help="start scene number of dataset")
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    trajectory_path = os.path.join(args.save_path, f'{args.dataset}_rl_data')
    label_path = args.label_path or f"/data/after_cvpr/linear_probe_data/scene_100/{args.dataset}_rl_data/label"

    env_config = EnvConfig(
        ego_state=True,
        road_map_obs=True,
        partner_obs=True,
        norm_obs=True,
        bev_obs=False,
        reward_type="weighted_combination",
        dynamics_model="classic",
        collision_behavior="ignore",
        dist_to_goal_threshold=2.0,
        polyline_reduction_threshold=0.1,
        remove_non_vehicles=True,
        lidar_obs=False,
        disable_classic_obs=False,
        obs_radius=50.0,
        steer_actions=torch.round(
            torch.linspace(-torch.pi, torch.pi, 13), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-4.0, 4.0, 7), decimals=3
        ),
        num_stack=args.num_stack,
    )
    print()
    print("num_stack : ", args.num_stack)
    print("trajectory_path : ", trajectory_path)
    print("label_path : ", label_path)
    print("dataset : ", args.dataset)
    print("save_trajectory : ", args.save_trajectory)
    print("save_label : ", args.save_label)
    print('Scene Loader')
    train_loader = SceneDataLoader(
        root=f"/data/full_version/data/{args.dataset}/",
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        sample_with_replacement=False,
        shuffle=False,
        start_idx=args.start_idx,
    )
    print('Call Env')
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=128,
        device="cuda",
        action_type="discrete",
    )
    print('Load policy')
    config = load_config("baselines/ppo/config/ppo_base_puffer.yaml")
    params = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False)
    policy = NeuralNet(
        input_dim=64,
        action_dim=91,
        hidden_dim=128,
        dropout=0.01,
        config=config.environment,
    ).to("cuda")
    policy.load_state_dict(params["parameters"])
    policy.eval()

    total_iter = int(args.dataset_size // args.batch_size)
    init_iter = 0 if args.start_idx is None else args.start_idx // args.batch_size
    for i in tqdm(range(init_iter, total_iter), total=total_iter, initial=init_iter):
        run_and_save(
            env,
            policy,
            batch_idx=i,
            batch_size=args.batch_size,
            trajectory_path=trajectory_path if args.save_trajectory else None,
            label_path=label_path if args.save_label else None,
            save_trajectory=args.save_trajectory,
            save_label=args.save_label,
        )
        if i != total_iter - 1:
            env.swap_data_batch()
    env.close()
    del env
    del env_config