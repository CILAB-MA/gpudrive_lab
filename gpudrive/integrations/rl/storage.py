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


def save_label(env, save_path, idx, num_worlds, batch_size):
    """
    Generate and save behavior labels from expert trajectories (goal-achieved agents only).
    Uses expert (continuous) actions, no policy needed.
    """
    device = env.device
    obs = env.reset()
    goal_achieved = torch.zeros(env.cont_agent_mask.sum().item(), device=device)
    off_road = torch.zeros_like(goal_achieved)
    veh_collision = torch.zeros_like(goal_achieved)
    cont_agent_mask = env.cont_agent_mask.to(device)
    alive_agent_mask = env.cont_agent_mask.clone()
    index_array = torch.tensor(
        [i for i, count in enumerate(alive_agent_mask.sum(-1).tolist()) for _ in range(count)],
        device=device
    )
    expert_actions, _, _, _, _ = env.get_expert_actions()
    log_actions = expert_actions[alive_agent_mask]
    done_step = torch.zeros(len(log_actions), device=device)
    ego_ids = env.ego_ids.clone()[alive_agent_mask]
    alive_agent_num = env.cont_agent_mask.sum().item()
    expert_partner_id_lst = torch.full(
        (alive_agent_num, env.episode_len, 127), -1, device=device, dtype=torch.long
    )

    for t in tqdm(range(env.episode_len)):
        expert_partner_id_lst[:, t] = env.partner_ids.clone()[alive_agent_mask].int()
        expert_actions, _, _, _, _ = env.get_expert_actions()
        env.step_dynamics(expert_actions[:, :, t, :])
        obs = env.get_obs()
        done = env.get_dones()
        infos = env.get_infos()
        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        off_road = torch.clamp(off_road, max=1.0)
        veh_collision = torch.clamp(veh_collision, max=1.0)
        mask = (done[alive_agent_mask] == 1.0) & (done_step == 0)
        done_step[mask] = t
        if done.all():
            break

    goal_mask = goal_achieved > 0
    scene_labels = get_label(
        log_actions.cpu().numpy(), idx * num_worlds, (idx + 1) * num_worlds, done_step, index_array
    )
    index_array = index_array + idx * num_worlds
    index_array_np = index_array.cpu().numpy()
    ego_ids_np = ego_ids.cpu().int().numpy()
    N, T, M = expert_partner_id_lst.shape
    scene_ego_keys = list(zip(index_array_np, ego_ids_np))
    expert_partner_id_lst_np = expert_partner_id_lst.cpu().numpy()
    id_to_label = {key: label for key, label in zip(scene_ego_keys, scene_labels)}
    scene_idx_expanded = np.repeat(index_array[:, None, None].cpu().numpy(), T * M).reshape(N, T, M)
    partner_ids_flat = expert_partner_id_lst_np.reshape(-1)
    scene_idx_flat = scene_idx_expanded.reshape(-1)
    labels_flat = np.array([id_to_label.get((s, pid), -1) for s, pid in zip(scene_idx_flat, partner_ids_flat)])
    partner_labels = labels_flat.reshape(N, T, M)[goal_mask.cpu().numpy()]
    scene_labels_filtered = scene_labels[goal_mask.cpu().numpy()]

    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(
        f'{save_path}/label_trajectory_{batch_size * idx}.npz',
        partner_label=partner_labels,
        ego_label=scene_labels_filtered
    )
    print(f'alive agent: {len(scene_labels_filtered)}')


def save_trajectory(env, policy, save_path, save_index=0):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    Actions are RL policy actions (discrete indices 0-90), not expert actions.

    Args:
        env: GPUDriveTorchEnv (use vecenv.env when created via PufferGPUDrive).
        policy: NeuralNet policy that returns discrete actions.
    """
    obs = env.reset()
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    # partner_id = env.get_partner_id().unsqueeze(-1)
    device = env.device
    
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    alive_agent_indices = cont_agent_mask.nonzero(as_tuple=False)
    alive_agent_num = env.cont_agent_mask.sum().item()
    print("alive_agent_num : ", alive_agent_num)
    
    expert_trajectory_lst = torch.zeros((alive_agent_num, env.episode_len, obs.shape[-1]), device=device)
    rl_actions_lst = torch.zeros((alive_agent_num, env.episode_len), device=device, dtype=torch.long)
    expert_dead_mask_lst = torch.ones((alive_agent_num, env.episode_len), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.full((alive_agent_num, env.episode_len, 127), 2, device=device, dtype=torch.long)
    expert_road_mask_lst = torch.ones((alive_agent_num, env.episode_len, 200), device=device, dtype=torch.bool)
    expert_global_pos_lst = torch.zeros((alive_agent_num, env.episode_len, 2), device=device) # global pos (2)
    expert_global_rot_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    expert_partner_id_lst = torch.zeros((alive_agent_num, env.episode_len, 127), device=device) # global actions (1)
    expert_ego_id_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    expert_scene_id_lst = torch.zeros((alive_agent_num, env.episode_len, 1), device=device) # global actions (1)
    # Initialize dead agent mask
    agent_info = (
            env.sim.absolute_self_observation_tensor()
            .to_torch()
            .to(device)
        )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    road_mask = env.get_road_mask()
    goal_achieved = 0
    off_road = 0
    veh_collision = 0
    for time_step in tqdm(range(env.episode_len)):
        with torch.no_grad():
            alive_obs = obs[~dead_agent_mask]
            actions, *_ = policy(alive_obs)
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], device=device, dtype=torch.long)
        all_actions[~dead_agent_mask] = actions
        for idx, (world_idx, agent_idx) in enumerate(alive_agent_indices):
            if not dead_agent_mask[world_idx, agent_idx]:
                expert_trajectory_lst[idx][time_step] = obs[world_idx, agent_idx]
                rl_actions_lst[idx][time_step] = all_actions[world_idx, agent_idx]
                expert_partner_mask_lst[idx][time_step] = partner_mask[world_idx, agent_idx]
                expert_road_mask_lst[idx][time_step] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, time_step] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, time_step] = agent_info[world_idx, agent_idx, 7:8]
                expert_partner_id_lst[idx, time_step] = env.partner_ids[world_idx, agent_idx].clone()
                expert_ego_id_lst[idx, time_step] = agent_info[world_idx, agent_idx, -1]
                expert_scene_id_lst[idx, time_step] = world_idx
            expert_dead_mask_lst[idx][time_step] = dead_agent_mask[world_idx, agent_idx]

        env.step_dynamics(all_actions)
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs() 
        road_mask = env.get_road_mask()
        partner_mask = env.get_partner_mask()
        # partner_id = env.get_partner_id().unsqueeze(-1)
        agent_info = (
        env.sim.absolute_self_observation_tensor()
        .to_torch()
        .to(device)
        )
        infos = env.get_infos()
        
        goal_achieved += infos.goal_achieved[cont_agent_mask]
        off_road += infos.off_road[cont_agent_mask]
        veh_collision += infos.collided[cont_agent_mask]
        goal_achieved = torch.clamp(goal_achieved, max=1.0)

        if (dead_agent_mask == True).all():
            num_finished_agents = cont_agent_mask.sum().float()
            goal_rate = goal_achieved.sum().float() / num_finished_agents
            off_road_rate = (
                torch.where(off_road > 0, 1, 0).sum().float()
                / num_finished_agents
            )
            veh_coll_rate = (
                torch.where(veh_collision > 0, 1, 0).sum().float()
                / num_finished_agents
            )
            collision = veh_collision > 0
            goal_mask = goal_achieved > 0
            # Goal achieved AND no offroad AND no collision
            save_mask = goal_mask & (off_road <= 0) & (veh_collision <= 0)
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate} Save {save_mask.sum()}/{goal_mask.sum()}')
            break
    
    expert_trajectory_lst = expert_trajectory_lst[save_mask].to('cpu')
    rl_actions_lst = rl_actions_lst[save_mask].to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst[save_mask].to('cpu')
    expert_partner_mask_lst = expert_partner_mask_lst[save_mask].to('cpu')
    expert_road_mask_lst = expert_road_mask_lst[save_mask].to('cpu')
    # global pos
    expert_global_pos_lst = expert_global_pos_lst[save_mask].to('cpu')
    expert_global_rot_lst = expert_global_rot_lst[save_mask].to('cpu')
    expert_partner_id_lst = expert_partner_id_lst[save_mask].to('cpu')
    expert_ego_id_lst = expert_ego_id_lst[save_mask].to('cpu')
    expert_scene_id_lst = expert_scene_id_lst[save_mask].to('cpu')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/global', exist_ok=True)
    os.makedirs(save_path + '/id', exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz",
                        obs=expert_trajectory_lst,
                        actions=rl_actions_lst,
                        dead_mask=expert_dead_mask_lst,
                        partner_mask=expert_partner_mask_lst,
                        road_mask=expert_road_mask_lst)
    np.savez_compressed(f"{save_path}/global/global_trajectory_{save_index}.npz",
                        ego_global_pos=expert_global_pos_lst,
                        ego_global_rot=expert_global_rot_lst)
    np.savez_compressed(f"{save_path}/id/id_trajectory_{save_index}.npz",
                        ego_id=expert_ego_id_lst,
                        scene_id=expert_scene_id_lst,
                        partner_id=expert_partner_id_lst)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_stack', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='/data/after_cvpr/linear_probe_data/')
    parser.add_argument('--model-path', '-mp', type=str, default='/data/after_cvpr/rl/scene_10000/PPO____S_200__03_04_04_06_56_997')
    parser.add_argument('--model-name', '-mn', type=str, default='model_PPO____S_200__03_04_04_06_56_997_007604.pt') # early_attn_s11_0808_043910
    parser.add_argument('--dataset', type=str, default='training', choices=['training', 'validation', 'testing'],)
    parser.add_argument('--function', type=str, default='save_trajectory',
                        choices=['save_trajectory', 'save_label'])
    parser.add_argument('--dataset-size', type=int, default=10000) # total_world
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    parser.add_argument('--start-idx', type=int, default=None, help="start scene number of dataset")
    args = parser.parse_args()

    torch.set_printoptions(precision=3, sci_mode=False)
    if args.function == 'save_label':
        save_path = f"/data/full_version/processed/{args.dataset}_only_goal/label"
        env_config = EnvConfig(collision_behavior="remove")
    else:
        save_path = os.path.join(args.save_path, f'{args.dataset}_rl_data')
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
            num_stack=1,
        )
    print()
    print("num_stack : ", args.num_stack)
    print("save_path : ", save_path)
    print("dataset : ", args.dataset)
    print("function : ", args.function)
    print('Scene Loader')
    # Create data loader
    train_loader = SceneDataLoader(
        root=f"/data/full_version/data/{args.dataset}/",
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        sample_with_replacement=False,
        shuffle=False,
        start_idx=args.start_idx
    )
    print('Call Env')
    # Make env
    action_type = "continuous" if args.function == "save_label" else "discrete"
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=128,  # Number of agents to control
        device="cuda",
        action_type=action_type,
    )
    policy = None
    if args.function == "save_trajectory":
        print('Launch Env')
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
        if args.function == 'save_trajectory':
            save_trajectory(env, policy, save_path, i * args.batch_size)
        elif args.function == 'save_label':
            save_label(env, save_path, i, args.batch_size, args.batch_size)
        if i != total_iter - 1:
            env.swap_data_batch()
    env.close()
    del env
    del env_config