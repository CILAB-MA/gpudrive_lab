"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())

import logging, imageio
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import mediapy as media

# GPUDrive
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(args, env, bc_policy, expert_dict, dataset, scene_batch_idx):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.batch_size)]
    expert_actions, _, _, _, _  = env.get_expert_actions() 
    obs_stack1_feat_size = int(obs.shape[-1] / 5)
    poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(alive_agent_mask, dtype=torch.float32)
    infos = env.get_infos()
    collision_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to("cuda")
    goal_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.float32).to("cuda")
    off_road_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to("cuda")
    # Extract expert done step
    sorted_keys = sorted(expert_dict.keys())
    scene_labels = np.array([expert_dict[k]['label'] for k in sorted_keys])
    turn_mask = torch.from_numpy(scene_labels == 'TURN').to("cuda")
    normal_mask = torch.from_numpy(scene_labels == 'NORMAL').to("cuda")
    straight_mask = torch.from_numpy(scene_labels == 'STRAIGHT').to("cuda")
    reverse_mask = torch.from_numpy(scene_labels == 'RETREAT').to("cuda")
    abnormal_mask = torch.from_numpy(scene_labels == 'ABNORMAL').to("cuda")
    expert_timesteps = np.array([expert_dict[k]['done_step'] for k in sorted_keys])
    expert_timesteps = torch.from_numpy(expert_timesteps).to("cuda")
    alive_world = alive_agent_mask.sum(-1)
    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]
        dist = torch.linalg.norm(poss, dim=-1)
        dist_metrics[alive_agent_mask] = dist

        # Record when agent status changed(goal or collided)
        off_road = infos.off_road[alive_agent_mask]
        veh_collision = infos.collided[alive_agent_mask]
        goal_achieved = infos.goal_achieved[alive_agent_mask]
        collision_mask = (veh_collision > 0) & (collision_timesteps == -1)
        goal_mask = (goal_achieved > 0) & (goal_timesteps == -1)
        off_road_mask = (off_road > 0) & (off_road_timesteps == -1)
        collision_timesteps[collision_mask] = time_step
        goal_timesteps[goal_mask] = time_step / expert_timesteps[goal_mask]
        off_road_timesteps[off_road_mask] = time_step

        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            context, *_, = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=True)
            actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions

        if args.make_video:
            sim_states = env.vis.plot_simulator_state(
                    env_indices=list(range(args.batch_size)),
                    time_steps=[time_step]*args.batch_size,
                    plot_importance_weight=False,
                    plot_linear_probing=False,
                    plot_linear_probing_label=False
                )

            for i in range(args.batch_size):
                frames[i].append(
                    img_from_fig(sim_states[i])
                )

        env.step_dynamics(all_actions)
        loss = torch.abs(all_actions[~dead_agent_mask] - expert_actions[~dead_agent_mask][:, time_step, :])
        # print(f'TIME {time_step} LOSS: {loss.mean(0)}')

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        off_road = infos.off_road[alive_agent_mask]
        veh_collision = infos.collided[alive_agent_mask]
        goal_achieved = infos.goal_achieved[alive_agent_mask]
        off_road_rate = off_road.sum().float() / alive_agent_mask.sum().float()
        veh_coll_rate = veh_collision.sum().float() / alive_agent_mask.sum().float()
        goal_rate = goal_achieved.sum().float() / alive_agent_mask.sum().float()
        print(f'STEP: {time_step} OFF ROAD: {off_road_rate} VEH COLL: {veh_coll_rate} GOAL: {goal_rate}')
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    # Calculate average timesteps for status
    valid_collision_times = collision_timesteps[collision_timesteps >= 0].float()
    valid_goal_times = goal_timesteps[goal_timesteps >= 0].float()
    collision_time_avg = valid_collision_times.mean().item() if len(valid_collision_times) > 0 else -1
    goal_time_avg = valid_goal_times.mean().item() if len(valid_goal_times) > 0 else -1

    off_road = infos.off_road[alive_agent_mask]
    veh_collision = infos.collided[alive_agent_mask]
    goal_achieved = infos.goal_achieved[alive_agent_mask]
    collision = (veh_collision + off_road > 0)
    goal_progress_ratio = dist_metrics[alive_agent_mask] / init_goal_dist
    goal_progress_ratio[goal_achieved.bool()] = 0
    # calculate the different label
    label_masks = [turn_mask, normal_mask, reverse_mask, abnormal_mask]
    offroads, veh_colls, goals, goal_progresses, goal_time_avgs, num_labels = [], [], [], [], [], []
    collisions = []
    for label_mask in label_masks:
        num_labels.append(label_mask.sum())
        offroads.append(off_road[label_mask].sum())
        veh_colls.append(veh_collision[label_mask].sum())
        collisions.append(veh_collision[label_mask].sum() + off_road[label_mask].sum())
        goals.append(goal_achieved[label_mask].sum())
        goal_progresses.append((1 - goal_progress_ratio)[label_mask].sum())
        label_timesteps = goal_timesteps[label_mask]
        label_timesteps = label_timesteps[label_timesteps >= 0].float()
        label_time_avg = label_timesteps.mean().item() if len(label_timesteps) > 0 else -1
        goal_time_avgs.append(label_time_avg)

    goal_progress_ratio = (1 - goal_progress_ratio).mean()
    print('Agents Achieved Ratio to Goal', goal_progress_ratio)
    off_road_rate = off_road.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision.sum().float() / alive_agent_mask.sum().float()
    goal_rate = goal_achieved.sum().float() / alive_agent_mask.sum().float()
    collision_rate = off_road_rate + veh_coll_rate

    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
    print(f'Success World idx : ', torch.where(goal_achieved == 1)[0].tolist())
    print(f'Goal Reached Time : {goal_time_avg}')
    if args.make_csv:
        csv_path = f"{args.model_path}/result_{args.partner_portion_test}.csv"
        file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if file_is_empty:
                column_name = "Model,Dataset"
                labels = ["Total", "Turn", "Normal", "Reverse", "Abnormal"]
                metrics = ["Num","OffRoad","VehCollision","Goal","Collision","GoalProgress","GoalTime"]
                for l, label in enumerate(labels):
                    for metric in metrics:
                        if l == 0 and metric == "Num":
                            continue
                        column_name += f",{label}{metric}"
                f.write(column_name + ",\n")
            data = f"{args.model_name},{dataset},{off_road_rate},{veh_coll_rate},{goal_rate},{collision_rate},{goal_progress_ratio},{goal_time_avg},"
            for l in range(4):
                data += f"{num_labels[l]},{offroads[l]},{veh_colls[l]},{goals[l]},{collisions[l]},{goal_progresses[l]},{goal_time_avgs[l]},"
            f.write(data + ",\n")

    if args.make_video:
        video_path = os.path.join(args.video_path, args.dataset, args.model_name, str(args.partner_portion_test))
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        for world_render_idx in range(args.batch_size):
            if world_render_idx in torch.where(veh_collision >= 1)[0].tolist():
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(veh_col).mp4', np.array(frames[world_render_idx]), fps=10, codec='libx264')
            elif world_render_idx in torch.where(off_road >= 1)[0].tolist():
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(off_road).mp4', np.array(frames[world_render_idx]), fps=10, codec='libx264')
            elif world_render_idx in torch.where(goal_achieved >= 1)[0].tolist():
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(goal).mp4', np.array(frames[world_render_idx]), fps=10, codec='libx264')
            else:
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(non_goal).mp4', np.array(frames[world_render_idx]), fps=10, codec='libx264')
    return off_road_rate, veh_coll_rate, goal_rate, collision_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    
    parser.add_argument('--dataset-size', type=int, default=1000) # total_world
    parser.add_argument('--batch-size', type=int, default=50) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/data_cut_add')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_seed_3_0522_115334.pth')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/full_version/videos')
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    parser.add_argument('--sim-agent', '-sa', type=str, default='log_replay', choices=['log_replay', 'self_play'])
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    args = parser.parse_args()
    # Configurations
    num_cont_agents = 128 if args.sim_agent == 'self_play' else 1

    # Create data loader
    if args.dataset == 'training':
        scene_loader = SceneDataLoader(
            root=f"/data/full_version/data/training/",
            batch_size=args.batch_size,
            dataset_size=args.dataset_size,
            sample_with_replacement=False,
            shuffle=False,
        )
        dataset_size = args.dataset_size
    else:
        # Test Scene
        scene_loader = SceneDataLoader(
            root=f"/data/full_version/data/validation/",
            batch_size=args.batch_size,
            dataset_size=100,
            sample_with_replacement=False,
            shuffle=False,
        )
        dataset_size = 100
    print(f'{args.dataset} len scene loader {len(scene_loader)}')
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='ignore',
        num_stack=5

    )
    render_config = RenderConfig(
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=scene_loader,
        max_cont_agents=num_cont_agents,  # Number of agents to control
        device="cuda",
        render_config=render_config,
        action_type="continuous",
    )
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    num_iter = int(dataset_size // args.batch_size)

    # Train Scene
    env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=False)
    df = pd.read_csv(f'/data/full_version/expert_{args.dataset}_data.csv')
    scene_dict = df.set_index('scene_idx').to_dict(orient='index')
    for i in tqdm(range(num_iter)):
        expert_dict = {k: scene_dict[k + i * args.batch_size] for k in range(args.batch_size) if k + i * args.batch_size in scene_dict}
        run(args, env, bc_policy, expert_dict, dataset=args.dataset, scene_batch_idx=i)
        if i != num_iter - 1:
            print('SWAP!!')
            env.swap_data_batch()
            env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=False)
    env.close()

