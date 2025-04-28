"""Obtain a policy using behavioral cloning."""

import logging, imageio
import torch
import os, sys, json
import numpy as np
sys.path.append(os.getcwd())
import argparse

# GPUDrive
from gpudrive.env.config import EnvConfig, RenderConfig, SceneConfig, SelectionDiscipline
from gpudrive.env.config import DynamicsModel, ActionSpace

from pygpudrive.registration import make


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    # ENV
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    parser.add_argument('--start-idx', '-st', type=int, default=0)
    parser.add_argument('--num-world', '-w', type=int, default=10)
    # EXPERIMENT
    parser.add_argument('--dataset', type=str, default='train', choices=['train', 'valid'],)
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model/old_v2/new_aux_horizon')
    parser.add_argument('--model-name', '-mn', type=str, default='aux_attn_gmm_guide_weight_20250217_1245.pth')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/videos')
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=1.0)
    parser.add_argument('--shortest-path-test', '-spt', action='store_true')
    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(args, env):
    # To make video with expert trajectories footprint
    if env.render_config.draw_expert_footprint:
        obs = env.reset()
        alive_agent_mask = env.cont_agent_mask.clone()
        expert_actions, _, _ = env.get_expert_actions()
        infos = env.get_infos()
        exp_goal_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to(args.device)

        for time_step in range(env.episode_len):
            env.step_dynamics(expert_actions[:, :, time_step, :])
            infos = env.get_infos()
            print(time_step, alive_agent_mask.sum())
            controlled_agent_info = infos[alive_agent_mask]
            # Record when agent status changed(goal or collided)
            goal_achieved = controlled_agent_info[:, 3]
            goal_mask = (goal_achieved > 0) & (exp_goal_timesteps == -1)

            exp_goal_timesteps[goal_mask] = time_step
            obs = env.get_obs()
            dones = env.get_dones()
            for world_render_idx in range(NUM_WORLDS):
                env.save_footprint(world_render_idx=world_render_idx, time_step=time_step)
            if (dones == True).all():
                break

    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(NUM_WORLDS)]
    expert_actions, _, _ = env.get_expert_actions()
    obs_stack1_feat_size = int(obs.shape[-1] / 5)
    poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(init_goal_dist)
    infos = env.get_infos()
    collision_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to(args.device)
    goal_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.float32).to(args.device)
    off_road_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to(args.device)

    for time_step in range(env.episode_len):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to(args.device)
        
        # MASK
        ego_masks = env.get_stacked_controlled_agents_mask().to(args.device)
        partner_masks = env.get_stacked_partner_mask().to(args.device)
        road_masks = env.get_stacked_road_mask().to(args.device)
        ego_masks = ego_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN)
        partner_masks = partner_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, -1)
        partner_mask_bool = partner_masks == 2
        poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]
        dist = torch.linalg.norm(poss, dim=-1)
        controlled_agent_info = infos[alive_agent_mask]
        alive_agents = infos[alive_agent_mask][:, :4].sum(-1) == 0
        dist_metrics[alive_agents] = dist[alive_agents]

        # Record when agent status changed(goal or collided)
        veh_collision = controlled_agent_info[:, 1] 
        goal_achieved = controlled_agent_info[:, 3]
        off_road = controlled_agent_info[:, 0]
        collision_mask = (veh_collision > 0) & (collision_timesteps == -1)
        goal_mask = (goal_achieved > 0) & (goal_timesteps == -1)
        off_road_mask = (off_road > 0) & (off_road_timesteps == -1)

        collision_timesteps[collision_mask] = time_step
        goal_timesteps[goal_mask] = time_step / exp_goal_timesteps[goal_mask]
        off_road_timesteps[off_road_mask] = time_step

        road_masks = road_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, -1)
        # Mask all roads for shortest path test
        if args.shortest_path_test:
            road_masks = torch.full((NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, 200), True, dtype=torch.bool).to(args.device)
        all_masks = [ego_masks[~dead_agent_mask], partner_mask_bool[~dead_agent_mask], road_masks[~dead_agent_mask]]
        # Mask for alive partner with ratio
        alive_partner_mask = partner_masks[~dead_agent_mask]
        alive_partner_mask_now = alive_partner_mask[:, -1] != 2
        num_alive_per_world = alive_partner_mask_now.sum(dim=-1)
        num_to_remove_per_world = (num_alive_per_world * (1 - args.partner_portion_test)).long()
        for world_idx in range(len(num_to_remove_per_world)):
            num_to_remove = num_to_remove_per_world[world_idx].item()
            if num_to_remove > 0:
                partner_indices = alive_partner_mask_now[world_idx].nonzero(as_tuple=False).squeeze(-1)  # (num_alive,)
                selected_indices = partner_indices[torch.randperm(len(partner_indices))[:num_to_remove]]
                alive_partner_mask[world_idx, -1, selected_indices] = 2
        alive_partner_mask_bool = alive_partner_mask == 2
        all_masks[1] = alive_partner_mask_bool
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            num_alive = len(alive_obs)
            alive_obs = alive_obs.reshape(num_alive, 5, -1)
            alive_partner_obs = alive_obs[:, :, 6:1276].reshape(num_alive, 5, 127, 10)
            alive_partner_obs[alive_partner_mask_bool] = 0
            if args.shortest_path_test: # padding road
                alive_obs[:, :, 1276:] = 0
            alive_obs[:, :, 6:1276] = alive_partner_obs.reshape(num_alive, 5, -1)
            context, ego_attn_score, max_indices_rg = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=True)
            actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions

        if args.make_video:
            if render_config.draw_ego_attention:
                # Save ego attention score
                partner_idx = env.partner_id[~dead_agent_mask].clone()
                
                def fill_tensor(partner_idx, ego_attn_score, partner_mask):
                    multi_head_num = ego_attn_score.shape[1]
                    ego_attn_score = ego_attn_score.transpose(1, 2)
                    n, _ = partner_idx.shape
                    filled_tensor = torch.zeros((n, 128, multi_head_num)).to(args.device)

                    row_indices = torch.arange(n).unsqueeze(1).expand_as(partner_idx).to(args.device)
                    valid_rows = row_indices[~partner_mask]
                    valid_cols = partner_idx[~partner_mask].int()
                    valid_values = ego_attn_score[~partner_mask]

                    filled_tensor[valid_rows, valid_cols] = valid_values
                    
                    return filled_tensor
                
                viz_ego_attn = fill_tensor(partner_idx, ego_attn_score, partner_mask_bool[:,:,-1,:][~dead_agent_mask])
                world_viz_ego_attn = torch.zeros(NUM_WORLDS, NUM_PARTNER, ego_attn_score.shape[1]).to(args.device)
                world_viz_ego_attn[(~dead_agent_mask).sum(dim=-1) == 1] = viz_ego_attn
                world_viz_ego_attn = world_viz_ego_attn.transpose(1, 2)

                env.save_ego_attn_score(world_viz_ego_attn)

            for world_render_idx in range(NUM_WORLDS):
                frame = env.render(world_render_idx=world_render_idx)
                frames[world_render_idx].append(frame)

        env.step_dynamics(all_actions)
        loss = torch.abs(all_actions[~dead_agent_mask] - expert_actions[~dead_agent_mask][:, time_step, :])
        # print(f'TIME {time_step} LOSS: {loss.mean(0)}')

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()

        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        if (dead_agent_mask == True).all():
            break
    # Calculate average timesteps for status
    valid_collision_times = collision_timesteps[collision_timesteps >= 0].float()
    valid_goal_times = goal_timesteps[goal_timesteps >= 0].float()
    valid_off_road_times = off_road_timesteps[off_road_timesteps >= 0].float()
    collision_time_avg = valid_collision_times.mean().item() if len(valid_collision_times) > 0 else -1
    goal_time_avg = valid_goal_times.mean().item() if len(valid_goal_times) > 0 else -1
    off_road_time_avg = valid_off_road_times.mean().item() if len(valid_off_road_times) > 0 else -1

    controlled_agent_info = infos[alive_agent_mask]
    off_road = controlled_agent_info[:, 0]
    veh_collision = controlled_agent_info[:, 1]
    goal_achieved = controlled_agent_info[:, 3]
    collision = (veh_collision + off_road > 0)
    goal_progress_ratio = dist_metrics / init_goal_dist
    goal_progress_ratio[goal_achieved.bool()] = 0
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
        if args.shortest_path_test:
            csv_path = f"{args.model_path}/result_shortest.csv"
        file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if file_is_empty:
                f.write("Model,Dataset,OffRoad,VehicleCollsion,Goal,Collision,GoalProgress,VehColTime,GoalTime,OffRoadTime\n")
            f.write(f"{args.model_name},{args.dataset},{off_road_rate},{veh_coll_rate},{goal_rate},{collision_rate},{goal_progress_ratio},{collision_time_avg},{goal_time_avg},{off_road_time_avg}\n")

    if args.make_video:
        video_path = os.path.join(args.video_path, args.dataset, args.model_name, str(args.partner_portion_test))
        if args.shortest_path_test:
            video_path = os.path.join(args.video_path, args.dataset, args.model_name, 'road_masked')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        for world_render_idx in range(NUM_WORLDS):
            if world_render_idx in torch.where(veh_collision >= 1)[0].tolist():
                imageio.mimwrite(f'{video_path}/world_{world_render_idx + args.start_idx}(veh_col).mp4', np.array(frames[world_render_idx]), fps=10)
            elif world_render_idx in torch.where(off_road >= 1)[0].tolist():
                imageio.mimwrite(f'{video_path}/world_{world_render_idx + args.start_idx}(off_road).mp4', np.array(frames[world_render_idx]), fps=10)
            elif world_render_idx in torch.where(goal_achieved >= 1)[0].tolist():
                imageio.mimwrite(f'{video_path}/world_{world_render_idx + args.start_idx}(goal).mp4', np.array(frames[world_render_idx]), fps=10)
            else:
                imageio.mimwrite(f'{video_path}/world_{world_render_idx + args.start_idx}(non_goal).mp4', np.array(frames[world_render_idx]), fps=10)
    return off_road_rate, veh_coll_rate, goal_rate, collision_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    
    parser.add_argument('--dataset-size', type=int, default=80000) # total_world
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/new_aux_horizon')
    parser.add_argument('--model-name', '-mn', type=str, default='aux_attn_gmm_guide_weight_20250217_1245.pth')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/full_version/videos')
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=1.0)
    parser.add_argument('--sim-agent', '-sa', type=str, default='log_replay', choices=['log_replay', 'self_play'])
    args = parser.parse_args()
    # Configurations
    num_cont_agents = 128 if args.sim_agent == 'self_play' else 1

    print('Scene Loader')
    # Create data loader
    train_loader = SceneDataLoader(
        root=f"/data/full_version/data/training/",
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        sample_with_replacement=False,
        shuffle=False,
    )
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='remove'
        num_stack=5

    )
    render_config = RenderConfig(
        draw_obj_idx=True,
        draw_expert_footprint=True,
        draw_only_ego_footprint=True,
        draw_ego_attention=True,
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=num_cont_agents,  # Number of agents to control
        device="cuda",
        render_config=render_config,
        action_type="continuous",
    )
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to(args.device)
    bc_policy.eval()
    num_iter = int(args.dataset_size // args.batch_size)

    # Train Scene
    env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=False) #todo: remove_controlled_agents -> ?
    for i in tqdm(range(num_iter)):
        print(env.data_batch)
        run(args, env)
        if i != num_iter - 1:
            env.swap_data_batch()
    env.close()

    # Test Scene
    test_loader = SceneDataLoader(
        root=f"/data/full_version/data/testing/",
        batch_size=args.batch_size,
        dataset_size=10_000,
        sample_with_replacement=False,
        shuffle=False,
    )
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=test_loader,
        max_cont_agents=num_cont_agents,  # Number of agents to control
        device="cuda",
        render_config=render_config,
        action_type="continuous",
    )
    # Test Scene
    env.remove_agents_by_id(args.partner_portion_test, remove_controlled_agents=False) #todo: remove_controlled_agents -> ?
    for i in tqdm(range(num_iter)):
        print(env.data_batch)
        run(args, env)
        if i != num_iter - 1:
            env.swap_data_batch()
    env.close()
    # integrate the result
    csv_path = f"{model_path}/result_{args.partner_portion_test}.csv"

