"""Obtain a policy using behavioral cloning."""

import logging, imageio
import torch
import os, sys
import numpy as np
sys.path.append(os.getcwd())
import argparse

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig, SelectionDiscipline
from pygpudrive.env.config import DynamicsModel, ActionSpace
from algorithms.il.model.bc import *
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
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model/new_aux_horizon')
    parser.add_argument('--model-name', '-mn', type=str, default='aux_attn_gmm_guide_weight_20250217_1245.pth')
    parser.add_argument('--make-csv', '-mc', action='store_true')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/videos')

    args = parser.parse_args()
    return args

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    args = parse_args()
    
    # Configurations
    NUM_WORLDS = args.num_world
    NUM_PARTNER = 128
    MAX_NUM_OBJECTS = 1
    ROLLOUT_LEN = 5

    # Initialize configurations
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_{args.dataset}/",
                               num_scenes=NUM_WORLDS,
                               start_idx=args.start_idx,
                               discipline=SelectionDiscipline.RANGE_N)
    
    env_config = EnvConfig(
        dynamics_model="delta_local",
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dy=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
    )
    render_config = RenderConfig(
        draw_obj_idx=True,
        draw_expert_footprint=True,
        draw_only_ego_footprint=True,
        draw_ego_attention=True,
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
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}").to(args.device)
    bc_policy.eval()

    # To make video with expert trajectories footprint
    if render_config.draw_expert_footprint:
        obs = env.reset()
        expert_actions, _, _ = env.get_expert_actions()
        for time_step in range(env.episode_len):
            env.step_dynamics(expert_actions[:, :, time_step, :])
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
    for time_step in range(env.episode_len):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to(args.device)
        
        # MASK
        ego_masks = env.get_stacked_controlled_agents_mask().to(args.device)
        partner_masks = env.get_stacked_partner_mask().to(args.device)
        road_masks = env.get_stacked_road_mask().to(args.device)
        ego_masks = ego_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN)
        partner_masks = partner_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, -1)
        partner_mask_bool = (partner_masks == 2)
        road_masks = road_masks.reshape(NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, -1)
        # road_masks = torch.full((NUM_WORLDS, NUM_PARTNER, ROLLOUT_LEN, 200), True, dtype=torch.bool).to(args.device)
        all_masks = [ego_masks[~dead_agent_mask], partner_mask_bool[~dead_agent_mask], road_masks[~dead_agent_mask]]
            
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            num_alive = len(alive_obs)
            alive_obs = alive_obs.reshape(num_alive, 5, -1)
            alive_obs[:, :, 6:1276] = 0
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
        
        print(f'TIME {time_step} LOSS: {loss.mean(0)}')

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()

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

    if args.make_csv:
        csv_path = f"{args.model_path}/result.csv"
        file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
        with open(csv_path, 'a', encoding='utf-8') as f:
            if file_is_empty:
                f.write("model_name,dataset,off_road_rate,veh_coll_rate,goal_rate,collision_rate\n")
            f.write(f"{args.model_name},{args.dataset},{off_road_rate},{veh_coll_rate},{goal_rate},{collision_rate}\n")

    if args.make_video:
        video_path = os.path.join(args.video_path, args.dataset, args.model_name)
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
    