"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())

import logging
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import mediapy as media

# GPUDrive
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(args, env, bc_policy, expert_dict, scene_batch_idx):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[[] for _ in range(4)] for _ in range(args.batch_size)]
    obs_stack1_feat_size = int(obs.shape[-1] / 5)
    poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3:obs_stack1_feat_size * 4 + 5]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(alive_agent_mask, dtype=torch.float32)
    infos = env.get_infos()
    goal_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.float32).to("cuda")
    off_road_timesteps = torch.full((alive_agent_mask.sum(), ), fill_value=-1, dtype=torch.int32).to("cuda")
    
    # Extract expert done step
    sorted_keys = sorted(expert_dict.keys())[args.batch_size * scene_batch_idx:args.batch_size * (scene_batch_idx + 1)]
    expert_timesteps = np.array([expert_dict[k]['done_step'] for k in sorted_keys])
    expert_timesteps = torch.from_numpy(expert_timesteps).to(dtype=goal_timesteps.dtype).to("cuda")
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    goal_achieved_ep = infos.goal_achieved[alive_agent_mask]

    num_head = bc_policy.net_config.num_head
    partner_num = bc_policy.config.max_num_agents_in_scene
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
        goal_mask = (goal_achieved > 0) & (goal_timesteps == -1)
        off_road_mask = (off_road > 0) & (off_road_timesteps == -1)
        goal_timesteps[goal_mask] = time_step / expert_timesteps[goal_mask]
        off_road_timesteps[off_road_mask] = time_step

        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            context, importance_weight, _ = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=True)
            actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions
        
        # Set importance weight to visualization
        world_importance_weight = torch.zeros((args.batch_size, num_head, partner_num)).to("cuda")
        multi_head_mask = dead_agent_mask.unsqueeze(1).repeat(1, 4, 1)
        world_mask = (~dead_agent_mask).sum(dim=-1) == 1
        world_importance_weight[world_mask] = world_importance_weight[world_mask].masked_scatter(multi_head_mask[world_mask], importance_weight)
        setattr(env.vis, "importance_weight", world_importance_weight.detach().cpu())
        
        sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(args.batch_size)),
                time_steps=[time_step]*args.batch_size,
                plot_importance_weight=True,
                plot_linear_probing=False,
                plot_linear_probing_label=False
            )

        for i in range(args.batch_size):
            for j in range(4):
                frames[i][j].append(
                    img_from_fig(sim_states[i][j])
                )

        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        off_road_ep += infos.off_road[alive_agent_mask]
        veh_collision_ep += infos.collided[alive_agent_mask]
        goal_achieved_ep += infos.goal_achieved[alive_agent_mask]
        off_road_ep = torch.clamp(off_road_ep, max=1)
        veh_collision_ep = torch.clamp(veh_collision_ep, max=1)
        goal_achieved_ep = torch.clamp(goal_achieved_ep, max=1)
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        if (dead_agent_mask == True).all():
            break
        
    # Calculate Goal Reached Time
    valid_goal_times = goal_timesteps[goal_timesteps >= 0].float()
    goal_time_avg = valid_goal_times.mean().item() if len(valid_goal_times) > 0 else -1
    print(f'Goal Reached Time : {goal_time_avg}')
    
    # Calculate Achieved Ratio to Goal
    goal_progress_ratio = dist_metrics[alive_agent_mask] / init_goal_dist
    goal_progress_ratio[goal_achieved.bool()] = 0
    goal_progress_ratio = (1 - goal_progress_ratio).mean()
    print('Agents Achieved Ratio to Goal', goal_progress_ratio)
    
    # Calculate Collision Rate
    off_road_rate = off_road_ep.sum().float() / alive_agent_mask.sum().float()
    veh_coll_rate = veh_collision_ep.sum().float() / alive_agent_mask.sum().float()
    goal_rate = goal_achieved_ep.sum().float() / alive_agent_mask.sum().float()
    print(f'Offroad {off_road_rate} VehCol {veh_coll_rate} Goal {goal_rate}')
    print(f'Success World idx : ', torch.where(goal_achieved_ep == 1)[0].tolist())

    # Make video
    root = os.path.join(args.video_path, args.dataset, args.model_name)
    for world_render_idx in range(args.batch_size):
        for head_idx in range(4):
            video_path = os.path.join(root, f"head_{head_idx}")
            os.makedirs(video_path, exist_ok=True)
            if world_render_idx in torch.where(veh_collision >= 1)[0].tolist():
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(veh_col).mp4', np.array(frames[world_render_idx][head_idx]), fps=10, codec='libx264')
            elif world_render_idx in torch.where(off_road >= 1)[0].tolist():
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(off_road).mp4', np.array(frames[world_render_idx][head_idx]), fps=10, codec='libx264')
            elif world_render_idx in torch.where(goal_achieved >= 1)[0].tolist():
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(goal).mp4', np.array(frames[world_render_idx][head_idx]), fps=10, codec='libx264')
            else:
                media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}(non_goal).mp4', np.array(frames[world_render_idx][head_idx]), fps=10, codec='libx264')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--dataset', '-d', type=str, default='training', choices=['training', 'validation'])
    parser.add_argument('--dataset-size', type=int, default=10) # total_world
    parser.add_argument('--batch-size', type=int, default=10) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/cov1792_clip10')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s3_0630_072820_60000.pth')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/full_version/videos/importance_weight')
    parser.add_argument('--linear-probing', '-lp', type=str, default='other_linear_prob')
    args = parser.parse_args()

    # Make scene loader
    scene_loader = SceneDataLoader(
        root=f"/data/full_version/data/{args.dataset}/",
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
        sample_with_replacement=False,
        shuffle=False,
    )
    dataset_size = args.dataset_size
    print(f'{args.dataset} len scene loader {len(scene_loader)}')
    
    # Make env
    env = GPUDriveTorchEnv(
        config=EnvConfig(
            dynamics_model="delta_local",
            dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
            dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
            dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
            collision_behavior='ignore',
            num_stack=5
        ),
        data_loader=scene_loader,
        max_cont_agents=1,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )
    
    # Load policy
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    lp_model = torch.load(f"{args.model_path}/other_linear_prob/{args.model_name[:-4]}/seed11/pos_early_lp_10.pth", weights_only=False).to("cuda")
    # Simulate the environment with the policy
    df = pd.read_csv(f'/data/full_version/expert_{args.dataset}_data_v2.csv')
    expert_dict = df.set_index('scene_idx').to_dict(orient='index')
    for i, batch in enumerate(scene_loader):
        env.swap_data_batch(batch)
        run(args, env, bc_policy, expert_dict, scene_batch_idx=i)
    env.close()

