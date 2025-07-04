"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())

import logging, functools
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import mediapy as media
import torch.nn.functional as F
# GPUDrive
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS
from collections import OrderedDict
# linear_probing

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def digitize(t, bins):
    return torch.bucketize(t, bins, right=False)

def transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step):
    """transform time t relative pos to current relative pos"""
    # 1. transform t-relative pos to t-global pos
    # get partner's relative pos and rot at time t
    t_partner_pos = aux_info[..., 1:3] * MAX_REL_AGENT_POS
    
    # get ego's global pos and rot at time t
    t_ego_global_pos = np.zeros_like(ego_global_pos)
    t_ego_global_rot = np.zeros_like(ego_global_rot)
    t_ego_global_pos[:, :-future_step] = ego_global_pos[:, future_step:]
    t_ego_global_rot[:, :-future_step] = ego_global_rot[:, future_step:]
    
    t_partner_global_pos_x = t_ego_global_pos[..., 0, None] + t_partner_pos[..., 0] * np.cos(t_ego_global_rot) - t_partner_pos[..., 1] * np.sin(t_ego_global_rot)
    t_partner_global_pos_y = t_ego_global_pos[..., 1, None] + t_partner_pos[..., 0] * np.sin(t_ego_global_rot) + t_partner_pos[..., 1] * np.cos(t_ego_global_rot)
    
    # 2. transform t-global pos to current relative pos
    delta_x = t_partner_global_pos_x - ego_global_pos[..., 0, None]
    delta_y = t_partner_global_pos_y - ego_global_pos[..., 1, None]
    
    cos_theta = np.cos(-ego_global_rot)
    sin_theta = np.sin(-ego_global_rot)
    
    current_relative_pos_x = delta_x * cos_theta + delta_y * sin_theta
    current_relative_pos_y = -delta_x * sin_theta + delta_y * cos_theta
    current_relative_pos_x = 2 * ((current_relative_pos_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
    current_relative_pos_y = 2 * ((current_relative_pos_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
    current_relative_pos = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
    
    return current_relative_pos



def register_all_layers_forward_hook(model):
    hidden_vector_dict = OrderedDict()

    def hook_fn(module, input, output, name):
        try:
            hidden_vector_dict[name] = output.detach()
        except AttributeError:
            hidden_vector_dict[name] = output['last_hidden_state'].detach()

    def _register(module, prefix=""):
        for name, layer in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer.register_forward_hook(functools.partial(hook_fn, name=full_name))

            _register(layer, full_name)

    _register(model)

    return hidden_vector_dict

def run(args, env, bc_policy, lp_model, expert_dict, scene_batch_idx):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.batch_size)]
    obs_stack1_feat_size = int(obs.shape[-1] / 5)
    
    # Extract Linear Probing
    layers = register_all_layers_forward_hook(bc_policy.fusion_attn)
    global_ego_pos = torch.zeros((args.batch_size, 91, 127, 2))
    future_other_mask = torch.zeros((args.batch_size, 91, 127)).bool().cuda()
    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        # last_obs = obs[alive_agent_mask][:, -int(obs.shape[-1] / 5):]
        # partner_obs = last_obs[:, 6:128 * 6].reshape(args.batch_size, 127, -1)
        pos = env.get_global_state()
        future_other_pos[:, time_step] = pos
        future_other_mask[:, time_step] = partner_mask_bool[alive_agent_mask]
        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            context, *_ = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            actions = bc_policy.get_action(context, deterministic=True)
            actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions
        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        if (dead_agent_mask == True).all():
            break
    print('ONE LOOP FINISHED!')
    # Transform the other pos to label
    x = future_other_pos[..., 0]
    y = future_other_pos[..., 1]
    xbins = torch.linspace(-0.05, 0.05, steps=9)
    ybins = torch.linspace(-0.05, 0.05, steps=9)
    x_bins = digitize(x, xbins) - 1
    y_bins = digitize(y, ybins) - 1
    x_bins = torch.clamp(x_bins, 0, 7)
    y_bins = torch.clamp(y_bins, 0, 7)
    discrete_pos = x_bins * 8 + y_bins
    discrete_pos = discrete_pos.cuda()
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    partner_num = bc_policy.config.max_num_agents_in_scene
    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        world_mask = (~dead_agent_mask).sum(dim=-1) == 1
        partner_mask_bool = partner_mask == 2
        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            context, *_ = (lambda *args: (args[0], args[-2], args[-1]))(*bc_policy.get_context(alive_obs, all_masks))
            if time_step < 82: 
                nth_layer =list(layers.keys())[-1]
                lp_input = layers[nth_layer][:,1:128,:]
                pred_pos = lp_model(lp_input)
                future_mask = future_other_mask[:, time_step + 9]
                loss = F.cross_entropy(pred_pos.reshape(-1, 64), discrete_pos[world_mask, time_step + 9].reshape(-1), reduction='none')
                loss = -loss.reshape(-1, 127)
                masked_loss = loss.masked_fill(future_mask[world_mask], float('-inf'))  # zero인 곳은 -inf로
                softmax_loss = torch.softmax(masked_loss, dim=1).unsqueeze(1)
        actions = bc_policy.get_action(context, deterministic=True)
        actions = actions.squeeze(1)
        all_actions[~dead_agent_mask, :] = actions
        # Set importance weight to visualization
        world_importance_weight = torch.zeros((args.batch_size, 1, partner_num)).to("cuda")
        multi_head_mask = dead_agent_mask.unsqueeze(1)
        world_mask = (~dead_agent_mask).sum(dim=-1) == 1
        world_importance_weight[world_mask] = world_importance_weight[world_mask].masked_scatter(multi_head_mask[world_mask], softmax_loss)
        setattr(env.vis, "importance_weight", world_importance_weight.detach().cpu())
        
        sim_states = env.vis.plot_simulator_state(
                env_indices=list(range(args.batch_size)),
                time_steps=[time_step]*args.batch_size,
                plot_importance_weight=True,
                plot_linear_probing=False,
                plot_linear_probing_label=False
            )

        for i in range(args.batch_size):
                frames[i].append(
                    img_from_fig(sim_states[i][0])
                )

        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        if (dead_agent_mask == True).all():
            break

    # Make video
    root = os.path.join(args.video_path, args.dataset, args.model_name)
    for world_render_idx in range(args.batch_size):
        video_path = os.path.join(root, f"lp_loss")
        os.makedirs(video_path, exist_ok=True)
        media.write_video(f'{video_path}/world_{world_render_idx + scene_batch_idx * args.batch_size}.mp4', np.array(frames[world_render_idx][0]), fps=10, codec='libx264')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--dataset', '-d', type=str, default='training', choices=['training', 'validation'])
    parser.add_argument('--dataset-size', type=int, default=5) # total_world
    parser.add_argument('--batch-size', type=int, default=5) # num_world
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
        run(args, env, bc_policy, lp_model, expert_dict, scene_batch_idx=i)
    env.close()

