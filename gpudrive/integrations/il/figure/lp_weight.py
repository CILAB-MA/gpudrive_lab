"""Obtain a policy using behavioral cloning."""
import os, sys
sys.path.append(os.getcwd())
from concurrent.futures import ThreadPoolExecutor
import logging, functools
import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import mediapy as media
from pathlib import Path
import torch.nn.functional as F
# GPUDrive
from gpudrive.env.config import EnvConfig
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS
from collections import OrderedDict, defaultdict
# linear_probing
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def digitize(t, bins):
    return torch.bucketize(t, bins, right=False)

def save_svg(path, fig):
    fig.savefig(path, format="svg")

def save_frames_parallel(frames_list, out_dir, stem="frame"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        futures = []
        for t, fig in enumerate(frames_list):     # fig: matplotlib Figure
            fpath = out_dir / f"{stem}_{t:06d}.svg"
            futures.append(ex.submit(save_svg, fpath, fig))
        for f in futures: f.result()  # join

def transform_relative_other_pos(partner_relative_pos, ego_global_pos, ego_global_rot, future_step):
    """transform time t relative pos to current relative pos"""
    # 1. transform t-relative pos to t-global pos
    # get partner's relative pos and rot at time t
    t_partner_pos = partner_relative_pos * MAX_REL_AGENT_POS

    # get ego's global pos and rot at time t
    t_ego_global_pos = torch.zeros_like(ego_global_pos)
    t_ego_global_rot = torch.zeros_like(ego_global_rot)
    t_ego_global_pos[:, :-future_step] = ego_global_pos[:, future_step:]
    t_ego_global_rot[:, :-future_step] = ego_global_rot[:, future_step:]

    t_partner_global_pos_x = t_ego_global_pos[..., 0].unsqueeze(-1) + t_partner_pos[..., 0] * torch.cos(t_ego_global_rot.unsqueeze(-1)) - t_partner_pos[..., 1] * torch.sin(t_ego_global_rot.unsqueeze(-1))
    t_partner_global_pos_y = t_ego_global_pos[..., 1].unsqueeze(-1) + t_partner_pos[..., 0] * torch.sin(t_ego_global_rot.unsqueeze(-1)) + t_partner_pos[..., 1] * torch.cos(t_ego_global_rot.unsqueeze(-1))

    # 2. transform t-global pos to current relative pos
    delta_x = t_partner_global_pos_x - ego_global_pos[..., 0].unsqueeze(-1)
    delta_y = t_partner_global_pos_y - ego_global_pos[..., 1].unsqueeze(-1)

    cos_theta = torch.cos(-ego_global_rot.unsqueeze(-1))
    sin_theta = torch.sin(-ego_global_rot.unsqueeze(-1))

    current_relative_pos_x = delta_x * cos_theta + delta_y * sin_theta
    current_relative_pos_y = -delta_x * sin_theta + delta_y * cos_theta
    current_relative_pos_x = 2 * ((current_relative_pos_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
    current_relative_pos_y = 2 * ((current_relative_pos_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
    current_relative_pos = torch.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)

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

def run(args, env, bc_policy, lp_model, scene_batch_idx, sweep_name, exp):
    obs = env.reset()
    ego_idx = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0])
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.batch_size)]
    NUM_WORLD = alive_agent_mask.shape[0]
    alive_ego_idx = ego_idx[:NUM_WORLD].to('cuda')
    # Extract Linear Probing
    layers = register_all_layers_forward_hook(bc_policy.fusion_attn)
    future_step = 10
    # =============== save data for linear probing ===============
    ego_global_pos = torch.zeros((args.batch_size, env.episode_len, 2)).cuda()
    ego_global_rot = torch.zeros((args.batch_size, env.episode_len)).cuda()
    other_relative_pos = torch.zeros((args.batch_size, env.episode_len, 127, 2)).cuda()
    other_relative_mask = torch.zeros((args.batch_size, env.episode_len, 127)).bool().cuda()
    # ============================================================
    
    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        lp_partner_mask_bool = torch.logical_or(partner_mask == 2, partner_mask == 1)
        ego_global_state = env.get_global_state()
        ego_global_pos[:, time_step][alive_agent_mask.sum(-1) == 1] = torch.stack((ego_global_state.pos_x, ego_global_state.pos_y), dim=-1)[alive_agent_mask]
        ego_global_rot[:, time_step][alive_agent_mask.sum(-1) == 1] = ego_global_state.rotation_angle[alive_agent_mask]
        partner_pos = env.get_partner_pos()
        other_relative_pos[:, time_step][alive_agent_mask.sum(-1) == 1] = partner_pos[alive_agent_mask]
        other_relative_mask[:, time_step][alive_agent_mask.sum(-1) == 1] = lp_partner_mask_bool[alive_agent_mask]
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
    current_relative_pos = transform_relative_other_pos(other_relative_pos, ego_global_pos, ego_global_rot, future_step=future_step)
    # Transform the other pos to label
    x = current_relative_pos[..., 0]
    y = current_relative_pos[..., 1]
    xbins = torch.linspace(-0.05, 0.05, steps=9).cuda()
    ybins = torch.linspace(-0.05, 0.05, steps=9).cuda()
    x_bins = digitize(x, xbins) - 1
    y_bins = digitize(y, ybins) - 1
    x_bins = torch.clamp(x_bins, 0, 7)
    y_bins = torch.clamp(y_bins, 0, 7)
    label_discrete_pos = x_bins * 8 + y_bins
    label_discrete_pos = label_discrete_pos.cuda()
    
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    partner_num = bc_policy.config.max_num_agents_in_scene
    expert_actions, _, _, _, _  = env.get_expert_actions() 
    for time_step in tqdm(range(env.episode_len)):
        # all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        all_actions = expert_actions[:, :, time_step].clone()
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
            if time_step < env.episode_len - future_step: 
                nth_layer = list(layers.keys())[-1]
                lp_input = layers[nth_layer][:,1:128,:] if exp == 'other' else layers[nth_layer][:,0,:] 
                wm = world_mask
                pred_logit_10 = lp_models[0](lp_input)

                futm = other_relative_mask[:, time_step + future_step]     
                labels = label_discrete_pos[wm, time_step + future_step]   

                B_active = wm.sum().item()
                num_cls = pred_logit_10.shape[-1]
                num_obj = 127
                logits = pred_logit_10.view(B_active, num_obj, num_cls)
                if args.weight_type == 'loss':
                    ce = F.cross_entropy(
                        logits.reshape(-1, num_cls), 
                        labels.reshape(-1),              
                        reduction='none'
                    ).view(B_active, num_obj)
                    scores = -ce 
                    scores = scores.masked_fill(futm[wm], float('-inf'))
                    weights = torch.softmax(scores, dim=-1) 

                elif args.weight_type == 'acc':
                    prob = torch.softmax(logits, dim=-1)                              
                    p_true = prob.gather(-1, labels.unsqueeze(-1)).squeeze(-1) 
                    p_true = p_true.masked_fill(futm[wm], float('-inf'))
                    weights = p_true

                else:  # args.weight_type == 'acc_hard'
                    pred_cls = logits.argmax(dim=-1)                                   
                    correct = (pred_cls == labels).float()                            
                    correct = correct.masked_fill(futm[wm], float('-inf'))
                    all_masked = torch.isinf(correct).all(dim=-1)                  
                    if all_masked.any():
                        correct[all_masked] = 0.0
                    weights = torch.softmax(correct, dim=-1)

                softmax_loss_masked = weights.unsqueeze(1)
                softmax_loss_masked = softmax_loss_masked.masked_fill(futm[world_mask].unsqueeze(1), float('-inf'))
        # actions = bc_policy.get_action(context, deterministic=True)
        # actions = actions.squeeze(1)
        # all_actions[~dead_agent_mask, :] = actions
        
        # Set importance weight to visualization
        world_importance_weight = torch.zeros((args.batch_size, 1, partner_num)).to("cuda")
        multi_head_mask = ~alive_agent_mask.unsqueeze(1)
        world_mask = (~dead_agent_mask).sum(dim=-1) == 1
        world_importance_weight[world_mask] = world_importance_weight[world_mask].masked_scatter(multi_head_mask[world_mask], softmax_loss_masked)
        setattr(env.vis, "importance_weight", world_importance_weight.detach().cpu())
        if time_step % 5 == 0:
            sim_states = env.vis.plot_simulator_state(
                    env_indices=list(range(args.batch_size)),
                    time_steps=[time_step]*args.batch_size,
                    plot_importance_weight=True,
                    center_agent_indices=alive_ego_idx,
                    plot_log_replay_trajectory=True,
                    zoom_radius=args.zoom_radius,
                )
    
            for i in range(args.batch_size):
                    frames[i].append(
                        sim_states[i][0]
                    )

        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        if (dead_agent_mask == True).all():
            break

    # Make video
    root = os.path.join(args.image_path, args.dataset, sweep_name, args.model_name)
    os.makedirs(root, exist_ok=True)
    for i in range(args.batch_size):
        out_dir = os.path.join(root, f"lp_{args.weight_type}_world{i}")
        save_frames_parallel(frames[i], out_dir, stem=f"lp_{args.weight_type}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    parser.add_argument('--dataset-size', type=int, default=90) # total_world
    parser.add_argument('--batch-size', type=int, default=90) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/exp_10000_v2')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s11_0802_051525.pth')
    parser.add_argument('--lp-model-name', '-lpn', type=str, default='pos_early_lp')
    parser.add_argument('--image-path', '-vp', type=str, default='/data/full_version/images/importance_weight_log')
    parser.add_argument('--linear-probing', '-lp', type=str, default='other')
    parser.add_argument('--zoom-radius', type=int, default=70)
    parser.add_argument('--weight-type', type=str,
                    choices=['acc', 'loss', 'acc_hard'],
                    default='acc')
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
    
    sweep_name = Path(args.model_path).name    
    # Load policy
    model_path = os.path.join(args.model_path, args.model_name)
    print(f'model: {model_path}')
    bc_policy = torch.load(f"{model_path}", weights_only=False).to("cuda")
    bc_policy.eval()
    
    # Load linear probing model
    lp_model_root = os.path.join(args.model_path, f'{args.linear_probing}_linear_prob', args.model_name.replace('.pth', ''))
    seed = int(args.model_name.split('_')[2][1:])
    lp_models = []
    lp_model = torch.load(os.path.join(lp_model_root, f'seed{seed}', f'{args.lp_model_name}_10.pth'), weights_only=False).to("cuda")
    lp_model.eval()
    lp_models.append(lp_model)
    
    # Simulate the environment with the policy
    df = pd.read_csv(f'/data/full_version/expert_{args.dataset}_data_v2.csv')
    expert_dict = df.set_index('scene_idx').to_dict(orient='index')
    for i, batch in enumerate(scene_loader):
        env.swap_data_batch(batch)
        run(args, env, bc_policy, lp_model, scene_batch_idx=i, sweep_name=sweep_name, exp=args.linear_probing)
    env.close()

