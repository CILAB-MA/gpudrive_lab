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
import os
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def digitize(t, bins):
    return torch.bucketize(t, bins, right=False)

def save_svg(path, arr):
    import io, base64
    from PIL import Image
    im = Image.fromarray(np.asarray(arr))
    w,h = im.size
    buf = io.BytesIO(); im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    open(path, "w", encoding="utf-8").write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">'
        f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="{w}" height="{h}"/></svg>'
    )

def save_frames_parallel(frames_list, out_dir, stem="frame", diff_cls_total=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        futures = []
        for t, fig in enumerate(frames_list):     # fig: matplotlib Figure
            fpath = out_dir / f"{stem}_{t:06d}_{diff_cls_total[t]}.svg"
            futures.append(ex.submit(save_svg, fpath, fig))
        for f in futures: f.result()  # join

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

def run(args, env, bc_policy, ego_lp_models, other_lp_models, scene_batch_idx, sweep_name, intervention_idx,  intervention_label):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.batch_size)]
    NUM_WORLD = alive_agent_mask.shape[0]
    diff_cls_total = np.zeros((30, NUM_WORLD)).astype('bool')
    # Extract Linear Probing
    other_layers = register_all_layers_forward_hook(bc_policy.fusion_attn)
    ego_layers = register_all_layers_forward_hook(bc_policy.ro_attn)
    # =============== save data for linear probing ===============
    ego_global_pos = torch.zeros((args.batch_size, env.episode_len, 2)).cuda()
    ego_global_rot = torch.zeros((args.batch_size, env.episode_len)).cuda()
    other_relative_pos = torch.zeros((args.batch_size, env.episode_len, 127, 2)).cuda()
    other_relative_mask = torch.zeros((args.batch_size, env.episode_len, 127)).bool().cuda()
    # ============================================================
    ego_idx = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0])
    alive_ego_idx = ego_idx[:NUM_WORLD].to('cuda')
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
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    expert_actions, _, _, _, _  = env.get_expert_actions()
    intervention_label = torch.as_tensor(intervention_label, dtype=torch.long).to('cuda').transpose(0, 1)
    intervention_idx = torch.as_tensor(intervention_idx, device='cuda', dtype=torch.long)
    
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
            if time_step < env.episode_len - 40: 
                other_nth_layer = list(other_layers.keys())[-1]
                ego_nth_layer = list(ego_layers.keys())[-1]
                other_lp_input = other_layers[other_nth_layer][:,1:128,:] 
                ego_lp_input = ego_layers[ego_nth_layer][:,0,:] 
                wm = world_mask
                orig_dict = defaultdict(dict)
                prime_dict = defaultdict(dict)
                other_dict = defaultdict(dict)
                intervention_dict = defaultdict(dict)
                full_weights = torch.zeros((NUM_WORLD, 128, 4)).to('cuda')[wm]
                batch = torch.arange(len(full_weights), device='cuda')
                for i, other_lp in enumerate(other_lp_models):
                    w = other_lp.head.weight
                    weight_label = w.index_select(0, intervention_label[i])[wm]
                    full_weights[..., i] = weight_label
                if args.intervention == 'mean':
                    full_weights_combined = full_weights.mean(-1)
                elif args.intervention == 'sum':
                    full_weights_combined = full_weights.sum(-1)
                for i, (other_lp, ego_lp, future_step) in enumerate(zip(other_lp_models, ego_lp_models, future_steps)):
                    futm = other_relative_mask[:, time_step + future_step]   
                    other_pred = other_lp(other_lp_input)
                    w = other_lp.head.weight 
                    # weight_label = w.index_select(0, intervention_label[i])[wm]
                    g_prime = other_lp_input.clone()
                    g_prime[batch, intervention_idx[wm], :] += full_weights_combined
                    g_prime = torch.cat([other_layers[other_nth_layer][:, 0, :].unsqueeze(1), g_prime], dim=1)
                    h_prime = bc_policy.ro_attn(g_prime)
                    ego_input_prime = h_prime['last_hidden_state'][:, 0, :]
                    ego_orig_pred = ego_lp(ego_lp_input)
                    ego_prime_pred = ego_lp(ego_input_prime) # todo: intervention idx applying

                    orig_alive_world = torch.zeros((NUM_WORLD, 1)).long().to("cuda")
                    other_alive_world = torch.zeros((NUM_WORLD, 127)).long().to("cuda")
                    intevention_alive_world = torch.zeros((NUM_WORLD, 127)).long().to("cuda")
                    prime_alive_world = torch.zeros((NUM_WORLD, 1)).long().to("cuda")
                    ego_orig_cls = ego_orig_pred.argmax(dim=-1) 
                    other_cls = other_pred.argmax(dim=-1) 
                    ego_prime_cls = ego_prime_pred.argmax(dim=-1) 
                    other_cls = other_cls.masked_fill(futm[wm], -1)
                    intevention_alive_world[torch.arange(NUM_WORLD), intervention_idx] = intervention_label[i]
                    other_alive_world[wm] = other_cls 

                    orig_alive_world[wm] = ego_orig_cls.unsqueeze(-1)
                    prime_alive_world[wm] = ego_prime_cls.unsqueeze(-1)
                    orig_dict[ego_lp.future_step] = orig_alive_world
                    intervention_dict[ego_lp.future_step] = intevention_alive_world
                    prime_dict[ego_lp.future_step] = prime_alive_world
                    other_dict[ego_lp.future_step] = other_alive_world

        setattr(env.vis, f"ego_pred_pos", orig_dict)
        setattr(env.vis, f"other_pred_pos", other_dict)
        setattr(env.vis, f"intervention_ego", prime_dict)
        setattr(env.vis, f"intervention_other", intervention_dict)
        setattr(env.vis, f"target_non_ego_rank", intervention_idx)
        if args.linear_probing == 'original':
            plot_intervention = False
            plot_ego_lp = True
            plot_other_lp = True
        else:
            plot_intervention = True
            plot_ego_lp = False
            plot_other_lp = False
        if time_step % 3 == 0:
            diff_cls = (orig_alive_world - prime_alive_world) != 0
            diff_cls_total[int(time_step // 3)] = diff_cls.cpu().numpy().reshape(-1)
            sim_states = env.vis.plot_simulator_state(
                    env_indices=list(range(args.batch_size)),
                    time_steps=[time_step]*args.batch_size,
                    plot_importance_weight=False,
                    plot_ego_linear_probing=plot_ego_lp,
                    plot_other_linear_probing=plot_other_lp,
                    center_agent_indices=alive_ego_idx,
                    plot_linear_probing_label=False,
                    plot_log_replay_trajectory=True,
                    plot_intervention=plot_intervention,
                    zoom_radius=args.zoom_radius,
                )
    
            for i in range(args.batch_size):
                    frames[i].append(
                        img_from_fig(sim_states[i])
                    )

        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        if (dead_agent_mask == True).all():
            break

    # Make video
    root = os.path.join(args.image_path, args.dataset, sweep_name, args.model_name, str(args.partner_portion_test))
    os.makedirs(root, exist_ok=True)
    for i in range(args.batch_size):
        out_dir = os.path.join(root, f"lp_world{i + world_mask.shape[0] * scene_batch_idx}")
        if args.linear_probing == 'intervention':
            out_dir = os.path.join(out_dir, f"{args.intervention}")
        save_frames_parallel(frames[i], out_dir, stem=f"lp_{args.linear_probing}", diff_cls_total=diff_cls_total[:, i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    parser.add_argument('--dataset-size', type=int, default=80) # total_world
    parser.add_argument('--batch-size', type=int, default=80) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/exp_80000_subset_aix') #80000_subset_aix
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s3_0908_113203.pth') # \early_attn_s3_0908_113203.pth.pth
    parser.add_argument('--lp-model-name', '-lpn', type=str, default='pos_early_lp')
    parser.add_argument('--image-path', '-vp', type=str, default='/data/full_version/images/intervention')
    parser.add_argument('--linear-probing', '-lp', type=str, default='intervention', choices=['original', 
    'intervention'])
    parser.add_argument('--intervention', '-i', type=str, default='mean', choices=['mean', 
    'sum'])
    parser.add_argument('--zoom-radius', type=int, default=70)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()
    dump = [0]*4
    dump_idx = -1
    cols = [f"step{i}" for i in (10, 20, 30, 40)]
    df = pd.read_csv("/data/full_version/intervention.csv")
    intervention_idx = df['intervention_idx'].tolist() 
    pad_len = args.dataset_size - len(intervention_idx)
    intervention_idx += [0] * pad_len
    intervention_label = np.stack([df[c].to_numpy() for c in cols], axis=1)
    intervention_label = np.pad(intervention_label, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)
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
    num_iter = int(dataset_size // args.batch_size) if dataset_size != 0 else 0
    # Load linear probing model
    lp_ego_root = os.path.join(args.model_path, f'ego_linear_prob', args.model_name.replace('.pth', ''))
    lp_other_root = os.path.join(args.model_path, f'other_linear_prob', args.model_name.replace('.pth', ''))
    seed = int(args.model_name.split('_')[2][1:])
    future_steps = [10, 20, 30, 40]
    other_lp_models, ego_lp_models = [], []
    for future_step in future_steps:
        ego_model = torch.load(os.path.join(lp_ego_root, f'seed{seed}', f'pos_final_lp_{future_step}.pth'), weights_only=False).to("cuda")
        ego_model.eval()
        other_model = torch.load(os.path.join(lp_other_root, f'seed{seed}', f'pos_early_lp_{future_step}.pth'), weights_only=False).to("cuda")
        other_model.eval()
        other_lp_models.append(other_model)
        ego_lp_models.append(ego_model)
    
    # Simulate the environment with the policy
    df = pd.read_csv(f'/data/full_version/expert_{args.dataset}_data_v2.csv')
    expert_dict = df.set_index('scene_idx').to_dict(orient='index')
    total_iter = int(args.dataset_size // args.batch_size)
    for i in range(total_iter):
        run(args, env, bc_policy, ego_lp_models, other_lp_models, scene_batch_idx=i, sweep_name=sweep_name,
            intervention_idx=intervention_idx, intervention_label=intervention_label)
        if i != num_iter - 1:
            env.swap_data_batch()
    env.close()

