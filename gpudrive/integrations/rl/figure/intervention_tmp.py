"""Obtain a policy using behavioral cloning."""
import os, sys
from typing import Any
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
import matplotlib.pyplot as plt
from gpudrive.networks.late_fusion import NeuralNet
import pufferlib, yaml
from box import Box

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)

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

def run(args, env, policy, ego_lp_models, other_lp_models, intervention_idx, 
    intervention_label, intervention_other_indices, intervention_other_labels, model_name,
    scene_batch_idx=0):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.batch_size)]
    NUM_WORLD = alive_agent_mask.shape[0]
    diff_cls_total = np.zeros((31, NUM_WORLD)).astype('bool')
    # Extract Linear Probing
    ego_layers = register_all_layers_forward_hook(policy.shared_embed)
    ego_embed = register_all_layers_forward_hook(policy.ego_embed)
    road_embed = register_all_layers_forward_hook(policy.road_map_embed)
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
    expert_actions, _, _, _, _  = env.get_expert_actions()
    for time_step in tqdm(range(env.episode_len)):
        all_actions = expert_actions[:, :, time_step].clone()
        # MASK
        partner_mask = env.get_partner_mask().to("cuda")
        lp_partner_mask_bool = torch.logical_or(partner_mask == 2, partner_mask == 1)
        ego_global_state = env.get_global_state()
        ego_global_pos[:, time_step][alive_agent_mask.sum(-1) == 1] = torch.stack((ego_global_state.pos_x, ego_global_state.pos_y), dim=-1)[alive_agent_mask]
        ego_global_rot[:, time_step][alive_agent_mask.sum(-1) == 1] = ego_global_state.rotation_angle[alive_agent_mask]
        partner_pos = env.get_partner_pos()
        other_relative_pos[:, time_step][alive_agent_mask.sum(-1) == 1] = partner_pos[alive_agent_mask]
        other_relative_mask[:, time_step][alive_agent_mask.sum(-1) == 1] = lp_partner_mask_bool[alive_agent_mask]

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
    intervention_other_indices = torch.as_tensor(intervention_other_indices, device='cuda', dtype=torch.long)#[:, :50]
    intervention_other_labels = torch.as_tensor(intervention_other_labels, dtype=torch.long).to('cuda').transpose(0, 1)#[:, :50]
    intervention_label = torch.as_tensor(intervention_label, dtype=torch.long).to('cuda').transpose(0, 1)
    intervention_idx = torch.as_tensor(intervention_idx, device='cuda', dtype=torch.long)
    
    # TMP should be change
    for time_step in tqdm(range(env.episode_len)):
        # all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        all_actions = expert_actions[:, :, time_step].clone()
        # MASK
        partner_mask = env.get_partner_mask().to("cuda")
        world_mask = (~dead_agent_mask).sum(dim=-1) == 1
        alpha = 5
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            _ = policy(alive_obs, deterministic=True)
            if time_step < env.episode_len - 40 and time_step % 3 == 0: 
                ego_nth_layer = list(ego_layers.keys())[-1]
                road_embed_nth_layer = list(road_embed.keys())[-1]
                ego_embed_nth_layer = list(ego_embed.keys())[-1]
                pobs = alive_obs[..., 6:6 * 128].view(-1, 127, 6)
                other_lp_input = policy.partner_embed(pobs)
                ego_lp_input = ego_layers[ego_nth_layer]
                ego_embed_input = ego_embed[ego_embed_nth_layer]
                road_embed_input = road_embed[road_embed_nth_layer]
                wm = world_mask
                orig_dict = defaultdict(dict)
                prime_dict = defaultdict(dict)
                other_dict = defaultdict(dict)
                intervention_dict = defaultdict[Any, dict](dict)
                full_weights = torch.zeros((NUM_WORLD, 64, 4, 4)).to('cuda')[wm]
                batch = torch.arange(len(full_weights), device='cuda')
                for i, other_lp in enumerate(other_lp_models):
                    w = other_lp.head.weight
                    weight_label = w.index_select(0, intervention_label[i])[wm]
                    weight_label2 = w.index_select(0, intervention_other_labels[i])[wm]
                    weight_label3 = w.index_select(0, intervention_other_labels[4 + i])[wm]
                    weight_label4 = w.index_select(0, intervention_other_labels[8 + i])[wm]
                    full_weights[..., 0, i] = weight_label * alpha
                    full_weights[..., 1, i] = weight_label2 * alpha
                    full_weights[..., 2, i] = weight_label3 * alpha
                    full_weights[..., 3, i] = weight_label4 * alpha
                    # full_weights2[..., i] = weight_label2 * 5
                if args.intervention == 'mean':
                    full_weights_combined = full_weights.mean(-1)
                    # full_weights_combined2 = full_weights2.mean(-1)
                elif args.intervention == 'sum':
                    full_weights_combined = full_weights.sum(-1)
                else:
                    full_weights_combined = full_weights
                
                for i, (other_lp, ego_lp, future_step) in enumerate(zip(other_lp_models, ego_lp_models, future_steps)):
                    futm = other_relative_mask[:, time_step + future_step]   
                    other_pred = other_lp(other_lp_input)
                    w = other_lp.head.weight 
                    # weight_label = w.index_select(0, intervention_label[i])[wm]
                    g_prime = other_lp_input.clone()
                    # g_prime2 = other_lp_input.clone()
                    if args.intervention == 'one':
                        g_prime[batch, intervention_idx[wm], :] += full_weights_combined[..., i]
                    else:
                        g_prime[batch, intervention_idx[wm], :] += full_weights_combined[..., 0]

                        idx0 = intervention_other_indices[0, wm]
                        m0 = idx0.ge(0)
                        g_prime[batch[m0], idx0[m0], :] += full_weights_combined[..., 1][m0]

                        idx1 = intervention_other_indices[1, wm]
                        m1 = idx1.ge(0)
                        g_prime[batch[m1], idx1[m1], :] += full_weights_combined[..., 2][m1]
                        idx2 = intervention_other_indices[2, wm]
                        m2 = idx2.ge(0)
                        g_prime[batch[m2], idx2[m2], :] += full_weights_combined[..., 3][m2]
                        # g_prime2[batch, intervention_idx[wm], :] += full_weights_combined2
                    g_prime = torch.cat([ego_embed_input, g_prime.max(dim=1)[0], road_embed_input.max(dim=1)[0]], dim=1)
                    # g_prime2 = torch.cat([other_layers[other_nth_layer][:, 0, :].unsqueeze(1), g_prime2], dim=1)
                    h_prime = policy.shared_embed(g_prime)
                    # h_prime2 = bc_policy.ro_attn(g_prime2)
                    ego_input_prime = h_prime
                    # ego_input_prime2 = h_prime2['last_hidden_state'][:, 0, :]
                    ego_orig_pred = ego_lp(ego_lp_input)
                    ego_prime_pred = ego_lp(ego_input_prime) # todo: intervention idx applying
                    # ego_prime_pred2 = ego_lp(ego_input_prime2) # todo: intervention idx applying
                    orig_alive_world = torch.zeros((NUM_WORLD, 1)).long().to("cuda")
                    other_alive_world = torch.zeros((NUM_WORLD, 127)).long().to("cuda")
                    intevention_alive_world = torch.zeros((NUM_WORLD, 127)).long().to("cuda")
                    prime_alive_world = torch.zeros((NUM_WORLD, 1)).long().to("cuda")
                    ego_orig_cls = ego_orig_pred.argmax(dim=-1) 
                    other_cls = other_pred.argmax(dim=-1) 
                    ego_prime_cls = ego_prime_pred.argmax(dim=-1) 
                    other_cls = other_cls.masked_fill(futm[wm], -1)
                    intevention_alive_world[torch.arange(NUM_WORLD), intervention_idx] = intervention_label[i]
                    intevention_alive_world[torch.arange(NUM_WORLD), intervention_other_indices[0]] = intervention_other_labels[i]
                    intevention_alive_world[torch.arange(NUM_WORLD), intervention_other_indices[1]] = intervention_other_labels[4 + i]
                    intevention_alive_world[torch.arange(NUM_WORLD), intervention_other_indices[2]] = intervention_other_labels[8 + i]
                    other_alive_world[wm] = other_cls 
                    orig_alive_world[wm] = ego_orig_cls.unsqueeze(-1)
                    prime_alive_world[wm] = ego_prime_cls.unsqueeze(-1)
                    orig_dict[ego_lp.future_step] = orig_alive_world
                    intervention_dict[ego_lp.future_step] = intevention_alive_world
                    prime_dict[ego_lp.future_step] = prime_alive_world
                    other_dict[ego_lp.future_step] = other_alive_world
                    del g_prime
                # print(f'Diff LP {(ego_prime_pred2 - ego_prime_pred).abs().mean()} {(ego_prime_pred2 - ego_prime_pred).abs().std()}')
        intervention_idx_total = torch.cat([intervention_idx.unsqueeze(0), intervention_other_indices], axis=0)
        setattr(env.vis, f"ego_pred_pos", orig_dict)
        setattr(env.vis, f"other_pred_pos", other_dict)
        setattr(env.vis, f"intervention_ego", prime_dict)
        setattr(env.vis, f"intervention_other", intervention_dict)
        setattr(env.vis, f"target_non_ego_rank", intervention_idx_total)
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

    # Make video (absolute scene id so lp_world folders never overlap across batches)
    root = os.path.join(args.image_path, args.dataset, model_name)
    os.makedirs(root, exist_ok=True)
    for i in range(args.batch_size):
        scene_id = args.start_idx + i + args.batch_size * scene_batch_idx
        out_dir = os.path.join(root, f"lp_world{scene_id}")
        if args.random:
            random = '_random'
        else:
            random = '4'
        if args.linear_probing == 'intervention':
            out_dir = os.path.join(out_dir, f"{args.intervention}{random}")
        save_frames_parallel(frames[i], out_dir, stem=f"lp_{args.linear_probing}", diff_cls_total=diff_cls_total[:, i])


def pad_df_to_length(df, end_idx, fill_values):
    """Pad dataframe so row_index is preserved up to end_idx (missing rows filled)."""
    df = df.copy()
    for col, fill in fill_values.items():
        if col not in df.columns:
            df[col] = fill
        df[col] = df[col].fillna(fill)
    pad_len = end_idx - len(df)
    if pad_len > 0:
        pad_df = pd.DataFrame({col: [fill] * pad_len for col, fill in fill_values.items()})
        df = pd.concat([df, pad_df], ignore_index=True)
        logger.info(f"Padded CSV with {pad_len} rows (total={len(df)})")
    return df.iloc[:end_idx].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    parser.add_argument('--dataset-size', type=int, default=400, help='number of scenes to run from start-idx')
    parser.add_argument('--start-idx', '-s', type=int, default=200, help='first scene / CSV row_index; default 200 avoids existing lp_world100-199')
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    # EXPERIMENT
    parser.add_argument('--base-path', '-bp', type=str, default='/data/after_cvpr/rl/scene_10000') #80000_subset_aix
    parser.add_argument('--image-path', '-vp', type=str, default='/data/after_cvpr/images/intervention_original')
    parser.add_argument('--linear-probing', '-lp', type=str, default='original', choices=['original', 
    'intervention'])
    parser.add_argument('--intervention', '-i', type=str, default='mean', choices=['mean', 
    'sum', 'one'])
    parser.add_argument('--random', '-r', action='store_true')
    parser.add_argument('--zoom-radius', type=int, default=50)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()

    end_idx = args.start_idx + args.dataset_size
    cols = [f"step{i}" for i in (10, 20, 30, 40)]
    cols1 = [f"step{i}_0" for i in (10, 20, 30, 40)]
    cols2 = [f"step{i}_1" for i in (10, 20, 30, 40)]
    cols3 = [f"step{i}_2" for i in (10, 20, 30, 40)]

    df = pad_df_to_length(
        pd.read_csv("/data/after_cvpr/intervention.csv"),
        end_idx,
        {**{"intervention_idx": 0}, **{c: 0 for c in cols}},
    )
    df_more = pad_df_to_length(
        pd.read_csv("/data/after_cvpr/intervention_others.csv"),
        end_idx,
        {
            "intervention_idx_0": -1, "intervention_idx_1": -1, "intervention_idx_2": -1,
            **{c: 0 for c in cols1 + cols2 + cols3},
        },
    )

    # Keep row_index == scene id, then take [start_idx, end_idx)
    df = df.iloc[args.start_idx:end_idx].reset_index(drop=True)
    df_more = df_more.iloc[args.start_idx:end_idx].reset_index(drop=True)

    intervention_idx = df['intervention_idx'].astype(int).tolist()
    intervention_other_indices = np.stack([
        df_more['intervention_idx_0'].astype(int).to_numpy(),
        df_more['intervention_idx_1'].astype(int).to_numpy(),
        df_more['intervention_idx_2'].astype(int).to_numpy(),
    ], axis=0)
    intervention_label = np.stack([df[c].astype(int).to_numpy() for c in cols], axis=1)
    intervention_other_labels = np.concatenate([
        np.stack([df_more[c].astype(int).to_numpy() for c in cols1], axis=1),
        np.stack([df_more[c].astype(int).to_numpy() for c in cols2], axis=1),
        np.stack([df_more[c].astype(int).to_numpy() for c in cols3], axis=1),
    ], axis=1)
    if args.random:
        np.random.seed(42)
        intervention_label = np.random.randint(
            low=0, high=64, size=intervention_label.shape, dtype=intervention_label.dtype
        )
        intervention_other_labels = np.random.randint(
            low=0, high=64, size=intervention_other_labels.shape, dtype=intervention_other_labels.dtype
        )

    # SceneDataLoader: start_idx : dataset_size (dataset_size acts as exclusive end)
    scene_loader = SceneDataLoader(
        root=f"/data/full_version/data/{args.dataset}/",
        batch_size=args.batch_size,
        dataset_size=end_idx,
        start_idx=args.start_idx,
        sample_with_replacement=False,
        shuffle=False,
    )
    dataset_size = args.dataset_size
    print(f'{args.dataset} start_idx={args.start_idx} end_idx={end_idx} len scene loader {len(scene_loader)}')
    
    # Make env
    env_config = EnvConfig(
        dynamics_model="delta_local",
        collision_behavior='ignore',
        steer_actions=torch.round(
                torch.linspace(-torch.pi, torch.pi, 13),
                decimals=3,
            ),
        accel_actions=torch.round(
                torch.linspace(-4.0, 4.0, 7), decimals=3
            ),
        num_stack=1

    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=scene_loader,
        max_cont_agents=1,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )
    lp_path = os.path.join(args.base_path, f'other_linear_prob')
    model_name = os.listdir(lp_path)[-1]
    # Load policy
    model_path = os.path.join(args.base_path, f"{model_name}.pt")
    print(f'model: {model_path}')
    config = load_config("baselines/ppo/config/ppo_base_puffer.yaml")
    params = torch.load(model_path, weights_only=False)
    policy = NeuralNet(
        input_dim=64,
        action_dim=91,
        hidden_dim=128,
        config=config.environment,
    ).to("cuda")
    policy.load_state_dict(params["parameters"])
    policy.eval()
    num_iter = int(dataset_size // args.batch_size) if dataset_size != 0 else 0
    # Load linear probing model
    lp_ego_root = os.path.join(args.base_path, f'ego_linear_prob', model_name.replace('.pth', ''))
    lp_other_root = os.path.join(args.base_path, f'other_linear_prob', model_name.replace('.pth', ''))
    seed = 3
    future_steps = [10, 20, 30, 40]
    other_lp_models, ego_lp_models = [], []
    for future_step in future_steps:
        ego_model = torch.load(os.path.join(lp_ego_root, f'seed{seed}', f'pos_ego_final_lp_{future_step}.pth'), weights_only=False).to("cuda")
        ego_model.eval()
        other_model = torch.load(os.path.join(lp_other_root, f'seed{seed}', f'pos_lp_{future_step}.pth'), weights_only=False).to("cuda")
        other_model.eval()
        other_lp_models.append(other_model)
        ego_lp_models.append(ego_model)
    
    # Simulate the environment with the policy
    total_iter = int(args.dataset_size // args.batch_size)
    for i in range(total_iter):
        sl = slice(i * args.batch_size, (i + 1) * args.batch_size)
        intervention_idx_batch = intervention_idx[sl]
        intervention_label_batch = intervention_label[sl]
        intervention_other_indices_batch = intervention_other_indices[:, sl]
        intervention_other_labels_batch = intervention_other_labels[sl]
        run(args, env, policy, ego_lp_models, other_lp_models,
            scene_batch_idx=i,
            intervention_idx=intervention_idx_batch, intervention_label=intervention_label_batch,
            intervention_other_indices=intervention_other_indices_batch,
            intervention_other_labels=intervention_other_labels_batch,
            model_name=model_name)
        if i != num_iter - 1:
            env.swap_data_batch()
    env.close()


