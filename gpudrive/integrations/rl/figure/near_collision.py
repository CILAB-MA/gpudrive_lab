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
import os, torch, numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib import cm
from matplotlib.colors import TwoSlopeNorm
from gpudrive.networks.late_fusion import NeuralNet
import pufferlib, yaml
from box import Box 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CELLS = 8
C = CELLS * CELLS
WINDOW = 15
def make_bucket():
    return {"num": torch.zeros(C, device="cuda"),
            "prob": torch.zeros(C, device="cuda"),
            "den": torch.zeros(C, device="cuda"),
            }

def make_group_dict():
    return {
        "all": make_bucket(),
        "goal": make_bucket(),
        "off_road": make_bucket(),
        "veh_collision": make_bucket(),
        "not_collision": make_bucket(),
        "not_offroad": make_bucket(),
    }
acc = defaultdict(make_group_dict)

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)

@torch.no_grad()
def _update_bucket(bucket, y, p, v, pr):
    yv = y[v].long()
    pv = p[v].long()
    corr = (pv == yv).float()
    bucket["num"] += torch.bincount(yv, weights=corr, minlength=C)
    bucket["prob"] += torch.bincount(yv, weights=pr[v], minlength=C)
    bucket["den"] += torch.bincount(yv, minlength=C)

@torch.no_grad()
def update_accumulators_for_step(future_step,
                                 other_cls, other_prob, discrete_pos, futm,
                                 goal_achieved_ep, off_road_ep, veh_collision_ep,
                                 veh_coll_step, off_road_step, time_step):
    valid = futm.bool() & (discrete_pos >= 0)
    prob_at_discrete_pos = other_prob.gather(
        dim=-1,
        index=discrete_pos.unsqueeze(-1)         # [B, 127, 1]
    ).squeeze(-1) 
    buckets = acc[future_step]

    _update_bucket(buckets["all"], discrete_pos, other_cls, valid, prob_at_discrete_pos)

    def upd_group(name, ep_mask):
        if ep_mask.any():
            rows = ep_mask.nonzero(as_tuple=False).squeeze(-1)
            _update_bucket(buckets[name],
                           discrete_pos[rows], other_cls[rows], valid[rows], 
                           prob_at_discrete_pos[rows])

    upd_group(
    "goal",
    (goal_achieved_ep == 1)
    & (off_road_ep != 1)
    & (veh_collision_ep != 1)
    )
    in_window = (
            (off_road_ep == 1)   
            & (goal_achieved_ep != 1)
            & (veh_collision_ep != 1)
            & (off_road_step >= 0)
            & (off_road_step - WINDOW - future_step < time_step)
            & (off_road_step - future_step >= time_step) # e.g. 30 - 10 - 5 <= time_step < 30 - 10 + 5 -> 15 <= time_step < 25 -> 25 <= time_step < 30
        )
    out_window = (
        (off_road_ep == 1)   
        & (goal_achieved_ep != 1)
        & (veh_collision_ep != 1) 
        & (off_road_step >= 0)      
        & ~((off_road_step - WINDOW - future_step <= time_step) & (off_road_step - future_step > time_step))
    )
    upd_group(
        "off_road",
        in_window
    )
    upd_group(
        "not_offroad",
        in_window
    )
    in_window = (
            (veh_collision_ep == 1)   
            & (goal_achieved_ep != 1)
            & (off_road_ep != 1) 
            & (veh_coll_step >= 0)
            & (veh_coll_step - WINDOW - future_step < time_step)
            & (veh_coll_step - future_step >= time_step)
        )
    out_window = (
        (veh_collision_ep == 1)   
        & (goal_achieved_ep != 1)
        & (off_road_ep != 1)  
        & (veh_coll_step >= 0)      
        & ~((veh_coll_step - WINDOW - future_step <= time_step) & (veh_coll_step - future_step > time_step))
    )
    upd_group(
        "veh_collision",
        in_window
    )

    upd_group(
        "not_collision",
        out_window
    )
def digitize(t, bins):
    return torch.bucketize(t, bins, right=False)

def save_png(path, arr):
    Image.fromarray(np.asarray(arr)).save(path, format="PNG", optimize=False, compress_level=0)

def save_frames_parallel(frames_list, out_dir, stem="frame"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as ex:
        futures = []
        for t, frame in enumerate(frames_list):
            fpath = out_dir / f"{stem}_{t:06d}.png"
            futures.append(ex.submit(save_png, fpath, frame))
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

def run(args, env, policy, raw_lp_models, other_lp_models, scene_batch_idx, sweep_name, exp):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    frames = [[] for _ in range(args.batch_size)]
    NUM_WORLD = alive_agent_mask.shape[0]
    # Extract Linear Probing
    layers = register_all_layers_forward_hook(bc_policy.fusion_attn)
    
    # =============== save data for linear probing ===============
    ego_global_pos = torch.zeros((args.batch_size, env.episode_len, 2)).cuda()
    ego_global_rot = torch.zeros((args.batch_size, env.episode_len)).cuda()
    other_relative_pos = torch.zeros((args.batch_size, env.episode_len, 127, 2)).cuda()
    other_relative_mask = torch.zeros((args.batch_size, env.episode_len, 127)).bool().cuda()
    # ============================================================
    infos = env.get_infos()
    off_road_ep = torch.zeros((args.batch_size, )).cuda()
    veh_collision_ep = torch.zeros((args.batch_size, )).cuda()
    veh_collision_step = torch.full((args.batch_size, ), fill_value=-1, dtype=torch.float32).to("cuda")
    off_road_step = torch.full((args.batch_size, ), fill_value=-1, dtype=torch.float32).to("cuda")
    goal_achieved_ep = torch.zeros((args.batch_size, )).cuda()
    alive_world = alive_agent_mask.sum(-1).bool()
    global_ego = torch.zeros((args.batch_size, env.episode_len, 3)).cuda()
    global_other = torch.zeros((args.batch_size, env.episode_len, 127, 2)).cuda()
    for time_step in tqdm(range(env.episode_len)):
        global_ego_t, global_other_t = env.get_probing_obs()
        if global_ego_t.shape[0] != len(alive_world):
            continue
        global_ego[:, time_step][alive_world] = global_ego_t[alive_world]
        global_other[:, time_step] = global_other_t
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
            actions, *_ = policy(alive_obs)
        all_actions[~dead_agent_mask, :] = actions
        env.step_dynamics(all_actions)
        infos = env.get_infos()
        off_road_ep[alive_world] += infos.off_road[~dead_agent_mask]
        veh_collision_ep[alive_world] += infos.collided[~dead_agent_mask]
        goal_achieved_ep[alive_world] += infos.goal_achieved[~dead_agent_mask]
        veh_coll_mask = (infos.collided[~dead_agent_mask] > 0).bool() & (veh_collision_step[alive_world] == -1)
        off_road_mask = (infos.off_road[~dead_agent_mask] > 0).bool() & (off_road_step[alive_world] == -1)
        alive_idx = alive_world.nonzero(as_tuple=True)[0] 
        coll_idx = alive_idx[veh_coll_mask] 
        veh_collision_step[coll_idx] = time_step
        offroad_idx = alive_idx[off_road_mask] 
        off_road_step[offroad_idx] = time_step
        off_road_ep = torch.clamp(off_road_ep, max=1)
        veh_collision_ep = torch.clamp(veh_collision_ep, max=1)
        goal_achieved_ep = torch.clamp(goal_achieved_ep, max=1)
        obs = env.get_obs()
        dones = env.get_dones()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        alive_world = (~dead_agent_mask).sum(-1).bool()
        if (dead_agent_mask == True).all():
            break
    print('ONE LOOP FINISHED!')
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    expert_actions, _, _, _, _  = env.get_expert_actions()
    goal_mask = (goal_achieved_ep == 1)
    offroad_mask = (off_road_ep == 1)
    veh_coll_mask = (veh_collision_ep  == 1)
    for time_step in tqdm(range(env.episode_len)):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3).to("cuda")
        # all_actions = expert_actions[:, :, time_step].clone()
        global_ego_t = global_ego[:, time_step]
        # MASK
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        world_mask = (~dead_agent_mask).sum(dim=-1) == 1
        partner_mask_bool = partner_mask == 2
        all_masks = [partner_mask_bool[~dead_agent_mask].unsqueeze(1), road_mask[~dead_agent_mask].unsqueeze(1)]
        with torch.no_grad():
            # for padding zero
            alive_obs = obs[~dead_agent_mask]
            actions, *_ = policy(alive_obs)
            actions = actions.squeeze(1)
            partner_obs = alive_obs[..., 6:128*6]         # (100, 5, 762)
            ego_obs = alive_obs[..., :6]
            ego_obs = ego_obs.reshape(100, 6).unsqueeze(1).repeat(1, 127, 1)
            po_input = partner_obs.reshape(100, 127, 6) # (100, 5, 127, 6)
            raw_lp_input = torch.cat([ego_obs, po_input], dim=-1)
            if time_step < env.episode_len - 10: 
                other_lp_input = policy.partner_embed(po_input)
                wm = world_mask
                for i, (other_lp, raw_lp, future_step) in enumerate(zip(other_lp_models, raw_lp_models, future_steps)):
                    if time_step >= env.episode_len - future_step:
                        continue
                    global_others_ft  = global_other[: ,time_step + future_step]
                    rel = global_others_ft - global_ego_t[:, :2].unsqueeze(1) 
                    c = torch.cos(global_ego_t[:, 2])                         # (W,)
                    s = torch.sin(global_ego_t[:, 2])                         # (W,)
                    Rinv = torch.stack((
                            torch.stack(( c,  s), dim=-1),          # (W,2)
                            torch.stack((-s,  c), dim=-1)           # (W,2)
                        ), dim=-2)                                   # (W,2,2)

                    rel_ego = torch.einsum('bij,bnj->bni', Rinv, rel)
                    rel_pos = 2 * ((rel_ego - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
                    x, y = rel_pos[..., 0], rel_pos[..., 1]
        
                    xbins = torch.linspace(-0.05, 0.05, 9).to(rel_pos.device)
                    ybins = torch.linspace(-0.05, 0.05, 9).to(rel_pos.device)

                    x_bins = torch.bucketize(x, xbins) - 1
                    y_bins = torch.bucketize(y, ybins) - 1
                    
                    x_bins = torch.clip(x_bins, 0, 7)
                    y_bins = torch.clip(y_bins, 0, 7)
                    discrete_pos = x_bins * 8 + y_bins
                    futm = other_relative_mask[:, time_step + future_step]   
                    other_pred = other_lp(other_lp_input)
                    raw_pred = raw_lp(raw_lp_input)
                    raw_prob = raw_pred.softmax(dim=-1) 
                    other_prob = other_pred.softmax(dim=-1) 
                    # ego_orig_cls = ego_orig_pred.argmax(dim=-1) 
                    other_cls = other_pred.argmax(dim=-1) 
                    if wm.shape[0] != discrete_pos.shape[0]:
                        update_accumulators_for_step(future_step,
                                other_cls, other_prob, discrete_pos, futm,
                                goal_achieved_ep, off_road_ep, veh_collision_ep,
                                veh_collision_step, off_road_step,
                                time_step)
                    else:
                        update_accumulators_for_step(future_step,
                                other_cls, other_prob, discrete_pos[wm], futm[wm],
                                goal_achieved_ep[wm], off_road_ep[wm], veh_collision_ep[wm],
                                veh_collision_step[wm],off_road_step[wm],
                                time_step)


        all_actions[~dead_agent_mask, :] = actions
        env.step_dynamics(all_actions)

        obs = env.get_obs()
        dones = env.get_dones()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)

        if (dead_agent_mask == True).all():
            break

def to_heat(bucket, CELLS, acc_or_prob='prob'):
    acc_vec = (bucket[acc_or_prob] / bucket["den"].clamp_min(1)).nan_to_num(np.nan)
    cnt_vec = bucket["den"]
    return acc_vec.view(CELLS, CELLS), cnt_vec.view(CELLS, CELLS)

def compute_overall_acc(bucket, acc_or_prob='prob') -> float:
    num = bucket[acc_or_prob].sum()
    den = bucket["den"].sum().clamp_min(1)
    return (num / den).item()

@torch.no_grad()
def save_diff_heatmaps(
    acc,
    future_steps,
    CELLS,
    outdir="./heatmaps_lp",
    collision_group="veh_collision",  
    non_collision_group="not_collision", 
    transpose=True,
    origin="lower",
    min_count=10,
    acc_or_prob="prob"
):
    os.makedirs(outdir, exist_ok=True)

    # future_steps: 40,30,20,10
    fs_list = sorted(list(future_steps), reverse=True)  # [40,30,20,10]

    diffs = []
    for fs in fs_list:
        h_coll, cnt_coll = to_heat(acc[fs][collision_group], CELLS, acc_or_prob)
        h_non,  cnt_non  = to_heat(acc[fs][non_collision_group], CELLS, acc_or_prob)
        avg_all = compute_overall_acc(acc[fs]['all'], acc_or_prob)
        mask_valid = (cnt_coll >= min_count) & (cnt_non >= min_count)
        diff = (h_coll - h_non) / avg_all
        diff = diff.masked_fill(~mask_valid, torch.nan)
        if transpose:
            diff = diff.T
        diffs.append(diff)

    if not diffs:
        return

    stacked = torch.stack(diffs)      # [num_fs, CELLS, CELLS]
    if torch.isfinite(stacked).any():
        mask = torch.isfinite(stacked)
        abs_max = stacked[mask].abs().max().item()
    else:
        abs_max = 1.0

    vlim = abs_max if abs_max > 0 else 1.0

    cmap = cm.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#9e9e9e")
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    n_fs = len(fs_list)
    fig, axes = plt.subplots(
        1, n_fs,
        figsize=(4.5 * n_fs, 4.2),
        dpi=300,
        gridspec_kw={"wspace": 0.10},
        constrained_layout=False,
    )
    if n_fs == 1:
        axes = [axes]

    last_im = None
    for ax, fs, diff in zip(axes, fs_list, diffs):
        im = ax.imshow(
            diff.cpu().numpy(), origin=origin,
            cmap=cmap, norm=norm, aspect="equal"
        )
        ticks = np.arange(CELLS)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])
        ax.set_title(f"{fs} Step Before")
        ax.set_xlabel(""); ax.set_ylabel("")
        last_im = im

    cbar_ax = fig.add_axes([0.92, 0.20, 0.012, 0.60])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label("Normalized Difference")

    fig.text(0.02, 0.5, "Y", va="center", ha="left", rotation="vertical")
    fig.text(0.5, 0.03, "X", va="center", ha="center")
    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.12)

    fig.savefig(os.path.join(outdir, f"heat_diff_{collision_group}_vs_non_{acc_or_prob}.svg"))
    plt.close(fig)

def save_acc_torch(path, acc, future_steps, CELLS, groups=("all","goal","off_road","veh_collision", "not_collision", "not_offroad")):
    state = {
        "future_steps": list(future_steps),
        "CELLS": CELLS,
        "groups": list(groups),
        "acc": {
            int(fs): {
                g: {
                    "num": acc[fs][g]["num"].cpu(),
                    "den": acc[fs][g]["den"].cpu(),
                    "prob":acc[fs][g]["prob"].cpu(),
                }
                for g in groups
            }
            for fs in future_steps
        },
    }
    torch.save(state, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    parser.add_argument('--dataset-size', type=int, default=9987)
    parser.add_argument('--batch-size', type=int, default=100)
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/exp_80000_subset_aix')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s3_0908_113203.pth')
    parser.add_argument('--lp-model-name', '-lpn', type=str, default='pos_early_lp')
    parser.add_argument('--image-path', '-vp', type=str, default='/data/full_version/images/intervention')
    parser.add_argument('--linear-probing', '-lp', type=str, default='other')
    parser.add_argument('--zoom-radius', type=int, default=50)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    import matplotlib as mpl
    args = parser.parse_args()
    mpl.rcParams.update({
        'font.family': 'Times New Roman',
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 19,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "font.size": 15,
        "axes.linewidth": 0.8,
        "axes.titlepad": 10,
        "figure.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
    })
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
    env_config = EnvConfig(
        dynamics_model="classic",
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
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=scene_loader,
        max_cont_agents=128,  # Number of agents to control
        device="cuda",
        action_type="discrete",
    )
    
    sweep_name = Path(args.model_path).name    
    # Load policy
    model_path = os.path.join(args.model_path, args.model_name)
    print(f'model: {model_path}')
    config = load_config("baselines/ppo/config/ppo_base_puffer.yaml")
    params = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False)
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
    lp_ego_root = os.path.join(args.model_path, f'ego_linear_prob', args.model_name.replace('.pth', ''))
    lp_other_root = os.path.join(args.model_path, f'other_linear_prob', args.model_name.replace('.pth', ''))
    seed = int(args.model_name.split('_')[2][1:])
    future_steps = [10, 20, 30, 40]
    other_lp_models, raw_lp_models = [], []
    for future_step in future_steps:
        raw_model = torch.load(os.path.join(lp_other_root, f'seed{seed}', f'pos_baseline_{future_step}.pth'), weights_only=False).to("cuda")
        raw_model.eval()
        other_model = torch.load(os.path.join(lp_other_root, f'seed{seed}', f'{args.lp_model_name}_{future_step}.pth'), weights_only=False).to("cuda")
        other_model.eval()
        other_lp_models.append(other_model)
        raw_lp_models.append(raw_model)
    
    # Simulate the environment with the policy
    total_iter = int(args.dataset_size // args.batch_size)
    for i in range(total_iter):
        run(args, env, policy, raw_lp_models, other_lp_models, scene_batch_idx=i, sweep_name=sweep_name, exp=args.linear_probing)
        if i != num_iter - 1:
            env.swap_data_batch()
    env.close()
    save_acc_torch(f"lp_grid_prob_raw_il_{WINDOW}.pt", acc, future_steps, CELLS)

    save_diff_heatmaps(
        acc,
        future_steps=[10, 20, 30, 40],
        CELLS=CELLS,
        outdir=f"./heatmaps_lp_exclusive_full_{WINDOW}_gap",
        collision_group="veh_collision",
        non_collision_group="all", 
        transpose=True,
        origin="lower",
        min_count=10,
    )