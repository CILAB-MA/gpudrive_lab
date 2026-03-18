"""Obtain a policy using behavioral cloning (2x1 closer/farther plot)."""
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, sys, torch
torch.backends.cudnn.benchmark = True
sys.path.append(os.getcwd())
import argparse, functools
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')

from gpudrive.integrations.il.linear_probing.lp_model import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS
import matplotlib as mpl
from gpudrive.integrations.rl.run_lp import get_dataloader
from gpudrive.integrations.rl.run_lp import load_config
from gpudrive.networks.late_fusion import NeuralNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ------------------------------ Dataset ------------------------------
class FutureDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, ego_global_pos, ego_global_rot, masks=None, partner_mask=None,
                 future_step=1, exp='other', xy_range=None, 
                 partner_labels=None, ego_labels=None):
        self.obs = obs
        B, T, F = obs.shape
        self.actions = actions

        valid_masks = 1 - masks
        if actions.shape[-1] == 3:
            action_mask = (np.abs(actions[..., 1]) >  0.5) | (np.abs(actions[..., 0]) >  5) | (np.abs(actions[..., -1]) > 0.2)
            valid_masks[action_mask] = 0
        self.valid_masks = valid_masks.astype('bool')
        self.future_step = future_step
        print(self.obs.shape, self.actions.shape, self.valid_masks.shape)

        if partner_labels is not None:
            partner_labels_pad = np.zeros((partner_labels.shape[0], future_step, *partner_labels.shape[2:]), dtype=np.int16)
            self.partner_labels = np.concatenate([partner_labels, partner_labels_pad], axis=1)[:, future_step:]
        if ego_labels is not None:
            self.ego_labels = ego_labels

        if exp == 'other':
            partner_info = obs[..., 6:128 * 6].reshape(B, T, 127, 6)[..., :4]
            aux_info, aux_mask = self._make_aux_info(partner_mask, partner_info, future_timestep=future_step)
            self.aux_mask = aux_mask.astype('bool')

        if exp == 'ego':
            future_valid_mask_pad = np.zeros((self.valid_masks.shape[0], future_step, *self.valid_masks.shape[2:]), dtype=np.float32)
            future_valid_masks = np.concatenate([valid_masks, future_valid_mask_pad], axis=1).astype('bool')[:, future_step:]
            self.future_valid_mask = self.valid_masks & future_valid_masks
        
        if exp == 'other':
            current_relative_other_pos = self._transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step=future_step)
            current_relative_other_pos[aux_mask] = 0
            self.other_pos = self._get_multi_class_pos(current_relative_other_pos)
        else:
            current_relative_ego_pos = self._transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step=future_step)
            self.ego_pos = self._get_multi_class_pos(current_relative_ego_pos, xy_range)

        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'valid_masks']
        if exp == 'other':
            self.full_var += ['aux_mask', 'other_pos']
        else:
            self.full_var += ['future_valid_mask', 'ego_pos']
        if partner_labels is not None: self.full_var += ['partner_labels']
        if ego_labels is not None:     self.full_var += ['ego_labels']
        self.future_step = future_step

    def __len__(self): return len(self.valid_indices)

    def _make_aux_info(self, partner_mask, partner_info, future_timestep):
        partner_mask_bool = np.where(partner_mask == 0, 0, 1).astype(bool)
        partner_info_pad = np.zeros((partner_info.shape[0], future_timestep, *partner_info.shape[2:]), dtype=np.float32)
        partner_mask_pad = np.full((partner_mask.shape[0], future_timestep, *partner_mask.shape[2:]), 2, dtype=np.float32)
        future_mask = np.concatenate([partner_mask, partner_mask_pad], axis=1)
        future_mask_bool = np.where(future_mask == 0, 0, 1).astype(bool)[:, future_timestep:]
        partner_info = np.concatenate([partner_info, partner_info_pad], axis=1)[:, future_timestep:]
        combined_mask = np.logical_or(future_mask_bool, partner_mask_bool).astype('bool')
        return partner_info, combined_mask
    
    @staticmethod
    def _transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step):
        t_partner_pos = aux_info[..., 1:3] * MAX_REL_AGENT_POS
        t_ego_global_pos = np.zeros_like(ego_global_pos)
        t_ego_global_rot = np.zeros_like(ego_global_rot)
        t_ego_global_pos[:, :-future_step] = ego_global_pos[:, future_step:]
        t_ego_global_rot[:, :-future_step] = ego_global_rot[:, future_step:]
        t_partner_global_pos_x = t_ego_global_pos[..., 0, None] + t_partner_pos[..., 0] * np.cos(t_ego_global_rot) - t_partner_pos[..., 1] * np.sin(t_ego_global_rot)
        t_partner_global_pos_y = t_ego_global_pos[..., 1, None] + t_partner_pos[..., 0] * np.sin(t_ego_global_rot) + t_partner_pos[..., 1] * np.cos(t_ego_global_rot)
        delta_x = t_partner_global_pos_x - ego_global_pos[..., 0, None]
        delta_y = t_partner_global_pos_y - ego_global_pos[..., 1, None]
        cos_theta = np.cos(-ego_global_rot); sin_theta = np.sin(-ego_global_rot)
        current_relative_pos_x = delta_x * cos_theta + delta_y * sin_theta
        current_relative_pos_y = -delta_x * sin_theta + delta_y * cos_theta
        current_relative_pos_x = 2 * ((current_relative_pos_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((current_relative_pos_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        return np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)

    @staticmethod
    def _transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step):
        current_relative_pos = np.zeros_like(ego_global_pos)
        ego_current_pos = ego_global_pos[:, :-future_step]
        ego_future_pos = ego_global_pos[:, future_step:]
        delta_x = ego_future_pos[..., 0] - ego_current_pos[..., 0]
        delta_y = ego_future_pos[..., 1] - ego_current_pos[..., 1]
        ego_current_rot = ego_global_rot[:, :-future_step]
        cos_theta = np.cos(ego_current_rot); sin_theta = np.sin(ego_current_rot)
        rel_x = delta_x * cos_theta.squeeze(-1) + delta_y * sin_theta.squeeze(-1)
        rel_y = -delta_x * sin_theta.squeeze(-1) + delta_y * cos_theta.squeeze(-1)
        current_relative_pos_x = 2 * ((rel_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((rel_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos[:, :-future_step, :] = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
        return current_relative_pos

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T)
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))

    @staticmethod
    def _get_multi_class_pos(pos, xy_range=None):
        x, y = pos[..., 0], pos[..., 1]
        if xy_range is not None:
            xrange = xy_range[0]; yrange = xy_range[1]
            xbins = np.linspace(xrange[0], xrange[1], 9)
            ybins = np.linspace(yrange[0], yrange[1], 9)
        else:
            xbins = np.linspace(-0.05, 0.05, 9)
            ybins = np.linspace(-0.05, 0.05, 9)
        x_bins = np.digitize(x, xbins) - 1
        y_bins = np.digitize(y, ybins) - 1
        x_bins = np.clip(x_bins, 0, 7); y_bins = np.clip(y_bins, 0, 7)
        return x_bins * 8 + y_bins
        
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        idx1 = int(idx1); idx2 = int(idx2)
        batch = ()
        for var_name in self.full_var:
            if self.__dict__[var_name] is not None:
                if var_name in ['obs', 'actions', 'valid_masks', 'aux_mask', 'other_pos', 'future_valid_mask', 'ego_pos', 'partner_labels']:
                    data = self.__dict__[var_name][idx1, idx2]
                elif var_name == 'ego_labels':
                    data = self.__dict__[var_name][idx1]
                else:
                    raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                batch = batch + (data, )
                if var_name == 'obs':
                    if idx2 < 91 - self.future_step:
                        future_dist = self.__dict__[var_name][idx1, idx2 +  self.future_step][6: 128 * 6].reshape(127, -1)
                        future_dist = future_dist[:, 1:3]
                    else:
                        future_dist = np.zeros((127, 2)).astype(self.obs.dtype)
                    batch = batch + (future_dist, )
        return batch

# ------------------------------ Utils ------------------------------
def get_dataloader(data_path, data_file, config, isshuffle=True):
    with np.load(os.path.join(data_path, data_file)) as npz:
        ego_labels = None; partner_labels = None
        expert_obs = npz['obs']; expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        if config['exp'] == 'ego':
            ego_labels = npz['ego_labels'].astype('int') if 'ego_labels' in npz.keys() else None
        if config['exp'] == 'other':
            partner_labels = npz['partner_labels'].astype('int') if 'partner_labels' in npz.keys() else None
    with np.load(os.path.join(data_path, "global", "global_" + data_file)) as global_npz:
        ego_global_pos = global_npz['ego_global_pos']
        ego_global_rot = global_npz['ego_global_rot']
    dataset = FutureDataset(
        expert_obs, expert_actions, ego_global_pos, ego_global_rot, expert_masks, partner_mask,
        future_step=config['future_step'], exp=config['exp'], partner_labels=partner_labels, ego_labels=ego_labels
    )
    return DataLoader(dataset, batch_size=1024, shuffle=isshuffle, num_workers=8, prefetch_factor=4, pin_memory=True)


# ------------------------------ Evaluate & Plot ------------------------------
def evaluate(exp_config):
    # Backbone / LP head
    if exp_config["model"] == 'baseline':
        hidden_dim = 6 if exp_config.exp == 'ego' else 12 # ego info
        backbone = None
    else:
        # Load default configs
        config = load_config("baselines/ppo/config/ppo_base_puffer.yaml")
        params = torch.load(f"{exp_config['backbone_path']}", weights_only=False)
        backbone = NeuralNet(
            input_dim=64,
            action_dim=91,
            hidden_dim=128,
            config=config.environment,
        ).to("cuda")
        backbone.load_state_dict(params["parameters"])
        backbone.eval()
        hidden_dim = 64

    pos_linear_model = torch.load(exp_config["lp_path"], weights_only=False)
    raw_linear_model = torch.load(exp_config["raw_path"], weights_only=False)
    # Data
    eval_data_path = f'/data/after_cvpr/linear_probe_data/scene_{exp_config["num_scene"]}/'
    eval_data_file =  "validation_trajectory_2500.npz"
    loader = get_dataloader(eval_data_path, eval_data_file, exp_config, isshuffle=True)
    print(f'EXP CONFIG {exp_config}')

    per_batch = 100  # sampling while collecting
    all_true_probs, all_fdists, all_cdists = [], [], []
    total_pick = 0
    pos_linear_model.eval()
    raw_linear_model.eval()
    for batch in loader:
        obs, future_dist, _, valid_mask, future_mask, future_pos, labels = batch
        batch_size = batch[0].size(0)
        with torch.no_grad():
            obs = obs.to("cuda"); future_dist = future_dist.to("cuda")
            future_pos = future_pos.to("cuda")
            current_dist = obs[:, 6:128 * 6].reshape(-1, 127, 6)[..., 1:3]
            valid_mask = valid_mask.to("cuda"); future_mask = future_mask.to("cuda")
            labels = labels.to("cuda")

            if exp_config["exp"] == "other":
                partner_obs = obs[..., 6:6 * 128].reshape(batch_size, 127, 6)
                ego_obs = obs[..., :6].unsqueeze(1).repeat(1, 127, 1)
                raw_input = torch.cat([ego_obs, partner_obs], dim=-1).reshape(batch_size, 127, -1)
                with torch.no_grad():
                    lp_input =backbone.partner_embed(partner_obs)
            else:
                raw_input = obs[..., :6].reshape(-1, 6)
                with torch.no_grad():
                    lp_input = backbone.ego_embed(obs[..., :6])

            pred_pos = pos_linear_model(lp_input)
            raw_pos = raw_linear_model(raw_input)
            future_mask = ~future_mask if exp_config["exp"] == 'other' else future_mask
            masked_pos = pred_pos[future_mask]
            masked_raw = raw_pos[future_mask]
            masked_label = labels[future_mask]
            future_pos = future_pos.clone()
            masked_pos_label = future_pos[future_mask]
            if future_mask.sum() == 0: 
                continue

            pred_probs = torch.softmax(masked_pos, dim=-1)
            raw_probs = torch.softmax(masked_raw, dim=-1)
            probs = pred_probs - raw_probs
            true_prob = probs[torch.arange(probs.size(0), device=probs.device), masked_pos_label.long()]
            future_dists = torch.linalg.norm(future_dist[future_mask], dim=-1)
            curr_dists   = torch.linalg.norm(current_dist[future_mask], dim=-1)
            how_closer   = curr_dists - future_dists

            valid = torch.isfinite(how_closer) & torch.isfinite(true_prob)
            if valid.any():
                idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                num_pick = min(per_batch, idx.numel())
                pick = idx[torch.randperm(idx.numel(), device=idx.device)[:num_pick]]
                all_fdists.append(future_dists[pick].detach().cpu())
                all_true_probs.append(true_prob[pick].detach().cpu())
                all_cdists.append(curr_dists[pick].detach().cpu())
                total_pick += len(pick)

    # Gather arrays
    probs_all  = torch.cat(all_true_probs).numpy()
    fdists_all = torch.cat(all_fdists).numpy()
    cdists_all = torch.cat(all_cdists).numpy()

    # Relative change & masks (40%)
    eps = 1e-12
    rel_change_raw = (cdists_all - fdists_all) / (cdists_all + eps)  # 1 - future/current

    closer_mask  = (rel_change_raw >= +0.4)
    farther_mask = (rel_change_raw <= -0.4)

    rel_change_clr = np.clip(rel_change_raw, -1.0, 1.0)

    Y_LIM = (-0.2, 0.5)
    MAX_SAMPLES = 2000
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 9.0), sharex=False)

    panels = [
        (ax_top,  closer_mask,  f"Closer (Scene {exp_config['num_scene']})"),
        (ax_bot,  farther_mask, f"Farther (Scene {exp_config['num_scene']})"),
    ]

    norm = mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")

    def corr_linfit(x, y):
        if len(x) < 3:
            return np.nan, np.nan, np.nan, np.nan, (np.array([0, 1]), np.array([y.mean() if len(y)>0 else 0]*2))
        r, p = pearsonr(x, y)
        slope, intercept, *_ = linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 200)
        ys = slope * xs + intercept
        return r, p, slope, intercept, (xs, ys)

    for ax, msk, title in panels:
        x = fdists_all[msk]
        y = probs_all[msk]
        c = rel_change_clr[msk] 

        n = len(x)
        if n > 0:
            keep = np.random.permutation(n)[:min(MAX_SAMPLES, n)]
            x, y, c = x[keep], y[keep], c[keep]

        if len(x) > 0:
            ax.scatter(x, y, s=10, c=c, cmap=cmap, norm=norm, alpha=0.35, linewidths=0)
            r, p, slope, intercept, (xs, ys) = corr_linfit(x, y)
            ax.plot(xs, ys, color="#DE8F05", linewidth=2.0)
            stats_txt = f"Pearson r={r:.3f}\n y={slope:.3f}x+{intercept:.3f}"
        else:
            stats_txt = "n=0"

        ax.text(0.98, 0.98, stats_txt, transform=ax.transAxes,
                ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="#888", alpha=0.95, pad=0.35), linespacing=1.15)

        ax.set_title(f"{title}", pad=12)
        ax.set_ylabel("Prob. Difference \n(IL - Raw)")
        ax.set_ylim(*Y_LIM)
        ax.grid(True, linestyle="--", linewidth=0.6)
        for spine in ["top","right"]:
            ax.spines[spine].set_visible(False)

    ax_bot.set_xlabel("Future Distance")

    fig.subplots_adjust(right=0.8, hspace=0.28)

    cax = fig.add_axes([0.83, 0.12, 0.025, 0.76])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel("Relative distance change", rotation=90, labelpad=16)
    out_base = f"{exp_config['base_path']}corr_fig_{exp_config['num_scene']}_prob"
    plt.savefig(out_base + "_diff.pdf", dpi=300, bbox_inches="tight", pad_inches=0.1)
    print(f"[Saved] {out_base}.pdf")

# ------------------------------ Main ------------------------------
def set_seed(seed=42, deterministic=False):
    np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--exp', type=str, default='other', choices=['other', 'ego'])
    parser.add_argument('--model', type=str, default='lp', choices=['lp', 'baseline'])
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--num-scene', '-n', type=int, default=100)
    parser.add_argument('--future-step', '-f', type=int, default=10)
    args = parser.parse_args()

    mpl.rcParams.update({
        # 'font.family': 'Times New Roman',
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.titlesize": 34,
        "axes.labelsize": 28,
        "xtick.labelsize": 28,
        "ytick.labelsize": 28,
        "legend.fontsize": 24,
        "font.size": 24,
        "axes.linewidth": 0.8,
        "axes.titlepad": 10,
        "figure.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",
    })
    base_path = f'/data/after_cvpr/rl/scene_{args.num_scene}/'
    lp_base_path = os.path.join(base_path,  f'{args.exp}_linear_prob')
    backbone_name = os.listdir(lp_base_path)[-1]
    lp_path = os.path.join(lp_base_path, backbone_name, f'seed{args.seed}')
    backbone_path = f'{base_path}/{backbone_name}' if 'pt' in backbone_name else f'{base_path}/{backbone_name}.pt'
    raw_path = f'{lp_path}/pos_baseline_{args.future_step}.pth'
    lp_path = f'{lp_path}/pos_{args.model}_{args.future_step}.pth'

    set_seed(0)
    exp_config = {
        "lp_path": lp_path,
        "raw_path": raw_path,
        "backbone_path": backbone_path,
        "base_path": base_path,
        "model": args.model,
        "exp": args.exp,
        "future_step": args.future_step,
        "num_scene": args.num_scene,
        "batch_size": 2048,
        "num_workers": 8,
        "prefetch_factor": 4,
        "pin_memory": True,
    }
    evaluate(exp_config)
