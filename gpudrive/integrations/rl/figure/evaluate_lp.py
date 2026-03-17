import logging
import os
import sys

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, linregress
from box import Box
import yaml

torch.backends.cudnn.benchmark = True
sys.path.append(os.getcwd())

from gpudrive.integrations.rl.linear_probing.lp_model import LinearProbPosition
from gpudrive.integrations.rl.run_lp import get_dataloader, load_config, set_seed
from gpudrive.networks.late_fusion import NeuralNet

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _future_pos_to_dists(future_pos_np, num_bins=8):
    """Convert discrete future position labels (0..num_bins^2-1) to proxy distances.
    Uses same 8x8 bin centers in [-1, 1] as in linear_probing dataloader."""
    centers = np.linspace(-1.0, 1.0, num_bins + 1)
    centers = (centers[:-1] + centers[1:]) / 2
    k = np.clip(future_pos_np.astype(int), 0, num_bins * num_bins - 1)
    x = centers[k // num_bins]
    y = centers[k % num_bins]
    return np.sqrt(x * x + y * y)


def evaluate(exp_config, num_scene=None):
    """Run evaluation and save closer/farther prob-diff plot."""
    if exp_config.model == "baseline":
        backbone = None
    else:
        config = load_config("baselines/ppo/config/ppo_base_puffer.yaml")
        params = torch.load(
            f"{exp_config.model_path}/{exp_config.model_name}.pt",
            weights_only=False,
        )
        backbone = NeuralNet(
            input_dim=64,
            action_dim=91,
            hidden_dim=128,
            config=config.environment,
        ).to("cuda")
        backbone.load_state_dict(params["parameters"])
        backbone.eval()

    pos_linear_model = torch.load(exp_config.lp_path, weights_only=False)
    raw_linear_model = torch.load(exp_config.raw_path, weights_only=False)

    eval_data_path = os.path.join(exp_config.base_path, f"scene_{num_scene}")
    print(eval_data_path)
    eval_data_file = "validation_trajectory_2500.npz"
    loader = get_dataloader(
        eval_data_path, eval_data_file, exp_config, isshuffle=True
    )
    logger.info("EXP CONFIG %s", exp_config)

    per_batch = 100
    all_true_probs, all_fdists, all_cdists = [], [], []
    pos_linear_model.eval()
    raw_linear_model.eval()

    for batch in loader:
        obs, _, valid_mask, future_mask, future_pos, labels = batch
        batch_size = obs.size(0)
        with torch.no_grad():
            obs = obs.to("cuda")
            future_pos = future_pos.to("cuda")
            valid_mask = valid_mask.to("cuda")
            future_mask = future_mask.to("cuda")
            labels = labels.to("cuda")

            current_dist = (
                obs[:, 6 : 128 * 6]
                .reshape(-1, 127, 6)[..., 1:3]
                .float()
            )
            current_dists = torch.linalg.norm(current_dist, dim=-1)

            if exp_config.model == "baseline":
                if exp_config.exp == "other":
                    ego_obs = obs[:, :6].unsqueeze(1).repeat(1, 127, 1)
                    partner_obs = obs[:, 6 : 6 * 128].reshape(batch_size, 127, 6)
                    raw_input = torch.cat(
                        [ego_obs, partner_obs], dim=-1
                    )
                    lp_input = raw_input
                else:
                    lp_input = obs[:, :6]
                    raw_input = lp_input
            else:
                with torch.no_grad():
                    if exp_config.exp == "other":
                        partner_obs = obs[:, 6 : 6 * 128].view(
                            batch_size, 127, 6
                        )
                        lp_input = backbone.partner_embed(partner_obs)
                        ego_obs = obs[:, :6].unsqueeze(1).repeat(1, 127, 1)
                        raw_input = torch.cat(
                            [ego_obs, partner_obs], dim=-1
                        )
                    else:
                        lp_input = backbone.ego_embed(obs[:, :6])
                        raw_input = lp_input

            pred_pos = pos_linear_model(lp_input)
            raw_pos = raw_linear_model(raw_input)
            future_mask_eval = (
                ~future_mask if exp_config.exp == "other" else future_mask
            )
            masked_pos = pred_pos[future_mask_eval]
            masked_raw = raw_pos[future_mask_eval]
            masked_label = labels[future_mask_eval]
            masked_pos_label = future_pos[future_mask_eval]
            current_dists_masked = current_dists[future_mask_eval]

            if future_mask_eval.sum() == 0:
                continue

            # Future distance: from discrete labels (RL dataloader has no continuous future_dist)
            future_dists = torch.from_numpy(
                _future_pos_to_dists(
                    masked_pos_label.detach().cpu().numpy()
                )
            ).to(masked_pos.device, dtype=masked_pos.dtype)

            pred_probs = torch.softmax(masked_pos, dim=-1)
            raw_probs = torch.softmax(masked_raw, dim=-1)
            probs = pred_probs - raw_probs
            true_prob = probs[
                torch.arange(
                    probs.size(0), device=probs.device
                ),
                masked_pos_label.long(),
            ]
            how_closer = current_dists_masked - future_dists

            valid = torch.isfinite(how_closer) & torch.isfinite(true_prob)
            if valid.any():
                idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
                num_pick = min(per_batch, idx.numel())
                pick = idx[
                    torch.randperm(idx.numel(), device=idx.device)[:num_pick]
                ]
                all_fdists.append(future_dists[pick].detach().cpu())
                all_true_probs.append(true_prob[pick].detach().cpu())
                all_cdists.append(current_dists_masked[pick].detach().cpu())

    if not all_true_probs:
        logger.warning("No valid samples collected; skipping plot.")
        return

    probs_all = torch.cat(all_true_probs).numpy()
    fdists_all = torch.cat(all_fdists).numpy()
    cdists_all = torch.cat(all_cdists).numpy()

    eps = 1e-12
    rel_change_raw = (cdists_all - fdists_all) / (cdists_all + eps)
    closer_mask = rel_change_raw >= 0.4
    farther_mask = rel_change_raw <= -0.4
    rel_change_clr = np.clip(rel_change_raw, -1.0, 1.0)

    Y_LIM = (-0.2, 0.5)
    MAX_SAMPLES = 2000
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 9.0), sharex=False)
    panels = [
        (ax_top, closer_mask, f"Closer (Scene {exp_config.num_scene})"),
        (ax_bot, farther_mask, f"Farther (Scene {exp_config.num_scene})"),
    ]

    norm = mpl.colors.TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    cmap = mpl.colormaps["coolwarm"] if hasattr(mpl, "colormaps") else mpl.cm.get_cmap("coolwarm")

    def corr_linfit(x, y):
        if len(x) < 3:
            return (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                (np.array([0, 1]), np.array([y.mean() if len(y) > 0 else 0] * 2)),
            )
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
            keep = np.random.permutation(n)[: min(MAX_SAMPLES, n)]
            x, y, c = x[keep], y[keep], c[keep]
        if len(x) > 0:
            ax.scatter(
                x, y, s=10, c=c, cmap=cmap, norm=norm, alpha=0.35, linewidths=0
            )
            r, p, slope, intercept, (xs, ys) = corr_linfit(x, y)
            ax.plot(xs, ys, color="#DE8F05", linewidth=2.0)
            stats_txt = f"Pearson r={r:.3f}\n y={slope:.3f}x+{intercept:.3f}"
        else:
            stats_txt = "n=0"
        ax.text(
            0.98,
            0.98,
            stats_txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                edgecolor="#888",
                alpha=0.95,
                pad=0.35,
            ),
            linespacing=1.15,
        )
        ax.set_title(title, pad=12)
        ax.set_ylabel("Prob. Difference \n(IL - Raw)")
        ax.set_ylim(*Y_LIM)
        ax.grid(True, linestyle="--", linewidth=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    ax_bot.set_xlabel("Future Distance")
    fig.subplots_adjust(right=0.8, hspace=0.28)
    cax = fig.add_axes([0.83, 0.12, 0.025, 0.76])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.set_ylabel("Relative distance change", rotation=90, labelpad=16)
    out_base = f"{exp_config.model_path}_prob"
    plt.savefig(
        out_base + "_diff.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()
    logger.info("[Saved] %s_diff.pdf", out_base)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate linear probe and plot closer/farther prob difference"
    )
    parser.add_argument("--exp", type=str, default=None, choices=["other", "ego"])
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["early_lp", "final_lp", "baseline"],
    )
    parser.add_argument("--model-path", "-mp", type=str, default="scene_10000/model_PPO____S_200__03_04_04_06_56_997_007604")
    parser.add_argument("--seed", "-s", type=int, default=None)
    parser.add_argument("--num-scene", "-n", type=int, default=None)
    parser.add_argument("--future-step", "-f", type=int, default=10)
    parser.add_argument("--config", "-c", type=str, default="gpudrive/integrations/rl/lp.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        exp_config = Box(yaml.safe_load(f))
    if args.exp is not None:
        exp_config.exp = args.exp
    if args.model is not None:
        exp_config.model = args.model
    if args.model_path is not None:
        exp_config.model_path = args.model_path
    if args.seed is not None:
        exp_config.seed = args.seed
    if args.num_scene is not None:
        exp_config.num_scene = args.num_scene
    if args.future_step is not None:
        exp_config.future_step = args.future_step

    exp_path = exp_config.model_path
    lp_base_path = os.path.join(exp_path, f"{exp_config.exp}_linear_prob")
    if not os.path.isdir(lp_base_path):
        raise FileNotFoundError(f"Linear probe dir not found: {lp_base_path}")
    backbone_name = sorted(os.listdir(lp_base_path))[-1]
    exp_config.model_name = backbone_name
    lp_dir = os.path.join(lp_base_path, backbone_name, f"seed{exp_config.seed}")
    exp_config.raw_path = os.path.join(
        lp_dir, f"pos_baseline_{exp_config.future_step}.pth"
    )
    exp_config.lp_path = os.path.join(
        lp_dir, f"pos_{exp_config.model}_{exp_config.future_step}.pth"
    )

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
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
        }
    )
    set_seed(exp_config.get("seed", 42))
    evaluate(exp_config, num_scene=exp_config.num_scene)
