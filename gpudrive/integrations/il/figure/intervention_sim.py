"""Evaluate driving performance on intervention CSV scenes (i/r by default).

Uses the same rollout metrics as baselines/il/test/simulation.py
(off-road, veh collision, goal, goal progress), restricted to
adaptiveness / recovery scenes from intervention.csv.

Writes a per-scene rollout CSV that can be fed to intervention_metrics.py
via --rollout-csv.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.getcwd())

import logging
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.integrations.il.figure.intervention_metrics import (
    attach_rollout_metrics,
    category_summary,
    load_and_normalize,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_batch(env, bc_policy, batch_size: int):
    """Roll out one batch; return per-world driving metrics (simulation.py style)."""
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()

    expert_actions, _, _, _, _ = env.get_expert_actions()
    obs_stack1_feat_size = int(obs.shape[-1] / 5)
    poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3 : obs_stack1_feat_size * 4 + 5]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(alive_agent_mask, dtype=torch.float32)

    infos = env.get_infos()
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    goal_achieved_ep = infos.goal_achieved[alive_agent_mask]

    for time_step in tqdm(range(env.episode_len), leave=False):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3, device="cuda")

        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        partner_mask_bool = partner_mask == 2
        poss = obs[alive_agent_mask][:, obs_stack1_feat_size * 4 + 3 : obs_stack1_feat_size * 4 + 5]
        dist = torch.linalg.norm(poss, dim=-1)
        dist_metrics[alive_agent_mask] = dist

        all_masks = [
            partner_mask_bool[~dead_agent_mask].unsqueeze(1),
            road_mask[~dead_agent_mask].unsqueeze(1),
        ]
        with torch.no_grad():
            alive_obs = obs[~dead_agent_mask]
            context, *_ = (lambda *a: (a[0], a[-2], a[-1]))(
                *bc_policy.get_context(alive_obs, all_masks)
            )
            actions = bc_policy.get_action(context, deterministic=True).squeeze(1)
        all_actions[~dead_agent_mask, :] = actions

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
        if dead_agent_mask.all():
            break

    # Per controlled agent (== per world when max_cont_agents=1)
    goal_progress = dist_metrics[alive_agent_mask] / init_goal_dist.clamp(min=1e-6)
    goal_progress[goal_achieved_ep.bool()] = 0
    goal_progress = (1 - goal_progress).clamp(0, 1)

    off_road = off_road_ep.float().cpu().numpy()
    veh_coll = veh_collision_ep.float().cpu().numpy()
    goal = goal_achieved_ep.float().cpu().numpy()
    gp = goal_progress.float().cpu().numpy()
    collision = off_road + veh_coll

    # Pad if some worlds had no controlled agent
    n = batch_size
    def _pad(x):
        out = np.full(n, np.nan, dtype=float)
        out[: len(x)] = x
        return out

    return {
        "off_road": _pad(off_road),
        "veh_collision": _pad(veh_coll),
        "goal": _pad(goal),
        "collision": _pad(collision),
        "goal_progress": _pad(gp),
    }


def parse_args():
    p = argparse.ArgumentParser("Intervention scene driving-performance sim")
    p.add_argument("--dataset", "-d", type=str, default="validation", choices=["training", "validation"])
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument(
        "--csv-path",
        "-cp",
        type=str,
        default="/data/full_version/intervention.csv",
    )
    p.add_argument("--start-idx", "-s", type=int, default=0)
    p.add_argument("--num-scenes", "-n", type=int, default=None)
    p.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=["adaptiveness", "recovery"],
        help="Which CSV categories to evaluate (default: i and r only)",
    )
    p.add_argument(
        "--model-path",
        "-mp",
        type=str,
        default="/data/full_version/model/exp_80000_subset_aix",
    )
    p.add_argument(
        "--model-name",
        "-mn",
        type=str,
        default="early_attn_s3_0908_113203.pth",
    )
    p.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    p.add_argument(
        "--out-csv",
        type=str,
        default="/data/after_cvpr/images/intervention_driving_per_scene.csv",
    )
    p.add_argument(
        "--out-summary",
        type=str,
        default="/data/after_cvpr/images/intervention_driving_summary.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    label_df = load_and_normalize(args.csv_path, args.start_idx, args.num_scenes)
    eval_df = label_df[label_df["category"].isin(args.categories)].copy()
    scene_ids = eval_df["scene_idx"].astype(int).tolist()
    if not scene_ids:
        raise SystemExit(f"No scenes for categories={args.categories}")

    print(
        f"Evaluating {len(scene_ids)} scenes "
        f"({eval_df['category'].value_counts().to_dict()}) "
        f"with model {args.model_path}/{args.model_name}"
    )

    data_root = f"/data/full_version/data/{args.dataset}/"
    max_scene = max(scene_ids) + 1
    batch_size = min(args.batch_size, len(scene_ids))

    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior="ignore",
        num_stack=5,
    )

    model_path = os.path.join(args.model_path, args.model_name)
    print(f"model: {model_path}")
    bc_policy = torch.load(model_path, weights_only=False).to("cuda")
    bc_policy.eval()

    # Pad to a multiple of batch_size; one env + swap_data_batch (like simulation.py)
    pad_n = (batch_size - len(scene_ids) % batch_size) % batch_size
    padded = scene_ids + [scene_ids[-1]] * pad_n
    valid = [True] * len(scene_ids) + [False] * pad_n
    num_iter = len(padded) // batch_size

    loader = SceneDataLoader(
        root=data_root,
        batch_size=batch_size,
        dataset_size=max_scene,
        sample_with_replacement=False,
        shuffle=False,
        scene_nums=padded,
    )
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=loader,
        max_cont_agents=1,
        device="cuda",
        render_config=RenderConfig(),
        action_type="continuous",
    )
    if args.partner_portion_test:
        env.remove_agents_by_id(
            args.partner_portion_test, remove_controlled_agents=True
        )

    rows = []
    cat_map = eval_df.set_index("scene_idx")["category"].to_dict()
    for bi in range(num_iter):
        metrics = run_batch(env, bc_policy, batch_size=batch_size)
        for j in range(batch_size):
            flat = bi * batch_size + j
            if not valid[flat]:
                continue
            sid = padded[flat]
            rows.append(
                {
                    "scene_idx": sid,
                    "category": cat_map[sid],
                    "off_road": metrics["off_road"][j],
                    "veh_collision": metrics["veh_collision"][j],
                    "collision": metrics["collision"][j],
                    "goal": metrics["goal"][j],
                    "goal_progress": metrics["goal_progress"][j],
                    "collision_orig": metrics["collision"][j],
                    "goal_progress_orig": metrics["goal_progress"][j],
                    "collision_intervened": np.nan,
                    "goal_progress_intervened": np.nan,
                }
            )
        if bi != num_iter - 1:
            env.swap_data_batch()
            if args.partner_portion_test:
                env.remove_agents_by_id(
                    args.partner_portion_test, remove_controlled_agents=True
                )
    env.close()

    per_scene = pd.DataFrame(rows).sort_values("scene_idx").reset_index(drop=True)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    per_scene.to_csv(args.out_csv, index=False)

    # Category-wise driving summary (simulation.py style rates)
    summary_rows = []
    for cat, g in per_scene.groupby("category"):
        summary_rows.append(
            {
                "category": cat,
                "n": len(g),
                "off_road_rate": float(g["off_road"].mean()),
                "veh_collision_rate": float(g["veh_collision"].mean()),
                "collision_rate": float(g["collision"].mean()),
                "goal_rate": float(g["goal"].mean()),
                "goal_progress_mean": float(g["goal_progress"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_summary, index=False)

    print("=== driving performance by category ===")
    print(summary.to_string(index=False))
    print(f"\nwrote per-scene: {args.out_csv}")
    print(f"wrote summary:   {args.out_summary}")

    # Merge into label metrics (CAG / recovery stay NaN until intervened cols filled)
    merged = attach_rollout_metrics(label_df, per_scene)
    print("\n=== intervention label summary (with driving cols attached) ===")
    print(category_summary(merged).to_string(index=False))


if __name__ == "__main__":
    main()
