"""Measure driving performance on alone-scene dataset (simulation.py metrics).

Reads scenes from the folder produced by extract_alone_scenes.py
(default: /data/full_version/data/validation_alone) and reports
off-road / veh-collision / goal / goal-progress rates.

When --out-dir is set, also appends a Model row to
  {out_dir}/result_alone.csv
so run_simulate_alone_scenes.py can sweep a whole model folder.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv


def zero_partner_obs_il(obs, num_stack=5, ego_dim=6, partner_dim=6, n_partners=127):
    """Zero partner features and ego collision flag in every stack frame.

    Partners may still exist in the simulator; masking them in obs alone would
    leak collisions via ego ``is_collided`` (last of the 6 ego feats).
    """
    out = obs.clone()
    feat = int(out.shape[-1] / num_stack)
    partner_size = partner_dim * n_partners
    # ego: [speed, length, width, goal_x, goal_y, is_collided]
    ego_collided_idx = 5
    for s in range(num_stack):
        base = s * feat
        out[..., base + ego_collided_idx] = 0
        out[..., base + ego_dim : base + ego_dim + partner_size] = 0
    return out


def run_batch(env, bc_policy, mask_partners=True, num_stack=5):
    """Roll out one batch; return per-world metrics (simulation.py style)."""
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    batch_size = alive_agent_mask.shape[0]
    n_ctrl = int(alive_agent_mask.sum().item())

    obs_stack1_feat_size = int(obs.shape[-1] / num_stack)
    poss = obs[alive_agent_mask][
        :, obs_stack1_feat_size * (num_stack - 1) + 3 : obs_stack1_feat_size * (num_stack - 1) + 5
    ]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(alive_agent_mask, dtype=torch.float32)

    infos = env.get_infos()
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    goal_achieved_ep = infos.goal_achieved[alive_agent_mask]
    # Absolute timestep when goal is first reached; -1 if never.
    goal_timesteps = torch.full((n_ctrl,), -1.0, dtype=torch.float32, device="cuda")

    for time_step in tqdm(range(env.episode_len), leave=False):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3, device="cuda")

        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")
        # partner_mask: 0=visible, 1=static, 2=non-exist. True in bool mask = pad/ignore.
        if mask_partners:
            partner_mask_bool = torch.ones_like(partner_mask, dtype=torch.bool)
        else:
            partner_mask_bool = partner_mask == 2
        poss = obs[alive_agent_mask][
            :,
            obs_stack1_feat_size * (num_stack - 1) + 3 : obs_stack1_feat_size * (num_stack - 1)
            + 5,
        ]
        dist = torch.linalg.norm(poss, dim=-1)
        dist_metrics[alive_agent_mask] = dist

        goal_now = infos.goal_achieved[alive_agent_mask]
        goal_mask = (goal_now > 0) & (goal_timesteps < 0)
        goal_timesteps[goal_mask] = float(time_step)

        all_masks = [
            partner_mask_bool[~dead_agent_mask].unsqueeze(1),
            road_mask[~dead_agent_mask].unsqueeze(1),
        ]
        with torch.no_grad():
            alive_obs = obs[~dead_agent_mask]
            if mask_partners:
                alive_obs = zero_partner_obs_il(alive_obs, num_stack=num_stack)
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

    goal_progress = dist_metrics[alive_agent_mask] / init_goal_dist.clamp(min=1e-6)
    goal_progress[goal_achieved_ep.bool()] = 0
    goal_progress = (1 - goal_progress).clamp(0, 1)

    off_road = torch.full((batch_size,), float("nan"), device="cuda")
    veh_coll = torch.full((batch_size,), float("nan"), device="cuda")
    goal = torch.full((batch_size,), float("nan"), device="cuda")
    gp = torch.full((batch_size,), float("nan"), device="cuda")
    goal_time = torch.full((batch_size,), float("nan"), device="cuda")
    ctrl_worlds = torch.where(alive_agent_mask)[0]
    off_road[ctrl_worlds] = off_road_ep.float()
    veh_coll[ctrl_worlds] = veh_collision_ep.float()
    goal[ctrl_worlds] = goal_achieved_ep.float()
    gp[ctrl_worlds] = goal_progress.float()
    goal_time[ctrl_worlds] = goal_timesteps

    return {
        "OffRoad": off_road.cpu().numpy(),
        "VehCollision": veh_coll.cpu().numpy(),
        "Collision": (off_road + veh_coll).cpu().numpy(),
        "Goal": goal.cpu().numpy(),
        "GoalProgress": gp.cpu().numpy(),
        "GoalTime": goal_time.cpu().numpy(),
        "has_controlled": alive_agent_mask.any(dim=-1).cpu().numpy().astype(int),
    }


def parse_args():
    p = argparse.ArgumentParser("Alone-scene driving performance")
    p.add_argument(
        "--data-dir",
        type=str,
        default="/data/full_version/data/validation_alone",
        help="Alone scene folder from extract_alone_scenes.py",
    )
    p.add_argument(
        "--alone-csv",
        type=str,
        default="/data/full_version/data/validation_alone/alone_scenes.csv",
        help="Optional metadata csv (copied==1 rows used for scene_idx labels)",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-stack", type=int, default=5)
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
        "--mask-partners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pad partners + zero partner obs and ego is_collided (default: on)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/after_cvpr/images/alone_driving",
        help="Directory for per-scene CSVs and appended result_alone.csv",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Override per-scene CSV path (default: {out_dir}/{model}_per_scene.csv)",
    )
    p.add_argument(
        "--out-summary",
        type=str,
        default=None,
        help="Override single-model summary path (default: {out_dir}/{model}_summary.csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"data dir not found: {args.data_dir}")

    scene_files = sorted(
        f for f in os.listdir(args.data_dir) if f.startswith("tfrecord") and f.endswith(".json")
    )
    dataset_size = len(scene_files)
    if dataset_size == 0:
        raise SystemExit(f"no tfrecord*.json in {args.data_dir}")

    batch_size = min(args.batch_size, dataset_size)
    while dataset_size % batch_size != 0 and batch_size > 1:
        batch_size -= 1

    model_stem = args.model_name.replace(".pth", "")
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = args.out_csv or os.path.join(args.out_dir, f"{model_stem}_per_scene.csv")
    out_summary = args.out_summary or os.path.join(
        args.out_dir, f"{model_stem}_summary.csv"
    )
    result_csv = os.path.join(args.out_dir, "result_alone.csv")

    print(f"alone scenes: {dataset_size} from {args.data_dir} (batch_size={batch_size})")
    print(f"model: {args.model_path}/{args.model_name}")
    print(f"mask_partners: {args.mask_partners}")

    loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=batch_size,
        dataset_size=dataset_size,
        sample_with_replacement=False,
        shuffle=False,
    )
    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior="ignore",
        num_stack=args.num_stack,
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

    bc_policy = torch.load(
        os.path.join(args.model_path, args.model_name), weights_only=False
    ).to("cuda")
    bc_policy.eval()

    fname_to_meta = {}
    if os.path.exists(args.alone_csv):
        meta = pd.read_csv(args.alone_csv)
        if "copied" in meta.columns:
            meta = meta[meta["copied"] == 1]
        for _, r in meta.iterrows():
            fname_to_meta[r["filename"]] = r

    num_iter = max(1, dataset_size // batch_size)
    rows = []
    global_i = 0

    for bi in tqdm(range(num_iter), desc="batches"):
        metrics = run_batch(
            env, bc_policy, mask_partners=args.mask_partners, num_stack=args.num_stack
        )
        batch_files = [os.path.basename(p) for p in env.data_batch]
        for j, fname in enumerate(batch_files):
            if global_i >= dataset_size:
                break
            meta = fname_to_meta.get(fname, {})
            rows.append(
                {
                    "Model": args.model_name,
                    "local_idx": global_i,
                    "filename": fname,
                    "scene_idx": int(meta["scene_idx"]) if "scene_idx" in meta else global_i,
                    "always_alone": int(meta.get("always_alone", 1)),
                    "has_controlled": int(metrics["has_controlled"][j]),
                    "OffRoad": float(metrics["OffRoad"][j]),
                    "VehCollision": float(metrics["VehCollision"][j]),
                    "Collision": float(metrics["Collision"][j]),
                    "Goal": float(metrics["Goal"][j]),
                    "GoalProgress": float(metrics["GoalProgress"][j]),
                    "GoalTime": float(metrics["GoalTime"][j]),
                }
            )
            global_i += 1
        if bi != num_iter - 1:
            env.swap_data_batch()
            if args.partner_portion_test:
                env.remove_agents_by_id(
                    args.partner_portion_test, remove_controlled_agents=True
                )

    env.close()
    per_scene = pd.DataFrame(rows)
    valid = per_scene[per_scene["has_controlled"] == 1].copy()
    valid.to_csv(out_csv, index=False)

    goal_times = valid.loc[valid["Goal"] > 0, "GoalTime"]
    goal_time_avg = float(goal_times.mean()) if len(goal_times) > 0 else float("nan")
    summary_row = {
        "Model": args.model_name,
        "Dataset": "validation_alone",
        "Num": len(valid),
        "OffRoad": float(valid["OffRoad"].mean()),
        "VehCollision": float(valid["VehCollision"].mean()),
        "Collision": float(valid["Collision"].mean()),
        "Goal": float(valid["Goal"].mean()),
        "GoalProgress": float(valid["GoalProgress"].mean()),
        "GoalTime": goal_time_avg,
    }
    summary = pd.DataFrame([summary_row])
    summary.to_csv(out_summary, index=False)

    # Append to sweep result CSV (run_simulation.py style)
    write_header = (not os.path.exists(result_csv)) or (os.path.getsize(result_csv) == 0)
    summary.to_csv(result_csv, mode="a", header=write_header, index=False)

    print("=== alone-scene driving performance ===")
    print(summary.to_string(index=False))
    print(f"\nwrote per-scene: {out_csv}")
    print(f"wrote summary:   {out_summary}")
    print(f"appended:        {result_csv}")


if __name__ == "__main__":
    main()
