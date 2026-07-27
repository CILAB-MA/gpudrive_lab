"""Spurious-correlation probe: *delete* far partners, then roll out IL policy.

Uses ``env.remove_agents_by_distance(..., far_thresh_m=...)`` to physically
remove uncontrolled agents farther than ``--far-thresh`` meters from the
nearest controlled ego (global xy). Near partners stay in the sim.

Default: validation, 2000 scenes (trend check), ``max_cont_agents=1``, ``pp=0.0``.
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


def partner_counts(env, alive_agent_mask: torch.Tensor):
    """Mean existing partner count per controlled agent (after far deletion)."""
    partner_mask = env.get_partner_mask().to(alive_agent_mask.device)
    exists = partner_mask != 2
    alive = alive_agent_mask
    if not alive.any():
        return 0.0
    return float(exists[alive].float().sum().item() / alive.sum().item())


def run_batch(env, bc_policy, num_stack: int = 5):
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    batch_size = alive_agent_mask.shape[0]
    n_ctrl = int(alive_agent_mask.sum().item())

    mean_near = partner_counts(env, alive_agent_mask)

    feat = int(obs.shape[-1] / num_stack)
    poss = obs[alive_agent_mask][
        :, feat * (num_stack - 1) + 3 : feat * (num_stack - 1) + 5
    ]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(alive_agent_mask, dtype=torch.float32)

    infos = env.get_infos()
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    goal_achieved_ep = infos.goal_achieved[alive_agent_mask]
    goal_timesteps = torch.full((n_ctrl,), -1.0, dtype=torch.float32, device="cuda")

    for time_step in tqdm(range(env.episode_len), leave=False):
        all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3, device="cuda")
        road_mask = env.get_road_mask().to("cuda")
        partner_mask = env.get_partner_mask().to("cuda")

        poss = obs[alive_agent_mask][
            :, feat * (num_stack - 1) + 3 : feat * (num_stack - 1) + 5
        ]
        dist_metrics[alive_agent_mask] = torch.linalg.norm(poss, dim=-1)

        goal_now = infos.goal_achieved[alive_agent_mask]
        goal_hit = (goal_now > 0) & (goal_timesteps < 0)
        goal_timesteps[goal_hit] = float(time_step)

        all_masks = [
            (partner_mask == 2)[~dead_agent_mask].unsqueeze(1),
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
        "mean_near_per_ctrl": mean_near,
    }


def apply_distance_delete(
    env,
    *,
    far_thresh_m: float | None = None,
    remove_perc: float = 0.0,
    nearest_first: bool = False,
) -> int:
    """Delete uncontrolled partners by distance (far thresh or perc order)."""
    return env.remove_agents_by_distance(
        perc_to_rmv_per_scene=remove_perc,
        remove_controlled_agents=False,
        far_thresh_m=far_thresh_m,
        nearest_first=nearest_first,
    )


def parse_args():
    p = argparse.ArgumentParser("IL partner deletion by distance")
    p.add_argument("--dataset", "-d", type=str, default="validation", choices=["training", "validation"])
    p.add_argument("--dataset-size", type=int, default=2000, help="Cap scenes (default: 2000 for trend)")
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--num-stack", type=int, default=5)
    p.add_argument(
        "--far-thresh",
        type=float,
        default=None,
        help="If set (and not --nearest-first): delete partners farther than this (m).",
    )
    p.add_argument(
        "--remove-perc",
        type=float,
        default=0.2,
        help="Fraction of uncontrolled partners to remove when using perc mode.",
    )
    p.add_argument(
        "--nearest-first",
        action="store_true",
        help="Remove nearest partners first (perc mode). Default far-first if perc without this.",
    )
    p.add_argument(
        "--model-path",
        "-mp",
        type=str,
        default="/data/full_version/model/exp_100",
    )
    p.add_argument(
        "--model-name",
        "-mn",
        type=str,
        default="early_attn_s3_0901_064245.pth",
    )
    p.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    p.add_argument("--sim-agent", "-sa", type=str, default="log_replay",
                   choices=["log_replay", "self_play", "delta_replay"])
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/after_cvpr/images/mask_far_partners_il",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data_root = f"/data/full_version/data/{args.dataset}/"
    if not os.path.isdir(data_root):
        raise SystemExit(f"data dir not found: {data_root}")

    full_size = len(
        [f for f in os.listdir(data_root) if f.startswith("tfrecord") and f.endswith(".json")]
    )
    dataset_size = min(args.dataset_size, full_size)
    batch_size = min(args.batch_size, dataset_size)
    dataset_size = (dataset_size // batch_size) * batch_size
    if dataset_size == 0:
        raise SystemExit(f"dataset_size too small for batch_size={batch_size}")

    # Mode: nearest-first perc | far thresh | farthest perc
    if args.nearest_first:
        far_thresh_m = None
        remove_perc = args.remove_perc
        nearest_first = True
        tag = f"near{int(round(remove_perc * 100))}"
        mode_desc = f"nearest-first perc={remove_perc}"
    elif args.far_thresh is not None:
        far_thresh_m = args.far_thresh
        remove_perc = 0.0
        nearest_first = False
        tag = f"far{args.far_thresh:g}"
        mode_desc = f"far_thresh={args.far_thresh}m"
    elif args.remove_perc <= 0.0:
        far_thresh_m = None
        remove_perc = 0.0
        nearest_first = False
        tag = "normal"
        mode_desc = "no partner deletion (matched normal)"
    else:
        far_thresh_m = None
        remove_perc = args.remove_perc
        nearest_first = False
        tag = f"farperc{int(round(remove_perc * 100))}"
        mode_desc = f"farthest-first perc={remove_perc}"

    os.makedirs(args.out_dir, exist_ok=True)
    model_stem = args.model_name.replace(".pth", "")
    out_csv = os.path.join(args.out_dir, f"{model_stem}_{tag}_per_scene.csv")
    out_summary = os.path.join(args.out_dir, f"{model_stem}_{tag}_summary.csv")
    result_csv = os.path.join(args.out_dir, f"result_{tag}.csv")

    print(f"dataset: {args.dataset} size={dataset_size} batch={batch_size}")
    print(f"model: {args.model_path}/{args.model_name}")
    print(
        f"delete mode: {mode_desc} | sim_agent={args.sim_agent} pp={args.partner_portion_test}"
    )

    num_cont = 1 if args.sim_agent == "log_replay" else 128
    loader = SceneDataLoader(
        root=data_root,
        batch_size=batch_size,
        dataset_size=dataset_size,
        sample_with_replacement=False,
        shuffle=False,
    )
    env = GPUDriveTorchEnv(
        config=EnvConfig(
            dynamics_model="delta_local",
            dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
            dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
            dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
            collision_behavior="ignore",
            num_stack=args.num_stack,
        ),
        data_loader=loader,
        max_cont_agents=num_cont,
        device="cuda",
        render_config=RenderConfig(),
        action_type="continuous",
    )
    remove_controlled = args.sim_agent != "log_replay"
    if args.partner_portion_test:
        env.remove_agents_by_id(
            args.partner_portion_test, remove_controlled_agents=remove_controlled
        )
    n_del = apply_distance_delete(
        env,
        far_thresh_m=far_thresh_m,
        remove_perc=remove_perc,
        nearest_first=nearest_first,
    )
    print(f"deleted partners (first batch): {n_del}")

    bc_policy = torch.load(
        os.path.join(args.model_path, args.model_name), weights_only=False
    ).to("cuda")
    bc_policy.eval()

    num_iter = max(1, dataset_size // batch_size)
    rows = []
    remain_stats = []
    del_stats = [n_del]
    for bi in tqdm(range(num_iter), desc="batches"):
        metrics = run_batch(env, bc_policy, num_stack=args.num_stack)
        remain_stats.append(metrics["mean_near_per_ctrl"])
        for j in range(batch_size):
            if not metrics["has_controlled"][j]:
                continue
            rows.append(
                {
                    "Model": args.model_name,
                    "batch": bi,
                    "world": j,
                    "tag": tag,
                    "remove_perc": remove_perc,
                    "far_thresh_m": far_thresh_m if far_thresh_m is not None else float("nan"),
                    "nearest_first": int(nearest_first),
                    "OffRoad": float(metrics["OffRoad"][j]),
                    "VehCollision": float(metrics["VehCollision"][j]),
                    "Collision": float(metrics["Collision"][j]),
                    "Goal": float(metrics["Goal"][j]),
                    "GoalProgress": float(metrics["GoalProgress"][j]),
                    "GoalTime": float(metrics["GoalTime"][j]),
                }
            )
        if bi != num_iter - 1:
            env.swap_data_batch()
            if args.partner_portion_test:
                env.remove_agents_by_id(
                    args.partner_portion_test, remove_controlled_agents=remove_controlled
                )
            del_stats.append(
                apply_distance_delete(
                    env,
                    far_thresh_m=far_thresh_m,
                    remove_perc=remove_perc,
                    nearest_first=nearest_first,
                )
            )
    env.close()

    per_scene = pd.DataFrame(rows)
    per_scene.to_csv(out_csv, index=False)
    goal_times = per_scene.loc[per_scene["Goal"] > 0, "GoalTime"]
    n_worlds = batch_size
    summary = pd.DataFrame(
        [
            {
                "Model": args.model_name,
                "Dataset": args.dataset,
                "tag": tag,
                "remove_perc": remove_perc,
                "far_thresh_m": far_thresh_m if far_thresh_m is not None else float("nan"),
                "nearest_first": int(nearest_first),
                "Num": len(per_scene),
                "OffRoad": float(per_scene["OffRoad"].mean()),
                "VehCollision": float(per_scene["VehCollision"].mean()),
                "Collision": float(per_scene["Collision"].mean()),
                "Goal": float(per_scene["Goal"].mean()),
                "GoalProgress": float(per_scene["GoalProgress"].mean()),
                "GoalTime": float(goal_times.mean()) if len(goal_times) else float("nan"),
                "mean_deleted_partners": float(np.mean(del_stats) / max(n_worlds, 1)),
                "mean_remain_partners": float(np.mean(remain_stats)),
            }
        ]
    )
    summary.to_csv(out_summary, index=False)
    write_header = (not os.path.exists(result_csv)) or (os.path.getsize(result_csv) == 0)
    summary.to_csv(result_csv, mode="a", header=write_header, index=False)

    print("=== IL partner deletion (remove_agents_by_distance) ===")
    print(summary.to_string(index=False))
    print(f"\nwrote per-scene: {out_csv}")
    print(f"wrote summary:   {out_summary}")
    print(f"appended:        {result_csv}")


if __name__ == "__main__":
    main()
