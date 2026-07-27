"""Spurious-correlation probe: *delete* far partners, then roll out RL policy.

Uses ``env.remove_agents_by_distance(..., far_thresh_m=...)`` to physically
remove uncontrolled agents farther than ``--far-thresh`` meters from the
nearest controlled ego. Near partners stay in the sim.

Default: validation, 2000 scenes (trend check), ``pp=0.0``,
``max_cont_agents=1`` (so far partners are uncontrolled and deletable).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import pufferlib
import torch
import yaml
from box import Box
from tqdm import tqdm

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.networks.late_fusion import NeuralNet


def load_config(config_path):
    with open(config_path, "r") as f:
        config = Box(yaml.safe_load(f))
    return pufferlib.namespace(**config)


def partner_counts(env, alive_agent_mask: torch.Tensor):
    partner_mask = env.get_partner_mask().to(alive_agent_mask.device)
    exists = partner_mask != 2
    alive = alive_agent_mask
    if not alive.any():
        return 0.0
    return float(exists[alive].float().sum().item() / alive.sum().item())


def run_batch(env, policy):
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    obs = env.reset()
    batch_size = alive_agent_mask.shape[0]
    n_ctrl = int(alive_agent_mask.sum().item())

    mean_near = partner_counts(env, alive_agent_mask)

    poss = obs[alive_agent_mask][:, 3:5]
    init_goal_dist = torch.linalg.norm(poss, dim=-1)
    dist_metrics = torch.zeros_like(alive_agent_mask, dtype=torch.float32)

    infos = env.get_infos()
    off_road_ep = infos.off_road[alive_agent_mask]
    veh_collision_ep = infos.collided[alive_agent_mask]
    goal_achieved_ep = infos.goal_achieved[alive_agent_mask]
    goal_timesteps = torch.full((n_ctrl,), -1.0, dtype=torch.float32, device="cuda")

    all_actions = torch.zeros(obs.shape[0], obs.shape[1], device="cuda").long()

    for time_step in tqdm(range(env.episode_len), leave=False):
        poss = obs[..., 3:5]
        dist_metrics[alive_agent_mask] = torch.linalg.norm(poss, dim=-1)[alive_agent_mask]

        goal_achieved = infos.goal_achieved[alive_agent_mask]
        goal_hit = (goal_achieved > 0) & (goal_timesteps < 0)
        goal_timesteps[goal_hit] = float(time_step)

        with torch.no_grad():
            alive_obs = obs[~dead_agent_mask]
            actions, *_ = policy(alive_obs, deterministic=True)
        all_actions[~dead_agent_mask] = actions

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
    return env.remove_agents_by_distance(
        perc_to_rmv_per_scene=remove_perc,
        remove_controlled_agents=False,
        far_thresh_m=far_thresh_m,
        nearest_first=nearest_first,
    )


def parse_args():
    p = argparse.ArgumentParser("RL partner deletion by distance")
    p.add_argument("--dataset", "-d", type=str, default="validation", choices=["training", "validation"])
    p.add_argument("--dataset-size", type=int, default=2000, help="Cap scenes (default: 2000 for trend)")
    p.add_argument("--batch-size", type=int, default=50)
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
        help="Remove nearest partners first (perc mode).",
    )
    p.add_argument(
        "--model-path",
        "-mp",
        type=str,
        default="/data/after_cvpr/rl/scene_100_v2",
    )
    p.add_argument(
        "--model-name",
        "-mn",
        type=str,
        default="model_PPO____S_100__04_03_16_56_22_529_001519.pt",
    )
    p.add_argument(
        "--config-path",
        type=str,
        default="baselines/ppo/config/ppo_base_puffer.yaml",
    )
    p.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/after_cvpr/images/mask_far_partners_rl",
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
    else:
        far_thresh_m = None
        remove_perc = args.remove_perc
        nearest_first = False
        tag = f"farperc{int(round(remove_perc * 100))}"
        mode_desc = f"farthest-first perc={remove_perc}"

    os.makedirs(args.out_dir, exist_ok=True)
    model_stem = os.path.splitext(args.model_name)[0]
    out_csv = os.path.join(args.out_dir, f"{model_stem}_{tag}_per_scene.csv")
    out_summary = os.path.join(args.out_dir, f"{model_stem}_{tag}_summary.csv")
    result_csv = os.path.join(args.out_dir, f"result_{tag}.csv")

    num_cont = 1
    print(f"dataset: {args.dataset} size={dataset_size} batch={batch_size}")
    print(f"model: {args.model_path}/{args.model_name}")
    print(
        f"delete mode: {mode_desc} | pp={args.partner_portion_test} max_cont={num_cont}"
    )

    loader = SceneDataLoader(
        root=data_root,
        batch_size=batch_size,
        dataset_size=dataset_size,
        sample_with_replacement=False,
        shuffle=False,
    )
    env = GPUDriveTorchEnv(
        config=EnvConfig(
            dynamics_model="classic",
            collision_behavior="ignore",
            steer_actions=torch.round(torch.linspace(-torch.pi, torch.pi, 13), decimals=3),
            accel_actions=torch.round(torch.linspace(-4.0, 4.0, 7), decimals=3),
            num_stack=1,
        ),
        data_loader=loader,
        max_cont_agents=num_cont,
        device="cuda",
        render_config=RenderConfig(),
        action_type="discrete",
    )
    remove_controlled = args.partner_portion_test != 1.0
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

    config = load_config(args.config_path)
    params = torch.load(
        os.path.join(args.model_path, args.model_name), weights_only=False
    )
    sd = params["parameters"]
    input_dim = int(sd["ego_embed.0.weight"].shape[0])
    hidden_dim = int(sd["shared_embed.0.weight"].shape[0])
    policy = NeuralNet(
        input_dim=input_dim,
        action_dim=91,
        hidden_dim=hidden_dim,
        config=config.environment,
    ).to("cuda")
    policy.load_state_dict(sd)
    policy.eval()
    print(f"policy dims: input_dim={input_dim}, hidden_dim={hidden_dim}")

    num_iter = max(1, dataset_size // batch_size)
    rows = []
    remain_stats = []
    del_stats = [n_del]
    for bi in tqdm(range(num_iter), desc="batches"):
        metrics = run_batch(env, policy)
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
                    args.partner_portion_test,
                    remove_controlled_agents=remove_controlled,
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

    print("=== RL partner deletion (remove_agents_by_distance) ===")
    print(summary.to_string(index=False))
    print(f"\nwrote per-scene: {out_csv}")
    print(f"wrote summary:   {out_summary}")
    print(f"appended:        {result_csv}")


if __name__ == "__main__":
    main()
