"""Measure RL driving performance on alone-scene (single AV) dataset.

Mirrors gpudrive/integrations/rl/simulation.py (classic + discrete NeuralNet)
but rolls out on validation_alone with Goal / GoalTime / GoalProgress /
Collision / OffRoad recorded per scene and in a sweep result CSV.
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


EGO_FEAT_DIM = 6
PARTNER_FEAT_DIM = 6
N_PARTNERS = 127
PARTNER_OBS_SLICE = slice(EGO_FEAT_DIM, EGO_FEAT_DIM + PARTNER_FEAT_DIM * N_PARTNERS)
# ego: [speed, length, width, goal_x, goal_y, is_collided]
EGO_COLLIDED_IDX = 5


def zero_partner_obs_rl(obs):
    """Zero partner features and ego collision flag (num_stack=1).

    Partners may still exist in the simulator; masking them in obs alone would
    leak collisions via ego ``is_collided``.
    """
    out = obs.clone()
    out[..., EGO_COLLIDED_IDX] = 0
    out[..., PARTNER_OBS_SLICE] = 0
    return out


def run_batch(env, policy, mask_partners=True):
    """Roll out one batch; return per-world metrics (RL simulation.py style)."""
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~env.cont_agent_mask.clone()
    obs = env.reset()
    batch_size = alive_agent_mask.shape[0]
    n_ctrl = int(alive_agent_mask.sum().item())

    # RL uses num_stack=1 → goal relative pos at indices 3:5
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
        dist = torch.linalg.norm(poss, dim=-1)
        dist_metrics[alive_agent_mask] = dist[alive_agent_mask]

        goal_achieved = infos.goal_achieved[alive_agent_mask]
        goal_mask = (goal_achieved > 0) & (goal_timesteps < 0)
        goal_timesteps[goal_mask] = float(time_step)

        with torch.no_grad():
            alive_obs = obs[~dead_agent_mask]
            if mask_partners:
                # Zero partners + ego is_collided (sim may still have partners).
                alive_obs = zero_partner_obs_rl(alive_obs)
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
    }


def parse_args():
    p = argparse.ArgumentParser("RL alone-scene (single AV) driving performance")
    p.add_argument(
        "--data-dir",
        type=str,
        default="/data/full_version/data/validation_alone",
    )
    p.add_argument(
        "--alone-csv",
        type=str,
        default="/data/full_version/data/validation_alone/alone_scenes.csv",
    )
    p.add_argument("--batch-size", type=int, default=63)
    p.add_argument(
        "--model-path",
        "-mp",
        type=str,
        default="/data/after_cvpr/rl/scene_10000",
    )
    p.add_argument(
        "--model-name",
        "-mn",
        type=str,
        default="model_PPO____S_200__03_04_04_06_56_997_007604.pt",
    )
    p.add_argument(
        "--config-path",
        type=str,
        default="baselines/ppo/config/ppo_base_puffer.yaml",
    )
    p.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    p.add_argument(
        "--mask-partners",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zero partner obs and ego is_collided so policy sees empty partners (default: on)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="/data/after_cvpr/images/alone_driving_rl",
    )
    p.add_argument("--out-csv", type=str, default=None)
    p.add_argument("--out-summary", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"data dir not found: {args.data_dir}")

    scene_files = sorted(
        f
        for f in os.listdir(args.data_dir)
        if f.startswith("tfrecord") and f.endswith(".json")
    )
    dataset_size = len(scene_files)
    if dataset_size == 0:
        raise SystemExit(f"no tfrecord*.json in {args.data_dir}")

    batch_size = min(args.batch_size, dataset_size)
    while dataset_size % batch_size != 0 and batch_size > 1:
        batch_size -= 1

    model_stem = os.path.splitext(args.model_name)[0]
    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = args.out_csv or os.path.join(args.out_dir, f"{model_stem}_per_scene.csv")
    out_summary = args.out_summary or os.path.join(
        args.out_dir, f"{model_stem}_summary.csv"
    )
    result_csv = os.path.join(args.out_dir, "result_alone.csv")

    print(f"alone scenes: {dataset_size} from {args.data_dir} (batch_size={batch_size})")
    print(f"model: {args.model_path}/{args.model_name}")
    print(f"mask_partners: {args.mask_partners}")

    # Alone / log-replay style single-AV eval uses max_cont_agents=1.
    num_cont_agents = 1
    print(f"max_cont_agents: {num_cont_agents}")

    loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=batch_size,
        dataset_size=dataset_size,
        sample_with_replacement=False,
        shuffle=False,
    )
    env_config = EnvConfig(
        dynamics_model="classic",
        collision_behavior="ignore",
        steer_actions=torch.round(torch.linspace(-torch.pi, torch.pi, 13), decimals=3),
        accel_actions=torch.round(torch.linspace(-4.0, 4.0, 7), decimals=3),
        num_stack=1,
    )
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=loader,
        max_cont_agents=num_cont_agents,
        device="cuda",
        render_config=RenderConfig(),
        action_type="discrete",
    )
    if args.partner_portion_test:
        # pp==1.0 keeps only ego; otherwise remove partners by portion
        remove_controlled = args.partner_portion_test != 1.0
        env.remove_agents_by_id(
            args.partner_portion_test, remove_controlled_agents=remove_controlled
        )

    config = load_config(args.config_path)
    params = torch.load(
        os.path.join(args.model_path, args.model_name), weights_only=False
    )
    # Infer embed / shared dims from checkpoint (scene_10000 PPO uses input_dim=64)
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
        metrics = run_batch(env, policy, mask_partners=args.mask_partners)
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
                remove_controlled = args.partner_portion_test != 1.0
                env.remove_agents_by_id(
                    args.partner_portion_test,
                    remove_controlled_agents=remove_controlled,
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

    write_header = (not os.path.exists(result_csv)) or (os.path.getsize(result_csv) == 0)
    summary.to_csv(result_csv, mode="a", header=write_header, index=False)

    print("=== RL alone-scene (single AV) driving performance ===")
    print(summary.to_string(index=False))
    print(f"\nwrote per-scene: {out_csv}")
    print(f"wrote summary:   {out_summary}")
    print(f"appended:        {result_csv}")


if __name__ == "__main__":
    main()
