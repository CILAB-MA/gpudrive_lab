"""Find alone-driving validation scenes via log replay and copy them to a folder.

Alone = controlled ego has zero visible partners (partner_mask == 0) on every
alive timestep. Matches the earlier offline partner_mask analysis.
"""
import argparse
import csv
import os
import shutil
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from tqdm import tqdm

from gpudrive.env.config import EnvConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv


def check_batch_alone(env):
    """Return per-world flags: always_alone, ever_alone, alive_steps, alone_steps."""
    obs = env.reset()
    alive_agent_mask = env.cont_agent_mask.clone()
    dead_agent_mask = ~alive_agent_mask.clone()
    num_worlds = alive_agent_mask.shape[0]

    has_controlled = alive_agent_mask.any(dim=-1)  # (W,)
    ever_had_partner = torch.zeros(num_worlds, dtype=torch.bool, device=alive_agent_mask.device)
    alive_steps = torch.zeros(num_worlds, dtype=torch.int32, device=alive_agent_mask.device)
    alone_steps = torch.zeros(num_worlds, dtype=torch.int32, device=alive_agent_mask.device)

    expert_actions, *_ = env.get_expert_actions()

    for t in range(env.episode_len):
        partner_mask = env.get_partner_mask()  # (W, A, 127)
        world_alive = (~dead_agent_mask) & alive_agent_mask  # controlled & not done
        world_has_alive = world_alive.any(dim=-1)

        # partner present if any slot == 0 for any still-alive controlled agent
        # gather partner_mask only on currently alive controlled agents
        # For max_cont_agents=1 there is at most one per world.
        has_partner = torch.zeros(num_worlds, dtype=torch.bool, device=partner_mask.device)
        if world_has_alive.any():
            # (num_alive_agents,) -> map back to world
            w_idx, a_idx = torch.where(world_alive)
            pm = partner_mask[w_idx, a_idx]  # (N, 127)
            agent_has_partner = (pm == 0).any(dim=-1)
            has_partner.scatter_(0, w_idx, agent_has_partner)

        ever_had_partner |= world_has_alive & has_partner
        alive_steps += world_has_alive.to(torch.int32)
        alone_steps += (world_has_alive & ~has_partner).to(torch.int32)

        env.step_dynamics(expert_actions[:, :, t])
        _ = env.get_obs()
        dones = env.get_dones()
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones.bool())
        if dead_agent_mask.all():
            break

    always_alone = has_controlled & (alive_steps > 0) & ~ever_had_partner
    ever_alone = has_controlled & (alone_steps > 0)
    return {
        "has_controlled": has_controlled.cpu().numpy(),
        "always_alone": always_alone.cpu().numpy(),
        "ever_alone": ever_alone.cpu().numpy(),
        "alive_steps": alive_steps.cpu().numpy(),
        "alone_steps": alone_steps.cpu().numpy(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="/data/full_version/data/validation")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/data/full_version/data/validation_alone",
        help="Folder to copy alone scene json files into",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/data/full_version/data/validation_alone/alone_scenes.csv",
    )
    parser.add_argument("--dataset-size", type=int, default=9987)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-stack", type=int, default=1)
    parser.add_argument(
        "--mode",
        type=str,
        default="always",
        choices=["always", "ever"],
        help="always: no partners on all alive steps; ever: alone on at least one alive step",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Only write CSV stats; do not copy scene json files",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_path) or ".", exist_ok=True)

    scene_loader = SceneDataLoader(
        root=args.data_dir,
        batch_size=args.batch_size,
        dataset_size=args.dataset_size,
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
        data_loader=scene_loader,
        max_cont_agents=1,
        device="cuda",
        action_type="continuous",
    )

    num_iter = args.dataset_size // args.batch_size
    rows = []
    copy_count = 0

    for batch_idx in tqdm(range(num_iter), desc="batches"):
        stats = check_batch_alone(env)
        scene_files = list(env.data_batch)
        try:
            scenario_ids = env.get_scenario_ids()
        except Exception:
            scenario_ids = {i: "" for i in range(len(scene_files))}

        select = stats["always_alone"] if args.mode == "always" else stats["ever_alone"]

        for i in range(len(scene_files)):
            scene_idx = batch_idx * args.batch_size + i
            src = scene_files[i]
            fname = os.path.basename(src)
            row = {
                "scene_idx": scene_idx,
                "filename": fname,
                "src_path": src,
                "scenario_id": scenario_ids.get(i, ""),
                "has_controlled": int(stats["has_controlled"][i]),
                "always_alone": int(stats["always_alone"][i]),
                "ever_alone": int(stats["ever_alone"][i]),
                "alive_steps": int(stats["alive_steps"][i]),
                "alone_steps": int(stats["alone_steps"][i]),
                "copied": 0,
            }
            if select[i]:
                if not args.no_copy:
                    dst = os.path.join(args.out_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                row["copied"] = 1
                copy_count += 1
            rows.append(row)

        if batch_idx != num_iter - 1:
            env.swap_data_batch()

    env.close()

    fieldnames = [
        "scene_idx",
        "filename",
        "src_path",
        "scenario_id",
        "has_controlled",
        "always_alone",
        "ever_alone",
        "alive_steps",
        "alone_steps",
        "copied",
    ]
    with open(args.csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_always = sum(r["always_alone"] for r in rows)
    n_ever = sum(r["ever_alone"] for r in rows)
    n_ctrl = sum(r["has_controlled"] for r in rows)
    print(f"scanned scenes: {len(rows)}")
    print(f"with controlled ego: {n_ctrl}")
    print(f"always_alone: {n_always} ({100 * n_always / max(n_ctrl, 1):.2f}% of controlled)")
    print(f"ever_alone: {n_ever} ({100 * n_ever / max(n_ctrl, 1):.2f}% of controlled)")
    print(f"copied ({args.mode}): {copy_count} -> {args.out_dir}")
    print(f"csv: {args.csv_path}")


if __name__ == "__main__":
    main()
