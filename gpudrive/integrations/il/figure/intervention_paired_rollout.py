"""Paired baseline / semantic-intervention rollouts for adaptiveness & recovery.

Causal protocol
---------------
1. Replay the same episode prefix (expert actions) to ``--intervene-step``.
2. Snapshot ego pose / obs (feedforward policy → no recurrent state).
3. Branch:
   - baseline: normal ``get_context`` → ``get_action``
   - intervention: add LP probe-direction weights into fusion_attn partner
     tokens (same as ``intervention.py``), then finish context → action.
4. Default ``single_step`` applies the embed edit only at the branch step;
   ``persistent_k`` reapplies for K consecutive policy steps.
5. Local horizon default H=40 (probe horizons). Optionally continue to
   episode end for full driving metrics (recovery GPG).

Adaptiveness metric (no target-trajectory injection in the simulator)
---------------------------------------------------------------------
Builds a **counterfactual** target path from LP class labels
(``step10..step40``) and measures

    min_clearance_base / min_clearance_int / CAP

via ``intervention_metrics.compute_paired_clearances``. This is explicitly
labeled ``counterfactual_planning_clearance`` (not an on-sim collision rate).

``--inject-target-trajectory`` is accepted but currently unsupported; the
script refuses to claim injected collision metrics if set.
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
from gpudrive.integrations.il.figure.intervention_metrics import (
    attach_rollout_metrics,
    build_counterfactual_target_traj,
    category_summary,
    compute_paired_clearances,
    load_and_normalize,
)

FUTURE_STEPS = [10, 20, 30, 40]
ALPHA = 5.0


def get_context_with_partner_delta(policy, obs, masks, partner_delta=None):
    """Like ``policy.get_context``, optionally adding delta to fusion partner tokens.

    partner_delta: (B, ro_max, D) or None
    """
    batch = obs.shape[0]
    ego_state, road_objects, road_graph = policy._unpack_obs(obs, num_stack=policy.num_stack)
    ro_masks = masks[0][:, -1]
    rg_masks = masks[1][:, -1]

    ego_state = policy.ego_state_net(ego_state)
    road_objects = policy.road_object_net(road_objects)
    road_graph = policy.road_graph_net(road_graph)

    ego_mask = torch.zeros(len(obs), 1, dtype=torch.bool, device=ego_state.device)
    all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
    all_masks = torch.cat([ego_mask, ro_masks, rg_masks], dim=-1)
    obj_masks = torch.cat([ego_mask, ro_masks], dim=-1)

    all_attn = policy.fusion_attn(all_objs_map, pad_mask=all_masks)
    objects_attn = all_attn["last_hidden_state"][:, : policy.ro_max + 1].clone()
    road_graph_attn = all_attn["last_hidden_state"][:, policy.ro_max + 1 :]

    if partner_delta is not None:
        objects_attn[:, 1 : policy.ro_max + 1, :] = (
            objects_attn[:, 1 : policy.ro_max + 1, :] + partner_delta
        )

    all_objects_attn = policy.ro_attn(objects_attn, pad_mask=obj_masks)
    ego_attn = all_objects_attn["last_hidden_state"][:, 0].unsqueeze(1)
    objects_attn = all_objects_attn["last_hidden_state"][:, 1 : policy.ro_max + 1]
    road_graph_attn = policy.rg_attn(road_graph_attn, pad_mask=rg_masks)
    road_graph_attn = road_graph_attn["last_hidden_state"]

    objects_attn = policy.ego_ro_attn(ego_attn, objects_attn, pad_mask=ro_masks)
    road_graph_attn = policy.ego_rg_attn(ego_attn, road_graph_attn, pad_mask=rg_masks)

    road_objects = objects_attn["last_hidden_state"].reshape(batch, -1)
    road_graph = road_graph_attn["last_hidden_state"].reshape(batch, -1)
    context = torch.cat((ego_attn.squeeze(1), road_objects, road_graph), dim=1)
    return context


def build_probe_delta(
    other_lp_models,
    intervention_idx: int,
    labels_4: np.ndarray,
    other_indices,
    other_labels_12,
    mode: str = "mean",
    control: str = "semantic",
    embed_dim: int = 128,
    device: str = "cuda",
) -> torch.Tensor:
    """Build (1, 127, D) partner embed delta matching intervention.py."""
    labels_4 = np.asarray(labels_4, dtype=np.int64).reshape(4)
    if control == "wrong_label":
        labels_4 = (labels_4 + 32) % 64
    elif control == "random":
        rng = np.random.default_rng(int(labels_4.sum()) + 17)
        labels_4 = rng.integers(0, 64, size=4)

    w0 = other_lp_models[0].head.weight  # (64, D)
    D = w0.shape[1]
    weights = torch.zeros(127, D, 4, device=device)

    for i, lp in enumerate(other_lp_models):
        w = lp.head.weight
        lab = int(labels_4[i])
        weights[intervention_idx, :, i] = w[lab] * ALPHA
        if mode != "one" and other_indices is not None:
            for j, oi in enumerate(other_indices):
                if oi is None or int(oi) < 0:
                    continue
                olab = int(other_labels_12[j * 4 + i])
                if control == "wrong_label":
                    olab = (olab + 32) % 64
                elif control == "random":
                    olab = int((olab * 7 + 13) % 64)
                weights[int(oi), :, i] = w[olab] * ALPHA

    if mode == "mean":
        combined = weights.mean(-1)
    elif mode == "sum":
        combined = weights.sum(-1)
    else:  # one — only primary partner, first horizon already in weights[...,0] path
        combined = weights[..., 0]

    if control == "random":
        # Norm-matched random direction (preserve L2 of semantic delta)
        sem_norm = combined.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rnd = torch.randn_like(combined)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp(min=1e-6) * sem_norm
        # Only keep slots that had nonzero semantic mass
        mask = (combined.abs().sum(-1, keepdim=True) > 0).float()
        combined = rnd * mask

    return combined.unsqueeze(0)  # (1, 127, D)


@torch.no_grad()
def step_policy_actions(env, policy, obs, dead_agent_mask, partner_delta=None):
    all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3, device="cuda")
    road_mask = env.get_road_mask().to("cuda")
    partner_mask = env.get_partner_mask().to("cuda")
    partner_mask_bool = partner_mask == 2
    alive = ~dead_agent_mask
    if not alive.any():
        return all_actions
    all_masks = [
        partner_mask_bool[alive].unsqueeze(1),
        road_mask[alive].unsqueeze(1),
    ]
    alive_obs = obs[alive]
    delta = None
    if partner_delta is not None:
        # partner_delta is (1, 127, D) for single-world batch; expand to alive count
        n_alive = int(alive.sum().item())
        delta = partner_delta.expand(n_alive, -1, -1)
    context = get_context_with_partner_delta(policy, alive_obs, all_masks, delta)
    actions = policy.get_action(context, deterministic=True).squeeze(1)
    all_actions[alive] = actions
    return all_actions


def ego_pose(env, world_idx: int = 0):
    """Return ego (xy, yaw, length, width) for controlled agent in one world."""
    st = env.get_global_state()
    mask = env.cont_agent_mask.clone()
    # Keep only the requested world (padded batches duplicate the same scene).
    world_filter = torch.zeros_like(mask)
    world_filter[world_idx] = True
    mask = mask & world_filter
    if not mask.any():
        mask = env.cont_agent_mask.clone()
        mask[:] = False
        # fallback: first controlled anywhere
        w, a = torch.where(env.cont_agent_mask)
        if len(w):
            mask[w[0], a[0]] = True
    xy = torch.stack((st.pos_x, st.pos_y), dim=-1)[mask]
    yaw = st.rotation_angle[mask]
    length = torch.full((xy.shape[0],), 4.5, device=xy.device)
    width = torch.full((xy.shape[0],), 2.0, device=xy.device)
    if hasattr(st, "vehicle_length"):
        length = st.vehicle_length[mask]
    if hasattr(st, "vehicle_width"):
        width = st.vehicle_width[mask]
    return (
        xy.detach().cpu().numpy(),
        yaw.detach().cpu().numpy(),
        float(length[0].item()) if len(length) else 4.5,
        float(width[0].item()) if len(width) else 2.0,
    )


def replay_prefix(env, expert_actions, intervene_step: int):
    """Deterministically replay expert actions to intervene_step; return obs."""
    obs = env.reset()
    dead = ~env.cont_agent_mask.clone()
    for t in range(intervene_step):
        env.step_dynamics(expert_actions[:, :, t])
        obs = env.get_obs()
        dead = torch.logical_or(dead, env.get_dones())
        if dead.all():
            break
    return obs, dead


def rollout_branch(
    env,
    policy,
    expert_actions,
    intervene_step: int,
    horizon: int,
    partner_delta: torch.Tensor | None,
    mode: str,
    persistent_k: int,
    continue_episode: bool,
    world_idx: int = 0,
):
    """Replay prefix then roll policy for ``horizon`` (and optionally to end)."""
    obs, dead = replay_prefix(env, expert_actions, intervene_step)
    alive_mask = env.cont_agent_mask.clone()
    world_filter = torch.zeros_like(alive_mask)
    world_filter[world_idx] = True
    alive_mask = alive_mask & world_filter
    if not alive_mask.any():
        return None

    xy0, yaw0, L, W = ego_pose(env, world_idx=world_idx)
    traj_xy, traj_yaw = [], []
    off_road = 0.0
    veh_coll = 0.0
    goal = 0.0

    obs_stack1 = int(obs.shape[-1] / 5)
    poss0 = obs[alive_mask][:, obs_stack1 * 4 + 3 : obs_stack1 * 4 + 5]
    init_goal = torch.linalg.norm(poss0, dim=-1).clamp(min=1e-6)
    dist_last = init_goal.clone()

    max_t = env.episode_len - intervene_step
    local_h = min(horizon, max_t)
    total_steps = max_t if continue_episode else local_h

    for k in range(total_steps):
        apply_delta = False
        if partner_delta is not None:
            if mode == "single_step" and k == 0:
                apply_delta = True
            elif mode == "persistent_k" and k < persistent_k:
                apply_delta = True
        delta = partner_delta if apply_delta else None
        actions = step_policy_actions(env, policy, obs, dead, delta)
        env.step_dynamics(actions)
        obs = env.get_obs()
        infos = env.get_infos()
        dones = env.get_dones()
        dead = torch.logical_or(dead, dones)

        if k < local_h:
            xy, yaw, _, _ = ego_pose(env, world_idx=world_idx)
            if len(xy):
                traj_xy.append(xy[0].copy())
                traj_yaw.append(float(yaw[0]))

        off_road = min(
            1.0, off_road + float(infos.off_road[alive_mask].float().max().item())
        )
        veh_coll = min(
            1.0, veh_coll + float(infos.collided[alive_mask].float().max().item())
        )
        goal = min(
            1.0, goal + float(infos.goal_achieved[alive_mask].float().max().item())
        )
        if alive_mask.any() and (~dead & alive_mask).any():
            poss = obs[alive_mask][:, obs_stack1 * 4 + 3 : obs_stack1 * 4 + 5]
            dist_last = torch.linalg.norm(poss, dim=-1)
        if dead[world_idx].all():
            break

    if not traj_xy:
        traj_xy = [xy0[0].copy()]
        traj_yaw = [float(yaw0[0])]

    gp = float((1.0 - (dist_last / init_goal)).clamp(0, 1).mean().item())
    if goal >= 1.0:
        gp = 1.0

    return {
        "traj_xy": np.stack(traj_xy, axis=0),
        "traj_yaw": np.asarray(traj_yaw, dtype=float),
        "ego_xy0": xy0[0].copy(),
        "ego_yaw0": float(yaw0[0]),
        "ego_length": L,
        "ego_width": W,
        "off_road": off_road,
        "veh_collision": veh_coll,
        "collision": off_road + veh_coll,
        "goal": goal,
        "goal_progress": gp,
    }


def make_env(data_root, scene_ids, env_config, pad_batch=16):
    """One env for all scenes: each scene is a pad_batch of duplicates.

    Madrona cannot recreate sims in-process (``setCudaHeapSize`` aborts), so we
    build a flat ``scene_nums`` list and advance with ``swap_data_batch``.
    """
    if not scene_ids:
        raise ValueError("scene_ids must be non-empty")
    flat = []
    for sid in scene_ids:
        flat.extend([int(sid)] * pad_batch)
    loader = SceneDataLoader(
        root=data_root,
        batch_size=pad_batch,
        dataset_size=max(flat) + 1,
        sample_with_replacement=False,
        shuffle=False,
        scene_nums=flat,
    )
    return GPUDriveTorchEnv(
        config=env_config,
        data_loader=loader,
        max_cont_agents=1,
        device="cuda",
        render_config=RenderConfig(),
        action_type="continuous",
    )


def parse_args():
    p = argparse.ArgumentParser("Paired intervention rollouts")
    p.add_argument("--dataset", "-d", type=str, default="validation")
    p.add_argument("--csv-path", "-cp", type=str, default="/data/full_version/intervention.csv")
    p.add_argument(
        "--others-csv",
        type=str,
        default="/data/full_version/intervention_others.csv",
    )
    p.add_argument("--start-idx", "-s", type=int, default=0)
    p.add_argument("--num-scenes", "-n", type=int, default=None)
    p.add_argument(
        "--categories",
        nargs="+",
        default=["adaptiveness", "recovery"],
    )
    p.add_argument("--model-path", "-mp", type=str, default="/data/full_version/model/exp_80000_subset_aix")
    p.add_argument("--model-name", "-mn", type=str, default="early_attn_s3_0908_113203.pth")
    p.add_argument("--lp-model-name", type=str, default="pos_early_lp")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--intervene-step", type=int, default=0)
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument(
        "--mode",
        choices=["single_step", "persistent_k"],
        default="single_step",
    )
    p.add_argument("--persistent-k", type=int, default=1)
    p.add_argument(
        "--control",
        choices=["semantic", "wrong_label", "random"],
        default="semantic",
    )
    p.add_argument("--intervention-reduce", choices=["mean", "sum", "one"], default="mean")
    p.add_argument(
        "--inject-target-trajectory",
        action="store_true",
        help="Not supported yet; kept for API compatibility (will error if set).",
    )
    p.add_argument(
        "--continue-episode",
        action="store_true",
        help="After local horizon, keep rolling to episode end (full driving metrics).",
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="/data/after_cvpr/images/intervention_paired_rollout.csv",
    )
    p.add_argument(
        "--out-summary",
        type=str,
        default="/data/after_cvpr/images/intervention_paired_summary.csv",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.inject_target_trajectory:
        raise SystemExit(
            "--inject-target-trajectory is not supported by the current simulator API. "
            "Clearance metrics will be counterfactual_planning_clearance only."
        )

    label_df = load_and_normalize(args.csv_path, args.start_idx, args.num_scenes)
    eval_df = label_df[label_df["category"].isin(args.categories)].copy()
    scene_ids = eval_df["scene_idx"].astype(int).tolist()
    if not scene_ids:
        raise SystemExit(f"No scenes for categories={args.categories}")

    raw = pd.read_csv(args.csv_path)
    others = pd.read_csv(args.others_csv)
    # Pad others to raw length
    if len(others) < len(raw):
        pad = pd.DataFrame(
            {
                "intervention_idx_0": [-1] * (len(raw) - len(others)),
                "intervention_idx_1": [-1] * (len(raw) - len(others)),
                "intervention_idx_2": [-1] * (len(raw) - len(others)),
                **{f"step{h}_{j}": [0] * (len(raw) - len(others)) for j in range(3) for h in FUTURE_STEPS},
            }
        )
        others = pd.concat([others, pad], ignore_index=True)

    model_path = os.path.join(args.model_path, args.model_name)
    print(f"model: {model_path}")
    policy = torch.load(model_path, weights_only=False).to("cuda")
    policy.eval()

    lp_root = os.path.join(args.model_path, "other_linear_prob", args.model_name.replace(".pth", ""))
    other_lps = []
    for h in FUTURE_STEPS:
        path = os.path.join(lp_root, f"seed{args.seed}", f"{args.lp_model_name}_{h}.pth")
        if not os.path.exists(path):
            # fallback naming used in some sweeps
            path = os.path.join(lp_root, f"seed{args.seed}", f"pos_lp_{h}.pth")
        m = torch.load(path, weights_only=False).to("cuda")
        m.eval()
        other_lps.append(m)

    env_config = EnvConfig(
        dynamics_model="delta_local",
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior="ignore",
        num_stack=5,
    )

    rows = []
    data_root = f"/data/full_version/data/{args.dataset}/"
    cat_map = eval_df.set_index("scene_idx")["category"].to_dict()

    env = make_env(data_root, scene_ids, env_config)
    for si, sid in enumerate(tqdm(scene_ids, desc="scenes")):
        expert_actions, *_ = env.get_expert_actions()

        idx = int(raw.loc[sid, "intervention_idx"]) if sid < len(raw) else 0
        labels = np.array([int(raw.loc[sid, f"step{h}"]) for h in FUTURE_STEPS], dtype=np.int64)
        oidx = [
            int(others.loc[sid, f"intervention_idx_{j}"]) if sid < len(others) else -1
            for j in range(3)
        ]
        olabels = []
        for j in range(3):
            for h in FUTURE_STEPS:
                col = f"step{h}_{j}"
                olabels.append(int(others.loc[sid, col]) if sid < len(others) and col in others.columns else 0)

        delta = build_probe_delta(
            other_lps,
            intervention_idx=idx,
            labels_4=labels,
            other_indices=oidx,
            other_labels_12=olabels,
            mode=args.intervention_reduce,
            control=args.control,
            device="cuda",
        )

        base = rollout_branch(
            env,
            policy,
            expert_actions,
            args.intervene_step,
            args.horizon,
            partner_delta=None,
            mode=args.mode,
            persistent_k=args.persistent_k,
            continue_episode=args.continue_episode or (cat_map[sid] == "recovery"),
        )
        # Identical prefix via fresh reset on same env (no clone API)
        inter = rollout_branch(
            env,
            policy,
            expert_actions,
            args.intervene_step,
            args.horizon,
            partner_delta=delta,
            mode=args.mode,
            persistent_k=args.persistent_k,
            continue_episode=args.continue_episode or (cat_map[sid] == "recovery"),
        )

        if base is not None and inter is not None:
            # Verify prefix pose equality (both branches start from same intervene state)
            if not np.allclose(base["ego_xy0"], inter["ego_xy0"], atol=1e-3):
                print(f"[warn] scene {sid}: ego_xy0 mismatch after prefix replay")

            row = {
                "scene_idx": sid,
                "category": cat_map[sid],
                "intervene_step": args.intervene_step,
                "horizon": args.horizon,
                "mode": args.mode,
                "control": args.control,
                "inject_target_trajectory": 0,
                "collision_orig": base["collision"],
                "collision_intervened": inter["collision"],
                "goal_progress_orig": base["goal_progress"],
                "goal_progress_intervened": inter["goal_progress"],
                "off_road_orig": base["off_road"],
                "off_road_intervened": inter["off_road"],
                "goal_orig": base["goal"],
                "goal_intervened": inter["goal"],
            }

            if cat_map[sid] == "adaptiveness":
                labels_by_h = {h: int(labels[i]) for i, h in enumerate(FUTURE_STEPS)}
                tgt_cf = build_counterfactual_target_traj(
                    labels_by_h,
                    ego_xy0=base["ego_xy0"],
                    ego_yaw0=base["ego_yaw0"],
                    horizon=args.horizon,
                )
                H = min(len(base["traj_xy"]), len(inter["traj_xy"]), len(tgt_cf))
                clr = compute_paired_clearances(
                    base["traj_xy"][:H],
                    inter["traj_xy"][:H],
                    tgt_cf[:H],
                    ego_yaw_base=base["traj_yaw"][:H],
                    ego_yaw_int=inter["traj_yaw"][:H],
                    tgt_yaw=np.full(H, base["ego_yaw0"]),  # CF yaw unknown → hold intervene yaw
                    ego_length=base["ego_length"],
                    ego_width=base["ego_width"],
                    tgt_length=base["ego_length"],
                    tgt_width=base["ego_width"],
                    method="obb",
                )
                row.update(clr)
                row["clearance_metric_kind"] = "counterfactual_planning_clearance"

            rows.append(row)

        if si != len(scene_ids) - 1:
            env.swap_data_batch()

    env.close()
    del env
    torch.cuda.empty_cache()

    per_scene = pd.DataFrame(rows).sort_values("scene_idx").reset_index(drop=True)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    per_scene.to_csv(args.out_csv, index=False)

    merged = attach_rollout_metrics(label_df, per_scene)
    summary = category_summary(merged)
    summary.to_csv(args.out_summary, index=False)
    print("=== paired intervention summary ===")
    print(summary.to_string(index=False))
    print(f"\nwrote per-scene: {args.out_csv}")
    print(f"wrote summary:   {args.out_summary}")


if __name__ == "__main__":
    main()
