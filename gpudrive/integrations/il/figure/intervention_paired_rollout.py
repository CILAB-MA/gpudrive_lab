"""Paired baseline / semantic-intervention rollouts for adaptiveness & recovery.

Causal protocol
---------------
1. Expert prefix to branch step ``image_idx * 3`` (``--intervene-step`` fallback).
2. Branch baseline vs. intervention (LP probe weights into fusion partner tokens).
3. Default ``persistent_k`` with K=10, ``--alpha 0.5``. Distinct scenes pack into
   ``--scene-batch`` worlds.
4. Horizon H=40 for traj metrics; recovery may continue to episode end for GPG.

Final paper metrics
-------------------
**Adaptiveness (CAP):** continuous-target ADE form
  CAP = ADE(ego_int, x_tgt) - ADE(ego_base, x_tgt)
  Ref: ADE(ego_base, ego_int)
  Secondary: planning collision vs x_tgt if min_t ||ego-tgt|| < r_ego+r_partner
  (~4.92 m = sum of half-diagonals of default 4.5×2.0 m boxes).
  Also report LP flip rate. Control: semantic vs L2-norm-matched random.

**Recovery (GPG):** gp_int - gp_base; Δcollision / Δoff-road as side stats.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gpudrive.env.constants import MAX_REL_AGENT_POS
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.integrations.il.figure.intervention_metrics import (
    attach_rollout_metrics,
    build_continuous_target_traj,
    build_counterfactual_target_traj,
    category_summary,
    half_diagonal,
    load_and_normalize,
    lp_class_to_rel_meters,
    norm_xy_to_lp_class,
    pull_labels_closer,
    rotate2d,
)

FUTURE_STEPS = [10, 20, 30, 40]
ALPHA_DEFAULT = 0.5


def traj_ade(a: np.ndarray, b: np.ndarray) -> float:
    """Average displacement error between two (H, 2) trajectories."""
    H = min(len(a), len(b))
    if H == 0:
        return float("nan")
    d = np.linalg.norm(np.asarray(a[:H], dtype=float) - np.asarray(b[:H], dtype=float), axis=-1)
    return float(np.nanmean(d))


def _fusion_partner_tokens(policy, obs, masks):
    """Return fusion partner tokens (B, ro_max, D) before any LP embed edit."""
    ego_state, road_objects, road_graph = policy._unpack_obs(obs, num_stack=policy.num_stack)
    ro_masks = masks[0][:, -1]
    rg_masks = masks[1][:, -1]

    ego_state = policy.ego_state_net(ego_state)
    road_objects = policy.road_object_net(road_objects)
    road_graph = policy.road_graph_net(road_graph)

    ego_mask = torch.zeros(len(obs), 1, dtype=torch.bool, device=ego_state.device)
    all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
    all_masks = torch.cat([ego_mask, ro_masks, rg_masks], dim=-1)

    all_attn = policy.fusion_attn(all_objs_map, pad_mask=all_masks)
    objects_attn = all_attn["last_hidden_state"][:, : policy.ro_max + 1]
    return objects_attn[:, 1 : policy.ro_max + 1].clone()


def rel_meters_to_lp_class(rx: float, ry: float) -> int:
    """Inverse of ``lp_class_to_rel_meters`` (nx = rx / 1000)."""
    return int(norm_xy_to_lp_class(float(rx) / 1000.0, float(ry) / 1000.0))


def renormalize_adapt_labels(
    labels_4,
    ego_xy0,
    ego_yaw0: float,
    ego_xy_t,
    ego_yaw_t: float,
    local_k: int,
    horizons=None,
    path_horizon: int = 40,
) -> np.ndarray:
    """Re-express fixed CF waypoints in the *current* ego frame as LP classes.

    Labels were defined relative to the branch ego pose. We lift them to a
    global CF path, then at policy step ``local_k`` after branch, map the
    waypoints at absolute times ``local_k + h`` (h in 10/20/30/40) back into
    the current ego frame so residual targets track the same planned path.
    """
    horizons = list(horizons) if horizons is not None else list(FUTURE_STEPS)
    labs = np.asarray(labels_4, dtype=np.int64).reshape(-1)
    labels_by_h = {int(h): int(labs[i]) for i, h in enumerate(horizons) if i < len(labs)}
    cf = build_counterfactual_target_traj(
        labels_by_h,
        ego_xy0=ego_xy0,
        ego_yaw0=float(ego_yaw0),
        horizon=int(path_horizon),
        horizons=horizons,
    )
    ego_xy_t = np.asarray(ego_xy_t, dtype=float).reshape(2)
    c, s = np.cos(-float(ego_yaw_t)), np.sin(-float(ego_yaw_t))
    out = []
    for i, h in enumerate(horizons):
        abs_step = int(local_k) + int(h)  # 1-indexed step after branch
        idx = int(np.clip(abs_step - 1, 0, path_horizon - 1))
        tgt = cf[idx]
        if not np.isfinite(tgt).all():
            out.append(int(labs[i]) if i < len(labs) else 0)
            continue
        dx = float(tgt[0] - ego_xy_t[0])
        dy = float(tgt[1] - ego_xy_t[1])
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        out.append(rel_meters_to_lp_class(rx, ry))
    return np.asarray(out, dtype=np.int64)


def build_residual_probe_deltas(
    other_lp_models,
    partner_tokens: torch.Tensor,
    intervention_indices,
    labels_4,
    alpha: float = ALPHA_DEFAULT,
    control: str = "semantic",
    seed: int = 0,
) -> torch.Tensor:

    B, R, D = partner_tokens.shape
    device = partner_tokens.device
    dtype = partner_tokens.dtype
    out = torch.zeros(B, R, D, device=device, dtype=dtype)
    labs = torch.as_tensor(labels_4, device=device, dtype=torch.long).view(B, 4)
    idxs = torch.as_tensor(intervention_indices, device=device, dtype=torch.long).view(B)
    n_h = max(len(other_lp_models), 1)

    for b in range(B):
        idx = int(idxs[b].item())
        if idx < 0 or idx >= R:
            continue
        tok = partner_tokens[b, idx]
        acc = torch.zeros(D, device=device, dtype=dtype)
        for i, lp in enumerate(other_lp_models):
            lab = int(labs[b, i].item())
            if control == "wrong_label":
                lab = (lab + 32) % 64
            pred = int(lp(tok.unsqueeze(0)).squeeze(0).argmax().item())
            w = lp.head.weight
            acc = acc + float(alpha) * (w[lab] - w[pred])
        out[b, idx] = acc / float(n_h)

    if control in ("semantic", "wrong_label"):
        return out

    if control == "random":
        rng = np.random.default_rng(int(seed))
        rnd_np = rng.standard_normal(size=(B, R, D)).astype(np.float32)
        rnd = torch.from_numpy(rnd_np).to(device=device, dtype=dtype)
        sem_norm = out.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp(min=1e-6) * sem_norm
        mask = (out.abs().sum(-1, keepdim=True) > 0).float()
        return rnd * mask

    raise ValueError(f"Unknown control={control}")


def get_context_with_partner_delta(policy, obs, masks, partner_delta=None):
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
    seed: int = 0,
    alpha: float = ALPHA_DEFAULT,
    sign: float = 1.0,
) -> torch.Tensor:
    """Build (1, 127, D) partner embed delta matching intervention.py.

    Controls
    --------
    semantic: LP class-weight directions for the labeled future classes.
    wrong_label: same geometry with class ids shifted by +32 (mod 64).
    random: direction is Gaussian noise, L2-matched per partner slot to the
      *semantic* delta (matched magnitude, scrambled semantics).

    ``sign``: +1 injects class weights (toward labels); -1 suppresses them
    (Bush et al. short-route / NEVER analogue: push away from L₀).
    """
    labels_sem = np.asarray(labels_4, dtype=np.int64).reshape(4)
    other_labels_sem = np.asarray(other_labels_12, dtype=np.int64).reshape(-1)
    sgn = float(sign)

    def _labels_for(control_name: str):
        if control_name == "wrong_label":
            return (labels_sem + 32) % 64, (other_labels_sem + 32) % 64
        return labels_sem.copy(), other_labels_sem.copy()

    def _combined_from_labels(labels_use, other_labs_use) -> torch.Tensor:
        w0 = other_lp_models[0].head.weight  # (64, D)
        D = w0.shape[1]
        weights = torch.zeros(127, D, 4, device=device)
        for i, lp in enumerate(other_lp_models):
            w = lp.head.weight
            lab = int(labels_use[i])
            weights[intervention_idx, :, i] = w[lab] * alpha * sgn
            if mode != "one" and other_indices is not None:
                for j, oi in enumerate(other_indices):
                    if oi is None or int(oi) < 0:
                        continue
                    olab = int(other_labs_use[j * 4 + i])
                    weights[int(oi), :, i] = w[olab] * alpha * sgn
        if mode == "mean":
            return weights.mean(-1)
        if mode == "sum":
            return weights.sum(-1)
        return weights[..., 0]

    # Always build semantic first (needed for random norm-matching / slot mask).
    sem = _combined_from_labels(labels_sem, other_labels_sem)

    if control == "semantic":
        combined = sem
    elif control == "wrong_label":
        lab_w, olab_w = _labels_for("wrong_label")
        combined = _combined_from_labels(lab_w, olab_w)
    elif control == "random":
        rng = np.random.default_rng(int(seed))
        rnd_np = rng.standard_normal(size=tuple(sem.shape)).astype(np.float32)
        rnd = torch.from_numpy(rnd_np).to(device=device, dtype=sem.dtype)
        sem_norm = sem.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        rnd = rnd / rnd.norm(dim=-1, keepdim=True).clamp(min=1e-6) * sem_norm
        mask = (sem.abs().sum(-1, keepdim=True) > 0).float()
        combined = rnd * mask
    else:
        raise ValueError(f"Unknown control={control}")

    return combined.unsqueeze(0)  # (1, 127, D)


@torch.no_grad()
def read_base_lp_preds(
    other_lp_models,
    partner_tokens: torch.Tensor,
    intervention_idx: int,
) -> list[int]:
    """Argmax other-LP predictions on one partner token (pre-edit)."""
    tok = partner_tokens[0] if partner_tokens.dim() == 3 else partner_tokens
    idx = int(intervention_idx)
    x = tok[idx].unsqueeze(0)
    preds = []
    for lp in other_lp_models:
        preds.append(int(lp(x).squeeze(0).argmax().item()))
    return preds


def transfer_label_delta(
    labels_star_pri: Sequence[int],
    labels_base_pri: Sequence[int],
    labels_base_near: Sequence[int],
) -> list[int]:
    """Map primary belief-update onto a nearby agent in LP class space.

    L★_near = L₀_near + (L★_pri − L₀_pri)  (mod 64, per horizon).
    """
    out = []
    for s, b, n in zip(labels_star_pri, labels_base_pri, labels_base_near):
        out.append(int((int(n) + (int(s) - int(b))) % 64))
    return out


def select_nearby_partner_indices(
    dist_m: np.ndarray,
    valid: np.ndarray,
    primary_idx: int,
    nearby_k: int,
    max_dist_m: float,
) -> list[int]:
    """Return up to ``nearby_k`` nearest valid partner slots (excluding primary)."""
    dist_m = np.asarray(dist_m, dtype=float).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    n = min(len(dist_m), len(valid))
    cands = []
    for i in range(n):
        if not valid[i]:
            continue
        if int(i) == int(primary_idx):
            continue
        d = float(dist_m[i])
        if not np.isfinite(d) or d > float(max_dist_m):
            continue
        cands.append((d, int(i)))
    cands.sort(key=lambda x: x[0])
    return [i for _, i in cands[: max(0, int(nearby_k))]]


def expand_others_with_probe_labels(
    other_lp_models,
    partner_tokens: torch.Tensor,
    primary_idx: int,
    labels_star_pri: np.ndarray,
    dist_m: np.ndarray | None,
    valid: np.ndarray | None,
    nearby_k: int,
    max_dist_m: float,
    csv_other_indices=None,
    keep_csv_others: bool = False,
) -> tuple[list[int], list[int], dict]:
    """Pick nearby partners and auto-build their L★ from probe preds.

    Returns (other_indices, other_labels_flat_4n, info).
    """
    labels_star_pri = np.asarray(labels_star_pri, dtype=np.int64).reshape(4)
    base_pri = read_base_lp_preds(other_lp_models, partner_tokens, primary_idx)

    indices: list[int] = []
    if keep_csv_others and csv_other_indices is not None:
        for oi in csv_other_indices:
            if oi is None:
                continue
            oi = int(oi)
            if oi < 0 or oi == int(primary_idx):
                continue
            if oi not in indices:
                indices.append(oi)

    if (
        nearby_k > 0
        and dist_m is not None
        and valid is not None
        and partner_tokens is not None
    ):
        for oi in select_nearby_partner_indices(
            dist_m, valid, primary_idx, nearby_k, max_dist_m
        ):
            if oi not in indices:
                indices.append(oi)

    labels_flat: list[int] = []
    per_slot = []
    for oi in indices:
        base_near = read_base_lp_preds(other_lp_models, partner_tokens, oi)
        star_near = transfer_label_delta(labels_star_pri, base_pri, base_near)
        labels_flat.extend(star_near)
        per_slot.append(
            {
                "idx": oi,
                "base": base_near,
                "star": star_near,
                "dist_m": float(dist_m[oi])
                if dist_m is not None and oi < len(dist_m)
                else float("nan"),
            }
        )

    info = {
        "nearby_n": len(indices),
        "primary_base": base_pri,
        "slots": per_slot,
    }
    return indices, labels_flat, info


def build_paper_dual_deltas(
    other_lp_models,
    intervention_idx: int,
    labels_star: np.ndarray,
    labels_base: list[int] | np.ndarray,
    other_indices,
    other_labels_12,
    mode: str,
    control: str,
    device: str,
    seed: int,
    alpha: float,
    suppress_alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (delta_suppress, delta_inject) each (1, 127, D).

    Inject: +α w_{L★} (directional / new plan).
    Suppress: −α_s w_{L₀} (short-route analogue / discourage old plan).
    Random: each arm is L2-matched to its semantic counterpart.
    """
    alpha_s = float(alpha if suppress_alpha is None else suppress_alpha)
    inj = build_probe_delta(
        other_lp_models,
        intervention_idx=intervention_idx,
        labels_4=labels_star,
        other_indices=other_indices,
        other_labels_12=other_labels_12,
        mode=mode,
        control=control,
        device=device,
        seed=seed,
        alpha=alpha,
        sign=1.0,
    )
    # Suppress only the primary partner slot (−w_L0); other partners stay untouched
    # (Bush short-route acts on specific squares, not the whole board).
    # For random, still L2-match to the semantic primary-only suppress.
    sup = build_probe_delta(
        other_lp_models,
        intervention_idx=intervention_idx,
        labels_4=np.asarray(labels_base, dtype=np.int64),
        other_indices=None,
        other_labels_12=other_labels_12,
        mode="one",
        control=control,
        device=device,
        seed=seed + 9973,
        alpha=alpha_s,
        sign=-1.0,
    )
    return sup, inj


@torch.no_grad()
def fusion_partner_tokens(policy, obs, masks):
    """Return fusion_attn partner tokens (B, ro_max, D) before the embed edit."""
    ego_state, road_objects, road_graph = policy._unpack_obs(obs, num_stack=policy.num_stack)
    ro_masks = masks[0][:, -1]
    rg_masks = masks[1][:, -1]

    ego_state = policy.ego_state_net(ego_state)
    road_objects = policy.road_object_net(road_objects)
    road_graph = policy.road_graph_net(road_graph)

    ego_mask = torch.zeros(len(obs), 1, dtype=torch.bool, device=ego_state.device)
    all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
    all_masks = torch.cat([ego_mask, ro_masks, rg_masks], dim=-1)

    all_attn = policy.fusion_attn(all_objs_map, pad_mask=all_masks)
    objects_attn = all_attn["last_hidden_state"][:, : policy.ro_max + 1]
    return objects_attn[:, 1 : policy.ro_max + 1].clone()


@torch.no_grad()
def branch_obs_and_masks(env, obs, dead, world_idx: int = 0):
    """Alive obs/masks for the controlled agent in ``world_idx`` at branch time."""
    road_mask = env.get_road_mask().to("cuda")
    partner_mask = env.get_partner_mask().to("cuda")
    partner_mask_bool = partner_mask == 2
    alive = (~dead) & env.cont_agent_mask
    world_filter = torch.zeros_like(alive)
    world_filter[world_idx] = True
    alive = alive & world_filter
    if not alive.any():
        alive = (~dead) & env.cont_agent_mask
        if not alive.any():
            return None, None
    masks = [
        partner_mask_bool[alive].unsqueeze(1),
        road_mask[alive].unsqueeze(1),
    ]
    return obs[alive], masks


@torch.no_grad()
def evaluate_probe_chain(
    other_lp_models,
    partner_tokens: torch.Tensor,
    partner_delta: torch.Tensor,
    intervention_idx: int,
    labels_4: np.ndarray,
    control: str,
) -> dict:
    """LP readout before/after delta on the primary intervened partner token.

    Measures whether intervention moves the *other* linear probe toward the
    semantic target class (and toward the control's own target when applicable).
    """
    labels_sem = np.asarray(labels_4, dtype=np.int64).reshape(4)
    if control == "wrong_label":
        labels_ctrl = (labels_sem + 32) % 64
    elif control == "semantic":
        labels_ctrl = labels_sem.copy()
    else:
        labels_ctrl = None  # random: no semantic control target

    # Use first batch row (single controlled agent).
    tok0 = partner_tokens[0]
    delta0 = partner_delta[0]
    tok1 = tok0 + delta0
    idx = int(intervention_idx)

    match_base, match_int = [], []
    logit_gain_sem, logit_gain_ctrl = [], []
    match_ctrl_int = []
    pred_changed = []
    preds_base, preds_int = [], []

    for i, lp in enumerate(other_lp_models):
        t = int(labels_sem[i])
        x0 = tok0[idx].unsqueeze(0)
        x1 = tok1[idx].unsqueeze(0)
        logits0 = lp(x0).squeeze(0)
        logits1 = lp(x1).squeeze(0)
        p0 = int(logits0.argmax().item())
        p1 = int(logits1.argmax().item())
        preds_base.append(p0)
        preds_int.append(p1)
        match_base.append(float(p0 == t))
        match_int.append(float(p1 == t))
        logit_gain_sem.append(float((logits1[t] - logits0[t]).item()))
        pred_changed.append(float(p0 != p1))
        if labels_ctrl is not None:
            tc = int(labels_ctrl[i])
            match_ctrl_int.append(float(p1 == tc))
            logit_gain_ctrl.append(float((logits1[tc] - logits0[tc]).item()))

    sem_match_base = float(np.mean(match_base))
    sem_match_int = float(np.mean(match_int))
    n_label_changed = int(sum(int(p != t) for p, t in zip(preds_base, labels_sem)))
    out = {
        "lp_sem_match_base": sem_match_base,
        "lp_sem_match_int": sem_match_int,
        "lp_sem_logit_gain": float(np.mean(logit_gain_sem)),
        "lp_sem_match_gain": sem_match_int - sem_match_base,
        "lp_flipped": float(sem_match_int > sem_match_base),
        "lp_fully_matched_int": float(sem_match_int >= 1.0 - 1e-8),
        "lp_pred_changed": float(any(pred_changed)),
        "lp_base_preds": preds_base,  # consumed for CAP #2/#3; dropped before CSV
        "lp_n_label_changed": float(n_label_changed),
    }
    for h, p0, p1 in zip(FUTURE_STEPS, preds_base, preds_int):
        out[f"lp_base_pred_h{h}"] = int(p0)
        out[f"lp_int_pred_h{h}"] = int(p1)
    if labels_ctrl is not None:
        out["lp_ctrl_match_int"] = float(np.mean(match_ctrl_int))
        out["lp_ctrl_logit_gain"] = float(np.mean(logit_gain_ctrl))
    else:
        out["lp_ctrl_match_int"] = float("nan")
        out["lp_ctrl_logit_gain"] = float("nan")
    return out


def metrics_row_from_pair(
    sid: int,
    category: str,
    image_idx,
    intervene_step: int,
    args,
    control: str,
    base: dict,
    inter: dict,
    labels: np.ndarray,
    probe: dict | None = None,
    tgt_rel_xy: np.ndarray | None = None,
) -> dict:
    """Build one per-scene result row for a (baseline, intervened) pair."""
    row = {
        "scene_idx": sid,
        "category": category,
        "image_idx": image_idx if image_idx is not None else -1,
        "intervene_step": intervene_step,
        "horizon": args.horizon,
        "mode": args.mode,
        "persistent_k": (
            args.persistent_k
            if args.mode in ("persistent_k", "per_step_residual")
            else (getattr(args, "suppress_k", args.horizon) if args.mode == "paper_dual" else 1)
        ),
        "residual_renorm": int(bool(getattr(args, "residual_renorm", False))),
        "inject_until_k": getattr(args, "inject_until_k", -1) if args.mode == "paper_dual" else -1,
        "suppress_k": getattr(args, "suppress_k", -1) if args.mode == "paper_dual" else -1,
        "alpha": args.alpha,
        "control": control,
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
    if category == "adaptiveness":
        labels_by_h = {h: int(labels[i]) for i, h in enumerate(FUTURE_STEPS)}
        # Prefer continuous ego-frame target (random-cells); else LP-class CF.
        use_cont = (
            tgt_rel_xy is not None
            and np.asarray(tgt_rel_xy).shape[0] >= len(FUTURE_STEPS)
            and np.isfinite(np.asarray(tgt_rel_xy, dtype=float)[: len(FUTURE_STEPS)]).all()
        )
        if use_cont:
            rel_by_h = {
                h: (
                    float(tgt_rel_xy[i, 0]),
                    float(tgt_rel_xy[i, 1]),
                )
                for i, h in enumerate(FUTURE_STEPS)
            }
            tgt_cf = build_continuous_target_traj(
                rel_by_h,
                ego_xy0=base["ego_xy0"],
                ego_yaw0=base["ego_yaw0"],
                horizon=args.horizon,
            )
            metric_kind = "continuous_tgt_ade"
        else:
            tgt_cf = build_counterfactual_target_traj(
                labels_by_h,
                ego_xy0=base["ego_xy0"],
                ego_yaw0=base["ego_yaw0"],
                horizon=args.horizon,
            )
            metric_kind = "counterfactual_xy_distance"
        H = min(len(base["traj_xy"]), len(inter["traj_xy"]), len(tgt_cf))
        ego_b = np.asarray(base["traj_xy"][:H], dtype=float)
        ego_i = np.asarray(inter["traj_xy"][:H], dtype=float)
        tgt = np.asarray(tgt_cf[:H], dtype=float)
        # Primary CAP = ADE(ego, tgt)_int - ADE(ego, tgt)_base
        ade_base = traj_ade(ego_b, tgt)
        ade_int = traj_ade(ego_i, tgt)
        cap_ade = (
            float(ade_int - ade_base)
            if np.isfinite(ade_int) and np.isfinite(ade_base)
            else float("nan")
        )
        row["ade_base_to_tgt"] = ade_base
        row["ade_int_to_tgt"] = ade_int
        row["cap_ade"] = cap_ade
        row["ade_base_to_int"] = traj_ade(ego_b, ego_i)
        row["clearance_metric_kind"] = metric_kind

        # Planning collision vs continuous/CF target (not on-sim partner).
        r_sum = half_diagonal(base["ego_length"], base["ego_width"]) + half_diagonal(
            base["ego_length"], base["ego_width"]
        )
        d_base = np.linalg.norm(ego_b - tgt, axis=-1)
        d_int = np.linalg.norm(ego_i - tgt, axis=-1)
        min_d_base = float(np.nanmin(d_base)) if len(d_base) else float("nan")
        min_d_int = float(np.nanmin(d_int)) if len(d_int) else float("nan")
        row["min_dist_base_to_tgt"] = min_d_base
        row["min_dist_int_to_tgt"] = min_d_int
        row["plan_coll_thresh_m"] = float(r_sum)
        row["plan_coll_base"] = (
            float(min_d_base < r_sum) if np.isfinite(min_d_base) else float("nan")
        )
        row["plan_coll_int"] = (
            float(min_d_int < r_sum) if np.isfinite(min_d_int) else float("nan")
        )
    if probe:
        probe_out = {k: v for k, v in probe.items() if k != "lp_base_preds"}
        row.update(probe_out)
    return row


def summarize_by_control(label_df: pd.DataFrame, per_scene: pd.DataFrame) -> pd.DataFrame:
    """Run category_summary separately for each control condition."""
    parts = []
    for ctrl, sub in per_scene.groupby("control", sort=False):
        merged = attach_rollout_metrics(label_df, sub)
        s = category_summary(merged)
        s.insert(0, "control", ctrl)
        a = sub[sub["category"] == "adaptiveness"]
        if len(a):
            if "cap_ade" in a.columns:
                s.loc[s["category"] == "adaptiveness", "cap_ade_mean"] = float(a["cap_ade"].mean())
            if "ade_base_to_int" in a.columns:
                s.loc[s["category"] == "adaptiveness", "ade_base_to_int_mean"] = float(
                    a["ade_base_to_int"].mean()
                )
            if "plan_coll_base" in a.columns:
                s.loc[s["category"] == "adaptiveness", "plan_coll_base_rate"] = float(
                    a["plan_coll_base"].mean()
                )
                s.loc[s["category"] == "adaptiveness", "plan_coll_int_rate"] = float(
                    a["plan_coll_int"].mean()
                )
            if "lp_flipped" in a.columns:
                s.loc[s["category"] == "adaptiveness", "lp_flip_rate"] = float(
                    a["lp_flipped"].mean()
                )
        r = sub[sub["category"] == "recovery"]
        if len(r):
            gpg = r["goal_progress_intervened"].astype(float) - r["goal_progress_orig"].astype(
                float
            )
            s.loc[s["category"] == "recovery", "gpg_mean"] = float(gpg.mean())
            s.loc[s["category"] == "recovery", "gpg_pos_rate"] = float((gpg > 0).mean())
            if "collision_orig" in r.columns:
                dcoll = r["collision_intervened"].astype(float) - r["collision_orig"].astype(float)
                s.loc[s["category"] == "recovery", "delta_collision_mean"] = float(dcoll.mean())
            if "off_road_orig" in r.columns:
                doff = r["off_road_intervened"].astype(float) - r["off_road_orig"].astype(float)
                s.loc[s["category"] == "recovery", "delta_off_road_mean"] = float(doff.mean())
            if "lp_flipped" in r.columns:
                s.loc[s["category"] == "recovery", "lp_flip_rate"] = float(r["lp_flipped"].mean())
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def print_probe_behavior_chain(per_scene: pd.DataFrame, controls: list[str]) -> None:
    """Contrast CAP ADE conditioned on LP probe flip."""
    if "lp_flipped" not in per_scene.columns:
        return
    print("\n=== probe → CAP chain (by control) ===")
    rows = []
    for ctrl in controls:
        sub = per_scene[per_scene["control"] == ctrl]
        if not len(sub):
            continue
        row = {
            "control": ctrl,
            "n": len(sub),
            "lp_flip_rate": float(sub["lp_flipped"].mean()),
        }
        a = sub[sub["category"] == "adaptiveness"]
        if len(a) and "cap_ade" in a.columns:
            flipped = a[a["lp_flipped"] > 0.5]
            not_flip = a[a["lp_flipped"] <= 0.5]
            row["cap_ade_if_flipped"] = (
                float(flipped["cap_ade"].mean()) if len(flipped) else float("nan")
            )
            row["cap_ade_if_not"] = (
                float(not_flip["cap_ade"].mean()) if len(not_flip) else float("nan")
            )
        r = sub[sub["category"] == "recovery"]
        if len(r):
            flipped = r[r["lp_flipped"] > 0.5]
            not_flip = r[r["lp_flipped"] <= 0.5]
            row["gpg_if_flipped"] = (
                float(
                    (flipped["goal_progress_intervened"] - flipped["goal_progress_orig"]).mean()
                )
                if len(flipped)
                else float("nan")
            )
            row["gpg_if_not"] = (
                float(
                    (not_flip["goal_progress_intervened"] - not_flip["goal_progress_orig"]).mean()
                )
                if len(not_flip)
                else float("nan")
            )
        rows.append(row)
    if rows:
        print(pd.DataFrame(rows).round(4).to_string(index=False))

@torch.no_grad()
def step_policy_actions(
    env,
    policy,
    obs,
    dead_agent_mask,
    partner_delta=None,
    world_apply_mask=None,
    residual_spec=None,
    step_seed: int = 0,
):
    """Policy actions for alive controlled agents.

    partner_delta: None | (1, 127, D) | (W, 127, D)
    world_apply_mask: optional (W,) bool — worlds that receive the delta
      (others get a zero delta). Ignored if partner_delta is None.
    residual_spec: optional dict for per-step residual correction:
      other_lps, indices (W,), labels (W,4), alpha, control, world_seeds (W,)
      Optional adaptiveness renorm fields:
        renorm=True, ego_xy0 (W,2), ego_yaw0 (W,), intervene_steps (W,),
        sim_t (int absolute env timestep).
      When set, ignores fixed partner_delta and rebuilds Δ from current preds.
    """
    all_actions = torch.zeros(obs.shape[0], obs.shape[1], 3, device="cuda")
    road_mask = env.get_road_mask().to("cuda")
    partner_mask = env.get_partner_mask().to("cuda")
    partner_mask_bool = partner_mask == 2
    alive = ~dead_agent_mask
    if not alive.any():
        return all_actions
    world_ids, _agent_ids = torch.where(alive)
    all_masks = [
        partner_mask_bool[alive].unsqueeze(1),
        road_mask[alive].unsqueeze(1),
    ]
    alive_obs = obs[alive]
    delta = None
    if residual_spec is not None:
        tokens = _fusion_partner_tokens(policy, alive_obs, all_masks)
        # Map world-level residual fields onto alive agent rows.
        w_idx = world_ids.detach().cpu().numpy().astype(int)
        labels_w = np.asarray(residual_spec["labels"], dtype=np.int64)
        indices_w = np.asarray(residual_spec["indices"], dtype=np.int64)
        seeds_w = np.asarray(
            residual_spec.get("world_seeds", np.zeros(len(indices_w), dtype=np.int64)),
            dtype=np.int64,
        )
        labels_alive = labels_w[w_idx].copy()
        if bool(residual_spec.get("renorm", False)):
            ego_xy0 = np.asarray(residual_spec["ego_xy0"], dtype=float)
            ego_yaw0 = np.asarray(residual_spec["ego_yaw0"], dtype=float)
            intervene_steps = np.asarray(residual_spec["intervene_steps"], dtype=int)
            sim_t = int(residual_spec.get("sim_t", step_seed))
            for bi, w in enumerate(w_idx):
                pose = ego_pose(env, world_idx=int(w))
                if len(pose[0]) == 0:
                    continue
                local_k = int(sim_t) - int(intervene_steps[w])
                if local_k < 0:
                    continue
                labels_alive[bi] = renormalize_adapt_labels(
                    labels_w[w],
                    ego_xy0=ego_xy0[w],
                    ego_yaw0=float(ego_yaw0[w]),
                    ego_xy_t=pose[0][0],
                    ego_yaw_t=float(pose[1][0]),
                    local_k=local_k,
                )
        indices_alive = indices_w[w_idx]
        # One seed for the alive batch; mix step + first world seed.
        batch_seed = int(step_seed) * 10007 + int(seeds_w[w_idx[0]]) if len(w_idx) else int(step_seed)
        delta = build_residual_probe_deltas(
            residual_spec["other_lps"],
            tokens,
            indices_alive,
            labels_alive,
            alpha=float(residual_spec.get("alpha", ALPHA_DEFAULT)),
            control=str(residual_spec.get("control", "semantic")),
            seed=batch_seed,
        )
        if world_apply_mask is not None:
            apply = world_apply_mask[world_ids].to(dtype=delta.dtype).view(-1, 1, 1)
            delta = delta * apply
    elif partner_delta is not None:
        if partner_delta.shape[0] == 1:
            delta = partner_delta.expand(int(alive.sum().item()), -1, -1).clone()
        else:
            delta = partner_delta[world_ids].clone()
        if world_apply_mask is not None:
            apply = world_apply_mask[world_ids].to(dtype=delta.dtype).view(-1, 1, 1)
            delta = delta * apply
    context = get_context_with_partner_delta(policy, alive_obs, all_masks, delta)
    actions = policy.get_action(context, deterministic=True).squeeze(1)
    all_actions[alive] = actions
    return all_actions


def ego_pose(env, world_idx: int = 0):
    """Return ego (xy, yaw, length, width) for controlled agent in one world."""
    st = env.get_global_state()
    mask = env.cont_agent_mask.clone()
    world_filter = torch.zeros_like(mask)
    world_filter[world_idx] = True
    mask = mask & world_filter
    if not mask.any():
        mask = env.cont_agent_mask.clone()
        mask[:] = False
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


def ego_poses_all_worlds(env):
    """Per-world controlled-ego pose lists (None if no controlled agent)."""
    W = env.cont_agent_mask.shape[0]
    out = []
    for w in range(W):
        if not env.cont_agent_mask[w].any():
            out.append(None)
            continue
        xy, yaw, L, Wd = ego_pose(env, world_idx=w)
        if len(xy) == 0:
            out.append(None)
        else:
            out.append((xy[0].copy(), float(yaw[0]), L, Wd))
    return out


@torch.no_grad()
def rollout_batch(
    env,
    policy,
    expert_actions,
    intervene_steps,
    horizon: int,
    partner_deltas,  # None or (W, 127, D)  — used for single_step / persistent_k
    mode: str,
    persistent_k: int,
    continue_flags,
    valid_worlds,
    partner_deltas_suppress=None,  # (W, 127, D) for paper_dual
    partner_deltas_inject=None,  # (W, 127, D) for paper_dual
    inject_until_k: int = 10,
    suppress_k: int | None = None,
    residual_spec=None,  # per-step residual (recovery): see step_policy_actions
):
    """Vectorized paired rollout over a batch of (possibly different) scenes.

    Each world ``w`` branches at ``intervene_steps[w]``. Returns list[dict|None]
    of length W (None for invalid / failed worlds). Also returns per-world
    branch fusion partner tokens (or None) for probe readout.

    ``paper_dual`` (Bush et al.): apply suppress every step for ``suppress_k``
    (default=horizon) and inject only for the first ``inject_until_k`` steps.

    ``per_step_residual``: each policy step rebuild Δ = mean_i α(w[L★]-w[pred])
    from current LP preds (closed-loop toward fixed labels).
    """
    W = int(expert_actions.shape[0])
    device = "cuda"
    intervene = torch.as_tensor(intervene_steps, device=device, dtype=torch.long)
    cont = torch.as_tensor(continue_flags, device=device, dtype=torch.bool)
    valid = torch.as_tensor(valid_worlds, device=device, dtype=torch.bool)
    if suppress_k is None:
        suppress_k = int(horizon)
    suppress_k = int(suppress_k)
    inject_until_k = int(inject_until_k)

    obs = env.reset()
    dead = ~env.cont_agent_mask.clone()
    obs_stack1 = int(obs.shape[-1] / 5)

    traj_xy = [[] for _ in range(W)]
    traj_yaw = [[] for _ in range(W)]
    ego_xy0 = [None] * W
    ego_yaw0 = [None] * W
    ego_LW = [(4.5, 2.0)] * W
    off_road = np.zeros(W, dtype=np.float64)
    veh_coll = np.zeros(W, dtype=np.float64)
    goal = np.zeros(W, dtype=np.float64)
    init_goal = [None] * W
    dist_last = [None] * W
    branch_tokens = [None] * W
    branch_partner_dist = [None] * W
    branch_partner_valid = [None] * W
    branched = torch.zeros(W, dtype=torch.bool, device=device)
    done_world = ~valid  # padding worlds finished immediately

    ep_len = int(env.episode_len)
    for t in range(ep_len):
        # Check early-exit sparsely to avoid a CUDA sync every sim step.
        if (t & 3) == 3 or t + 1 >= ep_len:
            if bool(done_world.all().item()):
                break

        local_k = t - intervene  # (W,)
        in_prefix = (t < intervene) & (~done_world)
        in_policy = (t >= intervene) & (~done_world) & valid

        # Snapshot branch obs (first policy decision time) before stepping.
        hit = in_policy & (~branched) & (local_k == 0)
        if hit.any():
            road_mask = env.get_road_mask().to(device)
            partner_mask = env.get_partner_mask().to(device)
            partner_mask_bool = partner_mask == 2
            partner_pos = env.get_partner_pos()  # normalized relative XY
            for w in torch.where(hit)[0].tolist():
                alive_w = (~dead[w]) & env.cont_agent_mask[w]
                if not alive_w.any():
                    branched[w] = True
                    continue
                full_alive = torch.zeros_like(dead)
                full_alive[w] = alive_w
                am = [
                    partner_mask_bool[full_alive].unsqueeze(1),
                    road_mask[full_alive].unsqueeze(1),
                ]
                try:
                    branch_tokens[w] = fusion_partner_tokens(
                        policy, obs[full_alive], am
                    )
                except Exception:
                    branch_tokens[w] = None
                # Partner distances (meters) for nearby expansion.
                try:
                    agent_ids = torch.where(alive_w)[0]
                    a0 = int(agent_ids[0].item())
                    # partner_pos: [W, A, 127, 2] or [W, 127, 2] depending on API
                    pp = partner_pos[w]
                    pm = partner_mask[w]
                    if pp.dim() == 3:
                        pp = pp[a0]
                        pm = pm[a0]
                    dist = (
                        torch.linalg.norm(pp.float(), dim=-1) * float(MAX_REL_AGENT_POS)
                    )
                    # mask: 0=partner, 1=static, 2=non-exist
                    branch_partner_dist[w] = dist.detach().cpu().numpy()
                    branch_partner_valid[w] = (pm == 0).detach().cpu().numpy()
                except Exception:
                    branch_partner_dist[w] = None
                    branch_partner_valid[w] = None
                pose = ego_pose(env, world_idx=w)
                if len(pose[0]):
                    ego_xy0[w] = pose[0][0].copy()
                    ego_yaw0[w] = float(pose[1][0])
                    ego_LW[w] = (pose[2], pose[3])
                poss0 = obs[full_alive][:, obs_stack1 * 4 + 3 : obs_stack1 * 4 + 5]
                ig = torch.linalg.norm(poss0, dim=-1).clamp(min=1e-6)
                init_goal[w] = ig
                dist_last[w] = ig.clone()
                branched[w] = True

        actions = expert_actions[:, :, t].clone()
        if in_policy.any():
            apply_delta = torch.zeros(W, dtype=torch.bool, device=device)
            partner_delta_step = None
            use_residual = mode == "per_step_residual" and residual_spec is not None
            if (
                mode == "paper_dual"
                and partner_deltas_suppress is not None
                and partner_deltas_inject is not None
            ):
                apply_sup = in_policy & (local_k >= 0) & (local_k < suppress_k)
                apply_inj = in_policy & (local_k >= 0) & (local_k < inject_until_k)
                apply_delta = apply_sup | apply_inj
                # Per-world combined delta this step.
                partner_delta_step = torch.zeros_like(partner_deltas_inject)
                if apply_sup.any():
                    m = apply_sup.to(dtype=partner_deltas_suppress.dtype).view(W, 1, 1)
                    partner_delta_step = partner_delta_step + partner_deltas_suppress * m
                if apply_inj.any():
                    m = apply_inj.to(dtype=partner_deltas_inject.dtype).view(W, 1, 1)
                    partner_delta_step = partner_delta_step + partner_deltas_inject * m
            elif use_residual:
                apply_delta = in_policy & (local_k >= 0) & (local_k < persistent_k)
            elif partner_deltas is not None:
                partner_delta_step = partner_deltas
                if mode == "single_step":
                    apply_delta = in_policy & (local_k == 0)
                elif mode in ("persistent_k", "paper_dual"):
                    # paper_dual without dual tensors falls back to persistent_k
                    apply_delta = in_policy & (local_k >= 0) & (local_k < persistent_k)
            # Force prefix / finished worlds to stay on expert by masking them dead for policy fill,
            # then scatter policy actions only onto in_policy worlds.
            dead_for_pol = dead.clone()
            # Mark non-policy worlds as dead so they are skipped in policy forward
            non_pol = ~in_policy
            dead_for_pol[non_pol] = True
            step_residual = None
            if use_residual:
                step_residual = dict(residual_spec)
                step_residual["sim_t"] = int(t)
            pol_actions = step_policy_actions(
                env,
                policy,
                obs,
                dead_for_pol,
                partner_delta=partner_delta_step,
                world_apply_mask=apply_delta if (partner_delta_step is not None or use_residual) else None,
                residual_spec=step_residual,
                step_seed=int(t),
            )
            for w in torch.where(in_policy)[0].tolist():
                actions[w] = pol_actions[w]

        env.step_dynamics(actions)
        obs = env.get_obs()
        infos = env.get_infos()
        dones = env.get_dones()
        dead = torch.logical_or(dead, dones)

        poses = ego_poses_all_worlds(env)
        for w in range(W):
            if not valid_worlds[w] or done_world[w]:
                continue
            lk = int(t - intervene_steps[w])
            if 0 <= lk < horizon and poses[w] is not None:
                traj_xy[w].append(poses[w][0])
                traj_yaw[w].append(poses[w][1])

            alive_w = env.cont_agent_mask[w] & (~dead[w])
            if alive_w.any():
                off_road[w] = min(
                    1.0,
                    off_road[w]
                    + float(infos.off_road[w][alive_w].float().max().item()),
                )
                veh_coll[w] = min(
                    1.0,
                    veh_coll[w]
                    + float(infos.collided[w][alive_w].float().max().item()),
                )
                goal[w] = min(
                    1.0,
                    goal[w]
                    + float(infos.goal_achieved[w][alive_w].float().max().item()),
                )
                if init_goal[w] is not None:
                    # obs rows for this world's controlled agents
                    full_alive = torch.zeros_like(dead)
                    full_alive[w] = alive_w
                    if full_alive.any():
                        poss = obs[full_alive][
                            :, obs_stack1 * 4 + 3 : obs_stack1 * 4 + 5
                        ]
                        dist_last[w] = torch.linalg.norm(poss, dim=-1)

            # Completion: after this step at time t
            pol_steps = lk + 1 if lk >= 0 else 0
            max_pol = ep_len - intervene_steps[w]
            if lk >= 0:
                if continue_flags[w]:
                    if pol_steps >= max_pol or bool(dead[w].all().item()):
                        done_world[w] = True
                else:
                    if pol_steps >= horizon or bool(dead[w].all().item()):
                        done_world[w] = True
            # If never had controlled agent
            if not env.cont_agent_mask[w].any():
                done_world[w] = True

    results = []
    for w in range(W):
        if not valid_worlds[w]:
            results.append(None)
            continue
        if ego_xy0[w] is None:
            # Never reached branch / no agent
            results.append(None)
            continue
        if not traj_xy[w]:
            traj_xy[w] = [ego_xy0[w].copy()]
            traj_yaw[w] = [float(ego_yaw0[w])]
        if dist_last[w] is None or init_goal[w] is None:
            gp = 0.0
        else:
            gp = float((1.0 - (dist_last[w] / init_goal[w])).clamp(0, 1).mean().item())
        if goal[w] >= 1.0:
            gp = 1.0
        L, Wd = ego_LW[w]
        results.append(
            {
                "traj_xy": np.stack(traj_xy[w], axis=0),
                "traj_yaw": np.asarray(traj_yaw[w], dtype=float),
                "ego_xy0": ego_xy0[w],
                "ego_yaw0": float(ego_yaw0[w]),
                "ego_length": L,
                "ego_width": Wd,
                "off_road": float(off_road[w]),
                "veh_collision": float(veh_coll[w]),
                "collision": float(off_road[w] + veh_coll[w]),
                "goal": float(goal[w]),
                "goal_progress": gp,
                "partner_dist_m": branch_partner_dist[w],
                "partner_valid": branch_partner_valid[w],
            }
        )
    return results, branch_tokens


def make_env(data_root, scene_ids, env_config, batch_size=32):
    """One env: ``batch_size`` *distinct* scenes per Madrona batch.

    Scene list is padded to a multiple of ``batch_size`` (last ids repeated);
    callers must mark padded slots invalid. Advance with ``swap_data_batch``.
    """
    if not scene_ids:
        raise ValueError("scene_ids must be non-empty")
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    flat = [int(s) for s in scene_ids]
    pad_n = (-len(flat)) % batch_size
    if pad_n:
        flat = flat + [flat[-1]] * pad_n
    loader = SceneDataLoader(
        root=data_root,
        batch_size=batch_size,
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
    ), flat, pad_n


def parse_args():
    p = argparse.ArgumentParser("Paired intervention rollouts")
    p.add_argument("--dataset", "-d", type=str, default="validation")
    p.add_argument("--csv-path", "-cp", type=str, default="/data/full_version/intervention.csv")
    p.add_argument(
        "--others-csv",
        type=str,
        default="/data/full_version/intervention_others.csv",
    )
    p.add_argument(
        "--tgt-debug-csv",
        type=str,
        default="",
        help=(
            "Optional auto-label debug CSV with tgt_h{10,20,30,40}_{x,y} continuous "
            "ego-frame targets for ADE CAP (random-cells)."
        ),
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
    p.add_argument(
        "--intervene-step",
        type=int,
        default=0,
        help="Fallback branch timestep if CSV image_idx is missing. "
        "Normally use image_idx * 3 per scene.",
    )
    p.add_argument("--horizon", type=int, default=40)
    p.add_argument(
        "--mode",
        choices=["single_step", "persistent_k", "paper_dual", "per_step_residual"],
        default="persistent_k",
        help="How long to keep applying the LP embed edit. "
        "paper_dual: suppress (−w_L0) for suppress-k + inject (+w_L*) until inject-until-k. "
        "per_step_residual: each step rebuild Δ=mean_i α(w[L★_i]-w[pred_i]) for K steps.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=ALPHA_DEFAULT,
        help="LP weight scale for partner embed intervention.",
    )
    p.add_argument(
        "--persistent-k",
        type=int,
        default=10,
        help="Re-apply probe delta for this many steps when mode=persistent_k "
        "(also K for per_step_residual).",
    )
    p.add_argument(
        "--residual-renorm",
        action="store_true",
        help="With --mode per_step_residual: re-express CF label waypoints in the "
        "current ego frame each step (adaptiveness). Keeps the planned path fixed "
        "in global coords while ego moves.",
    )
    p.add_argument(
        "--inject-until-k",
        type=int,
        default=10,
        help="paper_dual: apply inject (+w_L*) for this many steps after branch "
        "(Bush directional / until-local-event analogue).",
    )
    p.add_argument(
        "--suppress-k",
        type=int,
        default=None,
        help="paper_dual: apply suppress (−w_L0) for this many steps after branch "
        "(default: --horizon). Bush short-route / every-step analogue.",
    )
    p.add_argument(
        "--suppress-alpha",
        type=float,
        default=None,
        help="paper_dual: scale for suppress arm (default: same as --alpha). "
        "Use a smaller value for a weaker NEVER-style push away from L0.",
    )
    p.add_argument(
        "--early-h",
        type=int,
        default=10,
        help="Horizon (steps after branch) for early min-CAP variant.",
    )
    p.add_argument(
        "--scene-batch",
        type=int,
        default=32,
        help="Distinct scenes per Madrona batch (parallel worlds). Default 32.",
    )
    p.add_argument(
        "--control",
        choices=["semantic", "wrong_label", "random"],
        default=None,
        help="Single control. If omitted, runs all --controls.",
    )
    p.add_argument(
        "--controls",
        nargs="+",
        choices=["semantic", "wrong_label", "random"],
        default=["semantic", "random"],
        help="Semantic-control suite (default: semantic random). Ignored if --control is set.",
    )
    p.add_argument("--intervention-reduce", choices=["mean", "sum", "one"], default="mean")
    p.add_argument(
        "--nearby-k",
        type=int,
        default=0,
        help="Also intervene on K nearest partners. Their L★ are auto-built from "
        "probe preds via primary label-delta transfer. 0 disables.",
    )
    p.add_argument(
        "--nearby-max-dist",
        type=float,
        default=50.0,
        help="Max distance (m) when selecting --nearby-k partners.",
    )
    p.add_argument(
        "--keep-csv-others",
        action="store_true",
        help="When --nearby-k>0, also keep CSV other indices (labels still "
        "auto-derived from probe preds).",
    )
    p.add_argument(
        "--label-close-scale",
        type=float,
        default=1.0,
        help="Pull semantic LP labels toward ego in norm space before intervene+CAP. "
        "1.0=original labels; 0.5=half relative offset (closer/harder CF threat).",
    )
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
        default=None,
        help="Per-scene CSV. Default: /data/full_version/intervention_paired_rollout_a{alpha}_k{K}.csv",
    )
    p.add_argument(
        "--out-summary",
        type=str,
        default=None,
        help="Summary CSV. Default: /data/full_version/intervention_paired_summary_a{alpha}_k{K}.csv",
    )
    return p.parse_args()


def _alpha_k_tag(alpha: float, persistent_k: int, mode: str = "persistent_k",
                 inject_until_k: int = 10, suppress_k: int | None = None,
                 suppress_alpha: float | None = None, nearby_k: int = 0,
                 label_close_scale: float = 1.0,
                 residual_renorm: bool = False) -> str:
    """Filesystem-safe tag, e.g. a3_k20 or a0p5_paper_i10_s40."""
    a = f"{float(alpha):g}".replace(".", "p")
    if mode == "paper_dual":
        sk = int(suppress_k) if suppress_k is not None else 40
        tag = f"a{a}_paper_i{int(inject_until_k)}_s{sk}"
        if suppress_alpha is not None and float(suppress_alpha) != float(alpha):
            sa = f"{float(suppress_alpha):g}".replace(".", "p")
            tag += f"_sa{sa}"
    elif mode == "per_step_residual":
        tag = f"a{a}_res_k{int(persistent_k)}"
        if residual_renorm:
            tag += "_renorm"
    else:
        tag = f"a{a}_k{int(persistent_k)}"
    if int(nearby_k) > 0:
        tag += f"_nb{int(nearby_k)}"
    if float(label_close_scale) < 1.0 - 1e-12:
        cs = f"{float(label_close_scale):g}".replace(".", "p")
        tag += f"_c{cs}"
    return tag


def _default_out_paths(
    alpha: float,
    persistent_k: int,
    mode: str = "persistent_k",
    inject_until_k: int = 10,
    suppress_k: int | None = None,
    suppress_alpha: float | None = None,
    nearby_k: int = 0,
    label_close_scale: float = 1.0,
    residual_renorm: bool = False,
) -> tuple[str, str]:
    tag = _alpha_k_tag(
        alpha,
        persistent_k,
        mode,
        inject_until_k,
        suppress_k,
        suppress_alpha,
        nearby_k,
        label_close_scale,
        residual_renorm=residual_renorm,
    )
    root = "/data/full_version"
    return (
        f"{root}/intervention_paired_rollout_{tag}.csv",
        f"{root}/intervention_paired_summary_{tag}.csv",
    )


def main():
    args = parse_args()
    if args.inject_target_trajectory:
        raise SystemExit(
            "--inject-target-trajectory is not supported by the current simulator API. "
            "Clearance metrics will be counterfactual_planning_clearance only."
        )

    if args.suppress_k is None:
        args.suppress_k = int(args.horizon)
    if args.suppress_alpha is None:
        args.suppress_alpha = float(args.alpha)

    if args.out_csv is None or args.out_summary is None:
        def_csv, def_sum = _default_out_paths(
            args.alpha,
            args.persistent_k,
            mode=args.mode,
            inject_until_k=args.inject_until_k,
            suppress_k=args.suppress_k,
            suppress_alpha=args.suppress_alpha,
            nearby_k=args.nearby_k,
            label_close_scale=args.label_close_scale,
            residual_renorm=bool(args.residual_renorm),
        )
        if args.out_csv is None:
            args.out_csv = def_csv
        if args.out_summary is None:
            args.out_summary = def_sum

    controls = [args.control] if args.control is not None else list(args.controls)
    print(f"controls: {controls}  alpha={args.alpha}  mode={args.mode}")
    if args.mode == "paper_dual":
        print(
            f"paper_dual: inject_until_k={args.inject_until_k}  "
            f"suppress_k={args.suppress_k}  suppress_alpha={args.suppress_alpha}"
        )
    else:
        print(f"persistent_k={args.persistent_k}")
    if args.mode == "per_step_residual" and args.residual_renorm:
        print("residual_renorm=True (adaptiveness CF path tracked in ego frame)")
    if args.nearby_k > 0:
        print(
            f"nearby: k={args.nearby_k}  max_dist={args.nearby_max_dist}m  "
            f"keep_csv_others={args.keep_csv_others}"
        )
    if float(args.label_close_scale) < 1.0 - 1e-12:
        print(f"label_close_scale={args.label_close_scale} (pull CF labels toward ego)")
    print(f"scene_batch: {args.scene_batch}")
    print(f"out_csv: {args.out_csv}")
    print(f"out_summary: {args.out_summary}")

    label_df = load_and_normalize(args.csv_path, args.start_idx, args.num_scenes)
    eval_df = label_df[label_df["category"].isin(args.categories)].copy()
    scene_ids = eval_df["scene_idx"].astype(int).tolist()
    if not scene_ids:
        raise SystemExit(f"No scenes for categories={args.categories}")

    raw = pd.read_csv(args.csv_path)
    others = pd.read_csv(args.others_csv)
    # Optional continuous targets from random-cells debug CSV.
    tgt_by_sid: dict[int, np.ndarray] = {}
    tgt_path = str(getattr(args, "tgt_debug_csv", "") or "").strip()
    if tgt_path and os.path.exists(tgt_path):
        tdf = pd.read_csv(tgt_path)
        for _, tr in tdf.iterrows():
            if str(tr.get("type", "")).strip().lower() not in ("i", "adaptiveness"):
                continue
            sid = int(tr["scene_idx"])
            arr = np.zeros((len(FUTURE_STEPS), 2), dtype=float)
            ok = True
            for i, h in enumerate(FUTURE_STEPS):
                for j, ax in enumerate(("x", "y")):
                    col = f"tgt_h{h}_{ax}"
                    if col not in tdf.columns or not np.isfinite(tr[col]):
                        ok = False
                        break
                    arr[i, j] = float(tr[col])
                if not ok:
                    break
            if ok:
                tgt_by_sid[sid] = arr
        print(f"continuous tgt loaded: {len(tgt_by_sid)} adaptiveness scenes from {tgt_path}")
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

    B = int(args.scene_batch)
    env, flat_ids, pad_n = make_env(data_root, scene_ids, env_config, batch_size=B)
    n_batches = len(flat_ids) // B
    print(f"scenes={len(scene_ids)}  batches={n_batches}  pad_slots={pad_n}")

    def _scene_meta(sid: int):
        idx = int(raw.loc[sid, "intervention_idx"]) if sid < len(raw) else 0
        labels = np.array(
            [int(raw.loc[sid, f"step{h}"]) for h in FUTURE_STEPS], dtype=np.int64
        )
        if float(args.label_close_scale) < 1.0 - 1e-12:
            labels = pull_labels_closer(labels, args.label_close_scale)
        oidx = [
            int(others.loc[sid, f"intervention_idx_{j}"]) if sid < len(others) else -1
            for j in range(3)
        ]
        olabels = []
        for j in range(3):
            for h in FUTURE_STEPS:
                col = f"step{h}_{j}"
                olabels.append(
                    int(others.loc[sid, col])
                    if sid < len(others) and col in others.columns
                    else 0
                )
        image_idx = None
        if sid < len(raw) and "image_idx" in raw.columns:
            raw_img = raw.loc[sid, "image_idx"]
            if pd.notna(raw_img):
                image_idx = int(raw_img)
        intervene_step = (
            image_idx * 3 if image_idx is not None else int(args.intervene_step)
        )
        continue_ep = args.continue_episode or (cat_map.get(sid) == "recovery")
        return {
            "sid": sid,
            "idx": idx,
            "labels": labels,
            "oidx": oidx,
            "olabels": olabels,
            "image_idx": image_idx,
            "intervene_step": intervene_step,
            "continue_ep": continue_ep,
            "category": cat_map.get(sid, "not_related"),
        }

    for bi in tqdm(range(n_batches), desc="batches"):
        batch_sids = flat_ids[bi * B : (bi + 1) * B]
        # Padding only on the last batch: repeated trailing ids are invalid.
        if bi == n_batches - 1 and pad_n:
            valid = [True] * (B - pad_n) + [False] * pad_n
        else:
            valid = [True] * B

        metas = [_scene_meta(sid) for sid in batch_sids]
        expert_actions, *_ = env.get_expert_actions()
        intervene_steps = [m["intervene_step"] for m in metas]
        continue_flags = [m["continue_ep"] for m in metas]

        # Baseline (no delta); keep branch tokens for L0 readout (paper_dual).
        bases, base_toks = rollout_batch(
            env,
            policy,
            expert_actions,
            intervene_steps,
            args.horizon,
            partner_deltas=None,
            mode="persistent_k",
            persistent_k=args.persistent_k,
            continue_flags=continue_flags,
            valid_worlds=valid,
        )

        for control in controls:
            partner_deltas = None
            partner_deltas_suppress = None
            partner_deltas_inject = None
            probe_deltas = []  # per-world (1,127,D) used for LP chain at branch
            nearby_ns = []

            def _resolve_others(w, m):
                """Return (oidx, olabels, nearby_n) with optional probe-auto nearby."""
                tok = base_toks[w]
                base_res = bases[w]
                if args.nearby_k > 0 and tok is not None and base_res is not None:
                    oidx, olabels, info = expand_others_with_probe_labels(
            other_lps,
                        tok,
                        primary_idx=m["idx"],
                        labels_star_pri=m["labels"],
                        dist_m=base_res.get("partner_dist_m"),
                        valid=base_res.get("partner_valid"),
                        nearby_k=args.nearby_k,
                        max_dist_m=args.nearby_max_dist,
                        csv_other_indices=m["oidx"],
                        keep_csv_others=bool(args.keep_csv_others),
                    )
                    return oidx, olabels, int(info["nearby_n"])
                return m["oidx"], m["olabels"], int(
                    sum(1 for x in m["oidx"] if x is not None and int(x) >= 0)
                )

            residual_spec = None
            if args.mode == "paper_dual":
                sup_list, inj_list = [], []
                for w, m in enumerate(metas):
                    tok = base_toks[w]
                    oidx, olabels, nb = _resolve_others(w, m)
                    nearby_ns.append(nb)
                    if tok is None:
                        # Fallback: no suppress labels → inject-only zeros suppress
                        labels_base = list(m["labels"])
                    else:
                        labels_base = read_base_lp_preds(other_lps, tok, m["idx"])
                    d_sup, d_inj = build_paper_dual_deltas(
                        other_lps,
                        intervention_idx=m["idx"],
                        labels_star=m["labels"],
                        labels_base=labels_base,
            other_indices=oidx,
            other_labels_12=olabels,
            mode=args.intervention_reduce,
                        control=control,
            device="cuda",
                        seed=int(m["sid"]) * 1009 + 17,
                        alpha=args.alpha,
                        suppress_alpha=args.suppress_alpha,
                    )
                    sup_list.append(d_sup.squeeze(0))
                    inj_list.append(d_inj.squeeze(0))
                    probe_deltas.append((d_sup + d_inj))
                partner_deltas_suppress = torch.stack(sup_list, dim=0)
                partner_deltas_inject = torch.stack(inj_list, dim=0)
            elif args.mode == "per_step_residual":
                labels_list = []
                idx_list = []
                seed_list = []
                ego_xy0_list = []
                ego_yaw0_list = []
                renorm = bool(args.residual_renorm)
                for w, m in enumerate(metas):
                    _oidx, _olabels, nb = _resolve_others(w, m)
                    nearby_ns.append(nb)
                    labels_list.append(
                        np.asarray(m["labels"], dtype=np.int64).reshape(4)
                    )
                    idx_list.append(int(m["idx"]))
                    seed_list.append(int(m["sid"]) * 1009 + 17)
                    base_res = bases[w]
                    if renorm and base_res is not None:
                        ego_xy0_list.append(
                            np.asarray(base_res["ego_xy0"], dtype=float).reshape(2)
                        )
                        ego_yaw0_list.append(float(base_res["ego_yaw0"]))
                    else:
                        ego_xy0_list.append(np.zeros(2, dtype=float))
                        ego_yaw0_list.append(0.0)
                    tok = base_toks[w]
                    if tok is not None:
                        tok_b = tok if tok.dim() == 3 else tok.unsqueeze(0)
                        d = build_residual_probe_deltas(
                            other_lps,
                            tok_b,
                            [int(m["idx"])],
                            [m["labels"]],
                            alpha=args.alpha,
                            control=control,
                            seed=seed_list[-1],
                        )
                        probe_deltas.append(d)
                    else:
                        # Fallback fixed delta for probe readout only.
                        d = build_probe_delta(
                            other_lps,
                            intervention_idx=m["idx"],
                            labels_4=m["labels"],
                            other_indices=_oidx,
                            other_labels_12=_olabels,
                            mode=args.intervention_reduce,
                            control=control,
                            device="cuda",
                            seed=seed_list[-1],
                            alpha=args.alpha,
                        )
                        probe_deltas.append(d)
                residual_spec = {
                    "other_lps": other_lps,
                    "indices": np.asarray(idx_list, dtype=np.int64),
                    "labels": np.stack(labels_list, axis=0),
                    "alpha": float(args.alpha),
                    "control": control,
                    "world_seeds": np.asarray(seed_list, dtype=np.int64),
                    "renorm": renorm,
                    "ego_xy0": np.stack(ego_xy0_list, axis=0),
                    "ego_yaw0": np.asarray(ego_yaw0_list, dtype=float),
                    "intervene_steps": np.asarray(intervene_steps, dtype=int),
                }
            else:
                deltas = []
                for w, m in enumerate(metas):
                    oidx, olabels, nb = _resolve_others(w, m)
                    nearby_ns.append(nb)
                    d = build_probe_delta(
                        other_lps,
                        intervention_idx=m["idx"],
                        labels_4=m["labels"],
                        other_indices=oidx,
                        other_labels_12=olabels,
                        mode=args.intervention_reduce,
                        control=control,
                        device="cuda",
                        seed=int(m["sid"]) * 1009 + 17,
                        alpha=args.alpha,
                    )
                    deltas.append(d.squeeze(0))
                    probe_deltas.append(d)
                partner_deltas = torch.stack(deltas, dim=0)  # (B, 127, D)

            inters, branch_toks = rollout_batch(
                env,
                policy,
                expert_actions,
                intervene_steps,
                args.horizon,
                partner_deltas=partner_deltas,
                mode=args.mode,
                persistent_k=args.persistent_k,
                continue_flags=continue_flags,
                valid_worlds=valid,
                partner_deltas_suppress=partner_deltas_suppress,
                partner_deltas_inject=partner_deltas_inject,
                inject_until_k=args.inject_until_k,
                suppress_k=args.suppress_k,
                residual_spec=residual_spec,
            )

            for w in range(B):
                if not valid[w]:
                    continue
                base = bases[w]
                inter = inters[w]
                if base is None or inter is None:
                    continue
                m = metas[w]
                if not np.allclose(base["ego_xy0"], inter["ego_xy0"], atol=1e-3):
                    print(
                        f"[warn] scene {m['sid']} control={control}: ego_xy0 mismatch"
                    )
                probe = {}
                # Prefer baseline branch tokens (identical prefix); fallback to int.
                tok = base_toks[w] if base_toks[w] is not None else branch_toks[w]
                if tok is not None and w < len(probe_deltas):
                    probe = evaluate_probe_chain(
                        other_lps,
                        tok,
                        probe_deltas[w],
                        intervention_idx=m["idx"],
                        labels_4=m["labels"],
                        control=control,
                    )
                row = metrics_row_from_pair(
                    m["sid"],
                    m["category"],
                    m["image_idx"],
                    m["intervene_step"],
                    args,
                    control,
                    base,
                    inter,
                    m["labels"],
                    probe=probe,
                    tgt_rel_xy=tgt_by_sid.get(int(m["sid"])),
                )
                if w < len(nearby_ns):
                    row["nearby_n"] = nearby_ns[w]
                rows.append(row)

        if bi != n_batches - 1:
            env.swap_data_batch()

    env.close()
    del env
    torch.cuda.empty_cache()

    per_scene = pd.DataFrame(rows)
    if len(per_scene):
        per_scene = per_scene.sort_values(["control", "scene_idx"]).reset_index(drop=True)
    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    per_scene.to_csv(args.out_csv, index=False)

    summary = summarize_by_control(label_df, per_scene) if len(per_scene) else pd.DataFrame()
    summary.to_csv(args.out_summary, index=False)
    print("=== paired intervention summary (by control) ===")
    print(summary.to_string(index=False))

    # Compact paper-metric contrast
    if len(per_scene) and "cap_ade" in per_scene.columns:
        a = per_scene[per_scene["category"] == "adaptiveness"]
        if len(a):
            print("\n=== adaptiveness (paper metrics) ===")
            named = dict(
                n=("scene_idx", "count"),
                cap_ade=("cap_ade", "mean"),
                ade_base_to_int=("ade_base_to_int", "mean"),
                plan_coll_base=("plan_coll_base", "mean"),
                plan_coll_int=("plan_coll_int", "mean"),
            )
            if "lp_flipped" in a.columns:
                named["lp_flip_rate"] = ("lp_flipped", "mean")
            named = {k: v for k, v in named.items() if v[0] in a.columns or k == "n"}
            cmp = (
                a.groupby("control")
                .agg(**named)
                .reindex([c for c in controls if c in set(a["control"])])
            )
            print(cmp.round(4).to_string())
            print("(CAP ADE / ADE(base,int) / plan-coll vs tgt / LP flip)")

    if len(per_scene) and "goal_progress_orig" in per_scene.columns:
        r = per_scene[per_scene["category"] == "recovery"]
        if len(r):
            print("\n=== recovery (paper metrics) ===")
            r = r.copy()
            r["gpg"] = r["goal_progress_intervened"].astype(float) - r[
                "goal_progress_orig"
            ].astype(float)
            named = dict(n=("scene_idx", "count"), gpg=("gpg", "mean"))
            if "collision_orig" in r.columns:
                r["dcoll"] = r["collision_intervened"].astype(float) - r["collision_orig"].astype(
                    float
                )
                named["delta_coll"] = ("dcoll", "mean")
            if "off_road_orig" in r.columns:
                r["doff"] = r["off_road_intervened"].astype(float) - r["off_road_orig"].astype(float)
                named["delta_off"] = ("doff", "mean")
            cmp = (
                r.groupby("control")
                .agg(**named)
                .reindex([c for c in controls if c in set(r["control"])])
            )
            print(cmp.round(4).to_string())

    if len(per_scene):
        print_probe_behavior_chain(per_scene, controls)

    print(f"\nwrote per-scene: {args.out_csv}")
    print(f"wrote summary:   {args.out_summary}")


if __name__ == "__main__":
    main()
