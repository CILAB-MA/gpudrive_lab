"""Auto-label intervention scenes (type=i/r/n) from RL probing + expert rollouts.

Mirrors the manual labeling rule of thumb:

* **recovery (r)** — ego planning looks weirdly wrong *and* nearby partners'
  other-LP labels look weird. Use those weird nearby probe preds as the
  intervention labels.
* **adaptiveness (i)** — otherwise, pick a frame/partner and *set* partner
  future labels to maximally overlap the ego-LP planned path (partner will be
  where ego plans to go).
* **not-related (n)** — neither condition holds.

Default: leave rows ``0..99`` untouched and label scenes ``100..199``
(append to CSV). Use ``--eval-existing`` to score the first 100 against
manual ``type`` for threshold sanity-check.

With ``--with-metrics`` / ``--pick-by-metrics``, also run paired
base/semantic/random rollouts in the same pass and keep labels that look
metric-friendly (CAP/pref gap, LP flip, hard-subset, etc.).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
from typing import Optional

sys.path.append(os.getcwd())

import functools
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.constants import MAX_REL_AGENT_POS
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.integrations.rl.figure.intervention_metrics import (
    half_diagonal,
    lp_class_to_norm_xy,
    lp_class_to_rel_meters,
    norm_xy_to_lp_class,
    pull_labels_toward_ego_n_cells,
)

FUTURE_STEPS = [10, 20, 30, 40]
CSV_COLS = [
    "intervention_idx",
    "step10",
    "step20",
    "step30",
    "step40",
    "done",
    "label_done",
    "type",
    "changed?",
    "image_idx",
]
OTHER_COLS = [
    "intervention_idx_0",
    "step10_0",
    "step20_0",
    "step30_0",
    "step40_0",
    "intervention_idx_1",
    "step10_1",
    "step20_1",
    "step30_1",
    "step40_1",
    "intervention_idx_2",
    "step10_2",
    "step20_2",
    "step30_2",
    "step40_2",
]


def register_all_layers_forward_hook(model):
    hidden_vector_dict = OrderedDict()

    def hook_fn(module, input, output, name):
        try:
            hidden_vector_dict[name] = output.detach()
        except AttributeError:
            hidden_vector_dict[name] = output["last_hidden_state"].detach()

    def _register(module, prefix=""):
        for name, layer in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer.register_forward_hook(functools.partial(hook_fn, name=full_name))
            _register(layer, full_name)

    _register(model)
    return hidden_vector_dict


def rel_meters_to_lp_class(rx: float, ry: float) -> int:
    """Map relative meters to LP class (same convention as lp_class_to_rel_meters)."""
    # lp_class_to_rel_meters: rx = 1000 * nx for nx in LP_NORM_EDGES span
    return int(norm_xy_to_lp_class(float(rx) / 1000.0, float(ry) / 1000.0))


def class_mismatch(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.int64).reshape(-1)
    b = np.asarray(b, dtype=np.int64).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    return float(np.mean(a[:n] != b[:n]))


def path_from_classes(labels: np.ndarray) -> np.ndarray:
    return np.asarray([lp_class_to_rel_meters(int(c)) for c in labels], dtype=float)


def overlap_score(ego_labels: np.ndarray, partner_labels: np.ndarray) -> float:
    """Higher => partner CF path closer to ego plan (mean -L2 in rel meters)."""
    e = path_from_classes(ego_labels)
    p = path_from_classes(partner_labels)
    d = np.linalg.norm(e - p, axis=-1)
    return float(-np.mean(d))


def path_inconsistency(labels: np.ndarray) -> float:
    """Large jumps across horizons => weird plan."""
    pts = path_from_classes(labels)
    if len(pts) < 2:
        return 0.0
    jumps = np.linalg.norm(np.diff(pts, axis=0), axis=-1)
    return float(np.mean(jumps))


def partner_closing_to_ego(
    partner_labels: np.ndarray,
    ego_labels: np.ndarray,
    min_close_m: float = 1.0,
) -> bool:
    """Partner CF path moves closer to ego plan across horizons (cut-in style)."""
    p = path_from_classes(partner_labels)
    e = path_from_classes(ego_labels)
    d = np.linalg.norm(p - e, axis=-1)
    if len(d) < 2:
        return False
    return float(d[-1]) < float(d[0]) - float(min_close_m)


def spatial_separated(
    dist_m: float,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    tgt_length: float = 4.5,
    tgt_width: float = 2.0,
) -> bool:
    """True if center distance exceeds sum of half-diagonals (not overlapping)."""
    clearance = float(dist_m) - half_diagonal(ego_length, ego_width) - half_diagonal(
        tgt_length, tgt_width
    )
    return clearance > 0.0


def sample_adapt_n_cells(scene_idx: int, lo: int = 1, hi: int = 3) -> int:
    """Reproducible Uniform{lo..hi} from scene index."""
    rng = np.random.RandomState(int(scene_idx) * 10007 + 17)
    return int(rng.randint(int(lo), int(hi) + 1))


def blend_labels_toward_ego(
    partner_labels: np.ndarray,
    ego_labels: np.ndarray,
    alpha_start: float = 0.2,
    alpha_end: float = 0.85,
) -> np.ndarray:
    """Blend partner probe labels toward ego plan (horizon-dependent cut-in)."""
    p = np.asarray(partner_labels, dtype=np.int64).reshape(-1)
    e = np.asarray(ego_labels, dtype=np.int64).reshape(-1)
    n = min(len(p), len(e))
    out = []
    for i in range(n):
        t = i / max(n - 1, 1)
        alpha = float(alpha_start) + (float(alpha_end) - float(alpha_start)) * t
        px, py = lp_class_to_norm_xy(int(p[i]))
        ex, ey = lp_class_to_norm_xy(int(e[i]))
        bx = (1.0 - alpha) * px + alpha * ex
        by = (1.0 - alpha) * py + alpha * ey
        out.append(norm_xy_to_lp_class(bx, by))
    return np.asarray(out, dtype=np.int64)


@torch.no_grad()
def collect_batch_probe_features(
    env,
    policy,
    ego_lps,
    other_lps,
    valid_worlds: list,
    nearby_max_dist: float = 50.0,
    nearby_k: int = 3,
):
    """Expert-prefix + RL partner/shared embeds for every valid world in the batch.

    Frames are only stored when ``t % 3 == 0`` and ``t + 40 < ep_len``.
    Returns ``{world_idx: [frame_dict, ...]}``.
    """
    device = "cuda"
    B = int(env.cont_agent_mask.shape[0])
    ep_len = int(env.episode_len)
    shared_hooks = register_all_layers_forward_hook(policy.shared_embed)

    obs = env.reset()
    dead = ~env.cont_agent_mask.clone()
    expert_actions, *_ = env.get_expert_actions()

    ego_xy = np.full((B, ep_len, 2), np.nan, dtype=float)
    ego_yaw = np.full((B, ep_len), np.nan, dtype=float)
    partner_rel = np.full((B, ep_len, 127, 2), np.nan, dtype=float)
    partner_valid = np.zeros((B, ep_len, 127), dtype=bool)

    # Pass 1: expert playback to log GT poses for all worlds.
    for t in range(ep_len):
        if bool(dead.all().item()):
            break
        st = env.get_global_state()
        for w in range(B):
            if not valid_worlds[w] or bool(dead[w].all().item()):
                continue
            mask = env.cont_agent_mask[w] & (~dead[w])
            if not mask.any():
                continue
            a0 = int(torch.where(mask)[0][0].item())
            ego_xy[w, t] = [
                float(st.pos_x[w, a0].item()),
                float(st.pos_y[w, a0].item()),
            ]
            ego_yaw[w, t] = float(st.rotation_angle[w, a0].item())
            pp = env.get_partner_pos()[w]
            pm = env.get_partner_mask()[w]
            if pp.dim() == 3:
                pp = pp[a0]
                pm = pm[a0]
            partner_rel[w, t] = (pp.float() * float(MAX_REL_AGENT_POS)).detach().cpu().numpy()
            partner_valid[w, t] = (pm == 0).detach().cpu().numpy()

        actions = expert_actions[:, :, t].clone()
        env.step_dynamics(actions)
        obs = env.get_obs()
        dead = torch.logical_or(dead, env.get_dones())

    # Pass 2: reset and run again with partner/shared embeds for LP readout.
    obs = env.reset()
    dead = ~env.cont_agent_mask.clone()
    expert_actions, *_ = env.get_expert_actions()
    frames = {w: [] for w in range(B) if valid_worlds[w]}

    for t in range(ep_len):
        if bool(dead.all().item()):
            break
        if (t % 3 == 0) and (t + 40 < ep_len):
            alive = (~dead) & env.cont_agent_mask

            # Collect live controlled agents across valid worlds.
            live_ws = []
            for w in range(B):
                if not valid_worlds[w]:
                    continue
                if not bool(alive[w].any().item()):
                    continue
                live_ws.append(w)

            if live_ws:
                world_filter = torch.zeros_like(alive)
                for w in live_ws:
                    world_filter[w] = True
                alive_w = alive & world_filter
                alive_obs = obs[alive_w]
                # Forward once so shared_embed hooks populate ego LP input.
                with torch.no_grad():
                    _ = policy(alive_obs, deterministic=True)
                ego_key = list(shared_hooks.keys())[-1]
                ego_tok = shared_hooks[ego_key]  # (N, hidden_dim)
                # Partner tokens from partner_embed (same as intervention.py)
                _ego, road_objects, _rg = policy.unpack_obs(alive_obs)
                other_tok = policy.partner_embed(road_objects)  # (N,127,D)

                # Map batch-row -> world (alive_w True order is row-major over worlds).
                # With max_cont_agents=1, each live world contributes exactly one row.
                row = 0
                for w in live_ws:
                    ego_cls = []
                    for lp in ego_lps:
                        ego_cls.append(int(lp(ego_tok[row : row + 1]).argmax(dim=-1).item()))
                    ego_cls = np.asarray(ego_cls, dtype=np.int64)

                    other_cls = np.zeros((127, 4), dtype=np.int64)
                    for hi, lp in enumerate(other_lps):
                        pred = lp(other_tok[row]).argmax(dim=-1)
                        other_cls[:, hi] = pred.detach().cpu().numpy()

                    gt_ego = []
                    ok_gt = True
                    for h in FUTURE_STEPS:
                        if (
                            t + h >= ep_len
                            or not np.isfinite(ego_xy[w, t + h]).all()
                            or not np.isfinite(ego_xy[w, t]).all()
                        ):
                            ok_gt = False
                            break
                        dx = ego_xy[w, t + h, 0] - ego_xy[w, t, 0]
                        dy = ego_xy[w, t + h, 1] - ego_xy[w, t, 1]
                        yaw = ego_yaw[w, t]
                        c, s = np.cos(yaw), np.sin(yaw)
                        rx = dx * c + dy * s
                        ry = -dx * s + dy * c
                        gt_ego.append(rel_meters_to_lp_class(rx, ry))
                    row += 1
                    if not ok_gt:
                        continue
                    gt_ego = np.asarray(gt_ego, dtype=np.int64)

                    dist = np.linalg.norm(partner_rel[w, t], axis=-1)
                    valid = partner_valid[w, t] & np.isfinite(dist)
                    nearby = []
                    for pi in np.where(valid)[0].tolist():
                        d = float(dist[pi])
                        if d > float(nearby_max_dist):
                            continue
                        gt_o = []
                        ok = True
                        for hi, h in enumerate(FUTURE_STEPS):
                            if t + h >= ep_len or not partner_valid[w, t + h, pi]:
                                ok = False
                                break
                            if not np.isfinite(ego_xy[w, t + h]).all():
                                ok = False
                                break
                            eyaw = ego_yaw[w, t + h]
                            c, s = np.cos(eyaw), np.sin(eyaw)
                            prx, pry = partner_rel[w, t + h, pi]
                            pgx = ego_xy[w, t + h, 0] + prx * c - pry * s
                            pgy = ego_xy[w, t + h, 1] + prx * s + pry * c
                            dx = pgx - ego_xy[w, t, 0]
                            dy = pgy - ego_xy[w, t, 1]
                            c0, s0 = np.cos(ego_yaw[w, t]), np.sin(ego_yaw[w, t])
                            rx = dx * c0 + dy * s0
                            ry = -dx * s0 + dy * c0
                            gt_o.append(rel_meters_to_lp_class(rx, ry))
                        if not ok:
                            continue
                        gt_o = np.asarray(gt_o, dtype=np.int64)
                        pred_o = other_cls[pi]
                        nearby.append(
                            dict(
                                idx=int(pi),
                                dist=d,
                                pred=pred_o,
                                gt=gt_o,
                                err=class_mismatch(pred_o, gt_o),
                                inconsist=path_inconsistency(pred_o),
                            )
                        )
                    nearby.sort(key=lambda x: x["dist"])
                    frames[w].append(
                        dict(
                            t=int(t),
                            image_idx=int(t // 3),
                            ego_pred=ego_cls,
                            ego_gt=gt_ego,
                            ego_err=class_mismatch(ego_cls, gt_ego),
                            ego_inconsist=path_inconsistency(ego_cls),
                            nearby=nearby[: max(1, int(nearby_k))],
                        )
                    )

        actions = expert_actions[:, :, t].clone()
        env.step_dynamics(actions)
        obs = env.get_obs()
        dead = torch.logical_or(dead, env.get_dones())

    del shared_hooks
    return frames


def propose_label_candidates(
    frames: list,
    ego_err_thr: float,
    ego_inconsist_thr: float,
    other_err_thr: float,
    adapt_base_ov_thr: float,
    adapt_plausible: bool = False,
    adapt_closing_m: float = 1.0,
    adapt_blend_alpha_start: float = 0.10,
    adapt_blend_alpha_end: float = 0.65,
    adapt_min_label_mismatch: float = 0.5,
    adapt_min_label_shift_m: float = 5.0,
    adapt_random_cells: bool = False,
    adapt_n_cells: Optional[int] = None,
    scene_idx: int = 0,
    adapt_n_cells_lo: int = 1,
    adapt_n_cells_hi: int = 3,
):
    """Return (best_recovery, best_adaptiveness) without priority.

    Adaptiveness modes:
      * ``adapt_random_cells`` — gate C: nearby + spatially separated at t +
        LP paths not already overlapping (overlap < thr). Labels = pull partner
        toward ego by random N∈[lo,hi] grid cells; keep continuous tgt xy.
      * ``adapt_plausible`` — corridor OR cut-in; blend partner→ego.
      * else ego-copy with corridor gate + strength filters.
    """
    best_r = None
    best_i = None
    n_cells = (
        int(adapt_n_cells)
        if adapt_n_cells is not None
        else sample_adapt_n_cells(scene_idx, adapt_n_cells_lo, adapt_n_cells_hi)
    )

    for fr in frames:
        nearby = fr["nearby"]
        if not nearby:
            continue

        ego_lab = fr["ego_pred"]
        nearby_by_dist = sorted(nearby, key=lambda n: n["dist"])
        ego_weird = (fr["ego_err"] >= ego_err_thr) or (
            fr["ego_inconsist"] >= ego_inconsist_thr
        )

        if ego_weird:
            weird_partners = [
                n
                for n in nearby_by_dist
                if (n["err"] >= other_err_thr)
                or (n["inconsist"] >= ego_inconsist_thr)
            ]
            if weird_partners:
                primary = weird_partners[0]
                score = (
                    -primary["dist"]
                    + 8.0 * fr["ego_err"]
                    + 8.0 * primary["err"]
                    + 0.05 * fr["ego_inconsist"]
                    + 0.05 * primary["inconsist"]
                )
                others = [o for o in nearby_by_dist if o["idx"] != primary["idx"]][:2]
                cand = dict(
                    kind="r",
                    score=float(score),
                    image_idx=fr["image_idx"],
                    intervention_idx=primary["idx"],
                    labels=primary["pred"].copy(),
                    partner_gt=primary["gt"].copy(),
                    other_indices=[o["idx"] for o in others],
                    other_labels=[o["pred"].copy() for o in others],
                    ego_err=fr["ego_err"],
                    near_err=float(primary["err"]),
                    ego_inconsist=fr["ego_inconsist"],
                    near_inconsist=float(primary["inconsist"]),
                    overlap=overlap_score(ego_lab, primary["pred"]),
                    dist=float(primary["dist"]),
                )
                if best_r is None or cand["score"] > best_r["score"]:
                    best_r = cand

        for n in nearby_by_dist:
            base_ov = overlap_score(ego_lab, n["pred"])
            closing = partner_closing_to_ego(
                n["pred"], ego_lab, min_close_m=adapt_closing_m
            )
            others = [o for o in nearby_by_dist if o["idx"] != n["idx"]][:2]

            if adapt_random_cells:
                # Gate C: near (already), spatially separated, LP paths not overlapping.
                if not spatial_separated(float(n["dist"])):
                    continue
                if base_ov >= float(adapt_base_ov_thr):
                    continue
                labels, tgt_rel = pull_labels_toward_ego_n_cells(
                    n["pred"], ego_lab, n_cells
                )
                other_labels = []
                for o in others:
                    olab, _ = pull_labels_toward_ego_n_cells(o["pred"], ego_lab, n_cells)
                    other_labels.append(olab)
                label_mismatch = class_mismatch(labels, n["pred"])
                label_shift_m = float(
                    np.mean(
                        np.linalg.norm(
                            tgt_rel - path_from_classes(n["pred"]), axis=-1
                        )
                    )
                )
                # Require at least some movement
                if label_mismatch <= 0.0 and label_shift_m < 1e-6:
                    continue
                score = (
                    -n["dist"]
                    + 0.35 * overlap_score(ego_lab, labels)
                    - 0.05 * fr["ego_inconsist"]
                    + 0.1 * float(n_cells)
                )
                cand = dict(
                    kind="i",
                    score=float(score),
                    image_idx=fr["image_idx"],
                    intervention_idx=n["idx"],
                    labels=labels,
                    other_indices=[o["idx"] for o in others],
                    other_labels=other_labels,
                    ego_err=fr["ego_err"],
                    near_err=float(n["err"]),
                    ego_inconsist=fr["ego_inconsist"],
                    near_inconsist=float(n["inconsist"]),
                    overlap=float(overlap_score(ego_lab, labels)),
                    dist=float(n["dist"]),
                    adapt_subtype="random_cells",
                    gate_corridor=False,
                    gate_closing=bool(closing),
                    gate_spatial_sep=True,
                    gate_lp_apart=True,
                    label_mismatch=float(label_mismatch),
                    label_shift_m=float(label_shift_m),
                    adapt_n_cells=int(n_cells),
                    tgt_rel_xy=np.asarray(tgt_rel, dtype=float),
                )
                if best_i is None or cand["score"] > best_i["score"]:
                    best_i = cand
                continue

            if adapt_plausible:
                if base_ov < float(adapt_base_ov_thr) and not closing:
                    continue
            elif base_ov < float(adapt_base_ov_thr):
                continue

            corridor_ok = base_ov >= float(adapt_base_ov_thr)
            if adapt_plausible:
                labels = blend_labels_toward_ego(
                    n["pred"],
                    ego_lab,
                    alpha_start=adapt_blend_alpha_start,
                    alpha_end=adapt_blend_alpha_end,
                )
                other_labels = [
                    blend_labels_toward_ego(
                        o["pred"],
                        ego_lab,
                        alpha_start=adapt_blend_alpha_start,
                        alpha_end=adapt_blend_alpha_end,
                    )
                    for o in others
                ]
                blend_ov = overlap_score(ego_lab, labels)
                score = (
                    -n["dist"]
                    + 0.35 * blend_ov
                    - 0.05 * fr["ego_inconsist"]
                )
                if closing and not corridor_ok:
                    score += 0.25 * (blend_ov - float(adapt_base_ov_thr))
                if closing and not corridor_ok:
                    adapt_subtype = "cutin"
                elif corridor_ok and closing:
                    adapt_subtype = "corridor_cutin"
                elif corridor_ok:
                    adapt_subtype = "corridor"
                else:
                    adapt_subtype = "cutin"
            else:
                labels = ego_lab.copy()
                other_labels = [ego_lab.copy() for _ in others]
                score = (
                    -n["dist"]
                    + 0.35 * base_ov
                    - 0.05 * fr["ego_inconsist"]
                )
                adapt_subtype = "egocopy"

            label_mismatch = class_mismatch(labels, n["pred"])
            label_shift_m = float(
                np.mean(
                    np.linalg.norm(
                        path_from_classes(labels) - path_from_classes(n["pred"]),
                        axis=-1,
                    )
                )
            )
            if label_mismatch < float(adapt_min_label_mismatch):
                continue
            if label_shift_m < float(adapt_min_label_shift_m):
                continue

            cand = dict(
                kind="i",
                score=float(score),
                image_idx=fr["image_idx"],
                intervention_idx=n["idx"],
                labels=labels,
                other_indices=[o["idx"] for o in others],
                other_labels=other_labels,
                ego_err=fr["ego_err"],
                near_err=float(n["err"]),
                ego_inconsist=fr["ego_inconsist"],
                near_inconsist=float(n["inconsist"]),
                overlap=float(
                    overlap_score(ego_lab, labels) if adapt_plausible else base_ov
                ),
                dist=float(n["dist"]),
                adapt_subtype=adapt_subtype,
                gate_corridor=bool(corridor_ok),
                gate_closing=bool(closing),
                label_mismatch=float(label_mismatch),
                label_shift_m=float(label_shift_m),
            )
            if best_i is None or cand["score"] > best_i["score"]:
                best_i = cand

    return best_r, best_i


def decide_label(
    frames: list,
    ego_err_thr: float,
    ego_inconsist_thr: float,
    other_err_thr: float,
    adapt_base_ov_thr: float,
    recovery_score_thr: float = -20.0,
    adapt_plausible: bool = False,
    adapt_closing_m: float = 1.0,
    adapt_blend_alpha_start: float = 0.10,
    adapt_blend_alpha_end: float = 0.65,
    adapt_min_label_mismatch: float = 0.5,
    adapt_min_label_shift_m: float = 5.0,
    adapt_random_cells: bool = False,
    adapt_n_cells: Optional[int] = None,
    scene_idx: int = 0,
    adapt_n_cells_lo: int = 1,
    adapt_n_cells_hi: int = 3,
) -> dict:
    """Pick the single best intervention step in the episode."""
    best_r, best_i = propose_label_candidates(
        frames,
        ego_err_thr=ego_err_thr,
        ego_inconsist_thr=ego_inconsist_thr,
        other_err_thr=other_err_thr,
        adapt_base_ov_thr=adapt_base_ov_thr,
        adapt_plausible=adapt_plausible,
        adapt_closing_m=adapt_closing_m,
        adapt_blend_alpha_start=adapt_blend_alpha_start,
        adapt_blend_alpha_end=adapt_blend_alpha_end,
        adapt_min_label_mismatch=adapt_min_label_mismatch,
        adapt_min_label_shift_m=adapt_min_label_shift_m,
        adapt_random_cells=adapt_random_cells,
        adapt_n_cells=adapt_n_cells,
        scene_idx=scene_idx,
        adapt_n_cells_lo=adapt_n_cells_lo,
        adapt_n_cells_hi=adapt_n_cells_hi,
    )
    if best_r is not None and float(best_r["score"]) >= float(recovery_score_thr):
        return best_r
    if best_i is not None:
        return best_i
    return dict(
        kind="n",
        score=0.0,
        image_idx=-1,
        intervention_idx=0,
        labels=np.zeros(4, dtype=np.int64),
        other_indices=[],
        other_labels=[],
        ego_err=float("nan"),
        near_err=float("nan"),
        ego_inconsist=float("nan"),
        near_inconsist=float("nan"),
        overlap=float("nan"),
        dist=float("nan"),
    )


def empty_row(typ: str = "n") -> dict:
    row = {c: 0 for c in CSV_COLS}
    row["type"] = typ
    row["done"] = 1
    row["label_done"] = 1
    row["changed?"] = -1 if typ == "n" else 1
    row["image_idx"] = np.nan
    return row


def empty_other_row() -> dict:
    row = {c: 0 for c in OTHER_COLS}
    for j in range(3):
        row[f"intervention_idx_{j}"] = -1
    return row


def decision_to_rows(dec: dict) -> tuple[dict, dict]:
    row = empty_row(dec["kind"])
    oth = empty_other_row()
    if dec["kind"] == "n":
        return row, oth
    row["intervention_idx"] = int(dec["intervention_idx"])
    for i, h in enumerate(FUTURE_STEPS):
        row[f"step{h}"] = int(dec["labels"][i])
    row["image_idx"] = int(dec["image_idx"])
    row["changed?"] = 1
    row["label_done"] = 1
    row["done"] = 0
    for j, (oi, labs) in enumerate(
        zip(dec.get("other_indices", []), dec.get("other_labels", []))
    ):
        if j >= 3:
            break
        oth[f"intervention_idx_{j}"] = int(oi)
        for i, h in enumerate(FUTURE_STEPS):
            oth[f"step{h}_{j}"] = int(labs[i])
    return row, oth


def _load_label_tables(csv_path: str, others_csv: str, out_csv: str, out_oth: str):
    """Load base tables, preferring existing out files so high start_idx appends.

    Without this, ``--start-idx 400 --csv-path intervention.csv`` (100 rows)
    would wipe previously labeled rows 100..399 when writing a 500-row out CSV.
    """
    base_df = pd.read_csv(csv_path)
    base_oth = pd.read_csv(others_csv)

    if out_csv and os.path.exists(out_csv) and os.path.abspath(out_csv) != os.path.abspath(
        csv_path
    ):
        prev = pd.read_csv(out_csv)
        if len(prev) > len(base_df):
            print(
                f"preserving longer existing out CSV ({len(prev)} rows) as base: {out_csv}"
            )
            base_df = prev
    if out_oth and os.path.exists(out_oth) and os.path.abspath(out_oth) != os.path.abspath(
        others_csv
    ):
        prev_o = pd.read_csv(out_oth)
        if len(prev_o) > len(base_oth):
            print(
                f"preserving longer existing others CSV ({len(prev_o)} rows): {out_oth}"
            )
            base_oth = prev_o

    for c in CSV_COLS:
        if c not in base_df.columns:
            base_df[c] = np.nan if c == "image_idx" else 0
    for c in OTHER_COLS:
        if c not in base_oth.columns:
            base_oth[c] = -1 if c.startswith("intervention_idx") else 0
    # Align lengths
    if len(base_oth) < len(base_df):
        for _ in range(len(base_df) - len(base_oth)):
            base_oth = pd.concat(
                [base_oth, pd.DataFrame([empty_other_row()])], ignore_index=True
            )
    return base_df, base_oth


def make_env_for_ids(data_root, scene_ids, env_config, batch_size=32):
    """Build env for arbitrary scene indices (including high start_idx like 400+).

    ``SceneDataLoader`` first truncates to ``dataset_size`` then indexes with
    ``scene_nums``, so ``dataset_size`` must be ``> max(scene_nums)``.
    """
    flat = [int(s) for s in scene_ids]
    pad_n = (-len(flat)) % batch_size
    if pad_n:
        flat = flat + [flat[-1]] * pad_n
    # Need the prefix of the sorted file list to include every requested index.
    need = max(flat) + 1
    loader = SceneDataLoader(
        root=data_root,
        batch_size=batch_size,
        dataset_size=need,
        sample_with_replacement=False,
        shuffle=False,
        scene_nums=flat,
    )
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=loader,
        max_cont_agents=1,
        device="cuda",
        render_config=RenderConfig(),
        action_type="discrete",
    )
    return env, flat, pad_n


def parse_args():
    p = argparse.ArgumentParser("Auto-label intervention i/r/n from probing")
    p.add_argument("--dataset", default="validation")
    p.add_argument("--csv-path", default="/data/after_cvpr/intervention.csv")
    p.add_argument("--others-csv", default="/data/after_cvpr/intervention_others.csv")
    p.add_argument(
        "--out-csv",
        default=None,
        help="Default: overwrite --csv-path (keeps first 100, appends new).",
    )
    p.add_argument("--out-others-csv", default=None)
    p.add_argument("--start-idx", type=int, default=100, help="First scene index to label.")
    p.add_argument("--num-scenes", type=int, default=100)
    p.add_argument(
        "--scene-batch",
        type=int,
        default=48,
        help="Madrona/worlds per GPU batch (raise to use more VRAM / util).",
    )
    p.add_argument("--model-path", default="/data/after_cvpr/rl/scene_10000")
    p.add_argument("--model-name", default="", help="Checkpoint stem; auto from other_linear_prob if empty")
    p.add_argument("--config-path", default="baselines/ppo/config/ppo_base_puffer.yaml")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--nearby-k", type=int, default=3)
    p.add_argument("--nearby-max-dist", type=float, default=50.0)
    p.add_argument(
        "--ego-err-thr",
        type=float,
        default=0.5,
        help="Fraction of ego LP≠GT horizons for recovery candidate.",
    )
    p.add_argument(
        "--ego-inconsist-thr",
        type=float,
        default=25.0,
        help="Mean jump (m) across horizons for 'weird' plans.",
    )
    p.add_argument("--other-err-thr", type=float, default=0.5)
    p.add_argument(
        "--recovery-score-thr",
        type=float,
        default=-20.0,
        help=(
            "Min recovery quality score to accept r over i. "
            "Score ≈ -dist_m + 8*(ego_err+near_err). "
            "Raise (e.g. -10) for fewer / closer recovery labels."
        ),
    )
    p.add_argument(
        "--adapt-overlap-thr",
        type=float,
        default=-40.0,
        help=(
            "Corridor gate: min overlap_score(ego_pred, partner_pred) "
            "(≈ -mean L2 meters; higher / less negative = stricter). "
            "With --adapt-plausible, closing partners can pass below this."
        ),
    )
    p.add_argument(
        "--adapt-plausible",
        action="store_true",
        help=(
            "Adaptiveness mixture: gate = corridor OR closing cut-in; "
            "labels = blend partner_pred→ego_plan (not ego-copy). "
            "Subtypes: corridor / cutin / corridor_cutin."
        ),
    )
    p.add_argument(
        "--adapt-closing-m",
        type=float,
        default=1.0,
        help="Cut-in gate: min meters partner path closes toward ego (h40 vs h10).",
    )
    p.add_argument(
        "--adapt-blend-alpha-start",
        type=float,
        default=0.10,
        help="Blend weight at h=10 (0=partner pred, 1=ego plan). Default 0.10.",
    )
    p.add_argument(
        "--adapt-blend-alpha-end",
        type=float,
        default=0.65,
        help="Blend weight at h=40. Default 0.65.",
    )
    p.add_argument(
        "--adapt-min-label-mismatch",
        type=float,
        default=0.5,
        help=(
            "Keep adaptiveness only if L★ differs from partner_pred on at least this "
            "fraction of horizons (main A05: 0.5 = 2/4; strict A: 0.75)."
        ),
    )
    p.add_argument(
        "--adapt-min-label-shift-m",
        type=float,
        default=5.0,
        help=(
            "Keep adaptiveness only if mean L2(L★ path, partner_pred path) ≥ this "
            "(meters). Main A05: 5.0; strict A: 10.0; set 0 to disable."
        ),
    )
    p.add_argument(
        "--adapt-random-cells",
        action="store_true",
        help=(
            "Adaptiveness gate C: nearby + spatially separated + LP paths apart; "
            "labels = pull partner toward ego by random N grid cells."
        ),
    )
    p.add_argument(
        "--adapt-n-cells",
        type=int,
        default=None,
        help="Fix N cells for --adapt-random-cells (default: sample Uniform{lo..hi} per scene).",
    )
    p.add_argument("--adapt-n-cells-lo", type=int, default=1)
    p.add_argument("--adapt-n-cells-hi", type=int, default=3)
    p.add_argument(
        "--features-cache",
        default="/data/after_cvpr/intervention_auto_label_frames_rl.pkl",
        help="Cache collected probe frames (skip sim if file exists unless --refresh-cache).",
    )
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument(
        "--eval-existing",
        action="store_true",
        help="Also run on scenes 0..99 and print agreement with manual type.",
    )
    p.add_argument("--dry-run", action="store_true", help="Do not write CSV.")
    p.add_argument(
        "--sweep-thresholds",
        action="store_true",
        help="Offline threshold sweep vs manual types (uses --features-cache).",
    )
    p.add_argument(
        "--with-metrics",
        action="store_true",
        help="After labeling, run paired base/sem/rnd rollouts and score metrics.",
    )
    p.add_argument(
        "--pick-by-metrics",
        action="store_true",
        help=(
            "Evaluate both r and i candidates per scene; keep the one with best "
            "metric score (implies --with-metrics)."
        ),
    )
    p.add_argument(
        "--filter-by-metrics",
        action="store_true",
        help="Demote labels that fail metrics_ok to type=n (implies --with-metrics).",
    )
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--persistent-k", type=int, default=10)
    p.add_argument("--horizon", type=int, default=40)
    return p.parse_args()


def _frames_to_cpu(frames_by_sid: dict) -> dict:
    """Ensure cached frames are pickle-friendly numpy."""
    out = {}
    for sid, frames in frames_by_sid.items():
        out[sid] = []
        for fr in frames:
            nearby = []
            for n in fr["nearby"]:
                nearby.append(
                    dict(
                        idx=int(n["idx"]),
                        dist=float(n["dist"]),
                        pred=np.asarray(n["pred"], dtype=np.int64),
                        gt=np.asarray(n["gt"], dtype=np.int64),
                        err=float(n["err"]),
                        inconsist=float(n["inconsist"]),
                    )
                )
            out[sid].append(
                dict(
                    t=int(fr["t"]),
                    image_idx=int(fr["image_idx"]),
                    ego_pred=np.asarray(fr["ego_pred"], dtype=np.int64),
                    ego_gt=np.asarray(fr["ego_gt"], dtype=np.int64),
                    ego_err=float(fr["ego_err"]),
                    ego_inconsist=float(fr["ego_inconsist"]),
                    nearby=nearby,
                )
            )
    return out


def sweep_thresholds(frames_by_sid: dict, base_df: pd.DataFrame) -> None:
    """Grid-search recovery/adapt thresholds against manual labels 0..99."""
    manuals = {}
    for sid in range(min(100, len(base_df))):
        m = str(base_df.loc[sid, "type"]).strip().lower()
        if m in ("i", "r", "n"):
            manuals[sid] = m

    best = None
    rows = []
    for ego_err in (0.5, 0.75, 1.0):
        for ego_inc in (25.0, 35.0, 50.0, 1e9):  # 1e9 ≈ disable inconsist
            for other_err in (0.5, 0.75, 1.0):
                for adapt_ov in (-20.0, -40.0, -60.0, -1e9):
                    conf = {"i": 0, "r": 0, "n": 0}
                    agree = 0
                    by = {k: [0, 0] for k in "irn"}
                    for sid, manual in manuals.items():
                        dec = decide_label(
                            frames_by_sid.get(sid, []),
                            ego_err_thr=ego_err,
                            ego_inconsist_thr=ego_inc,
                            other_err_thr=other_err,
                            adapt_base_ov_thr=adapt_ov,
                            recovery_score_thr=0.0,
                        )
                        conf[dec["kind"]] += 1
                        hit = int(dec["kind"] == manual)
                        agree += hit
                        by[manual][1] += 1
                        by[manual][0] += hit
                    n = len(manuals)
                    # Prefer overall agree, then recovery recall, then closer r count~17
                    score = (
                        agree / n
                        + 0.15 * (by["r"][0] / max(1, by["r"][1]))
                        + 0.10 * (by["i"][0] / max(1, by["i"][1]))
                        - 0.002 * abs(conf["r"] - 17)
                    )
                    rows.append(
                        dict(
                            score=score,
                            agree=agree / n,
                            ego_err=ego_err,
                            ego_inc=ego_inc,
                            other_err=other_err,
                            adapt_ov=adapt_ov,
                            conf=conf,
                            recall_i=by["i"][0] / max(1, by["i"][1]),
                            recall_r=by["r"][0] / max(1, by["r"][1]),
                            recall_n=by["n"][0] / max(1, by["n"][1]),
                        )
                    )
                    if best is None or score > best["score"]:
                        best = rows[-1]
    rows = sorted(rows, key=lambda x: -x["score"])
    print("\n=== threshold sweep (top 10) ===")
    for r in rows[:10]:
        print(
            f"score={r['score']:.3f} agree={r['agree']:.3f} "
            f"ego_err>={r['ego_err']} ego_inc>={r['ego_inc']} "
            f"other_err>={r['other_err']} adapt_ov>={r['adapt_ov']} "
            f"conf={r['conf']} "
            f"rec[i/r/n]={r['recall_i']:.2f}/{r['recall_r']:.2f}/{r['recall_n']:.2f}"
        )
    print("best:", best)


def main():
    import pickle

    args = parse_args()

    # Prefer throughput on V100: TF32 matmul + cudnn autotune for fixed shapes.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    out_csv = args.out_csv or args.csv_path
    out_oth = args.out_others_csv or args.others_csv

    base_df, base_oth = _load_label_tables(
        args.csv_path, args.others_csv, out_csv, out_oth
    )

    scene_ids = list(range(int(args.start_idx), int(args.start_idx) + int(args.num_scenes)))
    eval_ids = list(range(100)) if args.eval_existing else []
    # When only calibrating, skip the 100..199 pass.
    if args.eval_existing and args.num_scenes == 0:
        all_ids = eval_ids
    else:
        all_ids = eval_ids + [s for s in scene_ids if s not in eval_ids]

    frames_by_sid = {}
    cache_path = args.features_cache
    need_sim = args.refresh_cache or (not os.path.exists(cache_path))
    if not need_sim:
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        missing = [sid for sid in all_ids if sid not in cached]
        if missing:
            print(f"cache missing {len(missing)} scenes → will simulate those")
            need_sim_ids = missing
            frames_by_sid = {k: v for k, v in cached.items()}
        else:
            print(f"loaded frame cache: {cache_path} ({len(cached)} scenes)")
            frames_by_sid = {sid: cached[sid] for sid in all_ids}
            need_sim_ids = []
    else:
        need_sim_ids = list(all_ids)
        if os.path.exists(cache_path) and not args.refresh_cache:
            with open(cache_path, "rb") as f:
                frames_by_sid = pickle.load(f)

    if need_sim_ids:
        from gpudrive.integrations.rl.figure.intervention_paired_rollout import (
            load_rl_policy,
        )

        model_name = args.model_name
        if not model_name:
            lp_dir = os.path.join(args.model_path, "other_linear_prob")
            cands = sorted(os.listdir(lp_dir)) if os.path.isdir(lp_dir) else []
            if not cands:
                raise SystemExit(f"No LP dirs under {lp_dir}; pass --model-name")
            model_name = cands[-1]
            print(f"auto model_name: {model_name}")
        model_stem = model_name.replace(".pth", "").replace(".pt", "")
        model_file = model_name if model_name.endswith((".pt", ".pth")) else f"{model_name}.pt"
        model_path = os.path.join(args.model_path, model_file)
        if not os.path.exists(model_path):
            alt = os.path.join(args.model_path, f"{model_stem}.pt")
            if os.path.exists(alt):
                model_path = alt
        print(f"model: {model_path}")
        policy = load_rl_policy(model_path, args.config_path)

        ego_root = os.path.join(args.model_path, "ego_linear_prob", model_stem)
        other_root = os.path.join(args.model_path, "other_linear_prob", model_stem)
        ego_lps, other_lps = [], []
        for h in FUTURE_STEPS:
            ep = os.path.join(ego_root, f"seed{args.seed}", f"pos_ego_final_lp_{h}.pth")
            if not os.path.exists(ep):
                ep = os.path.join(ego_root, f"seed{args.seed}", f"pos_final_lp_{h}.pth")
            if not os.path.exists(ep):
                ep = os.path.join(ego_root, f"seed{args.seed}", f"pos_early_lp_{h}.pth")
            op = os.path.join(other_root, f"seed{args.seed}", f"pos_lp_{h}.pth")
            if not os.path.exists(op):
                op = os.path.join(other_root, f"seed{args.seed}", f"pos_early_lp_{h}.pth")
            print(f"  ego LP h={h}: {ep}")
            print(f"  other LP h={h}: {op}")
            eg = torch.load(ep, weights_only=False).to("cuda")
            ot = torch.load(op, weights_only=False).to("cuda")
            eg.eval()
            ot.eval()
            ego_lps.append(eg)
            other_lps.append(ot)

        env_config = EnvConfig(
            dynamics_model="classic",
            collision_behavior="ignore",
            steer_actions=torch.round(torch.linspace(-torch.pi, torch.pi, 13), decimals=3),
            accel_actions=torch.round(torch.linspace(-4.0, 4.0, 7), decimals=3),
            num_stack=1,
        )
        data_root = f"/data/full_version/data/{args.dataset}/"
        B = int(args.scene_batch)
        env, flat_ids, pad_n = make_env_for_ids(
            data_root, need_sim_ids, env_config, batch_size=B
        )
        n_batches = len(flat_ids) // B
        print(f"simulate scenes={len(need_sim_ids)} batches={n_batches} pad={pad_n} B={B}")
        try:
            for bi in tqdm(range(n_batches), desc="collect-features"):
                batch_sids = flat_ids[bi * B : (bi + 1) * B]
                if bi == n_batches - 1 and pad_n:
                    valid = [True] * (B - pad_n) + [False] * pad_n
                else:
                    valid = [True] * B
                frames_by_w = collect_batch_probe_features(
                    env,
                    policy,
                    ego_lps,
                    other_lps,
                    valid_worlds=valid,
                    nearby_max_dist=float(args.nearby_max_dist),
                    nearby_k=int(args.nearby_k),
                )
                for w, sid in enumerate(batch_sids):
                    if not valid[w] or sid in frames_by_sid:
                        continue
                    frames_by_sid[sid] = frames_by_w.get(w, [])
                if bi < n_batches - 1:
                    env.swap_data_batch()
        finally:
            env.close()
            del env
            torch.cuda.empty_cache()

        # Merge into cache file.
        existing = {}
        if os.path.exists(cache_path) and not args.refresh_cache:
            with open(cache_path, "rb") as f:
                existing = pickle.load(f)
        existing.update(_frames_to_cpu(frames_by_sid))
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(existing, f)
        print(f"wrote frame cache: {cache_path} ({len(existing)} scenes)")
        frames_by_sid = {sid: existing[sid] for sid in all_ids if sid in existing}

        # Madrona cannot create a second env in the same process after close
        # (setCudaHeapSize invalid argument). Relaunch for metrics phase.
        want_metrics_early = bool(
            args.with_metrics or args.pick_by_metrics or args.filter_by_metrics
        )
        if want_metrics_early:
            import gc
            import subprocess

            del policy, ego_lps, other_lps
            gc.collect()
            torch.cuda.empty_cache()
            print(
                "feature cache ready — relaunching fresh process "
                "(Madrona cannot recreate env in-process)..."
            )
            cmd = [sys.executable, *sys.argv]
            if "--refresh-cache" in cmd:
                # Avoid infinite resim loop on relaunch.
                cmd = [c for c in cmd if c != "--refresh-cache"]
            rc = subprocess.call(cmd)
            raise SystemExit(rc)

    if args.sweep_thresholds:
        sweep_thresholds(frames_by_sid, base_df)
        if args.dry_run and args.num_scenes == 0 and not args.eval_existing:
            return

    want_metrics = bool(
        args.with_metrics or args.pick_by_metrics or args.filter_by_metrics
    )
    if args.pick_by_metrics or args.filter_by_metrics:
        args.with_metrics = True
        want_metrics = True

    decisions = {}
    metrics_dbg = None

    def _resolve_model_and_other_lps():
        from gpudrive.integrations.rl.figure.intervention_paired_rollout import (
            load_rl_policy,
        )

        model_name = args.model_name
        if not model_name:
            lp_dir = os.path.join(args.model_path, "other_linear_prob")
            cands = sorted(os.listdir(lp_dir)) if os.path.isdir(lp_dir) else []
            if not cands:
                raise SystemExit(f"No LP dirs under {lp_dir}; pass --model-name")
            model_name = cands[-1]
        model_stem = model_name.replace(".pth", "").replace(".pt", "")
        model_file = (
            model_name if model_name.endswith((".pt", ".pth")) else f"{model_name}.pt"
        )
        model_path = os.path.join(args.model_path, model_file)
        if not os.path.exists(model_path):
            alt = os.path.join(args.model_path, f"{model_stem}.pt")
            if os.path.exists(alt):
                model_path = alt
        print(f"model: {model_path}")
        policy = load_rl_policy(model_path, args.config_path)
        other_root = os.path.join(args.model_path, "other_linear_prob", model_stem)
        other_lps = []
        for h in FUTURE_STEPS:
            op = os.path.join(other_root, f"seed{args.seed}", f"pos_lp_{h}.pth")
            if not os.path.exists(op):
                op = os.path.join(other_root, f"seed{args.seed}", f"pos_early_lp_{h}.pth")
            ot = torch.load(op, weights_only=False).to("cuda")
            ot.eval()
            other_lps.append(ot)
        env_config = EnvConfig(
            dynamics_model="classic",
            collision_behavior="ignore",
            steer_actions=torch.round(
                torch.linspace(-torch.pi, torch.pi, 13), decimals=3
            ),
            accel_actions=torch.round(torch.linspace(-4.0, 4.0, 7), decimals=3),
            num_stack=1,
        )
        return policy, other_lps, env_config

    if args.pick_by_metrics:
        from gpudrive.integrations.rl.figure.intervention_label_eval import (
            eval_candidates_and_pick,
        )

        policy, other_lps, env_config = _resolve_model_and_other_lps()
        data_root = f"/data/full_version/data/{args.dataset}/"

        def _make_env(scene_ids, batch_size):
            return make_env_for_ids(data_root, scene_ids, env_config, batch_size=batch_size)

        cands = {}
        for sid in all_ids:
            br, bi = propose_label_candidates(
                frames_by_sid.get(sid, []),
                ego_err_thr=args.ego_err_thr,
                ego_inconsist_thr=args.ego_inconsist_thr,
                other_err_thr=args.other_err_thr,
                adapt_base_ov_thr=args.adapt_overlap_thr,
                adapt_plausible=bool(args.adapt_plausible),
                adapt_closing_m=float(args.adapt_closing_m),
                adapt_blend_alpha_start=float(args.adapt_blend_alpha_start),
                adapt_blend_alpha_end=float(args.adapt_blend_alpha_end),
                adapt_min_label_mismatch=float(args.adapt_min_label_mismatch),
                adapt_min_label_shift_m=float(args.adapt_min_label_shift_m),
                adapt_random_cells=bool(args.adapt_random_cells),
                adapt_n_cells=args.adapt_n_cells,
                scene_idx=int(sid),
                adapt_n_cells_lo=int(args.adapt_n_cells_lo),
                adapt_n_cells_hi=int(args.adapt_n_cells_hi),
            )
            lst = []
            # Prefer evaluating recovery only if it clears recovery_score_thr
            if br is not None and float(br["score"]) >= float(args.recovery_score_thr):
                lst.append(br)
            if bi is not None:
                lst.append(bi)
            cands[sid] = lst

        decisions, metrics_dbg = eval_candidates_and_pick(
            _make_env,
            policy,
            other_lps,
            cands,
            batch_size=int(args.scene_batch),
            alpha=float(args.alpha),
            persistent_k=int(args.persistent_k),
            horizon=int(args.horizon),
            require_ok=True,
        )
    else:
        for sid in all_ids:
            decisions[sid] = decide_label(
                frames_by_sid.get(sid, []),
                ego_err_thr=args.ego_err_thr,
                ego_inconsist_thr=args.ego_inconsist_thr,
                other_err_thr=args.other_err_thr,
                adapt_base_ov_thr=args.adapt_overlap_thr,
                recovery_score_thr=args.recovery_score_thr,
                adapt_plausible=bool(args.adapt_plausible),
                adapt_closing_m=float(args.adapt_closing_m),
                adapt_blend_alpha_start=float(args.adapt_blend_alpha_start),
                adapt_blend_alpha_end=float(args.adapt_blend_alpha_end),
                adapt_min_label_mismatch=float(args.adapt_min_label_mismatch),
                adapt_min_label_shift_m=float(args.adapt_min_label_shift_m),
                adapt_random_cells=bool(args.adapt_random_cells),
                adapt_n_cells=args.adapt_n_cells,
                scene_idx=int(sid),
                adapt_n_cells_lo=int(args.adapt_n_cells_lo),
                adapt_n_cells_hi=int(args.adapt_n_cells_hi),
            )

        if want_metrics:
            from gpudrive.integrations.rl.figure.intervention_label_eval import (
                eval_decisions_over_scenes,
            )

            policy, other_lps, env_config = _resolve_model_and_other_lps()
            data_root = f"/data/full_version/data/{args.dataset}/"

            def _make_env(scene_ids, batch_size):
                return make_env_for_ids(
                    data_root, scene_ids, env_config, batch_size=batch_size
                )

            judged = eval_decisions_over_scenes(
                _make_env,
                policy,
                other_lps,
                [(sid, decisions[sid]) for sid in all_ids],
                batch_size=int(args.scene_batch),
                alpha=float(args.alpha),
                persistent_k=int(args.persistent_k),
                horizon=int(args.horizon),
            )
            metrics_dbg = pd.DataFrame(list(judged.values())) if judged else pd.DataFrame()
            for sid, j in judged.items():
                decisions[sid].update(
                    {
                        "metrics_ok": j.get("metrics_ok"),
                        "metrics_score": j.get("metrics_score"),
                        "cap_gap": j.get("cap_gap"),
                        "lp_flip_sem": j.get("lp_flip_sem"),
                        "ade_sem": j.get("ade_sem"),
                        "hard": j.get("hard"),
                        "gpg_gap": j.get("gpg_gap"),
                        "plan_coll_base": j.get("plan_coll_base"),
                    }
                )
                if args.filter_by_metrics and decisions[sid]["kind"] in ("i", "r"):
                    if not j.get("metrics_ok"):
                        decisions[sid] = dict(
                            kind="n",
                            score=0.0,
                            image_idx=-1,
                            intervention_idx=0,
                            labels=np.zeros(4, dtype=np.int64),
                            other_indices=[],
                            other_labels=[],
                            ego_err=float("nan"),
                            near_err=float("nan"),
                            metrics_ok=False,
                            metrics_score=j.get("metrics_score"),
                            filtered_from=j.get("kind"),
                        )

    conf = {"i": 0, "r": 0, "n": 0}
    agree = 0
    eval_n = 0
    by_type = {"i": [0, 0], "r": [0, 0], "n": [0, 0]}
    ok_n = 0
    labeled_n = 0

    for sid in all_ids:
        dec = decisions[sid]
        conf[dec["kind"]] = conf.get(dec["kind"], 0) + 1
        if dec["kind"] in ("i", "r"):
            labeled_n += 1
            if dec.get("metrics_ok"):
                ok_n += 1
        if args.eval_existing and sid < 100 and sid < len(base_df):
            manual = str(base_df.loc[sid, "type"]).strip().lower()
            if manual in ("i", "r", "n"):
                eval_n += 1
                hit = int(manual == dec["kind"])
                agree += hit
                by_type[manual][1] += 1
                by_type[manual][0] += hit
        if sid % 10 == 0:
            print(
                f"  scene {sid}: type={dec['kind']} img={dec.get('image_idx')} "
                f"idx={dec.get('intervention_idx')} "
                f"ok={dec.get('metrics_ok')} "
                f"mscore={dec.get('metrics_score')} "
                f"cap_gap={dec.get('cap_gap')}"
            )

    print("\n=== auto-label counts ===")
    print(conf)
    if want_metrics and labeled_n:
        print(f"metrics_ok among i/r: {ok_n}/{labeled_n} = {ok_n / labeled_n:.3f}")
    if eval_n:
        print(f"agreement on first 100: {agree}/{eval_n} = {agree / eval_n:.3f}")
        for k in ("i", "r", "n"):
            a, t = by_type[k]
            if t:
                print(f"  recall[{k}]: {a}/{t} = {a / t:.3f}")

    dbg = os.path.join(
        os.path.dirname(out_csv) or ".", "intervention_auto_label_debug.csv"
    )
    pd.DataFrame(
        [
            {
                "scene_idx": sid,
                "type": d["kind"],
                "score": d.get("score"),
                "image_idx": d.get("image_idx"),
                "intervention_idx": d.get("intervention_idx"),
                "ego_err": d.get("ego_err"),
                "near_err": d.get("near_err"),
                "ego_inconsist": d.get("ego_inconsist"),
                "near_inconsist": d.get("near_inconsist"),
                "overlap": d.get("overlap"),
                "dist": d.get("dist"),
                "adapt_subtype": d.get("adapt_subtype"),
                "gate_corridor": d.get("gate_corridor"),
                "gate_closing": d.get("gate_closing"),
                "label_mismatch": d.get("label_mismatch"),
                "label_shift_m": d.get("label_shift_m"),
                "adapt_n_cells": d.get("adapt_n_cells"),
                "gate_spatial_sep": d.get("gate_spatial_sep"),
                "gate_lp_apart": d.get("gate_lp_apart"),
                **{
                    f"tgt_h{h}_{ax}": (
                        float(d["tgt_rel_xy"][i, j])
                        if d.get("tgt_rel_xy") is not None
                        and np.asarray(d["tgt_rel_xy"]).shape[0] > i
                        else np.nan
                    )
                    for i, h in enumerate(FUTURE_STEPS)
                    for j, ax in enumerate(("x", "y"))
                },
                "metrics_ok": d.get("metrics_ok"),
                "metrics_score": d.get("metrics_score"),
                "cap_gap": d.get("cap_gap"),
                "lp_flip_sem": d.get("lp_flip_sem"),
                "ade_sem": d.get("ade_sem"),
                "hard": d.get("hard"),
                "gpg_gap": d.get("gpg_gap"),
                "plan_coll_base": d.get("plan_coll_base"),
                "manual_type": (
                    str(base_df.loc[sid, "type"]).strip().lower()
                    if sid < len(base_df)
                    else ""
                ),
            }
            for sid, d in sorted(decisions.items())
        ]
    ).to_csv(dbg, index=False)
    print(f"wrote {dbg}")
    if metrics_dbg is not None and len(metrics_dbg):
        mpath = os.path.join(
            os.path.dirname(out_csv) or ".", "intervention_auto_label_metrics.csv"
        )
        metrics_dbg.to_csv(mpath, index=False)
        print(f"wrote {mpath}")

    if args.dry_run:
        print("dry-run: not writing intervention CSV")
        return

    # Never truncate previously labeled rows when appending a later block
    # (e.g. start_idx=200 must not wipe rows 300..499 written by another job).
    max_needed = max(all_ids) + 1 if all_ids else len(base_df)
    max_needed = max(max_needed, len(base_df), len(base_oth))
    rows = []
    orows = []
    for i in range(max_needed):
        if i in decisions:
            r, o = decision_to_rows(decisions[i])
            rows.append(r)
            orows.append(o)
        elif i < len(base_df):
            rows.append(base_df.loc[i, CSV_COLS].to_dict())
            orows.append(
                base_oth.loc[i, OTHER_COLS].to_dict()
                if i < len(base_oth)
                else empty_other_row()
            )
        else:
            rows.append(empty_row("n"))
            orows.append(empty_other_row())

    out_df = pd.DataFrame(rows)[CSV_COLS]
    out_o = pd.DataFrame(orows)[OTHER_COLS]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    out_o.to_csv(out_oth, index=False)
    print(f"wrote {out_csv} ({len(out_df)} rows)")
    print(f"wrote {out_oth} ({len(out_o)} rows)")


if __name__ == "__main__":
    main()
