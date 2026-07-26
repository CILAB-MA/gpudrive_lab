"""Intervention Adaptiveness / Recovery metrics from CSV (+ optional paired rollouts).

CSV `type`: i=adaptiveness, r=recovery, n=not-related (default when missing).
Row index == scene id.

Adaptiveness (i) — Collision-Avoidance Planning (preferred continuous metric)
---------------------------------------------------------------------------
Given paired ego rollouts and a counterfactual target trajectory over
horizon H (default 40):

    min_clearance_base = min_t clearance(x_ego_base[t], x_target_cf[t])
    min_clearance_int  = min_t clearance(x_ego_int[t],  x_target_cf[t])
    CAP = min_clearance_int - min_clearance_base   # higher => better avoidance

Clearance sign convention:
    >0  footprints separated
     0  touching
    <0  overlapping

Default clearance is **oriented-bbox SAT separation** when length/width/yaw
are provided. If only centers are available, falls back to:

    ||c_ego - c_tgt|| - r_ego - r_tgt

with r = half-diagonal of the vehicle box (documented approximation).

Without simulator target-trajectory injection, CAP is a
**counterfactual planning-clearance** metric (ego vs stored CF target path),
not an on-simulator collision-rate change. When
`inject_target_trajectory=True` columns are present, CAP is labeled as
injected / on-sim.

Legacy binary CAG = collision_orig - collision_intervened is still supported
when those columns exist.

Recovery (r)
------------
    GPG = goal_progress_intervened - goal_progress_orig  (higher => better)
"""
from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

TYPE_MAP = {
    "i": "adaptiveness",
    "r": "recovery",
    "n": "not_related",
}
DEFAULT_TYPE = "n"
DEFAULT_CATEGORY = "not_related"

# LP discrete bins used by IL linear probes (normalized relative coords).
LP_NORM_EDGES = np.linspace(-0.05, 0.05, 9)
LP_HORIZONS = (10, 20, 30, 40)
MIN_REL_AGENT_POS = -1000.0
MAX_REL_AGENT_POS = 1000.0

ArrayLike = Union[float, int, np.ndarray, pd.Series]


# ---------------------------------------------------------------------------
# Basic gains (legacy + recovery)
# ---------------------------------------------------------------------------

def collision_avoidance_gain(
    c_orig: ArrayLike, c_intervened: ArrayLike
) -> np.ndarray:
    """CAG = collision_orig - collision_intervened (higher => more avoidance)."""
    return np.asarray(c_orig, dtype=float) - np.asarray(c_intervened, dtype=float)


def goal_progress_gain(
    gp_orig: ArrayLike, gp_intervened: ArrayLike
) -> np.ndarray:
    """GPG = goal_progress_intervened - goal_progress_orig (higher => better recovery)."""
    return np.asarray(gp_intervened, dtype=float) - np.asarray(gp_orig, dtype=float)


def clearance_avoidance_gain(
    min_clearance_base: ArrayLike, min_clearance_int: ArrayLike
) -> np.ndarray:
    """CAP = min_clearance_int - min_clearance_base (higher => more clearance)."""
    return (
        np.asarray(min_clearance_int, dtype=float)
        - np.asarray(min_clearance_base, dtype=float)
    )


# ---------------------------------------------------------------------------
# Geometry / clearance
# ---------------------------------------------------------------------------

def half_diagonal(length: float, width: float) -> float:
    """Collision radius approximation: half of the oriented-bbox diagonal."""
    return 0.5 * float(np.hypot(length, width))


def clearance_radius(
    ego_xy: np.ndarray,
    tgt_xy: np.ndarray,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    tgt_length: float = 4.5,
    tgt_width: float = 2.0,
) -> float:
    """Center-distance minus sum of half-diagonals (approx footprint clearance).

    Positive => separated, 0 => touching, negative => overlapping (approx).
    """
    ego_xy = np.asarray(ego_xy, dtype=float).reshape(2)
    tgt_xy = np.asarray(tgt_xy, dtype=float).reshape(2)
    dist = float(np.linalg.norm(ego_xy - tgt_xy))
    return dist - half_diagonal(ego_length, ego_width) - half_diagonal(
        tgt_length, tgt_width
    )


def _obb_corners(
    x: float, y: float, length: float, width: float, yaw: float
) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    u = np.array([c, s])
    ut = np.array([-s, c])
    pt = np.array([x, y], dtype=float)
    return np.stack(
        [
            pt + (length / 2) * u - (width / 2) * ut,
            pt + (length / 2) * u + (width / 2) * ut,
            pt - (length / 2) * u + (width / 2) * ut,
            pt - (length / 2) * u - (width / 2) * ut,
        ],
        axis=0,
    )


def _project(poly: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    dots = poly @ axis
    return float(dots.min()), float(dots.max())


def _sat_overlap_and_separation(
    a: np.ndarray, b: np.ndarray
) -> Tuple[bool, float]:
    """Return (overlapping, signed_clearance) via SAT on convex quads.

    Signed clearance: +min_gap if separated, -min_penetration if overlapping.
    """
    axes = []
    for poly in (a, b):
        for i in range(len(poly)):
            edge = poly[(i + 1) % len(poly)] - poly[i]
            n = np.array([-edge[1], edge[0]], dtype=float)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            axes.append(n / norm)

    min_pos_gap = np.inf
    min_neg_pen = np.inf
    separated = False
    for axis in axes:
        amin, amax = _project(a, axis)
        bmin, bmax = _project(b, axis)
        if amax < bmin or bmax < amin:
            separated = True
            gap = bmin - amax if amax < bmin else amin - bmax
            min_pos_gap = min(min_pos_gap, gap)
        else:
            pen = min(amax - bmin, bmax - amin)
            min_neg_pen = min(min_neg_pen, pen)

    if separated:
        return False, float(min_pos_gap if np.isfinite(min_pos_gap) else 0.0)
    return True, float(-min_neg_pen if np.isfinite(min_neg_pen) else 0.0)


def clearance_obb(
    ego_xy: np.ndarray,
    ego_yaw: float,
    ego_length: float,
    ego_width: float,
    tgt_xy: np.ndarray,
    tgt_yaw: float,
    tgt_length: float,
    tgt_width: float,
) -> float:
    """Signed clearance between two oriented bounding boxes (SAT)."""
    a = _obb_corners(
        float(ego_xy[0]), float(ego_xy[1]), ego_length, ego_width, ego_yaw
    )
    b = _obb_corners(
        float(tgt_xy[0]), float(tgt_xy[1]), tgt_length, tgt_width, tgt_yaw
    )
    _, signed = _sat_overlap_and_separation(a, b)
    return signed


def clearance_pair(
    ego_xy: np.ndarray,
    tgt_xy: np.ndarray,
    ego_yaw: Optional[float] = None,
    tgt_yaw: Optional[float] = None,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    tgt_length: float = 4.5,
    tgt_width: float = 2.0,
    method: str = "auto",
) -> float:
    """Clearance between ego and target at one timestep.

    method:
      - 'obb': oriented-bbox SAT (requires yaw)
      - 'radius': center-distance half-diagonal approximation
      - 'auto': obb if yaws given else radius
    """
    if method == "auto":
        method = (
            "obb"
            if ego_yaw is not None and tgt_yaw is not None
            else "radius"
        )
    if method == "obb":
        return clearance_obb(
            ego_xy,
            float(ego_yaw),
            ego_length,
            ego_width,
            tgt_xy,
            float(tgt_yaw),
            tgt_length,
            tgt_width,
        )
    return clearance_radius(
        ego_xy, tgt_xy, ego_length, ego_width, tgt_length, tgt_width
    )


def min_clearance_over_horizon(
    ego_traj: np.ndarray,
    tgt_traj: np.ndarray,
    ego_yaw: Optional[np.ndarray] = None,
    tgt_yaw: Optional[np.ndarray] = None,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    tgt_length: float = 4.5,
    tgt_width: float = 2.0,
    method: str = "auto",
) -> float:
    """min_t clearance(ego[t], target[t]) over aligned horizon.

    ego_traj, tgt_traj: (H, 2)
    """
    ego_traj = np.asarray(ego_traj, dtype=float)
    tgt_traj = np.asarray(tgt_traj, dtype=float)
    if ego_traj.ndim != 2 or tgt_traj.ndim != 2:
        raise ValueError("ego_traj and tgt_traj must be (H, 2)")
    H = min(len(ego_traj), len(tgt_traj))
    if H == 0:
        return float("nan")
    vals = []
    for t in range(H):
        ey = None if ego_yaw is None else float(np.asarray(ego_yaw)[t])
        ty = None if tgt_yaw is None else float(np.asarray(tgt_yaw)[t])
        vals.append(
            clearance_pair(
                ego_traj[t],
                tgt_traj[t],
                ego_yaw=ey,
                tgt_yaw=ty,
                ego_length=ego_length,
                ego_width=ego_width,
                tgt_length=tgt_length,
                tgt_width=tgt_width,
                method=method,
            )
        )
    return float(np.min(vals))


# ---------------------------------------------------------------------------
# LP label ↔ counterfactual relative pose
# ---------------------------------------------------------------------------

def lp_class_to_norm_xy(cls: int) -> Tuple[float, float]:
    """Decode discrete LP class in [0, 63] to normalized relative (x, y)."""
    cls = int(np.clip(cls, 0, 63))
    x_bin, y_bin = divmod(cls, 8)
    edges = LP_NORM_EDGES
    x = 0.5 * (edges[x_bin] + edges[x_bin + 1])
    y = 0.5 * (edges[y_bin] + edges[y_bin + 1])
    return float(x), float(y)


def norm_xy_to_rel_meters(nx: float, ny: float) -> Tuple[float, float]:
    """Undo IL relative-pose normalization to meters."""
    span = MAX_REL_AGENT_POS - MIN_REL_AGENT_POS
    rx = ((nx + 1.0) / 2.0) * span + MIN_REL_AGENT_POS
    ry = ((ny + 1.0) / 2.0) * span + MIN_REL_AGENT_POS
    return float(rx), float(ry)


def lp_class_to_rel_meters(cls: int) -> Tuple[float, float]:
    nx, ny = lp_class_to_norm_xy(cls)
    return norm_xy_to_rel_meters(nx, ny)


def rotate2d(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
    c, s = np.cos(yaw), np.sin(yaw)
    return float(c * dx - s * dy), float(s * dx + c * dy)


def build_counterfactual_target_traj(
    labels_by_horizon: dict,
    ego_xy0: Sequence[float],
    ego_yaw0: float,
    horizon: int = 40,
    horizons: Sequence[int] = LP_HORIZONS,
) -> np.ndarray:
    """Interpolate CF target global XY from LP class labels at probe horizons.

    labels_by_horizon: {10: cls, 20: cls, ...} relative to ego at intervene time.
    Returns (horizon, 2) global positions. Missing labels => NaN row skipped in interp.
    """
    ego_xy0 = np.asarray(ego_xy0, dtype=float).reshape(2)
    ts, pts = [], []
    for h in horizons:
        if h not in labels_by_horizon or pd.isna(labels_by_horizon[h]):
            continue
        rx, ry = lp_class_to_rel_meters(int(labels_by_horizon[h]))
        gx, gy = rotate2d(rx, ry, ego_yaw0)
        ts.append(int(h))
        pts.append(ego_xy0 + np.array([gx, gy]))
    out = np.full((horizon, 2), np.nan, dtype=float)
    if not ts:
        return out
    ts = np.asarray(ts, dtype=float)
    pts = np.asarray(pts, dtype=float)
    # Include t=0 as current relative origin? CF is future — start from first label.
    query = np.arange(1, horizon + 1, dtype=float)
    for d in range(2):
        out[:, d] = np.interp(query, ts, pts[:, d], left=pts[0, d], right=pts[-1, d])
    return out


def compute_paired_clearances(
    ego_base: np.ndarray,
    ego_int: np.ndarray,
    tgt_cf: np.ndarray,
    ego_yaw_base: Optional[np.ndarray] = None,
    ego_yaw_int: Optional[np.ndarray] = None,
    tgt_yaw: Optional[np.ndarray] = None,
    ego_length: float = 4.5,
    ego_width: float = 2.0,
    tgt_length: float = 4.5,
    tgt_width: float = 2.0,
    method: str = "auto",
) -> dict:
    """Compute min clearances and CAP for one paired adaptiveness case."""
    min_base = min_clearance_over_horizon(
        ego_base,
        tgt_cf,
        ego_yaw=ego_yaw_base,
        tgt_yaw=tgt_yaw,
        ego_length=ego_length,
        ego_width=ego_width,
        tgt_length=tgt_length,
        tgt_width=tgt_width,
        method=method,
    )
    min_int = min_clearance_over_horizon(
        ego_int,
        tgt_cf,
        ego_yaw=ego_yaw_int,
        tgt_yaw=tgt_yaw,
        ego_length=ego_length,
        ego_width=ego_width,
        tgt_length=tgt_length,
        tgt_width=tgt_width,
        method=method,
    )
    cap = clearance_avoidance_gain(min_base, min_int)
    return {
        "min_clearance_base": min_base,
        "min_clearance_int": min_int,
        "clearance_gain": float(cap) if np.ndim(cap) == 0 else float(np.asarray(cap).reshape(-1)[0]),
        "clearance_improved": float(cap > 0) if np.isfinite(cap) else float("nan"),
    }


# ---------------------------------------------------------------------------
# CSV load / attach / summarize
# ---------------------------------------------------------------------------

def _find_changed_column(columns) -> Optional[str]:
    for c in columns:
        if c == "changed?" or str(c).startswith("changed?"):
            return c
    return None


def _parse_ox(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s == "o":
        return 1.0
    if s == "x":
        return 0.0
    return np.nan


def _extract_human_success(df: pd.DataFrame) -> pd.Series:
    ox_cols = []
    for c in df.columns:
        sample = df[c]
        if sample.dtype == object or str(c).startswith("Unnamed"):
            vals = sample.dropna().astype(str).str.strip().str.lower()
            if len(vals) and vals.isin(["o", "x"]).mean() > 0.5:
                ox_cols.append(c)
    if not ox_cols:
        return pd.Series(np.nan, index=df.index, dtype=float)
    parsed = pd.concat([df[c].map(_parse_ox) for c in ox_cols], axis=1)
    return parsed.max(axis=1, skipna=True)


def load_and_normalize(
    csv_path: str,
    start_idx: int = 0,
    num_scenes: Optional[int] = None,
) -> pd.DataFrame:
    """Load intervention CSV; preserve row_index as scene_idx. Missing type -> n."""
    df = pd.read_csv(csv_path)
    df = df.copy()
    df["scene_idx"] = df.index

    end_idx = (start_idx + num_scenes) if num_scenes is not None else (df.index.max() + 1)
    if end_idx > len(df):
        pad = pd.DataFrame({col: [np.nan] * (end_idx - len(df)) for col in df.columns})
        pad["scene_idx"] = range(len(df), end_idx)
        df = pd.concat([df, pad], ignore_index=True)

    df = df.iloc[start_idx:end_idx].copy()
    df["scene_idx"] = df["scene_idx"].astype(int)

    type_raw = df["type"] if "type" in df.columns else pd.Series(np.nan, index=df.index)
    cleaned = (
        type_raw.astype(str)
        .str.strip()
        .str.lower()
        .replace({"": DEFAULT_TYPE, "nan": DEFAULT_TYPE, "none": DEFAULT_TYPE})
    )
    cleaned = cleaned.where(cleaned.isin(TYPE_MAP), DEFAULT_TYPE)
    df["type_raw"] = cleaned
    df["category"] = cleaned.map(TYPE_MAP).fillna(DEFAULT_CATEGORY)

    changed_col = _find_changed_column(df.columns)
    if changed_col is not None:
        df["changed"] = pd.to_numeric(df[changed_col], errors="coerce")
    else:
        df["changed"] = np.nan

    df["human_success"] = _extract_human_success(df)

    # Adaptiveness placeholders
    df["cag"] = np.nan
    df["avoidance_success"] = np.nan
    df["min_clearance_base"] = np.nan
    df["min_clearance_int"] = np.nan
    df["clearance_gain"] = np.nan
    df["clearance_improved"] = np.nan
    df["clearance_metric_kind"] = pd.NA

    # Recovery placeholders
    df["goal_progress_gain"] = np.nan
    df["recovery_improved"] = np.nan
    return df.reset_index(drop=True)


def attach_rollout_metrics(
    df: pd.DataFrame,
    rollout: Union[str, pd.DataFrame, dict],
) -> pd.DataFrame:
    """Attach per-scene sim / paired-rollout outcomes and fill metrics.

    Expected columns (keyed by scene_idx), any subset:

    Adaptiveness (i):
      # continuous CAP (preferred)
      min_clearance_base, min_clearance_int
      # optional: precomputed clearance_gain
      # optional: inject_target_trajectory (bool) → metric kind
      # legacy binary
      collision_orig, collision_intervened

    Recovery (r):
      goal_progress_orig, goal_progress_intervened
      (fallback: goal_orig / goal_intervened)
    """
    out = df.copy()
    if isinstance(rollout, str):
        roll = pd.read_csv(rollout)
    elif isinstance(rollout, dict):
        roll = pd.DataFrame(rollout)
    else:
        roll = rollout.copy()

    if "scene_idx" not in roll.columns:
        roll = roll.reset_index().rename(columns={"index": "scene_idx"})

    roll = roll.set_index("scene_idx")
    out = out.set_index("scene_idx")

    for col in (
        "collision_orig",
        "collision_intervened",
        "goal_progress_orig",
        "goal_progress_intervened",
        "goal_orig",
        "goal_intervened",
        "min_clearance_base",
        "min_clearance_int",
        "clearance_gain",
        "clearance_improved",
        "inject_target_trajectory",
        "clearance_metric_kind",
        "off_road_orig",
        "off_road_intervened",
        "intervene_step",
        "horizon",
        "mode",
    ):
        if col in roll.columns:
            out[col] = roll[col]

    adapt_mask = out["category"] == "adaptiveness"

    # Continuous CAP from clearances
    if {"min_clearance_base", "min_clearance_int"}.issubset(out.columns):
        c0 = out["min_clearance_base"].astype(float)
        c1 = out["min_clearance_int"].astype(float)
        valid = adapt_mask & c0.notna() & c1.notna()
        if valid.any():
            if "clearance_gain" not in out.columns or out.loc[valid, "clearance_gain"].isna().all():
                cap = clearance_avoidance_gain(c0, c1)
                out.loc[valid, "clearance_gain"] = np.asarray(cap)[valid.to_numpy()]
            out.loc[valid, "clearance_improved"] = (
                out.loc[valid, "clearance_gain"] > 0
            ).astype(float)
            # Label metric kind
            if "inject_target_trajectory" in out.columns:
                inj = out["inject_target_trajectory"].astype(float).fillna(0) > 0.5
                out.loc[valid & inj, "clearance_metric_kind"] = "injected_target_clearance"
                out.loc[valid & ~inj, "clearance_metric_kind"] = (
                    "counterfactual_planning_clearance"
                )
            else:
                out.loc[valid, "clearance_metric_kind"] = (
                    "counterfactual_planning_clearance"
                )

    # Legacy binary CAG
    if {"collision_orig", "collision_intervened"}.issubset(out.columns):
        c0 = out["collision_orig"].astype(float)
        c1 = out["collision_intervened"].astype(float)
        valid = adapt_mask & c0.notna() & c1.notna()
        if valid.any():
            cag = collision_avoidance_gain(c0, c1)
            out.loc[valid, "cag"] = np.asarray(cag)[valid.to_numpy()]
            out.loc[valid, "avoidance_success"] = (out.loc[valid, "cag"] > 0).astype(
                float
            )

    # Recovery GPG
    rec_mask = out["category"] == "recovery"
    if {"goal_progress_orig", "goal_progress_intervened"}.issubset(out.columns):
        gp0, gp1 = out["goal_progress_orig"], out["goal_progress_intervened"]
    elif {"goal_orig", "goal_intervened"}.issubset(out.columns):
        gp0, gp1 = out["goal_orig"], out["goal_intervened"]
    else:
        gp0 = gp1 = None

    if gp0 is not None:
        gp0 = gp0.astype(float)
        gp1 = gp1.astype(float)
        valid = rec_mask & gp0.notna() & gp1.notna()
        if valid.any():
            gpg = goal_progress_gain(gp0, gp1)
            out.loc[valid, "goal_progress_gain"] = np.asarray(gpg)[valid.to_numpy()]
            out.loc[valid, "recovery_improved"] = (
                out.loc[valid, "goal_progress_gain"] > 0
            ).astype(float)

    return out.reset_index()


def _rate(mask: pd.Series, values: pd.Series) -> float:
    sub = values[mask]
    sub = sub.dropna()
    if len(sub) == 0:
        return float("nan")
    return float(sub.mean())


def category_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat in ("adaptiveness", "recovery", "not_related"):
        m = df["category"] == cat
        n = int(m.sum())
        changed_valid = m & df["changed"].isin([0, 1])
        changed_rate = _rate(changed_valid, (df["changed"] == 1).astype(float))
        human_rate = _rate(m & df["human_success"].notna(), df["human_success"])

        row = {
            "category": cat,
            "n": n,
            "changed_rate": changed_rate,
            "human_success_rate": human_rate,
            # Adaptiveness — continuous CAP
            "min_clearance_base_mean": float("nan"),
            "min_clearance_int_mean": float("nan"),
            "clearance_gain_mean": float("nan"),
            "clearance_improved_rate": float("nan"),
            "clearance_metric_kind": pd.NA,
            # Adaptiveness — legacy binary
            "cag_mean": float("nan"),
            "avoidance_success_rate": float("nan"),
            # Recovery
            "goal_progress_gain_mean": float("nan"),
            "recovery_improved_rate": float("nan"),
        }
        if cat == "adaptiveness" and m.any():
            row["min_clearance_base_mean"] = float(df.loc[m, "min_clearance_base"].mean())
            row["min_clearance_int_mean"] = float(df.loc[m, "min_clearance_int"].mean())
            row["clearance_gain_mean"] = float(df.loc[m, "clearance_gain"].mean())
            row["clearance_improved_rate"] = _rate(m, df["clearance_improved"])
            kinds = df.loc[m, "clearance_metric_kind"].dropna().unique()
            if len(kinds) == 1:
                row["clearance_metric_kind"] = kinds[0]
            elif len(kinds) > 1:
                row["clearance_metric_kind"] = "mixed"
            row["cag_mean"] = float(df.loc[m, "cag"].mean())
            row["avoidance_success_rate"] = _rate(m, df["avoidance_success"])
        if cat == "recovery" and m.any():
            row["goal_progress_gain_mean"] = float(df.loc[m, "goal_progress_gain"].mean())
            row["recovery_improved_rate"] = _rate(m, df["recovery_improved"])
        rows.append(row)
    return pd.DataFrame(rows)


def per_scene_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scene_idx",
        "type_raw",
        "category",
        "changed",
        "human_success",
        "min_clearance_base",
        "min_clearance_int",
        "clearance_gain",
        "clearance_improved",
        "clearance_metric_kind",
        "cag",
        "avoidance_success",
        "goal_progress_gain",
        "recovery_improved",
    ]
    for extra in (
        "collision_orig",
        "collision_intervened",
        "goal_progress_orig",
        "goal_progress_intervened",
        "goal_orig",
        "goal_intervened",
        "inject_target_trajectory",
        "intervene_step",
        "horizon",
        "mode",
    ):
        if extra in df.columns:
            cols.append(extra)
    return df[[c for c in cols if c in df.columns]].copy()


def parse_args():
    p = argparse.ArgumentParser("Intervention Adaptiveness / Recovery metrics")
    p.add_argument(
        "--csv-path",
        "-cp",
        type=str,
        default="/data/full_version/intervention.csv",
    )
    p.add_argument("--start-idx", "-s", type=int, default=0)
    p.add_argument("--num-scenes", "-n", type=int, default=None)
    p.add_argument(
        "--rollout-csv",
        type=str,
        default=None,
        help=(
            "Per-scene paired-rollout log. Adaptiveness prefers "
            "min_clearance_base/int (CAP); also accepts collision_orig/intervened. "
            "Recovery needs goal_progress_orig/intervened."
        ),
    )
    p.add_argument(
        "--out-csv",
        type=str,
        default="/data/after_cvpr/images/intervention_metrics_summary.csv",
    )
    p.add_argument(
        "--out-scenes-csv",
        type=str,
        default=None,
        help="Optional per-scene metrics path (default: alongside --out-csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = load_and_normalize(args.csv_path, args.start_idx, args.num_scenes)
    if args.rollout_csv:
        df = attach_rollout_metrics(df, args.rollout_csv)

    summary = category_summary(df)
    scenes = per_scene_metrics(df)

    print("=== category summary ===")
    print(summary.to_string(index=False))
    print("\ntype counts:", df["category"].value_counts(dropna=False).to_dict())

    out_csv = args.out_csv
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(out_csv, index=False)

    scenes_path = args.out_scenes_csv
    if scenes_path is None:
        root, ext = os.path.splitext(out_csv)
        scenes_path = f"{root}_per_scene{ext or '.csv'}"
    scenes.to_csv(scenes_path, index=False)
    print(f"\nwrote summary: {out_csv}")
    print(f"wrote per-scene: {scenes_path}")


if __name__ == "__main__":
    main()
