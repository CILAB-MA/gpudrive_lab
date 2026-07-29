"""Intervention Adaptiveness / Recovery metrics (paper final).

CSV `type`: i=adaptiveness, r=recovery, n=not-related (default when missing).
Row index == scene id.

Adaptiveness (CAP) — continuous-target ADE
------------------------------------------
    CAP = ADE(ego_int, x_tgt) - ADE(ego_base, x_tgt)
    Ref = ADE(ego_base, ego_int)
    plan_coll = 1[min_t ||ego_t - tgt_t|| < r_ego + r_partner]
      with r = half-diagonal of default 4.5×2.0 m boxes (~4.92 m sum).

Recovery (GPG)
--------------
    GPG = goal_progress_intervened - goal_progress_orig
    Δcollision / Δoff-road reported as side statistics only.

Geometry helpers (`half_diagonal`, `clearance_radius`) are kept for gate C
spatial separation in labeling.
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


def norm_xy_to_lp_class(nx: float, ny: float) -> int:
    """Quantize normalized relative (x, y) to nearest LP class in [0, 63]."""
    edges = LP_NORM_EDGES
    # Clamp into grid span then find bin index.
    nx = float(np.clip(nx, edges[0], edges[-1] - 1e-12))
    ny = float(np.clip(ny, edges[0], edges[-1] - 1e-12))
    x_bin = int(np.searchsorted(edges, nx, side="right") - 1)
    y_bin = int(np.searchsorted(edges, ny, side="right") - 1)
    x_bin = int(np.clip(x_bin, 0, 7))
    y_bin = int(np.clip(y_bin, 0, 7))
    return int(x_bin * 8 + y_bin)


def pull_lp_class_closer(cls: int, scale: float) -> int:
    """Move an LP class toward the ego (origin) by ``scale`` in norm space.

    scale=1 keeps the label; scale=0 maps to the nearest-to-origin bin;
    scale=0.5 halves the relative offset (stronger / closer CF threat).
    """
    scale = float(scale)
    if scale >= 1.0 - 1e-12:
        return int(cls)
    nx, ny = lp_class_to_norm_xy(cls)
    return norm_xy_to_lp_class(nx * scale, ny * scale)


def pull_labels_closer(labels: Sequence[int], scale: float) -> np.ndarray:
    """Apply ``pull_lp_class_closer`` to a length-4 (or N) label vector."""
    labs = np.asarray(labels, dtype=np.int64).reshape(-1)
    return np.asarray([pull_lp_class_closer(int(c), scale) for c in labs], dtype=np.int64)


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


def build_continuous_target_traj(
    rel_xy_by_horizon: dict,
    ego_xy0: Sequence[float],
    ego_yaw0: float,
    horizon: int = 40,
    horizons: Sequence[int] = LP_HORIZONS,
) -> np.ndarray:
    """Interpolate target global XY from continuous ego-frame relative meters.

    rel_xy_by_horizon: {10: (rx, ry), ...} — no LP grid midpoint decode.
    """
    ego_xy0 = np.asarray(ego_xy0, dtype=float).reshape(2)
    ts, pts = [], []
    for h in horizons:
        if h not in rel_xy_by_horizon:
            continue
        pair = rel_xy_by_horizon[h]
        if pair is None or (isinstance(pair, float) and np.isnan(pair)):
            continue
        rx, ry = float(pair[0]), float(pair[1])
        if not (np.isfinite(rx) and np.isfinite(ry)):
            continue
        gx, gy = rotate2d(rx, ry, ego_yaw0)
        ts.append(int(h))
        pts.append(ego_xy0 + np.array([gx, gy]))
    out = np.full((horizon, 2), np.nan, dtype=float)
    if not ts:
        return out
    ts = np.asarray(ts, dtype=float)
    pts = np.asarray(pts, dtype=float)
    query = np.arange(1, horizon + 1, dtype=float)
    for d in range(2):
        out[:, d] = np.interp(query, ts, pts[:, d], left=pts[0, d], right=pts[-1, d])
    return out


def pull_partner_toward_ego_n_cells(
    partner_cls: int, ego_cls: int, n_cells: int
) -> Tuple[int, Tuple[float, float]]:
    """Move partner LP toward ego by at most ``n_cells`` bins (continuous then quantize).

    Returns (quantized_class, continuous_rel_meters).
    """
    edges = LP_NORM_EDGES
    bin_w = float(edges[1] - edges[0])
    n_cells = int(max(0, n_cells))
    px, py = lp_class_to_norm_xy(int(partner_cls))
    ex, ey = lp_class_to_norm_xy(int(ego_cls))
    max_step = float(n_cells) * bin_w
    nx = px + float(np.clip(ex - px, -max_step, max_step))
    ny = py + float(np.clip(ey - py, -max_step, max_step))
    cls = norm_xy_to_lp_class(nx, ny)
    return int(cls), norm_xy_to_rel_meters(nx, ny)


def pull_labels_toward_ego_n_cells(
    partner_labels: Sequence[int],
    ego_labels: Sequence[int],
    n_cells: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-horizon N-cell pull. Returns (classes[H], continuous_rel_xy[H,2])."""
    p = np.asarray(partner_labels, dtype=np.int64).reshape(-1)
    e = np.asarray(ego_labels, dtype=np.int64).reshape(-1)
    n = min(len(p), len(e))
    classes = np.zeros(n, dtype=np.int64)
    rel = np.zeros((n, 2), dtype=float)
    for i in range(n):
        cls, (rx, ry) = pull_partner_toward_ego_n_cells(int(p[i]), int(e[i]), n_cells)
        classes[i] = cls
        rel[i] = (rx, ry)
    return classes, rel


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

    # Paper metric placeholders
    df["cap_ade"] = np.nan
    df["ade_base_to_int"] = np.nan
    df["plan_coll_base"] = np.nan
    df["plan_coll_int"] = np.nan
    df["clearance_metric_kind"] = pd.NA
    df["goal_progress_gain"] = np.nan
    df["recovery_improved"] = np.nan
    df["delta_collision"] = np.nan
    df["delta_off_road"] = np.nan
    return df.reset_index(drop=True)


def attach_rollout_metrics(
    df: pd.DataFrame,
    rollout: Union[str, pd.DataFrame, dict],
) -> pd.DataFrame:
    """Attach per-scene paired-rollout outcomes and fill paper metrics.

    Adaptiveness: cap_ade, ade_base_to_int, plan_coll_base/int
    Recovery: goal_progress_* → GPG; collision/off_road side deltas
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

    # One row per scene_idx: if multiple controls, keep semantic preferentially
    if "control" in roll.columns and roll["scene_idx"].duplicated().any():
        pref = {"semantic": 0, "intervention": 0, "random": 1, "wrong_label": 2}
        roll = roll.copy()
        roll["_ctrl_rank"] = roll["control"].map(lambda c: pref.get(str(c), 9))
        roll = (
            roll.sort_values(["scene_idx", "_ctrl_rank"])
            .drop_duplicates("scene_idx", keep="first")
            .drop(columns=["_ctrl_rank"])
        )

    roll = roll.set_index("scene_idx")
    out = out.set_index("scene_idx")

    for col in (
        "collision_orig",
        "collision_intervened",
        "goal_progress_orig",
        "goal_progress_intervened",
        "goal_orig",
        "goal_intervened",
        "off_road_orig",
        "off_road_intervened",
        "cap_ade",
        "ade_base_to_tgt",
        "ade_int_to_tgt",
        "ade_base_to_int",
        "plan_coll_base",
        "plan_coll_int",
        "plan_coll_thresh_m",
        "min_dist_base_to_tgt",
        "min_dist_int_to_tgt",
        "clearance_metric_kind",
        "lp_flipped",
        "intervene_step",
        "horizon",
        "mode",
        "control",
    ):
        if col in roll.columns:
            out[col] = roll[col]

    adapt_mask = out["category"] == "adaptiveness"
    if "cap_ade" in out.columns:
        valid = adapt_mask & out["cap_ade"].notna()
        if valid.any() and "clearance_metric_kind" in out.columns:
            out.loc[valid & out["clearance_metric_kind"].isna(), "clearance_metric_kind"] = (
                "continuous_tgt_ade"
            )

    # Recovery GPG + side deltas
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

    if {"collision_orig", "collision_intervened"}.issubset(out.columns):
        c0 = out["collision_orig"].astype(float)
        c1 = out["collision_intervened"].astype(float)
        valid = rec_mask & c0.notna() & c1.notna()
        if valid.any():
            out.loc[valid, "delta_collision"] = (c1 - c0)[valid]

    if {"off_road_orig", "off_road_intervened"}.issubset(out.columns):
        o0 = out["off_road_orig"].astype(float)
        o1 = out["off_road_intervened"].astype(float)
        valid = rec_mask & o0.notna() & o1.notna()
        if valid.any():
            out.loc[valid, "delta_off_road"] = (o1 - o0)[valid]

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
            "cap_ade_mean": float("nan"),
            "ade_base_to_int_mean": float("nan"),
            "plan_coll_base_rate": float("nan"),
            "plan_coll_int_rate": float("nan"),
            "clearance_metric_kind": pd.NA,
            "goal_progress_gain_mean": float("nan"),
            "recovery_improved_rate": float("nan"),
            "delta_collision_mean": float("nan"),
            "delta_off_road_mean": float("nan"),
        }
        if cat == "adaptiveness" and m.any():
            if "cap_ade" in df.columns:
                row["cap_ade_mean"] = float(df.loc[m, "cap_ade"].mean())
            if "ade_base_to_int" in df.columns:
                row["ade_base_to_int_mean"] = float(df.loc[m, "ade_base_to_int"].mean())
            if "plan_coll_base" in df.columns:
                row["plan_coll_base_rate"] = _rate(m, df["plan_coll_base"])
                row["plan_coll_int_rate"] = _rate(m, df["plan_coll_int"])
            if "clearance_metric_kind" in df.columns:
                kinds = df.loc[m, "clearance_metric_kind"].dropna().unique()
                if len(kinds) == 1:
                    row["clearance_metric_kind"] = kinds[0]
                elif len(kinds) > 1:
                    row["clearance_metric_kind"] = "mixed"
        if cat == "recovery" and m.any():
            if "goal_progress_gain" in df.columns:
                row["goal_progress_gain_mean"] = float(df.loc[m, "goal_progress_gain"].mean())
                row["recovery_improved_rate"] = _rate(m, df["recovery_improved"])
            if "delta_collision" in df.columns:
                row["delta_collision_mean"] = float(df.loc[m, "delta_collision"].mean())
            if "delta_off_road" in df.columns:
                row["delta_off_road_mean"] = float(df.loc[m, "delta_off_road"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def per_scene_metrics(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "scene_idx",
        "type_raw",
        "category",
        "changed",
        "human_success",
        "cap_ade",
        "ade_base_to_int",
        "plan_coll_base",
        "plan_coll_int",
        "clearance_metric_kind",
        "goal_progress_gain",
        "recovery_improved",
        "delta_collision",
        "delta_off_road",
    ]
    for extra in (
        "ade_base_to_tgt",
        "ade_int_to_tgt",
        "collision_orig",
        "collision_intervened",
        "goal_progress_orig",
        "goal_progress_intervened",
        "off_road_orig",
        "off_road_intervened",
        "lp_flipped",
        "intervene_step",
        "horizon",
        "mode",
        "control",
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
            "Per-scene paired-rollout log. Adaptiveness: cap_ade / ade_base_to_int "
            "/ plan_coll_*; recovery: goal_progress_orig/intervened."
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
