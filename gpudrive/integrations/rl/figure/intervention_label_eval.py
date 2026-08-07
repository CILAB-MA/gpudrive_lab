"""Paired-intervention metric check used during auto-labeling.

Runs base / semantic / random rollouts for candidate labels and scores whether
the label is likely to produce the paper metrics we care about — so labeling
and metric screening happen in one pass.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from gpudrive.integrations.rl.figure.intervention_paired_rollout import (
    ALPHA_DEFAULT,
    FUTURE_STEPS,
    build_probe_delta,
    evaluate_probe_chain,
    metrics_row_from_pair,
    rollout_batch,
)


def _finite(x, default=np.nan) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float(default)
    return v if np.isfinite(v) else float(default)


def judge_metrics(
    kind: str,
    sem: dict,
    rnd: dict,
    *,
    labels: np.ndarray | None = None,
    partner_gt: np.ndarray | None = None,
    near_err: float | None = None,
) -> dict:
    """Score whether a label looks metric-friendly for paper CAP / GPG."""
    kind = str(kind).lower()
    out = {
        "metrics_ok": False,
        "metrics_score": float("-inf"),
        "lp_flip_sem": _finite(sem.get("lp_flipped"), 0.0),
        "lp_flip_rnd": _finite(rnd.get("lp_flipped"), 0.0),
        "ade_sem": _finite(sem.get("ade_base_to_int")),
        "ade_rnd": _finite(rnd.get("ade_base_to_int")),
        "cap_ade_sem": _finite(sem.get("cap_ade")),
        "cap_ade_rnd": _finite(rnd.get("cap_ade")),
        "plan_coll_base": _finite(sem.get("plan_coll_base")),
        "gpg_sem": _finite(sem.get("goal_progress_intervened"))
        - _finite(sem.get("goal_progress_orig")),
        "gpg_rnd": _finite(rnd.get("goal_progress_intervened"))
        - _finite(rnd.get("goal_progress_orig")),
    }
    out["cap_gap"] = out["cap_ade_sem"] - out["cap_ade_rnd"]
    out["gpg_gap"] = out["gpg_sem"] - out["gpg_rnd"]
    out["hard"] = float(
        np.isfinite(out["plan_coll_base"]) and out["plan_coll_base"] > 0.5
    )

    if kind == "i":
        n_mis = _finite(sem.get("lp_n_label_changed"), 0.0)
        if not (np.isfinite(n_mis) and n_mis >= 1.0 - 1e-8):
            match_base = _finite(sem.get("lp_sem_match_base"), 1.0)
            if np.isfinite(match_base) and match_base < 1.0 - 1e-8:
                n_mis = max(1.0, round(4.0 * (1.0 - match_base)))
            else:
                n_mis = 0.0
        flip_ok = bool(n_mis >= 1.0 - 1e-8) or (out["lp_flip_sem"] > 0.5)
        out["lp_n_misaligned"] = float(n_mis)

        score = 0.0
        if np.isfinite(out["cap_gap"]):
            score += 2.0 * out["cap_gap"]
        score += 0.5 * float(flip_ok)
        score += 0.25 * min(float(n_mis), 4.0)
        if np.isfinite(out["ade_sem"]):
            score += 0.05 * min(out["ade_sem"], 10.0)
        score += 0.5 * out["hard"]

        ade_ok = np.isfinite(out["ade_sem"]) and out["ade_sem"] >= 0.05
        ctrl_ok = (
            (np.isfinite(out["cap_gap"]) and out["cap_gap"] > -0.35)
            or (out["lp_flip_sem"] > 0.5)
        )
        ok = flip_ok and (ade_ok or ctrl_ok)
        out["metrics_ok"] = bool(ok)
        out["metrics_score"] = float(score)
        return out

    if kind == "r":
        n_mis = 0.0
        if partner_gt is not None and labels is not None:
            gt = np.asarray(partner_gt, dtype=np.int64).reshape(-1)
            lab = np.asarray(labels, dtype=np.int64).reshape(-1)
            n = min(len(gt), len(lab), 4)
            if n > 0:
                n_mis = float(np.sum(gt[:n] != lab[:n]))
        elif near_err is not None and np.isfinite(float(near_err)):
            n_mis = float(round(4.0 * float(near_err)))
        traj_mismatch_ok = bool(n_mis >= 1.0 - 1e-8)
        out["lp_flip_sem"] = float(traj_mismatch_ok)
        out["lp_n_misaligned"] = float(n_mis)

        score = 0.0
        score += 1.0 * float(traj_mismatch_ok)
        score += 0.5 * min(float(n_mis), 4.0)
        if np.isfinite(out["gpg_gap"]):
            score += 1.0 * out["gpg_gap"]
        gpg_ok = (not np.isfinite(out["gpg_gap"])) or (out["gpg_gap"] > -0.05)
        ok = traj_mismatch_ok and gpg_ok
        out["metrics_ok"] = bool(ok)
        out["metrics_score"] = float(score)
        return out

    out["metrics_ok"] = False
    out["metrics_score"] = float("-inf")
    return out


def _dec_to_meta(sid: int, dec: dict) -> dict:
    kind = dec["kind"]
    cat = {"i": "adaptiveness", "r": "recovery", "n": "not_related"}[kind]
    oidx = list(dec.get("other_indices", []))
    while len(oidx) < 3:
        oidx.append(-1)
    olabels = []
    for labs in dec.get("other_labels", []):
        olabels.extend([int(x) for x in np.asarray(labs).reshape(-1)[:4]])
    while len(olabels) < 12:
        olabels.append(0)
    image_idx = dec.get("image_idx", -1)
    intervene_step = int(image_idx) * 3 if image_idx is not None and int(image_idx) >= 0 else 0
    partner_gt = dec.get("partner_gt", None)
    if partner_gt is not None:
        partner_gt = np.asarray(partner_gt, dtype=np.int64).reshape(-1)[:4]
    return {
        "sid": int(sid),
        "idx": int(dec.get("intervention_idx", 0)),
        "labels": np.asarray(dec.get("labels", [0, 0, 0, 0]), dtype=np.int64).reshape(4),
        "partner_gt": partner_gt,
        "near_err": (
            float(dec["near_err"])
            if dec.get("near_err", None) is not None
            else None
        ),
        "oidx": [int(x) for x in oidx[:3]],
        "olabels": [int(x) for x in olabels[:12]],
        "image_idx": int(image_idx) if image_idx is not None else -1,
        "intervene_step": intervene_step,
        "continue_ep": kind == "r",
        "category": cat,
        "kind": kind,
    }


@torch.no_grad()
def eval_label_batch(
    env,
    policy,
    other_lps,
    metas: list,
    valid: list,
    alpha: float = ALPHA_DEFAULT,
    persistent_k: int = 10,
    horizon: int = 40,
    intervention_reduce: str = "mean",
) -> list[dict]:
    """Run base+semantic+random for one env batch; return per-valid-world judges."""
    args_ns = SimpleNamespace(
        horizon=horizon,
        mode="persistent_k",
        persistent_k=persistent_k,
        alpha=alpha,
        early_h=10,
        inject_until_k=-1,
        suppress_k=-1,
    )
    B = len(metas)
    expert_actions, *_ = env.get_expert_actions()
    intervene_steps = [m["intervene_step"] for m in metas]
    continue_flags = [m["continue_ep"] for m in metas]

    bases, base_toks = rollout_batch(
        env,
        policy,
        expert_actions,
        intervene_steps,
        horizon,
        partner_deltas=None,
        mode="persistent_k",
        persistent_k=persistent_k,
        continue_flags=continue_flags,
        valid_worlds=valid,
    )

    control_rows = {"semantic": {}, "random": {}}
    for control in ("semantic", "random"):
        deltas = []
        probe_deltas = []
        for w, m in enumerate(metas):
            d = build_probe_delta(
                other_lps,
                intervention_idx=m["idx"],
                labels_4=m["labels"],
                other_indices=m["oidx"],
                other_labels_12=m["olabels"],
                mode=intervention_reduce,
                control=control,
                device="cuda",
                seed=int(m["sid"]) * 1009 + 17,
                alpha=alpha,
            )
            deltas.append(d.squeeze(0))
            probe_deltas.append(d)
        partner_deltas = torch.stack(deltas, dim=0)
        inters, branch_toks = rollout_batch(
            env,
            policy,
            expert_actions,
            intervene_steps,
            horizon,
            partner_deltas=partner_deltas,
            mode="persistent_k",
            persistent_k=persistent_k,
            continue_flags=continue_flags,
            valid_worlds=valid,
        )
        for w in range(B):
            if not valid[w]:
                continue
            base = bases[w]
            inter = inters[w]
            if base is None or inter is None:
                continue
            m = metas[w]
            tok = base_toks[w] if base_toks[w] is not None else branch_toks[w]
            probe = {}
            if tok is not None:
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
                args_ns,
                control,
                base,
                inter,
                m["labels"],
                probe=probe,
            )
            control_rows[control][w] = row

    results = [None] * B
    for w in range(B):
        if not valid[w]:
            continue
        m = metas[w]
        sem = control_rows["semantic"].get(w)
        rnd = control_rows["random"].get(w)
        if sem is None or rnd is None:
            results[w] = dict(
                sid=m["sid"],
                kind=m["kind"],
                metrics_ok=False,
                metrics_score=float("-inf"),
                reason="rollout_failed",
            )
            continue
        judged = judge_metrics(
            m["kind"],
            sem,
            rnd,
            labels=m.get("labels"),
            partner_gt=m.get("partner_gt"),
            near_err=m.get("near_err"),
        )
        judged["sid"] = m["sid"]
        judged["kind"] = m["kind"]
        judged["image_idx"] = m["image_idx"]
        judged["intervention_idx"] = m["idx"]
        results[w] = judged
    return results


def eval_decisions_over_scenes(
    make_env_fn,
    policy,
    other_lps,
    scene_decisions: list[tuple[int, dict]],
    batch_size: int = 48,
    alpha: float = ALPHA_DEFAULT,
    persistent_k: int = 10,
    horizon: int = 40,
) -> dict[int, dict]:
    """Evaluate (sid, decision) pairs; return sid -> judge dict.

    If multiple decisions share a sid, the last one wins unless caller
    disambiguates (use eval_candidates_pick instead).
    """
    # Filter n
    items = [(sid, dec) for sid, dec in scene_decisions if dec.get("kind") in ("i", "r")]
    out: dict[int, dict] = {}
    if not items:
        return out

    sids = [sid for sid, _ in items]
    env, flat, pad_n = make_env_fn(sids, batch_size)
    B = batch_size
    n_batches = len(flat) // B
    dec_by_sid = {sid: dec for sid, dec in items}

    try:
        for bi in tqdm(range(n_batches), desc="label-metrics"):
            batch_sids = flat[bi * B : (bi + 1) * B]
            if bi == n_batches - 1 and pad_n:
                valid = [True] * (B - pad_n) + [False] * pad_n
            else:
                valid = [True] * B
            metas = []
            seen = set()
            for w, sid in enumerate(batch_sids):
                if not valid[w]:
                    metas.append(
                        _dec_to_meta(
                            sid,
                            dict(
                                kind="n",
                                intervention_idx=0,
                                labels=np.zeros(4, dtype=np.int64),
                                other_indices=[],
                                other_labels=[],
                                image_idx=-1,
                            ),
                        )
                    )
                    continue
                if sid in seen:
                    # padded duplicate
                    valid[w] = False
                    metas.append(
                        _dec_to_meta(
                            sid,
                            dict(
                                kind="n",
                                intervention_idx=0,
                                labels=np.zeros(4, dtype=np.int64),
                                other_indices=[],
                                other_labels=[],
                                image_idx=-1,
                            ),
                        )
                    )
                    continue
                seen.add(sid)
                metas.append(_dec_to_meta(sid, dec_by_sid[sid]))
            judged = eval_label_batch(
                env,
                policy,
                other_lps,
                metas,
                valid,
                alpha=alpha,
                persistent_k=persistent_k,
                horizon=horizon,
            )
            for w, j in enumerate(judged):
                if j is None:
                    continue
                out[int(j["sid"])] = j
            if bi < n_batches - 1:
                env.swap_data_batch()
    finally:
        env.close()
        del env
        torch.cuda.empty_cache()
    return out


def eval_candidates_and_pick(
    make_env_fn,
    policy,
    other_lps,
    candidates_by_sid: dict[int, list],
    batch_size: int = 48,
    alpha: float = ALPHA_DEFAULT,
    persistent_k: int = 10,
    horizon: int = 40,
    require_ok: bool = True,
) -> tuple[dict[int, dict], pd.DataFrame]:
    """Evaluate all candidates; pick best metrics_score per scene.

    Returns (decisions, metrics_debug_df).
    """
    flat_items = []  # (sid, cand_i, dec)
    for sid, cands in candidates_by_sid.items():
        for ci, dec in enumerate(cands):
            if dec.get("kind") in ("i", "r"):
                flat_items.append((sid, ci, dec))

    if not flat_items:
        decisions = {
            sid: dict(
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
                metrics_score=float("-inf"),
            )
            for sid in candidates_by_sid
        }
        return decisions, pd.DataFrame()

    scene_list = [sid for sid, _, _ in flat_items]
    env, flat, pad_n = make_env_fn(scene_list, batch_size)
    B = batch_size
    n_batches = len(flat) // B
    # Pad flat_items to match env flat length
    pad_items = list(flat_items)
    while len(pad_items) < len(flat):
        pad_items.append(pad_items[-1])

    judged_rows = []
    best: dict = {}

    try:
        for bi in tqdm(range(n_batches), desc="pick-by-metrics"):
            sl = slice(bi * B, (bi + 1) * B)
            batch_items = pad_items[sl]
            if bi == n_batches - 1 and pad_n:
                valid = [True] * (B - pad_n) + [False] * pad_n
            else:
                valid = [True] * B
            real_n = len(scene_list)
            for w in range(B):
                if bi * B + w >= real_n:
                    valid[w] = False

            metas = []
            for w, (sid, ci, dec) in enumerate(batch_items):
                if not valid[w]:
                    metas.append(
                        _dec_to_meta(
                            sid,
                            dict(
                                kind="n",
                                intervention_idx=0,
                                labels=np.zeros(4, dtype=np.int64),
                                other_indices=[],
                                other_labels=[],
                                image_idx=-1,
                            ),
                        )
                    )
                else:
                    metas.append(_dec_to_meta(sid, dec))

            with torch.inference_mode():
                judged = eval_label_batch(
                    env,
                    policy,
                    other_lps,
                    metas,
                    valid,
                    alpha=alpha,
                    persistent_k=persistent_k,
                    horizon=horizon,
                )
            for w, (sid, ci, dec) in enumerate(batch_items):
                j = judged[w]
                if j is None:
                    continue
                row = dict(j)
                row["cand_i"] = ci
                row["label_score"] = dec.get("score")
                row["dist"] = dec.get("dist")
                judged_rows.append(row)
                sc = float(j.get("metrics_score", float("-inf")))
                ok = bool(j.get("metrics_ok", False))
                if require_ok and not ok:
                    continue
                prev = best.get(sid)
                # Recovery-first: if an r candidate is ok, prefer it over i
                # unless existing best is also r with higher score.
                if prev is None:
                    best[sid] = (sc, dec, j)
                else:
                    prev_sc, prev_dec, prev_j = prev
                    prev_r = prev_dec.get("kind") == "r"
                    cur_r = dec.get("kind") == "r"
                    if cur_r and not prev_r:
                        best[sid] = (sc, dec, j)
                    elif cur_r == prev_r and sc > prev_sc:
                        best[sid] = (sc, dec, j)
                    elif (not cur_r) and (not prev_r) and sc > prev_sc:
                        best[sid] = (sc, dec, j)

            if bi < n_batches - 1:
                env.swap_data_batch()
    finally:
        env.close()
        del env
        torch.cuda.empty_cache()

    decisions = {}
    for sid, cands in candidates_by_sid.items():
        if sid in best:
            sc, dec, j = best[sid]
            out_dec = dict(dec)
            out_dec.update(
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
            decisions[sid] = out_dec
        else:
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
                metrics_score=float("-inf"),
            )
    return decisions, pd.DataFrame(judged_rows)
