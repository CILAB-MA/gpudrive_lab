"""Sweep IL checkpoints: far-partner *deletion* via remove_agents_by_distance.

Default ``--dataset-size 2000`` for a quick trend check (not full val).
"""
from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def arg_parse():
    p = argparse.ArgumentParser("IL far-partner masking sweep")
    p.add_argument("--sweep-name", "-sn", type=str, default="exp_100")
    p.add_argument("--model-path", "-mp", type=str, default="/data/full_version/model")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: /data/after_cvpr/images/mask_far_partners_il/{sweep-name}",
    )
    p.add_argument("--dataset", "-d", type=str, default="validation")
    p.add_argument("--dataset-size", type=int, default=2000, help="Cap scenes (default 2000 for trend)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--num-stack", type=int, default=5)
    p.add_argument("--far-thresh", type=float, default=None,
                   help="Far-threshold delete (m). Ignored with --nearest-first.")
    p.add_argument("--remove-perc", type=float, default=0.2)
    p.add_argument(
        "--nearest-first",
        action="store_true",
        help="Delete nearest perc first (tag near{perc*100}).",
    )
    p.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    p.add_argument(
        "--sim-agent",
        "-sa",
        type=str,
        default="log_replay",
        choices=["log_replay", "self_play", "delta_replay"],
    )
    p.add_argument("--gpu-id", "-g", type=int, default=0)
    p.add_argument(
        "--filter-csv",
        type=str,
        default=None,
        help="Only Model names in this CSV (default: {model_dir}/log_replay/result_0.0_total.csv).",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Run every *.pth (ignore filter csv).",
    )
    p.add_argument(
        "--match-seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed ids to keep (e.g. '3,11,42').",
    )
    return p.parse_args()


def resolve_filter_models(model_dir: str, filter_csv: str | None, all_models: bool):
    if all_models:
        return None
    path = filter_csv
    if path is None:
        cand = os.path.join(model_dir, "log_replay", "result_0.0_total.csv")
        if os.path.isfile(cand):
            path = cand
    if path is None:
        return None
    if not os.path.isfile(path):
        raise SystemExit(f"filter csv not found: {path}")
    df = pd.read_csv(path)
    if "Model" not in df.columns:
        raise SystemExit(f"no Model column in {path}")
    if "Dataset" in df.columns:
        val = df[df["Dataset"].astype(str).str.contains("val", case=False, na=False)]
        if len(val):
            df = val
    models = sorted({str(m) for m in df["Model"].dropna().tolist()})
    print(f"filter csv: {path} ({len(models)} models)")
    return set(models)


def seed_of(model: str):
    import re
    m = re.search(r"_s(\d+)_", model)
    return int(m.group(1)) if m else None


if __name__ == "__main__":
    args = arg_parse()
    model_dir = os.path.join(args.model_path, args.sweep_name)
    out_dir = args.out_dir or os.path.join(
        "/data/after_cvpr/images/mask_far_partners_il", args.sweep_name
    )
    os.makedirs(out_dir, exist_ok=True)

    allow = resolve_filter_models(model_dir, args.filter_csv, args.all_models)
    seed_allow = None
    if args.match_seeds:
        seed_allow = {int(x.strip()) for x in args.match_seeds.split(",") if x.strip()}
        print(f"match seeds: {sorted(seed_allow)}")

    if args.nearest_first:
        tag = f"near{int(round(args.remove_perc * 100))}"
    elif args.far_thresh is not None:
        tag = f"far{args.far_thresh:g}"
    elif args.remove_perc <= 0.0:
        tag = "normal"
    else:
        tag = f"farperc{int(round(args.remove_perc * 100))}"
    result_csv = os.path.join(out_dir, f"result_{tag}.csv")
    if os.path.exists(result_csv):
        os.remove(result_csv)

    selected = []
    for model in sorted(os.listdir(model_dir)):
        if ".pth" not in model:
            continue
        if "optim" in model:
            continue
        if allow is not None and model not in allow:
            continue
        if seed_allow is not None and seed_of(model) not in seed_allow:
            continue
        selected.append(model)

    print(f"model dir: {model_dir}")
    print(f"out dir:   {out_dir}")
    print(f"tag:       {tag}")
    print(f"models to run ({len(selected)}): {selected}")
    if not selected:
        raise SystemExit("no models selected")

    for model in tqdm(selected):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
            f"python baselines/il/test/simulate_mask_far_partners.py "
            f"-d {args.dataset} "
            f"--dataset-size {args.dataset_size} "
            f"--batch-size {args.batch_size} "
            f"--num-stack {args.num_stack} "
            f"--remove-perc {args.remove_perc} "
            f"-mp {model_dir} "
            f"-mn {model} "
            f"-pp {args.partner_portion_test} "
            f"-sa {args.sim_agent} "
            f"--out-dir {out_dir}"
        )
        if args.nearest_first:
            cmd += " --nearest-first"
        if args.far_thresh is not None and not args.nearest_first:
            cmd += f" --far-thresh {args.far_thresh}"
        print(cmd)
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            print(f"Error: Command failed with return code {result.returncode} for {model}")

    if not os.path.exists(result_csv) or os.path.getsize(result_csv) == 0:
        print(f"CSV file {result_csv} does not exist or is empty. Exiting...")
        raise SystemExit(1)

    df = pd.read_csv(result_csv, index_col=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    print(df)
    out_total = os.path.join(out_dir, f"result_{tag}_total.csv")
    df.to_csv(out_total, index=False)
    print(f"Saved aggregated results to: {out_total}")
