"""Sweep IL checkpoints in an experiment folder with far-partner masking.

Mirrors ``baselines/il/test/run_simulation.py``:
  -sn <exp> under -mp → run simulate_mask_far_partners.py per *.pth
  (skip *optim*), then write result_far{thresh}_total.csv.
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
    p.add_argument("--dataset-size", type=int, default=9987)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--num-stack", type=int, default=5)
    p.add_argument("--far-thresh", type=float, default=20.0)
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
        help="Only Model names in this CSV (optional).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    model_dir = os.path.join(args.model_path, args.sweep_name)
    out_dir = args.out_dir or os.path.join(
        "/data/after_cvpr/images/mask_far_partners_il", args.sweep_name
    )
    os.makedirs(out_dir, exist_ok=True)

    allow = None
    if args.filter_csv:
        df_f = pd.read_csv(args.filter_csv)
        allow = set(df_f["Model"].dropna().astype(str).tolist())
        print(f"filter csv: {args.filter_csv} ({len(allow)} models)")

    tag = f"far{args.far_thresh:g}"
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
        selected.append(model)

    print(f"model dir: {model_dir}")
    print(f"out dir:   {out_dir}")
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
            f"--far-thresh {args.far_thresh} "
            f"-mp {model_dir} "
            f"-mn {model} "
            f"-pp {args.partner_portion_test} "
            f"-sa {args.sim_agent} "
            f"--out-dir {out_dir}"
        )
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
