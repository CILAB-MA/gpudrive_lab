"""Sweep RL checkpoints in an experiment folder with far-partner masking.

Mirrors ``run_simulation.py`` / ``run_simulate_alone_scenes.py``:
  -sn <exp> under -mp → run simulate_mask_far_partners.py per *.pt
  (skip *optim*), then write result_far{thresh}_total.csv.

By default only models listed in ``self_play/result_0.0_total.csv`` are run
(same filter as alone eval). Use ``--all-models`` to sweep every *.pt.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def arg_parse():
    p = argparse.ArgumentParser("RL far-partner masking sweep")
    p.add_argument("--sweep-name", "-sn", type=str, default="scene_100_v2")
    p.add_argument("--model-path", "-mp", type=str, default="/data/after_cvpr/rl")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Default: /data/after_cvpr/images/mask_far_partners_rl/{sweep-name}",
    )
    p.add_argument("--dataset", "-d", type=str, default="validation")
    p.add_argument("--dataset-size", type=int, default=9987)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--far-thresh", type=float, default=20.0)
    p.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    p.add_argument("--gpu-id", "-g", type=int, default=0)
    p.add_argument(
        "--filter-csv",
        type=str,
        default=None,
        help="Only Model names in this CSV (default: {model_dir}/self_play/result_0.0_total.csv).",
    )
    p.add_argument(
        "--all-models",
        action="store_true",
        help="Run every *.pt in the experiment folder.",
    )
    return p.parse_args()


def resolve_filter_models(model_dir: str, filter_csv: str | None, all_models: bool):
    if all_models:
        return None
    path = filter_csv
    if path is None:
        cand = os.path.join(model_dir, "self_play", "result_0.0_total.csv")
        if os.path.isfile(cand):
            path = cand
    if path is None:
        return None
    if not os.path.isfile(path):
        raise SystemExit(f"filter csv not found: {path}")
    df = pd.read_csv(path)
    if "Model" not in df.columns:
        raise SystemExit(f"no Model column in {path}")
    models = sorted({str(m) for m in df["Model"].dropna().tolist()})
    print(f"filter csv: {path} ({len(models)} models)")
    return set(models)


if __name__ == "__main__":
    args = arg_parse()
    model_dir = os.path.join(args.model_path, args.sweep_name)
    out_dir = args.out_dir or os.path.join(
        "/data/after_cvpr/images/mask_far_partners_rl", args.sweep_name
    )
    os.makedirs(out_dir, exist_ok=True)

    allow = resolve_filter_models(model_dir, args.filter_csv, args.all_models)
    tag = f"far{args.far_thresh:g}"
    result_csv = os.path.join(out_dir, f"result_{tag}.csv")
    if os.path.exists(result_csv):
        os.remove(result_csv)

    selected = []
    for model in sorted(os.listdir(model_dir)):
        if not model.endswith(".pt"):
            continue
        if "optim" in model:
            continue
        if allow is not None and model not in allow:
            continue
        selected.append(model)

    if allow is not None:
        missing = sorted(allow - set(selected))
        if missing:
            print(f"[warn] in filter csv but missing from model dir: {missing}")
    print(f"model dir: {model_dir}")
    print(f"out dir:   {out_dir}")
    print(f"models to run ({len(selected)}): {selected}")
    if not selected:
        raise SystemExit("no models selected")

    for model in tqdm(selected):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
            f"python gpudrive/integrations/rl/test/simulate_mask_far_partners.py "
            f"-d {args.dataset} "
            f"--dataset-size {args.dataset_size} "
            f"--batch-size {args.batch_size} "
            f"--far-thresh {args.far_thresh} "
            f"-mp {model_dir} "
            f"-mn {model} "
            f"-pp {args.partner_portion_test} "
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
