"""Sweep RL PPO checkpoints on alone (single AV) scenes.

Mirrors gpudrive/integrations/rl/test/run_simulation.py: list *.pt (skip *optim*),
run simulate_alone_scenes.py per model, then write result_alone_total.csv.

By default only models listed in ``--filter-csv`` (e.g. self_play
``result_0.0_total.csv``) are evaluated, so the alone set matches the
Normal-scene sweep.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-name",
        "-sn",
        type=str,
        default="scene_10000",
        help="Subfolder under --model-path (or '' to use model-path directly)",
    )
    parser.add_argument(
        "--model-path", "-mp", type=str, default="/data/after_cvpr/rl"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/data/full_version/data/validation_alone",
    )
    parser.add_argument(
        "--alone-csv",
        type=str,
        default="/data/full_version/data/validation_alone/alone_scenes.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/data/after_cvpr/images/alone_driving_rl",
    )
    parser.add_argument("--batch-size", type=int, default=63)
    parser.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument(
        "--mask-partners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--filter-csv",
        type=str,
        default=None,
        help=(
            "Only run Model names listed in this CSV "
            "(default: {model_dir}/self_play/result_0.0_total.csv if present)."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Ignore --filter-csv and run every *.pt in the model dir.",
    )
    return parser.parse_args()


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
    model_dir = (
        os.path.join(args.model_path, args.sweep_name)
        if args.sweep_name
        else args.model_path
    )
    allow = resolve_filter_models(model_dir, args.filter_csv, args.all_models)

    models = sorted(os.listdir(model_dir))
    print(f"model dir: {model_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    result_csv = os.path.join(args.out_dir, "result_alone.csv")
    if os.path.exists(result_csv):
        os.remove(result_csv)

    mask_flag = "--mask-partners" if args.mask_partners else "--no-mask-partners"
    selected = []
    for model in models:
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
    print(f"models to run ({len(selected)}): {selected}")
    if not selected:
        raise SystemExit("no models selected")

    for model in tqdm(selected):
        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
            f"python gpudrive/integrations/rl/test/simulate_alone_scenes.py "
            f"--data-dir {args.data_dir} "
            f"--alone-csv {args.alone_csv} "
            f"--batch-size {args.batch_size} "
            f"-mp {model_dir} "
            f"-mn {model} "
            f"-pp {args.partner_portion_test} "
            f"--out-dir {args.out_dir} "
            f"{mask_flag}"
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

    out_total = os.path.join(args.out_dir, "result_alone_total.csv")
    df.to_csv(out_total, index=False)
    print(f"Saved aggregated results to: {out_total}")
