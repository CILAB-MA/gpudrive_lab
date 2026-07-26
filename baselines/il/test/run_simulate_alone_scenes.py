"""Sweep all policy checkpoints in a model folder on alone scenes.

Mirrors baselines/il/test/run_simulation.py: list *.pth (skip *optim*),
run simulate_alone_scenes.py per model, then aggregate result_alone.csv
into result_alone_total.csv.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import pandas as pd
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-name", "-sn", type=str, default="exp_80000_subset_aix")
    parser.add_argument(
        "--model-path", "-mp", type=str, default="/data/full_version/model"
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
        default="/data/full_version/alone_driving",
    )
    parser.add_argument("--dataset-size", type=int, default=None, help="unused; kept for CLI parity")
    parser.add_argument("--batch-size", type=int, default=63)
    parser.add_argument("--num-stack", type=int, default=5)
    parser.add_argument("--partner-portion-test", "-pp", type=float, default=0.0)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument(
        "--mask-partners",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    model_dir = os.path.join(args.model_path, args.sweep_name)
    models = sorted(os.listdir(model_dir))
    print(f"model dir: {model_dir}")
    print(models)

    os.makedirs(args.out_dir, exist_ok=True)
    result_csv = os.path.join(args.out_dir, "result_alone.csv")
    # Fresh sweep each run (avoid duplicate Model rows on re-run)
    if os.path.exists(result_csv):
        os.remove(result_csv)

    mask_flag = "--mask-partners" if args.mask_partners else "--no-mask-partners"
    for model in tqdm(models):
        if ".pth" not in model:
            continue
        if "optim" in model:
            continue

        cmd = (
            f"CUDA_VISIBLE_DEVICES={args.gpu_id} "
            f"python baselines/il/test/simulate_alone_scenes.py "
            f"--data-dir {args.data_dir} "
            f"--alone-csv {args.alone_csv} "
            f"--batch-size {args.batch_size} "
            f"--num-stack {args.num_stack} "
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
