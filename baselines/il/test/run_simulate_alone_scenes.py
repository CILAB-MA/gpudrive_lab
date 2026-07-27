"""Sweep IL checkpoints on alone scenes.

Mirrors baselines/il/test/run_simulation.py: list *.pth (skip *optim*),
run simulate_alone_scenes.py per model, then aggregate result_alone.csv
into result_alone_total.csv.

By default only models listed in ``log_replay/result_0.0_total.csv`` are run
(same matching rule as RL alone ↔ self_play/result_0.0_total.csv).
Use ``--all-models`` to sweep every *.pth.
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
    parser.add_argument(
        "--filter-csv",
        type=str,
        default=None,
        help=(
            "Only run Model names listed in this CSV "
            "(default: {model_dir}/log_replay/result_0.0_total.csv if present)."
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Ignore --filter-csv and run every *.pth in the model dir.",
    )
    parser.add_argument(
        "--match-seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed ids to keep (e.g. '3,11,42').",
    )
    return parser.parse_args()


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
    # Prefer validation rows when Dataset column exists (train+val totals).
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
    allow = resolve_filter_models(model_dir, args.filter_csv, args.all_models)
    seed_allow = None
    if args.match_seeds:
        seed_allow = {int(x.strip()) for x in args.match_seeds.split(",") if x.strip()}
        print(f"match seeds: {sorted(seed_allow)}")

    models = sorted(os.listdir(model_dir))
    print(f"model dir: {model_dir}")

    os.makedirs(args.out_dir, exist_ok=True)
    result_csv = os.path.join(args.out_dir, "result_alone.csv")
    if os.path.exists(result_csv):
        os.remove(result_csv)

    mask_flag = "--mask-partners" if args.mask_partners else "--no-mask-partners"
    selected = []
    for model in models:
        if ".pth" not in model:
            continue
        if "optim" in model:
            continue
        if allow is not None and model not in allow:
            continue
        if seed_allow is not None and seed_of(model) not in seed_allow:
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
