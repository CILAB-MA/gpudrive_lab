"""Extract expert states and actions from Waymo Open Dataset."""
import logging
import subprocess
import argparse
import os
import json
from tqdm import tqdm
import glob
import pandas as pd
logging.getLogger(__name__)

def arg_parse():
    parser = argparse.ArgumentParser()
    # MODEL
    parser.add_argument('--sweep-name', '-sn', type=str, default='data_cut_add')
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model')
    parser.add_argument('--video-path', '-vp', type=str, default='/data/full_version/video')
    parser.add_argument('--dataset-size', type=int, default=1000) # total_world
    parser.add_argument('--batch-size', type=int, default=100) # num_world
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    parser.add_argument('--make-video', '-mv', action='store_true')
    # GPU SETTINGS
    parser.add_argument('--gpu-id', '-g', type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    models = os.listdir(os.path.join(args.model_path, args.sweep_name))[:5]
    print(models)
    for model in tqdm(models):
        for dataset in ['training', 'validation']:
            if '.pth' not in model:
                continue
            if args.partner_portion_test:
                video_path = args.video_path + f"_{args.partner_portion_test}"
            else:
                video_path = args.video_path

            video_path = os.path.join(video_path, args.sweep_name)
            model_path = os.path.join(args.model_path, args.sweep_name)
            arguments = f"-mc -d {dataset} --dataset-size {args.dataset_size} -mp {model_path} -vp {video_path} -mn {model} --batch-size {args.batch_size} -pp {args.partner_portion_test}"
            if args.make_video:
                arguments += ' -mv'
            command = f"CUDA_VISIBLE_DEVICES={args.gpu_id} /root/anaconda3/envs/gpudrive/bin/python baselines/il/test/simulation.py {arguments}"
            
            result = subprocess.run(command, shell=True)
            if result.returncode != 0:
                print(f"Error: Command failed with return code {result.returncode}")

    csv_path = f"{model_path}/result_{args.partner_portion_test}.csv"
    csv_path2 = f"{model_path}/result_{args.partner_portion_test}_total.csv"

    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        print(f"CSV file {csv_path} does not exist or is empty. Exiting...")
        exit()
    # Load CSV without index and remove Unnamed columns
    df = pd.read_csv(csv_path, index_col=False)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Result rows will be collected here
    rows = []

    # Group by (Model, Dataset)
    for (model_name, dataset_name), group in df.groupby(["Model", "Dataset"]):
        total_stats = group.mean(numeric_only=True)
        grouped_sum = group.sum(numeric_only=True)

        for prefix in ["Turn", "Normal", "Reverse", "Abnormal", "Straight"]:
            goal_col = f"{prefix}Goal"
            offroad_col = f"{prefix}OffRoad"
            vehcol_col = f"{prefix}VehCollision"
            coll_col = f"{prefix}Collision"
            goalprog_col = f"{prefix}GoalProgress"
            goaltime_col = f"{prefix}GoalTime"
            num_col = f"{prefix}Num"

            if goal_col in grouped_sum and num_col in grouped_sum:
                total_stats[goal_col] = grouped_sum[goal_col] / grouped_sum[num_col] if grouped_sum[num_col] > 0 else 0
                total_stats[offroad_col] = grouped_sum[offroad_col] / grouped_sum[num_col] if grouped_sum[num_col] > 0 else 0
                total_stats[vehcol_col] = grouped_sum[vehcol_col] / grouped_sum[num_col] if grouped_sum[num_col] > 0 else 0
                total_stats[coll_col] = grouped_sum[coll_col] / grouped_sum[num_col] if grouped_sum[num_col] > 0 else 0
                total_stats[goalprog_col] = grouped_sum[goalprog_col] / grouped_sum[num_col] if grouped_sum[num_col] > 0 else 0
                total_stats[goaltime_col] = grouped_sum[goaltime_col] / grouped_sum[goal_col] if grouped_sum[goal_col] > 0 else 0
                total_stats[num_col] = grouped_sum[num_col]

        total_stats["Model"] = model_name
        total_stats["Dataset"] = dataset_name
        rows.append(total_stats)

    # Create final DataFrame and reorder columns
    df_out = pd.DataFrame(rows)
    cols = ['Model', 'Dataset'] + [c for c in df_out.columns if c not in ['Model', 'Dataset']]
    df_out = df_out[cols]
    print(df_out)
    # Save
    df_out.to_csv(csv_path2, index=False)
    print(f"Saved aggregated results to: {csv_path2}")