import wandb, yaml, os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--save-name', '-sn', type=str, default='scene500')
    parser.add_argument('--linear-probing', type=str, default="partner")   
    parser.add_argument("--sweep-ids", '-s', nargs="+", type=str)
    args = parser.parse_args()
    
    return args

if __name__ == '__main__':
    args = parse_args()
    with open('private.yaml', "r") as file:
        wandb_config = yaml.safe_load(file)

    ENTITY = wandb_config.get("entity")
    PROJECT = wandb_config.get("main_project")
    sweep_ids = args.sweep_ids

    all_runs = []
    api = wandb.Api()
    for idx, sweep_id in enumerate(sweep_ids):
        sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
        
        for run in sweep.runs:
            summary = run.summary._json_dict
            config = run.config
            name = run.name

            summary.update(config)
            summary["run_name"] = name
            summary["sweep_id"] = sweep_id
            eval_summary = {k:v for k, v in summary.items() if k.startswith("eval/")}
            if args.linear_probing == 'ego':
                future_step = "ego_future_step"
            elif args.linear_probing == 'partner':
                future_step = "aux_future_step"
            eval_summary[future_step] = summary[future_step]
            if idx == 0:
                eval_summary["experiment"] = "baseline"
            else:
                eval_summary["experiment"] = "linear_probing"
            all_runs.append(eval_summary)

    df = pd.DataFrame(all_runs)

    csv_filename = f"/data/linear_probing/{args.linear_probing}_{args.save_name}.csv"
    os.makedirs(f"/data/linear_probing", exist_ok=True)
    df.to_csv(csv_filename, index=False)