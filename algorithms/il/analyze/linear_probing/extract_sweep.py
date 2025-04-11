import wandb, yaml, os
import pandas as pd
import argparse

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--save-name', '-sn', type=str, default='scene500')
    parser.add_argument('--linear-probing', '-lp', type=str, default="partner")   
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
    for sweep_id in sweep_ids:
        sweep = api.sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
        
        for run in sweep.runs:
            config = run.config
            name = run.name

            # step=20일 때의 로그 가져오기
            history = run.history()  # 충분히 커야 함
            if '_step' not in history.keys():
                continue
            step20 = history[history['_step'] == 20]

            if step20.empty:
                continue  # 해당 step 없으면 스킵

            row = step20.iloc[0].to_dict()
            eval_summary = {k: v for k, v in row.items() if k.startswith("eval/")}

            # 나머지 정보도 추가
            if args.linear_probing == 'ego':
                future_step = "ego_future_step"
            elif args.linear_probing == 'partner':
                future_step = "aux_future_step"

            # 여긴 여전히 summary에서 가져와야 함 (step과 무관하니까)
            summary = run.summary._json_dict
            eval_summary[future_step] = config.get(future_step)
            eval_summary["experiment"] = config.get("model")
            eval_summary["seed"] = config.get("seed")

            all_runs.append(eval_summary)


    df = pd.DataFrame(all_runs)
    print(df)
    csv_filename = f"/data/linear_probingv2/{args.linear_probing}_{args.save_name}.csv"
    os.makedirs(f"/data/linear_probingv2", exist_ok=True)
    df.to_csv(csv_filename, index=False)