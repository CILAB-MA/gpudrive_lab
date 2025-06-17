import os
import sys
import json
import argparse
import mediapy as media
import numpy as np

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="validation", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--total-scene-size", "-tss", type=int, default=10000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=100)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()

    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    json_folder = args.data_dir + '_interactive' if args.data_dir == 'validation' else args.data_dir
    base_folder =os.path.join("/data/womd-reasoning", json_folder, json_folder)
    json_list = os.listdir(base_folder)
    env_config = EnvConfig()
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=NUM_WORLDS,
        dataset_size=TOTAL_NUM_WORLDS,
        shuffle=False
    )

    # Make env
    env = GPUDriveTorchEnv(
        config=env_config,
        data_loader=train_loader,
        max_cont_agents=args.max_cont_agents,  # Number of agents to control
        device="cuda",
        action_type="continuous",
    )
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    for idx in tqdm(range(num_iter)):
        with open(f"/data/full_version/processed/reasoning/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
            jd = json.load(f)
        obs = env.reset()
        jd_scenario_ids = list(jd.keys())
        scenario_ids = env.get_scenario_ids()
        ego_ids = env.get_ego_ids()
        for scene_id, data in jd.items():
            print(ego_ids[data['batch_idx']].int()[:10], data['ego'], data['rel_id'], data['sid'], scenario_ids[data['batch_idx']])
        if idx != num_iter - 1:
            env.swap_data_batch()
    env.close()
