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
import torch

def filter_qa_by_id(questions, answers, qa_ids):
    remove_keywords = ["pedestrian", "traffic light"]

    qa = [
        [q, a]
        for q, a in zip(questions, answers)
        if not (
            any(f"#{qa_id}" in q or f"#{qa_id}" in a for qa_id in qa_ids) or
            any(keyword in q.lower() or keyword in a.lower() for keyword in remove_keywords)
        )
    ]
    return qa

def load_all_scenario_jsons(data_dir, scenario_ids, ego_ids, base_dir):
    scenario_data = {}

    for idx, sid in scenario_ids.items():
        matched_files = [
            f for f in data_dir
            if f.startswith(f"scid_{sid}__aid_")
        ]

        loaded_jsons = []
        if len(matched_files) != 0:
            for file in matched_files:
                file_path = os.path.join(base_dir, file)
                with open(file_path, 'r') as f:
                    jd = json.load(f)
                    egos = ego_ids[idx].int()
                    mask = torch.isin(egos, jd['ego'])
                    indices = mask.nonzero(as_tuple=True)[0]
                    rel_ids = jd['rel_id']
                    rel_qa_ids = jd['rel_qa_id']
                    jd['batch_idx'] = idx
                    if len(indices) == 0:
                        continue
                    jd['ego_idx'] = indices[0].item()
                    env_q = jd['env_q']
                    env_a = jd['env_a']
                    ego_q = jd['ego_q']
                    ego_a = jd['ego_a']
                    sur_q = jd['sur_q']
                    sur_a = jd['sur_a']
                    int_q = jd['int_q']
                    int_a = jd['int_a']
                    no_rel_id = []
                    for rel_id, qa_id in zip(rel_ids, rel_qa_ids):
                        if rel_id not in ego_ids[idx].int():
                            no_rel_id.append(qa_id)
                    jd['env_qa'] = filter_qa_by_id(env_q, env_a, no_rel_id)
                    jd['ego_qa'] = filter_qa_by_id(ego_q, ego_a, no_rel_id)
                    jd['sur_qa'] = filter_qa_by_id(sur_q, sur_a, no_rel_id)
                    jd['int_qa'] = filter_qa_by_id(int_q, int_a, no_rel_id)
                    
                    del jd['env_q']
                    del jd['env_a']
                    del jd['ego_q']
                    del jd['ego_a']
                    del jd['sur_q']
                    del jd['sur_a']
                    del jd['int_q']
                    del jd['int_a']

                    loaded_jsons.append(jd)
            if len(loaded_jsons) != 0:
                scenario_data[idx] = loaded_jsons[0]
    return scenario_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="validation", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--video-dir", "-vd", type=str, default="/data/full_version/expert_video/validation_log")
    parser.add_argument("--total-scene-size", "-tss", type=int, default=10000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=50)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()

    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    VIDEO_DIR = args.video_dir
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
        womd_reasoning_json = dict()
        obs = env.reset()
        scenario_ids = env.get_scenario_ids()
        ego_ids = env.get_ego_ids()
        scenario_data = load_all_scenario_jsons(json_list, scenario_ids, ego_ids, base_folder)
        womd_reasoning_json.update(scenario_data)
        with open(f"/data/full_version/processed/reasoning/{args.data_dir}/womd_reasoning_{100 * idx}.json", "w") as f:
            json.dump(womd_reasoning_json, f, indent=2)
        if idx != num_iter - 1:
            env.swap_data_batch()
    env.close()
