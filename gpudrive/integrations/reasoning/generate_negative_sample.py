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

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re


def generate_all_neg_ego(answers):
    neg_answers = []
    for answer in answers:
        neg_answer = generate_negative_ego(answer)
        neg_answers.append(neg_answer)
        if 'collision' in answer:
            print(answer)
    neg_answers = np.array(neg_answers)
    answer_pairs = np.concatenate([answers.reshape(-1, 1), neg_answers.reshape(-1, 1)], axis=-1)
    return answer_pairs

def generate_negative_ego(answer):
    neg_answer = answer.copy()
    change = False
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", answer)
    if len(numbers) > 0:
        change = True
        orig_number = [int(eval(n)) for n in numbers]
        max_range = max(orig_number) + 1 if max(orig_number) > 1 else 10
        candidate = np.arange(0, max_range)
        candidate = candidate[~np.isin(candidate, orig_number)]
        change_speed = np.random.choice(candidate, len(orig_number))
        for orig, new in zip(numbers, change_speed):
            neg_answer = neg_answer.replace(orig, str(new))
    # Direction
    if 'left' in answer:
        change = True
        neg_answer = neg_answer.replace('left', 'right')
    elif 'right' in answer:
        change = True
        neg_answer = neg_answer.replace('right', 'left')
    # Position
    if "ahead" in answer:
        change = True
        neg_answer = neg_answer.replace("ahead", "behind")
    elif "behind" in answer:
        change = True
        neg_answer = neg_answer.replace("behind", "ahead")    

    # accel
    if 'accelerating' in answer:
        change = True
        neg_answer = neg_answer.replace('accelerating', 'decelerating')
    elif 'decelerating' in answer:
        change = True
        neg_answer = neg_answer.replace('decelerating', 'accelerating')

    if 'going straight' in answer:
        change = True
        candidate = ['turn left', 'turn right']
        neg_candidate = np.random.choice(candidate)
        neg_answer = neg_answer.replace('going straight', neg_candidate)
    elif 'approaching' in answer:
        change = True
        neg_answer = neg_answer.replace("approaching", "departing from")
    elif 'departing from' in answer:
        change = True
        neg_answer = neg_answer.replace("departing from", "approaching")
    if not change:
        neg_answer = None
    return neg_answer


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--total-scene-size", "-tss", type=int, default=80000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=50)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    env_qs = []
    ego_qs = []
    sur_qs = []
    int_qs = []
    env_as = []
    ego_as = []
    sur_as = []
    int_as = []
    for idx in tqdm(range(num_iter)):
        if idx != num_iter - 1:
            with open(f"/data/full_version/processed/reasoning_raw/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
            for d in jd.values():
                env_qa = np.array(d['env_qa'])
                ego_qa = np.array(d['ego_qa'])
                sur_qa = np.array(d['sur_qa'])
                # print('Q:',ego_qa[:, 0])
                # print('A:', ego_qa[:, 1])
                answer_pairs = generate_all_neg_ego(ego_qa[:, 1])
                generation_ratio = (answer_pairs[:, 1] != None).sum() / len(answer_pairs)
                print(f"Negative Generation Ratio: {generation_ratio:.4f}")
                print('A:', answer_pairs)
                int_qa = np.array(d['int_qa'])
