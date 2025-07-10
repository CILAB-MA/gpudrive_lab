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
import seaborn as sns

@torch.no_grad()
def compute_sentence_embeddings(questions, model_name='all-MiniLM-L6-v2', device='cuda', name='env'):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    model.eval()
    embeddings  = model.encode(questions, convert_to_tensor=True)  
    return embeddings.cpu().numpy()  # (N, hidden_dim)

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
    positive = ["Yes, there is a stop sign.", "The stop sign is 1 meter ahead of the ego agent.",
                "The stop sign is 1 meter ahead of the ego agent.", "   The ego agent's current speed is 13 m/s.",
                "Surrounding agent #2 is 4 meters on the left and 1 meter behind the ego agent.",
                "Surrounding agent #2 is on the same lane as the ego agent.",
                "Surrounding agent #4 will be overtaken by the ego agent as they are both departing from the intersection but the ego agent is accelerating and will be closer to surrounding agent #4 in the future.",
                "Surrounding agent #0 will pass the ego agent as they are heading in opposite directions."
                ]
    negative = ["No, there isn't a stop sign.", "The stop sign is 2 meter behind the ego agent.",
                "The stop sign is 1 meter behind the ego agent.", "The ego agent's current speed is 7 m/s.",
                "Surrounding agent #2 is 4 meters on the right and 1 meter ahead of the ego agent.",
                "Surrounding agent #2 is on the different lane from the ego agent.",
                "Surrounding agent #4 will not be overtaken by the ego agent as they are both departing from the intersection but the ego agent is decelerating and will be farther from surrounding agent #4 in the future.",
                "Surrounding agent #0 will follow the ego agent as they are heading in same directions."
                ]
    positive_embedding = compute_sentence_embeddings(positive)
    negative_embedding = compute_sentence_embeddings(negative)

    num_pos = positive_embedding.shape[0]
    num_neg = negative_embedding.shape[0]

    heatmap = np.zeros((num_pos, num_neg))

    for i in range(num_pos):
        for j in range(num_neg):
            diff_vec = positive_embedding[i] - negative_embedding[j]
            score = np.linalg.norm(diff_vec)  # L2 norm
            heatmap[i, j] = score

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "L2 norm difference"})
    plt.xlabel("Negative sentence index")
    plt.ylabel("Positive sentence index")
    plt.title("Embedding Difference Heatmap (8x8)")
    plt.tight_layout()
    plt.savefig("embed_dist.png")