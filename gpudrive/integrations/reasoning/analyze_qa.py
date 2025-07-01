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

@torch.no_grad()
def compute_sentence_embeddings(questions, model_name='all-MiniLM-L6-v2', device='cuda', name='env'):
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    model.eval()
    embeddings  = model.encode(questions, convert_to_tensor=True)  
    return embeddings.cpu().numpy()  # (N, hidden_dim)

def analyze_semantic_distribution_hf(questions, n_clusters=10, save_path="tsne_plot.png"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = compute_sentence_embeddings(questions, device=device)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
    labels = kmeans.labels_
    sil_score = silhouette_score(embeddings, labels)
    print(f"Silhouette Score: {sil_score:.4f}")

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", s=30)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved t-SNE plot to: {os.path.abspath(save_path)}")
    for i in range(10):
        print(f"\n[Cluster {i}]")
        for q in np.array(questions)[labels == i][:5]:
            print(" -", q)
    return embeddings

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
            with open(f"/data/full_version/processed/reasoning/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
            for d in jd.values():
                env_qa = np.array(d['env_qa'])
                ego_qa = np.array(d['ego_qa'])
                sur_qa = np.array(d['sur_qa'])
                int_qa = np.array(d['int_qa'])
                if len(env_qa) > 0:
                    env_qs += env_qa[:, 0].tolist()
                    env_as += env_qa[:, 1].tolist()
                if len(ego_qa) > 0:
                    ego_qs += ego_qa[:, 0].tolist()
                    ego_as += ego_qa[:, 1].tolist()
                if len(sur_qa) > 0:
                    sur_qs += sur_qa[:, 0].tolist()
                    sur_as += sur_qa[:, 1].tolist()
                if len(int_qa) > 0:
                    int_qs += int_qa[:, 0].tolist()
                    int_as += int_qa[:, 1].tolist()
    
    env_qs = list(set(env_qs))
    env_as = list(set(env_as))
    ego_qs = list(set(ego_qs))
    ego_as = list(set(ego_as))
    sur_qs = list(set(sur_qs))
    sur_as = list(set(sur_as))
    int_qs = list(set(int_qs))
    int_as = list(set(int_as))

    analyze_semantic_distribution_hf(env_qs, save_path='env_qs.png')
    analyze_semantic_distribution_hf(env_as, save_path='env_as.png')
    analyze_semantic_distribution_hf(ego_qs, save_path='ego_qs.png')
    analyze_semantic_distribution_hf(ego_as, save_path='ego_as.png')
    analyze_semantic_distribution_hf(sur_qs, save_path='sur_qs.png')
    analyze_semantic_distribution_hf(sur_as, save_path='sur_as.png')
    analyze_semantic_distribution_hf(int_qs, save_path='int_qs.png')
    analyze_semantic_distribution_hf(int_as, save_path='int_as.png')