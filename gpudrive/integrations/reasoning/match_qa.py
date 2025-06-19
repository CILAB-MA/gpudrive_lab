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
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name).to(device)
    model = SentenceTransformer('all-MiniLM-L6-v2') 
    model.eval()
    embeddings  = model.encode(questions, convert_to_tensor=True)  
    # embeddings = []
    # for q in questions:
    #     inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True, max_length=64).to(device)
    #     outputs = model(**inputs).last_hidden_state  # (1, seq_len, hidden)
    #     cls_embedding = outputs[:, 0, :]  # [CLS] token
    #     embeddings.append(cls_embedding.cpu())
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
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
    # env = GPUDriveTorchEnv(
    #     config=env_config,
    #     data_loader=train_loader,
    #     max_cont_agents=args.max_cont_agents,  # Number of agents to control
    #     device="cuda",
    #     action_type="continuous",
    # )
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    all_env_questions = []
    all_ego_questions = []
    all_sur_questions = []
    all_int_questions = []
    all_env_answers = []
    all_ego_answers = []
    all_sur_answers = []
    all_int_answers = []
    for idx in tqdm(range(num_iter - 1)):
        with open(f"/data/full_version/processed/reasoning/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
            jd = json.load(f)
        # obs = env.reset()
        jd_scenario_ids = list(jd.keys())
        # scenario_ids = env.get_scenario_ids()
        # ego_ids = env.get_ego_ids()
        for scene_id, data in jd.items():
            env_questions = [qa[0] for qa in data['env_qa']]
            ego_questions = [qa[0] for qa in data['ego_qa']]
            sur_questions = [qa[0] for qa in data['sur_qa']]
            int_questions = [qa[0] for qa in data['int_qa']]
            env_answers = [qa[1] for qa in data['env_qa']]
            ego_answers = [qa[1] for qa in data['ego_qa']]
            sur_answers = [qa[1] for qa in data['sur_qa']]
            int_answers = [qa[1] for qa in data['int_qa']]
            all_env_questions += env_questions
            all_ego_questions += ego_questions
            all_sur_questions += sur_questions
            all_int_questions += int_questions
            all_env_answers += env_answers
            all_ego_answers += ego_answers
            all_sur_answers += sur_answers
            all_int_answers += int_answers
            
        all_env_questions = list(set(all_env_questions))
        all_ego_questions = list(set(all_ego_questions))
        all_sur_questions = list(set(all_sur_questions))
        all_int_questions = list(set(all_int_questions))
        all_env_answers = list(set(all_env_answers))
        all_ego_answers = list(set(all_ego_answers))
        all_sur_answers = list(set(all_sur_answers))
        all_int_answers = list(set(all_int_answers))
    env_q_embed = analyze_semantic_distribution_hf(all_env_questions, save_path="q_tsne_env.png")
    ego_q_embed = analyze_semantic_distribution_hf(all_ego_questions, save_path="q_tsne_ego.png")
    sur_q_embed = analyze_semantic_distribution_hf(all_sur_questions, save_path="q_tsne_sur.png")
    int_q_embed = analyze_semantic_distribution_hf(all_int_questions, save_path="q_tsne_int.png")
    env_a_embed = analyze_semantic_distribution_hf(all_env_answers, save_path="a_tsne_env.png")
    ego_a_embed = analyze_semantic_distribution_hf(all_ego_answers, save_path="a_tsne_ego.png")
    sur_a_embed = analyze_semantic_distribution_hf(all_sur_answers, save_path="a_tsne_sur.png")
    int_a_embed = analyze_semantic_distribution_hf(all_int_answers, save_path="a_tsne_int.png")

    save_path = '/data/full_version/processed/final'
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(f"{save_path}/womd_reasoning_embed_{args.data_dir}.npz", 
                    env_q=env_q_embed,
                    ego_q=ego_q_embed,
                    sur_q=sur_q_embed,
                    int_q=int_q_embed,
                    env_a=env_a_embed,
                    ego_a=ego_a_embed,
                    sur_a=sur_a_embed,
                    int_a=int_a_embed)
    #     if idx != num_iter - 1:
    #         env.swap_data_batch()
    # env.close()
