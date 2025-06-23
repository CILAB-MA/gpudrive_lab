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

def save_qa_trajectory(env, jd, save_path, save_index=0):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
    """
    qa_types = ["env", "ego", "sur", "int"]
    valid_agent = len(jd)
    qa_timesteps = 20
    expert_env_q_lst = np.zeros((valid_agent, 20, 384))
    expert_ego_q_lst = np.zeros((valid_agent, 20, 384))
    expert_sur_q_lst = np.zeros((valid_agent, 120, 384))
    expert_int_q_lst = np.zeros((valid_agent, 30, 384))

    expert_env_a_lst = np.zeros((valid_agent, 20, 384))
    expert_ego_a_lst = np.zeros((valid_agent, 20, 384))
    expert_sur_a_lst = np.zeros((valid_agent, 120, 384))
    expert_int_a_lst = np.zeros((valid_agent, 30, 384))
    q_npy = [expert_env_q_lst, expert_ego_q_lst, expert_sur_q_lst, expert_int_q_lst]
    a_npy = [expert_env_a_lst, expert_ego_a_lst, expert_sur_a_lst, expert_int_a_lst]

    expert_env_mask_lst = np.ones((valid_agent, 20), dtype=bool)
    expert_ego_mask_lst = np.ones((valid_agent, 20), dtype=bool)
    expert_sur_mask_lst = np.ones((valid_agent, 120), dtype=bool)
    expert_int_mask_lst = np.ones((valid_agent, 30), dtype=bool)
    qa_mask_npy = [expert_env_mask_lst, expert_ego_mask_lst, expert_sur_mask_lst, expert_int_mask_lst]
    for i, data in enumerate(jd.values()):
        for qa, qa_type in enumerate(qa_types):
            qas = data[f"{qa_type}_qa"]
            qs, ans = zip(*qas) if qas else ([], [])
            q_embeddings = compute_sentence_embeddings(qs, device='cuda')
            a_embeddings = compute_sentence_embeddings(ans, device='cuda')
            num = min(len(q_embeddings), qa_timesteps)
            q_npy[qa][i, :num] = q_embeddings[:num]
            a_npy[qa][i, :num] = a_embeddings[:num]
            qa_mask_npy[qa][i, :num] = False
    obs = env.reset()
    expert_actions, _, _, _ , _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    # partner_id = env.get_partner_id().unsqueeze(-1)
    device = env.device
    cont_agent_mask = env.cont_agent_mask.to(device)  # (num_worlds, num_agents)
    scene_idx = np.array([int(k) for k in jd.keys()])
    qa_ego_idx = np.array([int(jd[sid]['ego_idx']) for sid in jd.keys()])
    # trajectory information
    expert_trajectory_lst = torch.zeros((valid_agent, qa_timesteps, obs.shape[-1]), device=device)
    expert_actions_lst = torch.zeros((valid_agent, qa_timesteps, 3), device=device)
    expert_dead_mask_lst = torch.ones((valid_agent, qa_timesteps), device=device, dtype=torch.bool)
    expert_partner_mask_lst = torch.full((valid_agent, qa_timesteps, 127), 2, device=device, dtype=torch.long)
    expert_road_mask_lst = torch.ones((valid_agent, qa_timesteps, 200), device=device, dtype=torch.bool)
    expert_global_pos_lst = torch.zeros((valid_agent, qa_timesteps, 2), device=device) # global pos (2)
    expert_global_rot_lst = torch.zeros((valid_agent, qa_timesteps, 1), device=device) # global actions (1)

    # qa information
    # Initialize dead agent mask
    agent_info = (
            env.sim.absolute_self_observation_tensor()
            .to_torch()
            .to(device)
        )
    dead_agent_mask = ~env.cont_agent_mask.clone().to(device) # (num_worlds, num_agents)
    road_mask = env.get_road_mask()

    for time_step in tqdm(range(env.episode_len)):
        for idx, (world_idx, agent_idx) in enumerate(zip(scene_idx, qa_ego_idx)):
            if (not dead_agent_mask[world_idx, agent_idx]) and (time_step < 30) and (time_step >= 10):
                expert_trajectory_lst[idx][time_step - 10] = obs[world_idx, agent_idx]
                expert_actions_lst[idx][time_step - 10] = expert_actions[world_idx, agent_idx, time_step]
                expert_partner_mask_lst[idx][time_step - 10] = partner_mask[world_idx, agent_idx]
                expert_road_mask_lst[idx][time_step - 10] = road_mask[world_idx, agent_idx]
                expert_global_pos_lst[idx, time_step - 10] = agent_info[world_idx, agent_idx, 0:2]
                expert_global_rot_lst[idx, time_step - 10] = agent_info[world_idx, agent_idx, 7:8]
            expert_dead_mask_lst[idx][time_step - 10] = dead_agent_mask[world_idx, agent_idx]

        
        # env.step() -> gather next obs
        env.step_dynamics(expert_actions[:, :, time_step, :])
        dones = env.get_dones().to(device)
        
        dead_agent_mask = torch.logical_or(dead_agent_mask, dones)
        obs = env.get_obs() 
        road_mask = env.get_road_mask()
        partner_mask = env.get_partner_mask()
        # partner_id = env.get_partner_id().unsqueeze(-1)
        agent_info = (
        env.sim.absolute_self_observation_tensor()
        .to_torch()
        .to(device)
        )
        infos = env.get_infos()
        if time_step >= 29:
            off_road = infos.off_road[scene_idx, qa_ego_idx]
            veh_collision = infos.collided[scene_idx, qa_ego_idx]

            off_road_rate = off_road.sum().float() / valid_agent
            veh_coll_rate = veh_collision.sum().float() / valid_agent
            collision = (veh_collision + off_road > 0)
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate}')
            print(f'Save number w/o collision {len(expert_trajectory_lst[~collision])} / {len(expert_trajectory_lst)}')
            break
    
    expert_trajectory_lst = expert_trajectory_lst[~collision].to('cpu')
    expert_actions_lst = expert_actions_lst[~collision].to('cpu')
    expert_dead_mask_lst = expert_dead_mask_lst[~collision].to('cpu')
    expert_partner_mask_lst = expert_partner_mask_lst[~collision].to('cpu')
    expert_road_mask_lst = expert_road_mask_lst[~collision].to('cpu')
    # global pos
    expert_global_pos_lst = expert_global_pos_lst[~collision].to('cpu')
    expert_global_rot_lst = expert_global_rot_lst[~collision].to('cpu')

    expert_env_q_lst = q_npy[0][~collision.cpu().numpy()]
    expert_ego_q_lst = q_npy[1][~collision.cpu().numpy()]
    expert_sur_q_lst = q_npy[2][~collision.cpu().numpy()]
    expert_int_q_lst = q_npy[3][~collision.cpu().numpy()]

    expert_env_a_lst = a_npy[0][~collision.cpu().numpy()]
    expert_ego_a_lst = a_npy[1][~collision.cpu().numpy()]
    expert_sur_a_lst = a_npy[2][~collision.cpu().numpy()]
    expert_int_a_lst = a_npy[3][~collision.cpu().numpy()]

    expert_env_mask_lst = qa_mask_npy[0][~collision.cpu().numpy()]
    expert_ego_mask_lst = qa_mask_npy[1][~collision.cpu().numpy()]
    expert_sur_mask_lst = qa_mask_npy[2][~collision.cpu().numpy()]
    expert_int_mask_lst = qa_mask_npy[3][~collision.cpu().numpy()]
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(save_path + '/global', exist_ok=True)
    os.makedirs(save_path + '/reasoning', exist_ok=True)
    np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
                        obs=expert_trajectory_lst,
                        actions=expert_actions_lst,
                        dead_mask=expert_dead_mask_lst,
                        partner_mask=expert_partner_mask_lst,
                        road_mask=expert_road_mask_lst)
    np.savez_compressed(f"{save_path}/reasoning/reasoning_trajectory_{save_index}.npz", 
                        env_q=expert_env_q_lst,
                        ego_q=expert_ego_q_lst,
                        sur_q=expert_sur_q_lst,
                        int_q=expert_int_q_lst,
                        env_a=expert_env_a_lst,
                        ego_a=expert_ego_a_lst,
                        sur_a=expert_sur_a_lst,
                        int_a=expert_int_a_lst,
                        env_mask=expert_env_mask_lst,
                        ego_mask=expert_ego_mask_lst,
                        sur_mask=expert_sur_mask_lst,
                        int_mask=expert_int_mask_lst,
                        )
    np.savez_compressed(f"{save_path}/global/global_trajectory_{save_index}.npz", 
                        ego_global_pos=expert_global_pos_lst,
                        ego_global_rot=expert_global_rot_lst)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="training", help="training (80000) / testing (10000)")
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument("--total-scene-size", "-tss", type=int, default=10000)
    parser.add_argument("--scene-batch-size", "-sbs", type=int, default=50)
    parser.add_argument("--max-cont-agents", "-m", type=int, default=128)
    parser.add_argument('--partner-portion-test', '-pp', type=float, default=0.0)
    args = parser.parse_args()

    DATA_DIR = os.path.join("/data/full_version/data", args.data_dir)
    TOTAL_NUM_WORLDS = args.total_scene_size
    NUM_WORLDS = args.scene_batch_size
    json_folder = args.data_dir + '_interactive' if args.data_dir == 'validation' else args.data_dir
    env_config = EnvConfig()
    render_config = RenderConfig()

    # Create data loader
    train_loader = SceneDataLoader(
        root=DATA_DIR,
        batch_size=NUM_WORLDS,
        dataset_size=TOTAL_NUM_WORLDS,
        shuffle=False
    )
    env_config = EnvConfig(
        dynamics_model='delta_local',
        steer_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        accel_actions=torch.round(torch.tensor([-np.inf, np.inf]), decimals=3),
        dx=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dy=torch.round(torch.tensor([-6.0, 6.0]), decimals=3),
        dyaw=torch.round(torch.tensor([-np.pi, np.pi]), decimals=3),
        collision_behavior='remove'
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
    print('Launch Env')
    num_iter = int(args.total_scene_size // args.scene_batch_size)
    save_path = f'/data/full_version/processed/final/reasoning_{args.data_dir}_subset'
    os.makedirs(save_path, exist_ok=True)
    env_count, ego_count= [], []
    for idx in tqdm(range(num_iter)):
        if idx != num_iter - 1:
            with open(f"/data/full_version/processed/reasoning/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
        save_qa_trajectory(env, jd, save_path, idx * args.scene_batch_size)
        if idx != num_iter - 1:
            env.swap_data_batch()
    env.close()
    del env
    del env_config
