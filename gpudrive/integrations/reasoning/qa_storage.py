import os
import sys
import json
import argparse
import numpy as np

from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig, RenderConfig
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.visualize.utils import img_from_fig
from tqdm import tqdm

# from sentence_transformers import SentenceTransformer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# @torch.no_grad()
# def compute_sentence_embeddings(questions, model_name='all-MiniLM-L6-v2', device='cuda', name='env'):
#     model = SentenceTransformer('all-MiniLM-L6-v2') 
#     model.eval()
#     embeddings  = model.encode(questions, convert_to_tensor=True)  
#     return embeddings.cpu().numpy()  # (N, hidden_dim)

def save_qa_trajectory(env, reasoning_embedding, jd, save_path, save_index=0):
    """
    Save the trajectory, partner_mask and road_mask in the environment, distinguishing them by each scene and agent.
    
    Args:
        env (GPUDriveTorchEnv): Initialized environment class.
    """
    qa_types = ["env", "ego", "sur", "int"]

    valid_agent = len(reasoning_embedding)
    qa_timesteps = env.episode_len
    
    obs = env.reset()
    expert_actions, _, _, _ , _ = env.get_expert_actions() # (num_worlds, num_agents, episode_len, action_dim)
    road_mask = env.get_road_mask()
    partner_mask = env.get_partner_mask()
    # partner_id = env.get_partner_id().unsqueeze(-1)
    device = env.device
    scene_idx = np.array([int(k) for k in jd.keys()])
    qa_ego_idx = np.array([int(jd[sid]['ego_idx']) for sid in jd.keys()])
    
    env_q = reasoning_embedding['env_q']
    env_pos_a = reasoning_embedding['env_pos_a']
    env_neg_a = reasoning_embedding['env_neg_a']
    
    ego_q = reasoning_embedding['ego_q']
    ego_pos_a = reasoning_embedding['ego_pos_a']
    ego_neg_a = reasoning_embedding['ego_neg_a']
    
    sur_q = reasoning_embedding['sur_q']
    sur_pos_a = reasoning_embedding['sur_pos_a']
    sur_neg_a = reasoning_embedding['sur_neg_a']

    int_q = reasoning_embedding['int_q']
    int_pos_a = reasoning_embedding['int_pos_a']
    int_neg_a = reasoning_embedding['int_neg_a']

    env_mask = reasoning_embedding['env_mask']
    ego_mask = reasoning_embedding['ego_mask']
    sur_mask = reasoning_embedding['sur_mask']
    int_mask = reasoning_embedding['int_mask']

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
        if (dead_agent_mask == True).all():
            off_road = infos.off_road[scene_idx, qa_ego_idx]
            veh_collision = infos.collided[scene_idx, qa_ego_idx]

            off_road_rate = off_road.sum().float() / valid_agent
            veh_coll_rate = veh_collision.sum().float() / valid_agent
            collision = (veh_collision + off_road > 0)
            print(f'Offroad {off_road_rate} VehCol {veh_coll_rate}')
            break
    

    expert_env_q_lst = env_q[~collision.cpu().numpy()]
    expert_ego_q_lst = ego_q[~collision.cpu().numpy()]
    expert_sur_q_lst = sur_q[~collision.cpu().numpy()]
    expert_int_q_lst = int_q[~collision.cpu().numpy()]

    expert_env_pa_lst = env_pos_a[~collision.cpu().numpy()]
    expert_ego_pa_lst = ego_pos_a[~collision.cpu().numpy()]
    expert_sur_pa_lst = sur_pos_a[~collision.cpu().numpy()]
    expert_int_pa_lst = int_pos_a[~collision.cpu().numpy()]

    expert_env_na_lst = env_neg_a[~collision.cpu().numpy()]
    expert_ego_na_lst = ego_neg_a[~collision.cpu().numpy()]
    expert_sur_na_lst = sur_neg_a[~collision.cpu().numpy()]
    expert_int_na_lst = int_neg_a[~collision.cpu().numpy()]

    expert_env_mask_lst = env_mask[~collision.cpu().numpy()]
    expert_ego_mask_lst = ego_mask[~collision.cpu().numpy()]
    expert_sur_mask_lst = sur_mask[~collision.cpu().numpy()]
    expert_int_mask_lst = int_mask[~collision.cpu().numpy()]
    # os.makedirs(save_path, exist_ok=True)
    # os.makedirs(save_path + '/global', exist_ok=True)
    os.makedirs(save_path + '/reasoning_final', exist_ok=True)
    # np.savez_compressed(f"{save_path}/trajectory_{save_index}.npz", 
    #                     obs=expert_trajectory_lst,
    #                     actions=expert_actions_lst,
    #                     dead_mask=expert_dead_mask_lst,
    #                     partner_mask=expert_partner_mask_lst,
    #                     road_mask=expert_road_mask_lst)
    np.savez_compressed(f"{save_path}/reasoning_final/reasoning_trajectory_{save_index}.npz", 
                        env_q=expert_env_q_lst,
                        ego_q=expert_ego_q_lst,
                        sur_q=expert_sur_q_lst,
                        int_q=expert_int_q_lst,
                        env_pos_a=expert_env_pa_lst,
                        ego_pos_a=expert_ego_pa_lst,
                        sur_pos_a=expert_sur_pa_lst,
                        int_pos_a=expert_int_pa_lst,
                        env_neg_a=expert_env_na_lst,
                        ego_neg_a=expert_ego_na_lst,
                        sur_neg_a=expert_sur_na_lst,
                        int_neg_a=expert_int_na_lst,
                        env_mask=expert_env_mask_lst,
                        ego_mask=expert_ego_mask_lst,
                        sur_mask=expert_sur_mask_lst,
                        int_mask=expert_int_mask_lst,
                        )
    # np.savez_compressed(f"{save_path}/global/global_trajectory_{save_index}.npz", 
    #                     ego_global_pos=expert_global_pos_lst,
    #                     ego_global_rot=expert_global_rot_lst)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument("--data_dir", "-dd", type=str, default="validation", help="training (80000) / testing (10000)")
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
    # model = SentenceTransformer('all-MiniLM-L6-v2') 
    # model.eval()
    model = None
    num_iter = int(TOTAL_NUM_WORLDS // NUM_WORLDS)
    print('Launch Env')
    num_iter = int(args.total_scene_size // args.scene_batch_size)
    save_path = f'/data/full_version/processed/final/reasoning_{args.data_dir}_subset'
    os.makedirs(save_path, exist_ok=True)
    env_count, ego_count= [], []
    for idx in tqdm(range(num_iter)):
        if idx != num_iter - 1:
            with open(f"/data/full_version/processed/reasoning_raw/{args.data_dir}/womd_reasoning_{100 * idx}.json", "r") as f:
                jd = json.load(f)
            np_path = f"{save_path}/reasoning_posneg/reasoning_trajectory_{idx * args.scene_batch_size}.npz"
            reasoning_embedding = np.load(np_path)
        save_qa_trajectory(env, reasoning_embedding, jd, save_path, idx * args.scene_batch_size)
        if idx != num_iter - 1:
            env.swap_data_batch()
    env.close()
    del env
    del env_config
