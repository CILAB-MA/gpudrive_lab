import os, sys, torch
import numpy as np
sys.path.append(os.getcwd())
import wandb, yaml, argparse
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import random
# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv

from baselines.il.config import BehavCloningConfig
from baselines.il.run_bc_from_scratch import ContFeedForward

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='discrete', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],)
    parser.add_argument('--load-dir', '-l', type=str, default='models')
    parser.add_argument('--make-video', '-mv', action='store_true')
    parser.add_argument('--model-name', '-m', type=str, default='bc_policy')
    parser.add_argument('--action-scale', '-as', type=int, default=100)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    args = parser.parse_args()
    return args
def change_obs_prime(obs, veh_ind, features='partner'):

    # todo: make variables for getting specific chagnes of values
    if features == 'ego':
        obs_prime = change_ego_state(obs) # this chnges the ego state value in reasonable boundary.

    elif features == 'partner':
        obs_prime = change_partner_state(obs, veh_ind) # this chnges the ego state value in reasonable boundary.

    else:
        obs_prime = change_edge_state(obs) # this chnges the ego state value in reasonable boundary.

    return obs_prime

def change_ego_state(obs):
    '''
    Define the reasonable bound of ego
    - veh speed
    - veh length
    - veh width
    - relative goal pos
    - collision or not
    '''

    pass

def change_partner_state(obs, veh_ind, feat_ind=0, delta=0.1):
    '''
    obs: (num_world, num_veh, num_partner, partner_feat)
    delta: (N, )
    Define the reasonable bound of partner
    (num_control, 10)
    - speed 1
    - relative pos 2
    - relative orienttaion 1
    - length 1
    - width 1
    - type
    '''
    partner_obs = obs[:, :, 6:1276]

    obs_prime = partner_obs.reshape(-1, 128, 127, 10).clone()

    if feat_ind == 0:
        obs_prime[:, :, veh_ind, 0] += delta
        obs_prime[:, :, veh_ind, 0] = torch.clamp(obs_prime[:, :, veh_ind, 0], 0, 1)

    elif feat_ind == 1:
        obs_prime[:, veh_ind, 1:3] += delta
        obs_prime[:, veh_ind, 1:3] = torch.clamp(obs_prime[:, veh_ind, 1:3], -1, 1)
    obs_prime = obs_prime.reshape(-1, 128, 1270)
    obs[:, :, 6:1276] = obs_prime
    return obs

def visualize_heatmap(diffs, delta):

    diffs_sum = diffs.sum(dim=-1, keepdim=True)
    diffs_sum[diffs_sum == 0] = 1
    diffs_normalized = diffs / diffs_sum
    diffs_np = diffs_normalized.detach().cpu().numpy()
    diffs_sum_per_timestep = diffs.sum(dim=-1).cpu().numpy()

    plt.figure(figsize=(14, 6))

    gs = plt.GridSpec(1, 2, width_ratios=[3, 1])

    plt.subplot(gs[0])
    sns.heatmap(diffs_np, cmap='viridis', cbar=True)
    plt.title(f"Action difference heatmap\nDelta -0.5 ~ 0.5")
    plt.xlabel("Vehicle Index")
    plt.ylabel("Time Step")

    plt.subplot(gs[1])
    sns.heatmap(diffs_sum_per_timestep.reshape(-1, 1), cmap='viridis', cbar=True)
    plt.title("Total Action Difference per Timestep")
    plt.xlabel("Time Step")
    plt.ylabel("Total Action Difference")

    plt.tight_layout()
    plt.savefig('intervention_example_with_sum.jpg', dpi=300)
    plt.show()



def change_edge_state(obs):
    pass

if __name__ == "__main__":
    args = parse_args()
    # Configurations
    # Configurations
    env_config = EnvConfig(
        dynamics_model=args.dynamics_model,
        steer_actions=torch.round(
            torch.linspace(-0.3, 0.3, 7), decimals=3
        ),
        accel_actions=torch.round(
            torch.linspace(-6.0, 6.0, 7), decimals=3
        ),
        dx=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ),
        dyaw=torch.round(
            torch.linspace(-3.14, 3.14, 300), decimals=3
        ),
    )
    render_config = RenderConfig()
    bc_config = BehavCloningConfig()

    NUM_WORLDS = 1
    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    # print('Initializeing env....')
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=1,  # Number of agents to control
        device=args.device,
        render_config=render_config,
        action_type='continuous',
        num_stack=args.num_stack
    )
    bc_policy = torch.load(f"{bc_config.model_path}/{args.model_name}.pth").to(args.device)
    bc_policy.eval()
    alive_agent_mask = env.cont_agent_mask.clone()
    obs = env.reset()
    frames = []
    deltas = torch.linspace(-0.5, 0.5, 20)
    diffs = torch.zeros(NUM_WORLDS, 91, 127).to(args.device)
    for time_step in range(env.episode_len):
        with torch.no_grad():
            actions = bc_policy(obs, deterministic=True)
        for veh_ind in range(127):
            action_deltas = torch.zeros(len(deltas))
            for i, delta in enumerate(deltas):
                with torch.no_grad():
                    obs_prime = change_partner_state(obs, veh_ind=veh_ind, delta=delta)
                    actions_prime = bc_policy(obs_prime, deterministic=True)
                    action_deltas[i] = abs(actions - actions_prime).sum() / (3 * args.action_scale)
            diff = action_deltas.mean()
            diffs[:, time_step, veh_ind] = diff
        env.step_dynamics(actions / args.action_scale)

        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()
    visualize_heatmap(diffs[0], delta)
