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

def change_partner_state(obs, veh_ind, deltas, feat_size):
    '''
    obs: (num_world, num_veh, num_partner, partner_feat)
    '''
    obs = obs.repeat(20, 1, 1)
    for i in range(5):  # todo: should change for num_stack variable
        partner_obs = obs[:, :, i * feat_size + 6:i * feat_size + 1276]

        obs_prime = partner_obs.reshape(-1, 128, 127, 10).clone()

        # Expand obs_prime along the second dimension to match deltas
        # obs_prime = obs_prime.unsqueeze(1).repeat(1, len(deltas), 1, 1, 1)
        # Expand deltas to match the obs_prime dimensions
        deltas_expanded = deltas.view(len(deltas), 1).to(obs_prime.device)
        deltas_expanded = deltas_expanded.repeat(1, 128)
        # Adjust the delta along the vehicle index
        obs_prime[:, :, veh_ind, 0] += deltas_expanded
        obs_prime[:, :, veh_ind, 0] = torch.clamp(obs_prime[:, :, veh_ind, 0], 0, 1)

        # Reshape back to the original shape
        obs_prime = obs_prime.reshape(-1, len(deltas), 128, 1270)
        obs[:, :, i * feat_size + 6:i * feat_size + 1276] = obs_prime

    return obs


if __name__ == "__main__":
    args = parse_args()

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
    feat_size = 3876

    scene_config = SceneConfig(f"/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=1,
        device=args.device,
        render_config=render_config,
        action_type='continuous',
        num_stack=args.num_stack
    )

    bc_policy = torch.load(f"{bc_config.model_path}/{args.model_name}.pth", weights_only=False).to(args.device)
    bc_policy.eval()

    alive_agent_mask = env.cont_agent_mask.clone()
    obs = env.reset()
    frames = []
    deltas = torch.linspace(-0.5, 0.5, 20).to(args.device)
    diffs = torch.zeros(NUM_WORLDS, 91, 127).to(args.device)

    for time_step in range(env.episode_len):
        with torch.no_grad():
            actions = bc_policy(obs, deterministic=True)
        for veh_ind in range(127):
            action_deltas = torch.zeros(len(deltas))
            for i, delta in enumerate(deltas):
                with torch.no_grad():
                    obs_prime = change_partner_state(obs, veh_ind=veh_ind, deltas=deltas,
                                                     feat_size=feat_size)
                    actions_prime = bc_policy(obs_prime, deterministic=True)
                    action_deltas[i] = abs(actions - actions_prime).sum() / (3 * args.action_scale)
            diff = action_deltas.mean()
            diffs[:, time_step, veh_ind] = diff

        env.step_dynamics(actions / args.action_scale)
        obs = env.get_obs()
        dones = env.get_dones()
        infos = env.get_infos()

    visualize_heatmap(diffs[0], deltas)
