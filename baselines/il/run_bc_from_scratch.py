"""Obtain a policy using behavioral cloning."""

# Torch
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from datetime import datetime
import numpy  as np

# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from algorithms.il.data_generation import generate_state_action_pairs
from baselines.il.config import BehavCloningConfig
from algorithms.il.model.bc import *

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cpu', choices=['cpu', 'cuda'],)
    parser.add_argument('--model-name', '-m', type=str, default='bc_policy')
    parser.add_argument('--action-scale', '-as', type=int, default=100)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    parser.add_argument('--data-path', '-dp', type=str, default='/data')
    args = parser.parse_args()
    return args
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    # Get state action pairs
    expert_obs, expert_actions = [], []
    for f in os.listdir(args.data_path):
        with np.load(os.path.join(args.data_path, f)) as npz:
            expert_obs.append(npz['obs'])
            expert_actions.append(npz['actions'])

    NUM_WORLDS = 50
    scene_config = SceneConfig("/data/formatted_json_v2_no_tl_train/", NUM_WORLDS)
    env = GPUDriveTorchEnv(
        config=env_config,
        scene_config=scene_config,
        max_cont_agents=128,  # Number of agents to control
        device=args.device,
        action_type=args.action_type,
        num_stack=args.num_stack
    )
    # Generate expert actions and observations
    (
        expert_obs,
        expert_actions,
        next_expert_obs,
        expert_dones,
        goal_rate,
        collision_rate
    ) = generate_state_action_pairs(
        env=env,
        device='cpu',
        action_space_type=args.action_type,  # Discretize the expert actions
        use_action_indices=True,  # Map action values to joint action index
        make_video=True,  # Record the trajectories as sanity check
        render_index=[0, 0],  # start_idx, end_idx
        debug_world_idx=None,
        debug_veh_idx=None,
        save_path="run_bc_from_scratch",
    )
    print('Generating action pairs...')

    expert_obs = np.concatenate(expert_obs, axis=0)
    expert_actions = np.concatenate(expert_actions, axis=0)
    print(f'OBS SHAPE {expert_obs.shape} ACTIONS SHAPE {expert_actions.shape}')
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    filename = datetime.now().strftime("%Y%m%d%H%M%S")
    wandb.init(project=private_info['project'], entity=private_info['entity'],
               name=f'{filename}')
    wandb.config.update({
        'lr': bc_config.lr,
        'batch_size': bc_config.batch_size,
        'num_stack': args.num_stack,
        'num_scene': expert_actions.shape[0],
        'num_vehicle': 128
    })


    class ExpertDataset(torch.utils.data.Dataset):
        def __init__(self, obs, actions):
            self.obs = obs
            self.actions = actions

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, idx):
            return self.obs[idx], self.actions[idx]

    # Make dataloader
    expert_dataset = ExpertDataset(expert_obs, expert_actions)
    expert_data_loader = DataLoader(
        expert_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )

    # # Build model
    bc_policy = ContFeedForwardMSE(
        input_size=expert_obs.shape[-1],
        hidden_size=bc_config.hidden_size,
        output_size=3,
    ).to(args.device)
    #
    # bc_policy = WayForward(
    #         input_size=env.observation_space.shape[0],
    #         hidden_size=bc_config.hidden_size[1],
    #     ).to(args.device)

        # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=bc_config.lr)

    global_step = 0
    for epoch in range(bc_config.epochs):
        for i, (obs, expert_action) in enumerate(expert_data_loader):

            obs, expert_action = obs.to(args.device), expert_action.to(
                args.device
            )

            # # Forward pass
            pred_action = bc_policy(obs)
            # mu, vars, mixed_weights = bc_policy(obs)
            # log_prob = bc_policy._log_prob(obs, expert_action)
            # loss = -log_prob
            loss = F.smooth_l1_loss(pred_action, expert_action * args.action_scale)
            # loss = gmm_loss(mu, vars, mixed_weights, expert_actions)
            # Backward pass
            with torch.no_grad():
                pred_action = bc_policy(obs)
                action_loss = torch.abs(pred_action - expert_action * args.action_scale) / args.action_scale
                dx_loss = action_loss[:, 0].mean().item()
                dy_loss = action_loss[:, 1].mean().item()
                dyaw_loss = action_loss[:, 2].mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log(
                {
                    "global_step": global_step,
                    "loss": loss.item(),
                    "dx_loss":dx_loss,
                    "dy_loss":dy_loss,
                    "dyaw_loss":dyaw_loss,
                }
            )

            global_step += 1

    # Save policy
    if bc_config.save_model:
        torch.save(bc_policy, f"{bc_config.model_path}/{args.model_name}.pth")
