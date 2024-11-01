"""Obtain a policy using behavioral cloning."""

# Torch
import logging
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from datetime import datetime
import numpy  as np
import pyrallis
# GPUDrive
from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
from pygpudrive.env.env_torch import GPUDriveTorchEnv
from pygpudrive.env.wrappers.sb3_wrapper import SB3MultiAgentEnv
from algorithms.il.data_generation import generate_state_action_pairs
from baselines.il.config import BehavCloningConfig
from algorithms.il.model.bc import *
from baselines.ippo.config import ExperimentConfig
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--model-name', '-m', type=str, default='late_fusion_l1', choices=['late_fusion_l1', 
                                                                                      'bc_mse, bc_dist'])
    parser.add_argument('--action-scale', '-as', type=int, default=1)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    parser.add_argument('--data-path', '-dp', type=str, default='/data')
    parser.add_argument('--train-data-file', '-td', type=str, default='new_train_trajectory_1000.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='eval_trajectory_200.npz')
    args = parser.parse_args()
    return args
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    args = parse_args()
    bc_config = BehavCloningConfig()
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
    # Get state action pairs
    train_expert_obs, train_expert_actions = [], []
    eval_expert_obs, eval_expert_actions = [], []
    
    with np.load(os.path.join(args.data_path, args.train_data_file)) as npz:
        train_expert_obs.append(npz['obs'])
        train_expert_actions.append(npz['actions'])
    with np.load(os.path.join(args.data_path, args.eval_data_file)) as npz:
        eval_expert_obs.append(npz['obs'])
        eval_expert_actions.append(npz['actions'])

    train_expert_obs = np.concatenate(train_expert_obs)
    train_expert_actions = np.concatenate(train_expert_actions)
    eval_expert_obs = np.concatenate(eval_expert_obs)
    eval_expert_actions = np.concatenate(eval_expert_actions)

    class ExpertDataset(torch.utils.data.Dataset):
        def __init__(self, obs, actions):
            self.obs = obs
            self.actions = actions

        def __len__(self):
            return len(self.obs)

        def __getitem__(self, idx):
            return self.obs[idx], self.actions[idx]

    # Make dataloader
    expert_dataset = ExpertDataset(train_expert_obs, train_expert_actions)
    expert_data_loader = DataLoader(
        expert_dataset,
        batch_size=bc_config.batch_size,
        shuffle=True,  # Break temporal structure
    )
    eval_expert_dataset = ExpertDataset(eval_expert_obs, eval_expert_actions)
    eval_expert_data_loader = DataLoader(
        eval_expert_dataset,
        batch_size=bc_config.batch_size,
        shuffle=False,  # Break temporal structure
    )
    # Build Model
    if args.model_name == 'bc_l1':    
        bc_policy = ContFeedForwardMSE(
            input_size=train_expert_obs.shape[-1],
            hidden_size=bc_config.hidden_size,
            output_size=3,
        ).to(args.device)
    elif args.model_name == 'late_fusion_l1':
        exp_config = pyrallis.parse(config_class=ExperimentConfig)
        bc_policy = LateFusionBCNet(
            observation_space=None,
            exp_config=exp_config,
            env_config=env_config
        ).to(args.device)
    elif args.model_name == 'bc_dist':
        bc_policy = ContFeedForward(
            input_size=train_expert_obs.shape[-1],
            hidden_size=bc_config.hidden_size,
            output_size=3,
        ).to(args.device)
    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=bc_config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    sample_per_epoch = 50000

    # Logging
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    currenttime = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{type(bc_policy).__name__}_{currenttime}"
    wandb.init(
        project=private_info['main_project'],
        entity=private_info['entity'],
        name=run_id,
        id=run_id,
        group=f"{env_config.dynamics_model}_{args.action_type}",
        config={**bc_config.__dict__, **env_config.__dict__},
        tags=[args.model_name, args.action_type, env_config.dynamics_model, 'NORM']
    )
    
    wandb.config.update({
        'lr': bc_config.lr,
        'batch_size': bc_config.batch_size,
        'num_stack': args.num_stack,
        'num_scene': train_expert_actions.shape[0],
        'num_vehicle': 128
    })
    
    global_step = 0
    for epoch in tqdm(range(bc_config.epochs), desc="Epochs", unit="epoch"):
        bc_policy.train()
        total_samples = 0  # Initialize sample counter
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, (obs, expert_action) in enumerate(expert_data_loader):
            batch_size = obs.size(0)
            if total_samples + batch_size > 50000:  # Check if adding this batch exceeds 50,000
                break
            total_samples += batch_size

            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            # Normalize
            expert_action[:, 0] = (expert_action[:, 0] + 6) / 12
            expert_action[:, 1] = (expert_action[:, 1] + 6) / 12
            expert_action[:, 2] = (expert_action[:, 2] + np.pi) / 2 * np.pi

            # Forward pass
            pred_actions = bc_policy(obs)
            loss = F.smooth_l1_loss(pred_actions, expert_action * args.action_scale)
            
            # Backward pass
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()  # Update model parameters
            scheduler.step()

            with torch.no_grad():
                pred_action = bc_policy(obs)
                action_loss = torch.abs(pred_action - expert_action * args.action_scale) / args.action_scale
                dx_loss = action_loss[:, 0].mean().item()
                dy_loss = action_loss[:, 1].mean().item()
                dyaw_loss = action_loss[:, 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                
            losses += loss.mean().item()
        
        # Log training losses
        wandb.log(
            {
                "train/loss": losses / (i + 1) / args.action_scale,
                "train/dx_loss": dx_losses / (i + 1),
                "train/dy_loss": dy_losses / (i + 1),
                "train/dyaw_loss": dyaw_losses / (i + 1),
            }
        )

        # Evaluation loop
        bc_policy.eval()
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, (obs, expert_action) in enumerate(eval_expert_data_loader):
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            # Normalize
            expert_action[:, 0] = (expert_action[:, 0]  + 6) / 12
            expert_action[:, 1] = (expert_action[:, 1]  + 6) / 12
            expert_action[:, 2] = (expert_action[:, 2]  + np.pi) / (2 * np.pi)

            with torch.no_grad():
                pred_action = bc_policy(obs)
                action_loss = torch.abs(pred_action - expert_action * args.action_scale) / args.action_scale
                dx_loss = action_loss[:, 0].mean().item()
                dy_loss = action_loss[:, 1].mean().item()
                dyaw_loss = action_loss[:, 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss

            losses += loss.mean().item()
            
        # Log evaluation losses
        wandb.log(
            {
                "eval/loss": losses / (i + 1) / args.action_scale,
                "eval/dx_loss": dx_losses / (i + 1),
                "eval/dy_loss": dy_losses / (i + 1),
                "eval/dyaw_loss": dyaw_losses / (i + 1),
            }
        )

# Save policy
if bc_config.save_model:
    if not os.path.exists(bc_config.model_path):
        os.makedirs(bc_config.model_path)
    torch.save(bc_policy, f"{bc_config.model_path}/{args.model_name}_FIX_UNPACK.pth")
