"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim as optim
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from datetime import datetime
from tqdm import tqdm

# GPUDrive
from pygpudrive.env.config import EnvConfig
from baselines.il.config import ExperimentConfig
from algorithms.il.model.bc import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--dynamics-model', '-dm', type=str, default='delta_local', choices=['delta_local', 'bicycle', 'classic'],)
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/models')
    parser.add_argument('--model-name', '-m', type=str, default='late_fusion_gmm', choices=['late_fusion_l1', 'bc_l1', 'bc_dist', 'late_fusion_gmm'])
    
    # DATA
    parser.add_argument('--data-path', '-dp', type=str, default='/data')
    parser.add_argument('--train-data-file', '-td', type=str, default='new_train_trajectory_1000.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='eval_trajectory_200.npz')
    args = parser.parse_args()
    return args

def two_hot_encoding(value, bins):
    idx_upper = torch.searchsorted(bins, value, right=True).clamp(max=len(bins) - 1)
    idx_lower = torch.clamp(idx_upper - 1, min=0)
    
    lower_weight = (value - bins[idx_lower]) / (bins[idx_upper] - bins[idx_lower])
    upper_weight =  (bins[idx_upper] - value) / (bins[idx_upper] - bins[idx_lower])
    batch_indices = torch.arange(len(value), device=value.device)
    two_hot = torch.zeros(len(value), len(bins), device=value.device)
    two_hot[batch_indices, idx_lower] = lower_weight
    two_hot[batch_indices, idx_upper] = upper_weight
    
    return two_hot

def two_hot_loss(pred, targ, dx_bins, dy_bins, dyaw_bins):
    '''
    pred: real value of model output
    targ: real value of label
    dx_bins: 
    '''
    pred_dist = torch.zeros(len(pred), len(dx_bins), 3,  device=pred.device)
    targ_dist = torch.zeros(len(targ), len(dx_bins), 3, device=pred.device)
    pred_dist[..., 0] = two_hot_encoding(bins=dx_bins, value=pred[:, 0] )
    pred_dist[..., 1] = two_hot_encoding(bins=dy_bins, value=pred[:, 1] )
    pred_dist[..., 2] = two_hot_encoding(bins=dyaw_bins, value=pred[:, 2] )

    targ_dist[..., 0] = two_hot_encoding(bins=dx_bins, value=targ[:, 0] )
    targ_dist[...,1] = two_hot_encoding(bins=dy_bins, value=targ[:, 1] )
    targ_dist[...,2] = two_hot_encoding(bins=dyaw_bins, value=targ[:, 2] )
    epsilon = 1e-8
    log_targ_dist = torch.log(targ_dist + epsilon)

    loss_dx = (pred_dist[..., 0] * log_targ_dist[..., 0]).sum(dim=-1).mean()
    loss_dy = (pred_dist[..., 1] * log_targ_dist[..., 1]).sum(dim=-1).mean()
    loss_dyaw = (pred_dist[..., 2] * log_targ_dist[..., 2]).sum(dim=-1).mean()

    total_loss = (loss_dx + loss_dy + loss_dyaw) / 3

    return total_loss

def two_hot_encoding(value, bins):
    idx_upper = torch.searchsorted(bins, value, right=True).clamp(max=len(bins) - 1)
    idx_lower = torch.clamp(idx_upper - 1, min=0)
    
    lower_weight = (value - bins[idx_lower]) / (bins[idx_upper] - bins[idx_lower])
    upper_weight =  (bins[idx_upper] - value) / (bins[idx_upper] - bins[idx_lower])
    batch_indices = torch.arange(len(value), device=value.device)
    two_hot = torch.zeros(len(value), len(bins), device=value.device)
    two_hot[batch_indices, idx_lower] = lower_weight
    two_hot[batch_indices, idx_upper] = upper_weight
    
    return two_hot

def two_hot_loss(pred, targ, dx_bins, dy_bins, dyaw_bins):
    '''
    pred: real value of model output
    targ: real value of label
    dx_bins: 
    '''
    pred_dist = torch.zeros(len(pred), len(dx_bins), 3,  device=pred.device)
    targ_dist = torch.zeros(len(targ), len(dx_bins), 3, device=pred.device)
    pred_dist[..., 0] = two_hot_encoding(bins=dx_bins, value=pred[:, 0] )
    pred_dist[..., 1] = two_hot_encoding(bins=dy_bins, value=pred[:, 1] )
    pred_dist[..., 2] = two_hot_encoding(bins=dyaw_bins, value=pred[:, 2] )

    targ_dist[..., 0] = two_hot_encoding(bins=dx_bins, value=targ[:, 0] )
    targ_dist[...,1] = two_hot_encoding(bins=dy_bins, value=targ[:, 1] )
    targ_dist[...,2] = two_hot_encoding(bins=dyaw_bins, value=targ[:, 2] )
    epsilon = 1e-8
    log_targ_dist = torch.log(targ_dist + epsilon)

    loss_dx = (pred_dist[..., 0] * log_targ_dist[..., 0]).sum(dim=-1).mean()
    loss_dy = (pred_dist[..., 1] * log_targ_dist[..., 1]).sum(dim=-1).mean()
    loss_dyaw = (pred_dist[..., 2] * log_targ_dist[..., 2]).sum(dim=-1).mean()

    total_loss = (loss_dx + loss_dy + loss_dyaw) / 3

    return total_loss


if __name__ == "__main__":
    args = parse_args()
    exp_config = ExperimentConfig()
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
        ).to(args.device),
        dy=torch.round(
            torch.linspace(-6.0, 6.0, 100), decimals=3
        ).to(args.device),
        dyaw=torch.round(
            torch.linspace(-np.pi, np.pi, 100), decimals=3
        ).to(args.device),
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
        batch_size=exp_config.batch_size,
        shuffle=True,  # Break temporal structure
    )
    eval_expert_dataset = ExpertDataset(eval_expert_obs, eval_expert_actions)
    eval_expert_data_loader = DataLoader(
        eval_expert_dataset,
        batch_size=exp_config.batch_size,
        shuffle=False,  # Break temporal structure
    )
    
    # Build Model
    if args.model_name == 'bc_l1':    
        bc_policy = ContFeedForwardMSE(
            input_size=train_expert_obs.shape[-1],
            hidden_size=exp_config.hidden_size,
            output_size=3,
        ).to(args.device)
    elif args.model_name == 'late_fusion_l1':
        bc_policy = LateFusionBCNet(
            observation_space=None,
            exp_config=exp_config,
            env_config=env_config
        ).to(args.device)
    elif args.model_name == 'bc_dist':
        bc_policy = ContFeedForward(
            input_size=train_expert_obs.shape[-1],
            hidden_size=exp_config.hidden_size,
            output_size=3,
        ).to(args.device)
    elif args.model_name == 'attn_l1':
        bc_policy = LateFusionAttnBCNet(
            observation_space=None,
            exp_config=exp_config,
            env_config=env_config
        ).to(args.device)
    elif args.model_name == 'late_fusion_gmm':
        bc_policy = LateFusionGmmBCNet(
            observation_space=train_expert_obs.shape[-1],
            exp_config=exp_config,
            env_config=env_config
        ).to(args.device)
    else:
        raise ValueError(f"Model name {args.model_name} is not supported")
    
    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=exp_config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    sample_per_epoch = 50000
    dataset_len = len(expert_dataset)

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
        config={**exp_config.__dict__, **env_config.__dict__},
        tags=[args.model_name, args.action_type, env_config.dynamics_model, str(dataset_len)]
    )
    wandb.config.update({
        'lr': exp_config.lr,
        'batch_size': exp_config.batch_size,
        'num_stack': args.num_stack,
        'num_scene': train_expert_actions.shape[0],
        'num_vehicle': 128
    })
    
    global_step = 0
    for epoch in tqdm(range(exp_config.epochs), desc="Epochs", unit="epoch"):
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

            # Forward pass
            # pred_actions = bc_policy(obs)
            # loss = two_hot_loss(pred_actions, expert_action, 
            #                     dx_bins=env_config.dx,
            #                     dy_bins=env_config.dy,
            #                     dyaw_bins=env_config.dyaw)
            # loss = F.smooth_l1_loss(pred_actions, expert_action)
            loss = bc_policy.compute_loss(obs, expert_action)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model parameters

            with torch.no_grad():
                pred_action = bc_policy.sample(obs)
                action_loss = torch.abs(pred_action - expert_action)
                dx_loss = action_loss[:, 0].mean().item()
                dy_loss = action_loss[:, 1].mean().item()
                dyaw_loss = action_loss[:, 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                
            losses += loss.mean().item()
        scheduler.step()
        # Log training losses
        wandb.log(
            {   
                "train/loss": losses / (i + 1),
                "train/loss": losses / (i + 1),
                "train/dx_loss": dx_losses / (i + 1),
                "train/dy_loss": dy_losses / (i + 1),
                "train/dyaw_loss": dyaw_losses / (i + 1),
            }
        )

        # Evaluation loop
        bc_policy.eval()
        total_samples = 0  # Initialize sample counter
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, (obs, expert_action) in enumerate(eval_expert_data_loader):
            batch_size = obs.size(0)
            if total_samples + batch_size > 10000:  # Check if adding this batch exceeds 50,000
                break
            total_samples += batch_size
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)

            with torch.no_grad():
                pred_action = bc_policy.sample(obs)
                action_loss = torch.abs(pred_action - expert_action)
                dx_loss = action_loss[:, 0].mean().item()
                dy_loss = action_loss[:, 1].mean().item()
                dyaw_loss = action_loss[:, 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                losses += action_loss.mean().item()
            
        # Log evaluation losses
        wandb.log(
            {
                "eval/loss": losses / (i + 1) ,
                "eval/dx_loss": dx_losses / (i + 1),
                "eval/dy_loss": dy_losses / (i + 1),
                "eval/dyaw_loss": dyaw_losses / (i + 1),
            }
        )

    # Save policy
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    torch.save(bc_policy, f"{args.model_path}/{args.model_name}_scale_{dataset_len}.pth")
