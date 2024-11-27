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
from tqdm import tqdm
from datetime import datetime

# GPUDrive
from pygpudrive.env.config import EnvConfig
from baselines.il.config import ExperimentConfig
from algorithms.il import MODELS, LOSS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--model-name', '-m', type=str, default='bc', choices=['bc', 'late_fusion', 'attention', 'wayformer'])
    parser.add_argument('--loss-name', '-l', type=str, default='l1', choices=['l1', 'mse', 'twohot', 'gmm'])
    parser.add_argument('--action-scale', '-as', type=int, default=1)
    
    # DATA
    parser.add_argument('--data-path', '-dp', type=str, default='/data/train_trajectory_by_veh')
    parser.add_argument('--train-data-file', '-td', type=str, default='train_sorted_trajectory_1000.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='eval_sorted_trajectory_200.npz')
    
    # EXPERIMENT
    parser.add_argument('--exp-name', '-en', type=str, default='exp_description')
    args = parser.parse_args()
    
    return args

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None):
        self.obs = obs
        self.actions = actions
        self.masks = masks

        if self.masks is not None:
            valid_indices = self.masks.flatten() == 0
            self.obs = self.obs.reshape(-1, self.obs.shape[-1])[valid_indices]
            self.actions = self.actions.reshape(-1, self.actions.shape[-1])[valid_indices]


    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.actions[idx]

if __name__ == "__main__":
    args = parse_args()
    exp_config = ExperimentConfig()
    env_config = EnvConfig(
        dynamics_model='delta_local',
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
    eval_expert_obs, eval_expert_actions, = [], []
    
    # Additional data depends on model
    train_expert_masks, eval_expert_masks = [], []
    
    with np.load(os.path.join(args.data_path, args.train_data_file)) as npz:
        train_expert_obs.append(npz['obs'])
        train_expert_actions.append(npz['actions'])
        if 'dead_mask' in npz.keys():
            train_expert_masks.append(npz['dead_mask'])
    with np.load(os.path.join(args.data_path, args.eval_data_file)) as npz:
        eval_expert_obs.append(npz['obs'])
        eval_expert_actions.append(npz['actions'])
        if 'dead_mask' in npz.keys():
            eval_expert_masks.append(npz['dead_mask'])

    train_expert_obs = np.concatenate(train_expert_obs)
    train_expert_actions = np.concatenate(train_expert_actions)
    train_expert_masks = np.concatenate(train_expert_masks) if len(train_expert_masks) > 0 else None

    eval_expert_obs = np.concatenate(eval_expert_obs)
    eval_expert_actions = np.concatenate(eval_expert_actions)
    eval_expert_masks = np.concatenate(eval_expert_masks) if (len(eval_expert_masks) > 0) else None

    # Make dataloader
    expert_dataset = ExpertDataset(train_expert_obs, train_expert_actions, train_expert_masks)
    expert_data_loader = DataLoader(
        expert_dataset,
        batch_size=exp_config.batch_size,
        shuffle=True,
    )
    eval_expert_dataset = ExpertDataset(eval_expert_obs, eval_expert_actions, eval_expert_masks)
    eval_expert_data_loader = DataLoader(
        eval_expert_dataset,
        batch_size=exp_config.batch_size,
        shuffle=False,
    )
    
    # Build Model
    bc_policy = MODELS[args.model_name](env_config, exp_config, args.loss_name, args.num_stack).to(args.device)
    
    # Configure loss and optimizer
    optimizer = Adam(bc_policy.parameters(), lr=exp_config.lr)
    dataset_len = len(expert_dataset)

    # Logging
    with open("private.yaml") as f:
        private_info = yaml.load(f, Loader=yaml.FullLoader)
    wandb.login(key=private_info["wandb_key"])
    run_id = f"{type(bc_policy).__name__}_{args.exp_name}"
    model_save_path = f"{args.model_path}/{args.model_name}_{args.exp_name}.pth"
    wandb.init(
        project=private_info['main_project'],
        entity=private_info['entity'],
        name=run_id,
        id=run_id + "_" + datetime.now().strftime("%m%d%H%M"),
        group=f"{args.model_name}",
        config={**exp_config.__dict__, **env_config.__dict__},
        tags=[args.model_name, args.loss_name, args.exp_name, str(dataset_len)]
    )
    wandb.config.update({
        'num_stack': args.num_stack,
        'num_scene': train_expert_actions.shape[0],
        'num_vehicle': 128,
        'model_save_path': model_save_path})
    
    global_step = 0
    for epoch in tqdm(range(exp_config.epochs), desc="Epochs", unit="epoch"):
        bc_policy.train()
        total_samples = 0
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, (obs, expert_action) in enumerate(expert_data_loader):
            batch_size = obs.size(0)
            if total_samples + batch_size > exp_config.sample_per_epoch:  # Check if adding this batch exceeds 50,000
                break
            total_samples += batch_size

            obs, expert_action = obs.to(args.device), expert_action.to(args.device)
            
            # Forward pass
            expert_action *= args.action_scale
            loss = LOSS[args.loss_name](bc_policy, obs, expert_action)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # Update model parameters

            with torch.no_grad():
                pred_actions = bc_policy(obs, deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
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
            if total_samples + batch_size > int(exp_config.sample_per_epoch / 5):  # Check if adding this batch exceeds 50,000
                break
            total_samples += batch_size
            obs, expert_action = obs.to(args.device), expert_action.to(args.device)

            with torch.no_grad():
                pred_actions = bc_policy(obs, deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
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
    torch.save(bc_policy, f"{args.model_path}/{args.model_name}_{args.loss_name}_{args.exp_name}.pth")
