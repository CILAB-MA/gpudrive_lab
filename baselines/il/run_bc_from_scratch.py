"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import os, sys, torch
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from tqdm import tqdm
from datetime import datetime

# GPUDrive
from baselines.il.config import *
from baselines.il.dataloader import ExpertDataset
from algorithms.il import MODELS, LOSS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    # ENVIRONMENT
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/data/model')
    parser.add_argument('--model-name', '-m', type=str, default='aux_fusion', choices=['bc', 'late_fusion', 'attention', 'wayformer', 'aux_fusion'])
    parser.add_argument('--loss-name', '-l', type=str, default='gmm', choices=['l1', 'mse', 'twohot', 'nll', 'gmm'])
    parser.add_argument('--rollout-len', '-rl', type=int, default=5)
    parser.add_argument('--pred-len', '-pl', type=int, default=1)
    
    # DATA
    parser.add_argument('--data-path', '-dp', type=str, default='/data/tom')
    parser.add_argument('--train-data-file', '-td', type=str, default='train_trajectory_1000.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='test_trajectory_200.npz')
    
    # EXPERIMENT
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--use-tom', '-ut', type=str, default='none', choices=['none', 'oracle', 'aux_head'])
    args = parser.parse_args()
    
    return args

def train():
    net_config = NetworkConfig()
    env_config = EnvConfig()
    head_config = HeadConfig()

    if args.use_wandb:
        wandb.init()
        # Tag Update
        current_time = datetime.now().strftime("%Y%m%d_%H%M")
        wandb_tags = list(wandb.run.tags)
        wandb_tags.append(current_time)
        for key, value in wandb.config.items():
            wandb_tags.append(f"{key}_{value}")
        wandb.run.tags = tuple(wandb_tags)
        # Config Update
        for key, value in vars(args).items():
            if key not in wandb.config:
                wandb.config[key] = value
        config = wandb.config
        wandb.run.name = f"{config.model_name}_{config.loss_name}_{config.exp_name}"
        wandb.run.save()
        # NetConfig, HeadConfig Update (if sweep parameter is used)
        for key, value in config.items():
            if key in net_config.__dict__.keys():
                setattr(net_config, key, value)
            if key in head_config.__dict__.keys():
                setattr(head_config, key, value)
    else:
        config = ExperimentConfig()
        config.__dict__.update(vars(args))
    
    # Initialize model and optimizer
    bc_policy = MODELS[config.model_name](env_config, net_config, head_config, config.loss_name, config.num_stack).to(config.device)
    optimizer = Adam(bc_policy.parameters(), lr=config.lr)

    # Get state action pairs
    train_expert_obs, train_expert_actions = [], []
    eval_expert_obs, eval_expert_actions, = [], []
    
    # Additional data depends on model
    train_expert_masks, eval_expert_masks = [], []
    train_other_info, eval_other_info = [], []
    train_road_mask, eval_road_mask = [], []
    
    # Load cached data
    with np.load(os.path.join(config.data_path, config.train_data_file)) as npz:
        train_expert_obs = [npz['obs']]
        train_expert_actions = [npz['actions']]
        train_expert_masks = [npz['dead_mask']] if ('dead_mask' in npz.keys() and config.use_mask) else []
        train_other_info = [npz['other_info']] if ('other_info' in npz.keys() and config.use_tom) else []
        train_road_mask = [npz['road_mask']] if ('road_mask' in npz.keys() and config.use_mask) else []

    with np.load(os.path.join(config.data_path, config.eval_data_file)) as npz:
        eval_expert_obs = [npz['obs']]
        eval_expert_actions = [npz['actions']]
        eval_expert_masks = [npz['dead_mask']] if ('dead_mask' in npz.keys() and config.use_mask) else []
        eval_other_info = [npz['other_info']] if ('other_info' in npz.keys() and config.use_tom) else []
        eval_road_mask = [npz['road_mask']] if ('road_mask' in npz.keys() and config.use_mask) else []


    # Combine data (no changes)
    num_cpus = os.cpu_count()

    train_expert_obs = np.concatenate(train_expert_obs)
    train_expert_actions = np.concatenate(train_expert_actions)
    train_expert_masks = np.concatenate(train_expert_masks) if len(train_expert_masks) > 0 else None
    train_other_info = np.concatenate(train_other_info) if len(train_other_info) > 0 else None
    train_road_mask = np.concatenate(train_road_mask) if len(train_road_mask) > 0 else None
    expert_data_loader = DataLoader(
        ExpertDataset(
            train_expert_obs, train_expert_actions, train_expert_masks,
            other_info=train_other_info, road_mask=train_road_mask,
            rollout_len=config.rollout_len, pred_len=config.pred_len
        ),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=int(num_cpus / 2),
        pin_memory=True
    )
    del train_expert_obs
    del train_expert_actions
    del train_expert_masks

    eval_expert_obs = np.concatenate(eval_expert_obs)
    eval_expert_actions = np.concatenate(eval_expert_actions)
    eval_expert_masks = np.concatenate(eval_expert_masks) if len(eval_expert_masks) > 0 else None
    eval_other_info = np.concatenate(eval_other_info) if len(eval_other_info) > 0 else None
    eval_road_mask = np.concatenate(eval_road_mask) if len(eval_road_mask) > 0 else None
    eval_expert_data_loader = DataLoader(
        ExpertDataset(
            eval_expert_obs, eval_expert_actions, eval_expert_masks,
            other_info=eval_other_info, road_mask=eval_road_mask,
            rollout_len=config.rollout_len, pred_len=config.pred_len
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=int(num_cpus / 2),
        pin_memory=True
    )
    del eval_expert_obs
    del eval_expert_actions
    del eval_expert_masks

    # Training loop
    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="epoch"):
        bc_policy.train()
        total_samples = 0
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, batch in enumerate(expert_data_loader):
            batch_size = batch[0].size(0)
            if total_samples + batch_size > config.sample_per_epoch:
                break
            total_samples += batch_size
            
            if len(batch) == 7:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info = batch
            elif len(batch) == 6:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks = batch 
            elif len(batch) == 3:
                obs, expert_action, masks = batch
            else:
                obs, expert_action = batch
            obs, expert_action = obs.to(config.device), expert_action.to(config.device)
            masks = masks.to(config.device) if len(batch) > 2 else None
            ego_masks = ego_masks.to(config.device) if len(batch) > 3 else None
            partner_masks = partner_masks.to(config.device) if len(batch) > 3 else None
            road_masks = road_masks.to(config.device) if len(batch) > 3 else None
            other_info = other_info.to(config.device).transpose(1, 2).reshape(batch_size, 127, -1) if len(batch) > 6 else None
            all_masks= [masks, ego_masks, partner_masks, road_masks]
            
            # Forward pass
            if config.use_tom == 'oracle':
                context = bc_policy.get_context(obs, all_masks[1:], other_info=other_info)
                loss = LOSS[config.loss_name](bc_policy, context, expert_action, all_masks)
            elif config.use_tom == 'aux_head':
                context, other_embeds = bc_policy.get_context(obs, all_masks[1:], other_info=None)
                tom_a_loss = LOSS[config.loss_name](bc_policy, other_embeds, other_info[...,:3], all_masks, aux_head='action')
                tom_g_loss = LOSS[config.loss_name](bc_policy, other_embeds, other_info[...,3:], all_masks, aux_head='goal')
                pred_loss = LOSS[config.loss_name](bc_policy, context, expert_action, all_masks)
                loss = pred_loss + tom_a_loss + tom_g_loss
            else:
                context = bc_policy.get_context(obs, all_masks[1:])
                loss = LOSS[config.loss_name](bc_policy, context, expert_action, all_masks)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_actions = bc_policy.get_action(context, deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                
            losses += loss.mean().item()
        if config.use_wandb:
            wandb.log(
                {   
                    "train/loss": losses / (i + 1),
                    "train/dx_loss": dx_losses / (i + 1),
                    "train/dy_loss": dy_losses / (i + 1),
                    "train/dyaw_loss": dyaw_losses / (i + 1),
                }, step=epoch
            )

        # Evaluation loop
        bc_policy.eval()
        total_samples = 0
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        for i, batch in enumerate(eval_expert_data_loader):
            batch_size = batch[0].size(0)
            if total_samples + batch_size > int(config.sample_per_epoch / 5): 
                break
            total_samples += batch_size
            
            if len(batch) == 7:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info = batch  
            elif len(batch) == 6:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks = batch  
            elif len(batch) == 3:
                obs, expert_action, masks = batch
            else:
                obs, expert_action = batch
            obs, expert_action = obs.to(config.device), expert_action.to(config.device)
            masks = masks.to(config.device) if len(batch) > 2 else None
            ego_masks = ego_masks.to(config.device) if len(batch) > 3 else None
            partner_masks = partner_masks.to(config.device) if len(batch) > 3 else None
            road_masks = road_masks.to(config.device) if len(batch) > 3 else None
            other_info = other_info.to(config.device) if len(batch) > 6 else None
            all_masks= [masks, ego_masks, partner_masks, road_masks]
            
            with torch.no_grad():
                pred_actions = bc_policy(obs, all_masks[1:], deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                losses += action_loss.mean().item()
        
        if config.use_wandb:
            wandb.log(
                {
                    "eval/loss": losses / (i + 1) ,
                    "eval/dx_loss": dx_losses / (i + 1),
                    "eval/dy_loss": dy_losses / (i + 1),
                    "eval/dyaw_loss": dyaw_losses / (i + 1),
                }, step=epoch
            )

    # Save policy
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    torch.save(bc_policy, f"{config.model_path}/{config.model_name}_{config.loss_name}_{config.exp_name}_{current_time}.pth")


if __name__ == "__main__":
    args = parse_args()

    if args.use_wandb:
        with open("baselines/il/sweep.yaml") as f:
            exp_config = yaml.load(f, Loader=yaml.FullLoader)
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        sweep_id = wandb.sweep(exp_config, project=private_info['main_project'], entity=private_info['entity'])
        wandb.agent(sweep_id, function=train)
    else:
        train()
