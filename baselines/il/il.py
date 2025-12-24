"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, sys, torch
torch.backends.cudnn.benchmark = True
sys.path.append(os.getcwd())
import wandb, yaml, argparse
from tqdm import tqdm
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from baselines.il.config.config import EnvConfig
from baselines.il.il_utils import *
from box import Box
# GPUDrive
from gpudrive.integrations.il.dataloader import ExpertDataset
from gpudrive.integrations.il.model.model import EarlyFusionAttnBCNet
from gpudrive.integrations.il.loss import gmm_loss
# from algorithms.il.utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS = dict(early_attn=EarlyFusionAttnBCNet,)
def parse_args():
    parser = argparse.ArgumentParser("Most of vars are in il.yaml. These are for different server.")
    # DATALOADER
    parser.add_argument('--num-workers', '-nw', type=int, default=16)
    parser.add_argument('--prefetch-factor', '-pf', type=int, default=4)
    parser.add_argument('--pin-memory', '-pm', action='store_true')
    parser.add_argument('--load-path', type=str, default=None)
    # EXPERIMENT
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--sweep-id', type=str, default=None)
    args = parser.parse_args()
    
    return args

def load_config(config_path):
    """Load the configuration file."""
    with open(config_path, "r") as f:
        return Box(yaml.safe_load(f))
    
def set_seed(seed=42, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

def get_grad_norm(params, step=None):
    max_grad_norm = 0
    grad_name = None
    for name, param in params:
        if param.grad is not None:
            grad_norm = param.grad.view(-1).norm(2).item()  # L2 norm
            if grad_norm > max_grad_norm:
                max_grad_norm = grad_norm
                grad_name = str(name)
    return max_grad_norm, grad_name

def get_dataloader(data_path, data_file, config, isshuffle=True):
    with np.load(os.path.join(data_path, data_file), mmap_mode='r') as npz:
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None

    dataset = ExpertDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask,
        rollout_len=config.rollout_len, pred_len=config.pred_len, aux_future_step=config.aux_future_step,
        ego_global_pos=None, ego_global_rot=None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=isshuffle,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory,
        persistent_workers=True
    )
    del dataset
    return dataloader

def evaluate(eval_expert_data_loader, config, bc_policy, num_train_sample):
    total_samples = 0
    losses = 0
    dx_losses = 0
    dy_losses = 0
    dyaw_losses = 0
    dx_std2_losses = 0
    dy_std2_losses = 0
    dyaw_std2_losses = 0
    dx_values = 0
    dy_values = 0
    dyaw_values = 0
    dx_std2_count = 0
    dy_std2_count = 0
    dyaw_std2_count = 0
    cvar95_losses = 0
    for i, batch in enumerate(eval_expert_data_loader):
        batch_size = batch[0].size(0)
        total_samples += batch_size
        if len(batch) == 7:
            obs, expert_action, partner_masks, road_masks, other_pos, aux_mask, data_idx = batch
            other_pos = other_pos.to(exp_config.device)
        elif len(batch) == 5:
            obs, expert_action, partner_masks, road_masks, data_idx = batch 

        obs, expert_action = obs.to(config.device), expert_action.to(config.device)
        partner_masks = partner_masks.to(config.device) if len(batch) > 3 else None
        road_masks = road_masks.to(config.device) if len(batch) > 3 else None
        all_masks= [partner_masks, road_masks]
        with torch.no_grad():
            context, other_embeds, other_weights, *_  = bc_policy.get_context(obs, all_masks)
            pred_loss, _ = gmm_loss(bc_policy, context, expert_action)
            # pred_loss, _ = focal_loss(bc_policy, context, expert_action)
            loss = pred_loss
            pred_actions = bc_policy.get_action(context, deterministic=True)
            action_loss = torch.abs(pred_actions - expert_action)
            dx_std2_mask = expert_action[..., 0].abs() > 2 
            dy_std2_mask = expert_action[..., 1].abs() > 0.035 
            dyaw_std2_mask = expert_action[..., 2].abs() > 0.023
            dx_std2_mask = dx_std2_mask.cpu().numpy()
            dy_std2_mask = dy_std2_mask.cpu().numpy()
            dyaw_std2_mask = dyaw_std2_mask.cpu().numpy()
            action_loss = action_loss.cpu().numpy()
            dx_loss = action_loss[..., 0].mean()
            dy_loss = action_loss[..., 1].mean()
            dyaw_loss = action_loss[..., 2].mean()

            dx_std2_loss = action_loss[..., 0][dx_std2_mask].sum() if dx_std2_mask.sum() > 0 else 0
            dy_std2_loss = action_loss[..., 1][dy_std2_mask].sum() if dy_std2_mask.sum() > 0 else 0
            dyaw_std2_loss = action_loss[..., 2][dyaw_std2_mask].sum() if dyaw_std2_mask.sum() > 0 else 0
            # bsae loss
            dx_losses += dx_loss
            dy_losses += dy_loss
            dyaw_losses += dyaw_loss

            # std2 loss
            dx_std2_losses += dx_std2_loss
            dy_std2_losses += dy_std2_loss
            dyaw_std2_losses += dyaw_std2_loss

            # action values
            dx_mean = pred_actions[..., 0][dx_std2_mask].sum() if dx_std2_mask.sum() > 0 else 0
            dy_mean = pred_actions[..., 1][dy_std2_mask].sum() if dy_std2_mask.sum() > 0 else 0
            dyaw_mean = pred_actions[..., 2][dyaw_std2_mask].sum() if dyaw_std2_mask.sum() > 0 else 0

            dx_values += dx_mean
            dy_values += dy_mean
            dyaw_values += dyaw_mean

            dx_std2_count += dx_std2_mask.sum()
            dy_std2_count += dy_std2_mask.sum()
            dyaw_std2_count += dyaw_std2_mask.sum()

            losses += loss.mean().item()

    test_loss = losses / (i + 1) 
    dx_loss = dx_losses / (i + 1) 
    dy_loss = dy_losses / (i + 1) 
    dyaw_loss = dyaw_losses / (i + 1) 
    cvar95_losses = cvar95_losses / (i + 1)
    dx_loss = dx_losses / (i + 1) 
    dy_loss = dy_losses / (i + 1) 
    dyaw_loss = dyaw_losses / (i + 1) 

    dx_values = dx_values / dx_std2_count
    dy_values = dy_values / dy_std2_count
    dyaw_values = dyaw_values / dyaw_std2_count

    dx_std2_loss = dx_std2_losses / dx_std2_count
    dy_std2_loss = dy_std2_losses / dy_std2_count
    dyaw_std2_loss = dyaw_std2_losses / dyaw_std2_count
    return test_loss, dx_loss, dy_loss, dyaw_loss, dx_std2_loss, dy_std2_loss, dyaw_std2_loss, dx_values, dy_values, dyaw_values

def train(exp_config=None):
    env_config = EnvConfig()
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    if args.use_wandb:
        wandb.init()
        # Tag Update
        wandb_tags = list(wandb.run.tags)
        wandb_tags.append(current_time)
        wandb_tags.append('full_paper')
        wandb.run.tags = tuple(wandb_tags)
        # Config Update
        for key, value in vars(args).items():
            wandb.config[key] = value
        config = wandb.config
        wandb_dict = {}
        for k, v in dict(config).items():
            if isinstance(v, dict) and "listitems" in v and isinstance(v["listitems"], list):
                wandb_dict[k] = v['listitems']
            else:    
                wandb_dict[k] = v    
        exp_config = Box(wandb_dict)
        wandb.run.name = f"{exp_config.model_name}_{exp_config.seed}"
        wandb.run.save()
    exp_config.update(vars(args))
    set_seed(exp_config.seed)
    # scaler = torch.cuda.amp.GradScaler()
    # Initialize model and optimizer
    bc_policy = MODELS[exp_config.model_name](env_config, exp_config).to(exp_config.device)
    optimizer = AdamW(bc_policy.parameters(), lr=exp_config.lr, eps=0.0001)
    load_path = exp_config.get('load_path', None)
    if load_path and os.path.exists(load_path):
        ckpt = torch.load(load_path[:-4] + '_optim.pth', map_location=exp_config.device)
        bc_policy = torch.load(load_path)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        gradient_steps = ckpt.get('gradient_steps', 0)
        print(f"Loaded checkpoint from {load_path} with step {gradient_steps}")
    # Model Params wandb update
    trainable_params = sum(p.numel() for p in bc_policy.parameters() if p.requires_grad)
    print(f'Total params: {trainable_params}')
    if exp_config.use_wandb:
        wandb_tags = list(wandb.run.tags)
        wandb_tags.append(f"trainable_params_{trainable_params}")
        wandb.run.tags = tuple(wandb_tags)
    train_data_path = os.path.join(exp_config.base_path, exp_config.data_path)
    train_data_file = f"training_trajectory_{exp_config.num_scene}.npz"
    eval_data_path = os.path.join(exp_config.base_path, exp_config.data_path)
    eval_data_file =  f"validation_trajectory_2500.npz"
    expert_data_loader = get_dataloader(train_data_path, train_data_file, exp_config)
    eval_expert_data_loader = get_dataloader(eval_data_path, eval_data_file, exp_config,
                                            isshuffle=False)
    num_train_sample = len(expert_data_loader.dataset)
    best_loss = 9999999
    gradient_steps = 0
    model_path = f"{exp_config.model_path}/{exp_config.name}" if exp_config.use_wandb else exp_config.model_path
    model_path = os.path.join(exp_config.base_path, model_path)
    pbar = tqdm(total=exp_config.total_gradient_steps, desc="Gradient Steps", ncols=100)
    stop_training = False
    if not os.path.exists(model_path):
        os.makedirs(os.path.join(model_path, 'checkpoints'))
    while gradient_steps < exp_config.total_gradient_steps and not stop_training:
        bc_policy.train()
        train_losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        
        for n, batch in enumerate(expert_data_loader):
            if gradient_steps >= exp_config.total_gradient_steps:
                break
            if len(batch) == 7:
                obs, expert_action, partner_masks, road_masks, other_pos, aux_mask, data_idx = batch
                other_pos = other_pos.to(exp_config.device)
            elif len(batch) == 5:
                obs, expert_action, partner_masks, road_masks, data_idx = batch 
            other_pos = other_pos.to(exp_config.device) if len(batch) > 5 else None
            obs, expert_action = obs.to(exp_config.device), expert_action.to(exp_config.device)
            partner_masks = partner_masks.to(exp_config.device) if len(batch) > 3 else None
            road_masks = road_masks.to(exp_config.device) if len(batch) > 3 else None
            all_masks= [partner_masks, road_masks]
            context, other_embeds, other_weights, *_ = bc_policy.get_context(obs, all_masks)
            # l1 loss version

            pred_loss, _ = gmm_loss(bc_policy, context, expert_action)
            loss = pred_loss
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bc_policy.parameters(), exp_config.grad_norm)

            optimizer.step()
            gradient_steps += 1
            pbar.update(1)
            with torch.no_grad():
                pred_actions = bc_policy.get_action(context, deterministic=True)
                action_loss = torch.abs(pred_actions - expert_action).cpu().numpy()
                dx_loss = action_loss[..., 0].mean()
                dy_loss = action_loss[..., 1].mean()
                dyaw_loss = action_loss[..., 2].mean()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss

            train_losses += loss.item()
                
            # Evaluation loop
            if gradient_steps % exp_config.eval_freq == 0:
                bc_policy.eval()
                test_losses = evaluate(eval_expert_data_loader, exp_config, bc_policy, num_train_sample)
                test_loss, dx_loss, dy_loss, dyaw_loss, dx_std2_loss, dy_std2_loss, dyaw_std2_loss, dx_vals, dy_vals, dyaw_vals = test_losses
                if exp_config.use_wandb:
                    log_dict = {
                            "eval/loss": test_loss,
                            "eval/dx_loss": dx_loss,
                            "eval/dy_loss": dy_loss,
                            "eval/dyaw_loss": dyaw_loss,
                            "eval/dx_std2_loss": dx_std2_loss,
                            "eval/dy_std2_loss": dy_std2_loss,
                            "eval/dyaw_std2_loss": dyaw_std2_loss,
                            "eval/dx_values": dx_vals,
                            "eval/dy_values": dy_vals,
                            "eval/dyaw_values": dyaw_vals,
                        }

                    wandb.log(log_dict, step=gradient_steps)
                
                # model save
                save_dict = {
                    'optimizer_state_dict': optimizer.state_dict(),
                    'gradient_steps': gradient_steps,
                    'exp_config': dict(exp_config),
                }
                torch.save(bc_policy, f"{model_path}/checkpoints/{exp_config.model_name}_s{exp_config.seed}_{current_time}_step_{gradient_steps}.pth")
                torch.save(save_dict, f"{model_path}/checkpoints/{exp_config.model_name}_s{exp_config.seed}_{current_time}_step_{gradient_steps}_optim.pth")
                if test_loss < best_loss:
                    torch.save(save_dict, f"{model_path}/{exp_config.model_name}_s{exp_config.seed}_{current_time}_optim.pth")
                    torch.save(bc_policy, f"{model_path}/{exp_config.model_name}_s{exp_config.seed}_{current_time}.pth")
                    best_loss = test_loss
                    print(f'STEP {gradient_steps} gets BEST!')
                bc_policy.train()                
        if exp_config.use_wandb:
            log_dict = {   
                    "train/loss": train_losses / (n + 1),
                    "train/dx_loss": dx_losses / (n + 1),
                    "train/dy_loss": dy_losses / (n + 1),
                    "train/dyaw_loss": dyaw_losses / (n + 1),
                    # "train/max_grad_norm": max_norms / (n + 1),
                }
            wandb.log(log_dict, step=gradient_steps)
    wandb.finish()

if __name__ == "__main__":
    args = parse_args()
    with open('baselines/il/config/il.yaml', "r") as f:
        exp_config = Box(yaml.safe_load(f))
    if args.use_wandb:
        with open("baselines/il/sweep.yaml") as f:
            sweep_config = yaml.safe_load(f)
            sweep_params = sweep_config.setdefault("parameters", {})
            for k, v in exp_config.items():
                if k not in sweep_params:
                    sweep_params[k] = dict()
                    sweep_params[k]['value'] = v
            sweep_params['name'] = dict(value=sweep_config['name'])
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        
        if args.sweep_id is not None:
            wandb.agent(args.sweep_id, function=train, project=private_info['main_project'], entity=private_info['entity'])
        else:
            sweep_id = wandb.sweep(sweep_config, project=private_info['main_project'], entity=private_info['entity'])
            wandb.agent(sweep_id, function=train)
    else:
        train(exp_config)
