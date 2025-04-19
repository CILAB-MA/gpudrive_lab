"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
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
from gpudrive.integrations.il.loss import gmm_loss, aux_loss
# from algorithms.il.utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MODELS = dict(early_attn=EarlyFusionAttnBCNet,)
def parse_args():
    parser = argparse.ArgumentParser("Most of vars are in il.yaml. These are for different server.")
    # DATALOADER
    parser.add_argument('--num-workers', '-nw', type=int, default=4)
    parser.add_argument('--prefetch-factor', '-pf', type=int, default=4)
    parser.add_argument('--pin-memory', '-pm', action='store_true')
    
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
    with np.load(os.path.join(data_path, data_file)) as npz:
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
        other_info = npz['other_info'] if ('other_info' in npz.keys() and config.use_tom) else None
    ego_global_pos = None
    ego_global_rot = None
    if config.use_tom:
        with np.load(os.path.join(data_path, "linear_probing", "global_" + data_file)) as global_npz:
            ego_global_pos = global_npz['ego_global_pos']
            ego_global_rot = global_npz['ego_global_rot']
    dataset = ExpertDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask, other_info,
        rollout_len=config.rollout_len, pred_len=config.pred_len, aux_future_step=config.aux_future_step,
        ego_global_pos=ego_global_pos, ego_global_rot=ego_global_rot
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=isshuffle,
        num_workers=config.num_workers,
        prefetch_factor=config.prefetch_factor,
        pin_memory=config.pin_memory
    )
    del dataset
    return dataloader

def evaluate(eval_expert_data_loader, config, bc_policy, num_train_sample):
    total_samples = 0
    losses = 0
    dx_losses = 0
    dy_losses = 0
    dyaw_losses = 0
    tom_losses = 0
    for i, batch in enumerate(eval_expert_data_loader):
        batch_size = batch[0].size(0)
        total_samples += batch_size
        if len(batch) == 7:
            obs, expert_action, partner_masks, road_masks, other_pos, aux_mask, data_idx = batch
            other_pos = other_pos.to(exp_config.device)
        elif len(batch) == 5:
            obs, expert_action, partner_masks, road_masks, data_idx = batch 
        elif len(batch) == 3:
            obs, expert_action, data_idx = batch
        obs, expert_action = obs.to(config.device), expert_action.to(config.device)
        partner_masks = partner_masks.to(config.device) if len(batch) > 3 else None
        road_masks = road_masks.to(config.device) if len(batch) > 3 else None
        all_masks= [partner_masks, road_masks]
        with torch.no_grad():
            context, other_embeds, other_weights, *_  = bc_policy.get_context(obs, all_masks)
            if config.use_tom:
                aux_info  = ['pos', other_weights.mean(axis=-1), config.use_tom]
                if 'no_guide_no_weighted' in config.use_tom:
                    other_input = other_embeds
                else:
                    other_input = other_embeds # todo: [..., aux_ind * 32: (aux_ind + 1) * 32]
                tom_loss = aux_loss(bc_policy, other_input, other_pos, aux_mask, 
                    aux_info=aux_info)
            pred_loss, _ = gmm_loss(bc_policy, context, expert_action)
            loss = pred_loss
            pred_actions = bc_policy.get_action(context, deterministic=True)
            action_loss = torch.abs(pred_actions - expert_action)
            dx_loss = action_loss[..., 0].mean().item()
            dy_loss = action_loss[..., 1].mean().item()
            dyaw_loss = action_loss[..., 2].mean().item()
            dx_losses += dx_loss
            dy_losses += dy_loss
            dyaw_losses += dyaw_loss
            losses += loss.mean().item()
            if config.use_tom:
                tom_losses += tom_loss.mean().item()
    test_loss = losses / (i + 1) 
    dx_loss = dx_losses / (i + 1) 
    dy_loss = dy_losses / (i + 1) 
    dyaw_loss = dyaw_losses / (i + 1) 
    tom_losses = tom_losses / (i + 1) 
    return test_loss, dx_loss, dy_loss, dyaw_loss, tom_losses

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
            if key not in wandb.config:
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
    # Initialize model and optimizer
    bc_policy = MODELS[exp_config.model_name](env_config, exp_config).to(exp_config.device)
    optimizer = AdamW(bc_policy.parameters(), lr=exp_config.lr, eps=0.0001)
    print(bc_policy)
    
    # Model Params wandb update
    trainable_params = sum(p.numel() for p in bc_policy.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in bc_policy.parameters() if not p.requires_grad)
    print(f'Total params: {trainable_params + non_trainable_params}')
    if exp_config.use_wandb:
        wandb_tags = list(wandb.run.tags)
        wandb_tags.append(f"trainable_params_{trainable_params}")
        wandb_tags.append(f"non_trainable_params_{non_trainable_params}")
        wandb.run.tags = tuple(wandb_tags)
    train_data_file = os.path.join(exp_config.base_path, exp_config.data_path, f"train_trajectory_{exp_config.num_scene}.npz")
    eval_data_file = os.path.join(exp_config.base_path, exp_config.data_path, f"test_trajectory_200.npz")
    expert_data_loader = get_dataloader(exp_config.data_path, train_data_file, exp_config)
    eval_expert_data_loader = get_dataloader(exp_config.data_path, eval_data_file, exp_config,
                                            isshuffle=False)
    
    num_train_sample = len(expert_data_loader.dataset)
    best_loss = 9999999
    early_stopping = 0
    gradient_steps = 0
    model_path = f"{exp_config.model_path}/{exp_config.name}" if exp_config.use_wandb else exp_config.model_path
    model_path = os.path.join(exp_config.base_path, model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    pbar = tqdm(total=exp_config.total_gradient_steps, desc="Gradient Steps", ncols=100)
    stop_training = False
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    while gradient_steps < exp_config.total_gradient_steps and not stop_training:
        bc_policy.train()
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        tom_losses = 0
        max_norms = 0
        max_names = []
        
        for n, batch in enumerate(expert_data_loader):
            if gradient_steps >= exp_config.total_gradient_steps:
                break
            if len(batch) == 7:
                obs, expert_action, partner_masks, road_masks, other_pos, aux_mask, data_idx = batch
                other_pos = other_pos.to(exp_config.device)
            elif len(batch) == 5:
                obs, expert_action, partner_masks, road_masks, data_idx = batch 
            elif len(batch) == 3:
                obs, expert_action, data_idx = batch
            other_pos = other_pos.to(exp_config.device) if len(batch) > 5 else None
            obs, expert_action = obs.to(exp_config.device), expert_action.to(exp_config.device)
            partner_masks = partner_masks.to(exp_config.device) if len(batch) > 3 else None
            road_masks = road_masks.to(exp_config.device) if len(batch) > 3 else None
            all_masks= [partner_masks, road_masks]
            context, other_embeds, other_weights, *_ = bc_policy.get_context(obs, all_masks)
            pred_loss, _ = gmm_loss(bc_policy, context, expert_action)
            loss = pred_loss
            
            # To write data idx that has the highest loss
                
            if exp_config.use_tom:
                aux_info  = ['pos', other_weights.mean(axis=-1), config.use_tom]
                if 'no_guide_no_weighted' in exp_config.use_tom:
                    other_input = other_embeds
                else:
                    other_input = other_embeds# todo: [..., aux_ind * 32: (aux_ind + 1) * 32]
                tom_loss = aux_loss(bc_policy, other_input, other_pos, aux_mask, 
                    aux_info=aux_info)
                loss += 0.2 * tom_loss
            loss = loss.mean()
            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(bc_policy.parameters(), 20)
            max_norm, max_name = get_grad_norm(bc_policy.named_parameters())
            max_norms += max_norm
            max_names.append(max_name)

            optimizer.step()
            gradient_steps += 1
            pbar.update(1)
            with torch.no_grad():
                pred_actions = bc_policy.get_action(context, deterministic=True)
                # component_probs = bc_policy.head.get_component_probs().cpu().numpy() # gmm record part is now deactivated
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                if exp_config.use_tom:
                    tom_losses += tom_loss.mean().item()
            losses += pred_loss.mean().item()
                
            # Evaluation loop
            if gradient_steps % exp_config.eval_freq == 0:
                bc_policy.eval()
                test_loss, dx_loss, dy_loss, dyaw_loss, tom_loss = evaluate(eval_expert_data_loader, exp_config, bc_policy, num_train_sample)
                if exp_config.use_wandb:
                    log_dict = {
                            "eval/loss": test_loss,
                            "eval/dx_loss": dx_loss,
                            "eval/dy_loss": dy_loss,
                            "eval/dyaw_loss": dyaw_loss,
                            "eval/tom_loss": tom_loss,
                        }
                    wandb.log(log_dict, step=gradient_steps)
                if test_loss < best_loss:
                    torch.save(bc_policy, f"{model_path}/{exp_config.model_name}_seed_{exp_config.seed}_{current_time}.pth")
                    best_loss = test_loss
                    early_stopping = 0
                    print(f'STEP {gradient_steps} gets BEST!')
                else:
                    early_stopping += 1
                    if early_stopping > exp_config.early_stop_num + 1:
                        wandb.finish()
                        stop_training = True
                        break

        if exp_config.use_wandb:
            log_dict = {   
                    "train/loss": losses / (n + 1),
                    "train/dx_loss": dx_losses / (n + 1),
                    "train/dy_loss": dy_losses / (n + 1),
                    "train/dyaw_loss": dyaw_losses / (n + 1),
                    "train/tom_loss": tom_losses / (n + 1),
                    "train/max_grad_norm": max_norms / (n + 1),
                    # "gmm/max_component_probs": max(component_probs),
                    # "gmm/median_component_probs": np.median(component_probs),
                    # "gmm/min_component_probs": min(component_probs),
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
