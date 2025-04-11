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

# GPUDrive
from baselines.il.config import *
from baselines.il.dataloader import ExpertDataset
from algorithms.il import MODELS, LOSS
from algorithms.il.utils import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    # ENVIRONMENT
    parser.add_argument('--action-type', '-at', type=str, default='continuous', choices=['discrete', 'multi_discrete', 'continuous'],)
    parser.add_argument('--device', '-d', type=str, default='cuda', choices=['cpu', 'cuda'],)
    parser.add_argument('--num-stack', '-s', type=int, default=5)
    
    # MODEL
    parser.add_argument('--model-path', '-mp', type=str, default='/results/model')
    parser.add_argument('--model-name', '-m', type=str, default='early_attn', choices=['bc', 'late_fusion', 'attention', 'early_attn',
                                                                                         'wayformer',
                                                                                         'aux_fusion', 'aux_attn'])
    parser.add_argument('--loss-name', '-l', type=str, default='gmm', choices=['l1', 'mse', 'twohot', 'nll', 'gmm', 'new_gmm'])
    
    # DATA
    parser.add_argument('--data-path', '-dp', type=str, default='/data/tom_v5/')
    parser.add_argument('--train-data-file', '-td', type=str, default='test_trajectory_200.npz')
    parser.add_argument('--eval-data-file', '-ed', type=str, default='test_trajectory_200.npz')
    parser.add_argument('--num-workers', '-nw', type=int, default=4)
    parser.add_argument('--prefetch-factor', '-pf', type=int, default=4)
    parser.add_argument('--pin-memory', '-pm', action='store_true')
    parser.add_argument('--rollout-len', '-rl', type=int, default=5)
    parser.add_argument('--pred-len', '-pl', type=int, default=1)
    parser.add_argument('--aux-future-step', '-afs', type=int, default=10)
    
    # EXPERIMENT
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-tom', '-ut', default=None, choices=[None, 'guide_weighted', 'no_guide_no_weighted',
                                                                   'no_guide_weighted', 'guide_no_weighted'])
    args = parser.parse_args()
    
    return args

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
    print(f'DATA {data_path} {data_file}')
    with np.load(os.path.join(data_path, data_file)) as npz:
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
        other_info = npz['other_info'] if ('other_info' in npz.keys() and config.use_tom) else None
    dataset = ExpertDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask, other_info,
        rollout_len=config.rollout_len, pred_len=config.pred_len, aux_future_step=config.aux_future_step
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

def get_tsne_data(data_path, data_file, config):
    with np.load(os.path.join(data_path, data_file)) as npz:
        tsne_obs = npz['obs'][:10, 2:7]
        tsne_data_mask = npz['partner_mask'][:10, 6]
        tsne_partner_mask = np.where(tsne_data_mask == 2, 1, 0).astype('bool')
        tsne_road_mask = npz['road_mask'][:10, 6]
    if config.use_wandb:
        raw_fig, tsne_indices = visualize_partner_obs_final(tsne_obs[0], tsne_data_mask[0])
        wandb.log({"embedding/relative_positions_plot": wandb.Image(raw_fig)}, step=0)
        plt.close(raw_fig)
    tsne_obs = torch.from_numpy(tsne_obs).to(config.device)
    tsne_partner_mask = torch.from_numpy(tsne_partner_mask).to(config.device)
    tsne_road_mask = torch.from_numpy(tsne_road_mask).to(config.device)
    
    return tsne_obs, tsne_partner_mask, tsne_road_mask, tsne_indices

def evaluate(eval_expert_data_loader, config, bc_policy):
    total_samples = 0
    losses = 0
    dx_losses = 0
    dy_losses = 0
    dyaw_losses = 0
    for i, batch in enumerate(eval_expert_data_loader):
        batch_size = batch[0].size(0)
        total_samples += batch_size
        if len(batch) == 9:
            obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info, aux_mask, _ = batch  
        elif len(batch) == 7:
            obs, expert_action, masks, ego_masks, partner_masks, road_masks, _ = batch  
        elif len(batch) == 4:
            obs, expert_action, masks, _ = batch
        else:
            obs, expert_action, _ = batch
        obs, expert_action = obs.to(config.device), expert_action.to(config.device)
        masks = masks.to(config.device) if len(batch) > 2 else None
        ego_masks = ego_masks.to(config.device) if len(batch) > 3 else None
        partner_masks = partner_masks.to(config.device) if len(batch) > 3 else None
        road_masks = road_masks.to(config.device) if len(batch) > 3 else None
        all_masks= [masks, ego_masks, partner_masks, road_masks]
        with torch.no_grad():
            context, all_ratio, *_ = bc_policy.get_context(obs, all_masks[1:])
            pred_loss, _ = LOSS[config.loss_name](bc_policy, context, expert_action, all_masks)
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
    test_loss = losses / (i + 1) 
    dx_loss = dx_losses / (i + 1) 
    dy_loss = dy_losses / (i + 1) 
    dyaw_loss = dyaw_losses / (i + 1) 
    return test_loss, dx_loss, dy_loss, dyaw_loss

def train():
    env_config = EnvConfig()
    net_config = NetworkConfig()
    head_config = HeadConfig()
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    if args.use_wandb:
        wandb.init()
        # Tag Update
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
    set_seed(config.seed)
    
    # Initialize model and optimizer
    bc_policy = MODELS[config.model_name](env_config, net_config, head_config, config.loss_name, config.num_stack,
                                          config.use_tom).to(config.device)
    optimizer = AdamW(bc_policy.parameters(), lr=config.lr, eps=0.0001)
    print(bc_policy)
    
    # Model Params wandb update
    trainable_params = sum(p.numel() for p in bc_policy.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in bc_policy.parameters() if not p.requires_grad)
    print(f'Total params: {trainable_params + non_trainable_params}')
    if config.use_wandb:
        wandb_tags = list(wandb.run.tags)
        wandb_tags.append(f"trainable_params_{trainable_params}")
        wandb_tags.append(f"non_trainable_params_{non_trainable_params}")
        wandb.run.tags = tuple(wandb_tags)
    nth_train_file_list = sorted([
        f for f in os.listdir(config.data_path) if (f.endswith(".npz") and "test" not in f)
    ])
    eval_expert_data_loader = get_dataloader(config.data_path, config.eval_data_file, config, isshuffle=False)
    tsne_obs, tsne_partner_mask, tsne_road_mask, tsne_indices = get_tsne_data(config.data_path, config.eval_data_file, config)
    
    num_train_sample = len(expert_data_loader.dataset)
    best_loss = 9999999
    early_stopping = 0
    gradient_steps = 0
    nth_train_idx = 0
    model_path = f"{config.model_path}/{exp_config['name']}" if config.use_wandb else config.model_path
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    pbar = tqdm(total=config.total_gradient_steps, desc="Gradient Steps", ncols=100)
    stop_training = False
    while gradient_steps < config.total_gradient_steps and not stop_training:
        bc_policy.train()
        losses = 0
        dx_losses = 0
        dy_losses = 0
        dyaw_losses = 0
        max_norms = 0
        max_names = []
        max_losses = []
        current_file = nth_train_file_list[nth_train_idx]
        expert_data_loader = get_dataloader(config.data_path, current_file, config)
        for _, batch in enumerate(expert_data_loader):
            if gradient_steps >= config.total_gradient_steps:
                break
            if len(batch) == 9:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info, aux_mask, data_idx = batch
            elif len(batch) == 7:
                obs, expert_action, masks, ego_masks, partner_masks, road_masks, data_idx = batch 
            elif len(batch) == 4:
                obs, expert_action, masks, data_idx = batch
            else:
                obs, expert_action, data_idx = batch
            
            obs, expert_action = obs.to(config.device), expert_action.to(config.device)
            masks = masks.to(config.device) if len(batch) > 2 else None
            ego_masks = ego_masks.to(config.device) if len(batch) > 3 else None
            partner_masks = partner_masks.to(config.device) if len(batch) > 3 else None
            road_masks = road_masks.to(config.device) if len(batch) > 3 else None
            all_masks= [masks, ego_masks, partner_masks, road_masks]

            context, all_ratio, *_ = bc_policy.get_context(obs, all_masks[1:])
            pred_loss, pred_loss_wandb = LOSS[config.loss_name](bc_policy, context, expert_action, all_masks)
            loss = pred_loss
            
            # To write data idx that has the highest loss
            if config.use_wandb:
                max_loss, max_loss_idx = torch.max(pred_loss_wandb, dim=-1)
                max_loss_idx = data_idx[max_loss_idx]
                max_losses.append((max_loss.item(), (max_loss_idx.detach().cpu().numpy())))
            
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
                component_probs = bc_policy.head.get_component_probs().cpu().numpy()
                action_loss = torch.abs(pred_actions - expert_action)
                dx_loss = action_loss[..., 0].mean().item()
                dy_loss = action_loss[..., 1].mean().item()
                dyaw_loss = action_loss[..., 2].mean().item()
                dx_losses += dx_loss
                dy_losses += dy_loss
                dyaw_losses += dyaw_loss
                
            losses += pred_loss.mean().item()
            if config.use_wandb and (gradient_steps % config.log_freq == 0):
                log_dict = {   
                        "train/loss": losses / config.log_freq,
                        "train/dx_loss": dx_losses / config.log_freq,
                        "train/dy_loss": dy_losses / config.log_freq ,
                        "train/dyaw_loss": dyaw_losses / config.log_freq ,
                        "gmm/max_grad_norm": max_norms / config.log_freq ,
                        "gmm/max_component_probs": max(component_probs),
                        "gmm/median_component_probs": np.median(component_probs),
                        "gmm/min_component_probs": min(component_probs),
                    }
                wandb.log(log_dict, step=gradient_steps)
                
                # make csv file for max loss
                model_path = f"{config.model_path}/{exp_config['name']}"
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                csv_path = f"{model_path}/max_loss({config.exp_name}).csv"
                
                # write csv file for max loss
                max_loss_value, best_max_loss_idx = max(max_losses, key=lambda x: x[0])
                file_is_empty = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
                with open(csv_path, 'a') as f:
                    if file_is_empty:
                        f.write("gradient_step, max_loss, data_idx\n")
                    f.write(f"{gradient_steps}, {max_loss_value}, {best_max_loss_idx[0]}, {best_max_loss_idx[1]}\n")
            
            # Evaluation loop
            if gradient_steps % config.eval_freq == 0:
                bc_policy.eval()
                test_loss, dx_loss, dy_loss, dyaw_loss = evaluate(eval_expert_data_loader, config, bc_policy, num_train_sample)
                if config.use_wandb:
                    with torch.no_grad():
                        others_tsne, other_distance, other_speed, other_weights = bc_policy.get_tsne(tsne_obs, tsne_partner_mask, tsne_road_mask)
                    fig1, _ = visualize_embedding(others_tsne, other_distance, other_speed, tsne_indices, tsne_partner_mask, tsne_partner_mask)
                    wandb.log({"embedding/tsne_subplots": wandb.Image(fig1)}, step=gradient_steps)
                    plt.close(fig1)
                    log_dict = {
                            "eval/loss": test_loss,
                            "eval/dx_loss": dx_loss,
                            "eval/dy_loss": dy_loss,
                            "eval/dyaw_loss": dyaw_loss,
                        }
                    wandb.log(log_dict, step=gradient_steps)
                if test_loss < best_loss:
                    torch.save(bc_policy, f"{model_path}/{config.model_name}_{config.exp_name}_{current_time}.pth")
                    best_loss = test_loss
                    early_stopping = 0
                    print(f'STEP {gradient_steps} gets BEST!')
                else:
                    early_stopping += 1
                    if early_stopping > config.early_stop_num + 1:
                        wandb.finish()
                        stop_training = True
                        break
        nth_train_idx = (nth_train_idx + 1) % len(nth_train_file_list)
if __name__ == "__main__":
    args = parse_args()
    if args.use_wandb:
        with open("baselines/il/sweep_many.yaml") as f:
            exp_config = yaml.load(f, Loader=yaml.FullLoader)
        with open("private.yaml") as f:
            private_info = yaml.load(f, Loader=yaml.FullLoader)
        wandb.login(key=private_info["wandb_key"])
        
        if args.sweep_id is not None:
            wandb.agent(args.sweep_id, function=train, project=private_info['main_project'], entity=private_info['entity'])
        else:
            sweep_id = wandb.sweep(exp_config, project=private_info['main_project'], entity=private_info['entity'])
            wandb.agent(sweep_id, function=train)
    else:
        train()
