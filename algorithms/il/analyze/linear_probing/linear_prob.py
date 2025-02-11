"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os, sys, torch
torch.backends.cudnn.benchmark = True
sys.path.append(os.getcwd())
import wandb, yaml, argparse, functools
from tqdm import tqdm
from datetime import datetime
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
# GPUDrive
from algorithms.il.analyze.linear_probing.dataloader import ExpertDataset
from algorithms.il.analyze.linear_probing.config import ExperimentConfig
from algorithms.il.analyze.linear_probing.model import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def compute_correlation_scatter(dist, coll, y):
    data = np.vstack([dist, coll, y])
    corr_matrix = np.corrcoef(data)
    df_corr = pd.DataFrame(corr_matrix, index=['x1', 'x2', 'y'], columns=['x1', 'x2', 'y'])

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(dist, coll, c=y, cmap='viridis', edgecolor='k', alpha=0.75)
    plt.colorbar(scatter, label="Loss Value")
    ax.set_xlabel("Current Distance")
    ax.set_ylabel("Collision Risk")
    ax.set_title("Evaluation of Collision Risk")
    ax.grid(True)

    return df_corr, fig

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--use-tom', '-ut', default=None, choices=[None, 'oracle', 'aux_head'])
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

def get_dataloader(data_path, data_file, config, isshuffle=True):
    with np.load(os.path.join(data_path, data_file)) as npz:
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
        other_info = npz['other_info'] if 'other_info' in npz.keys() else None    

    dataloader = DataLoader(
        ExpertDataset(
            expert_obs, expert_actions, expert_masks, partner_mask, road_mask, other_info,
            rollout_len=config.rollout_len, pred_len=config.pred_len
        ),
        batch_size=config.batch_size,
        shuffle=isshuffle,
        num_workers=os.cpu_count(),
        prefetch_factor=4,
        pin_memory=True
    )

    return dataloader

def register_all_layers_forward_hook(model):
    hidden_vector_dict = OrderedDict()

    def hook_fn(module, input, output, name):
        try:
            hidden_vector_dict[name] = output.detach()
        except AttributeError:
            hidden_vector_dict[name] = output['last_hidden_state'].detach()

    def _register(module, prefix=""):
        for name, layer in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            layer.register_forward_hook(functools.partial(hook_fn, name=full_name))

            _register(layer, full_name)

    _register(model)

    return hidden_vector_dict

def train():
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
        wandb.run.name = f"{config.model_name}(linear_prob)"
        wandb.run.save()
    else:
        config = ExperimentConfig()
        config.__dict__.update(vars(args))
    set_seed(config.seed)
    
    # Backbone and heads
    backbone = torch.load(f"{config.model_path}/{config.model_name}.pth")
    backbone.eval()
    print(backbone)
    hidden_vector_dict = register_all_layers_forward_hook(backbone)
    linear_model_action = LinearProbAction(backbone.head.input_layer[0].in_features, 127).to("cuda")
    # linear_model_pos = LinearProbPosition(backbone.head.input_layer[0].in_features, 127).to("cuda")
    # linear_model_angle = LinearProbAngle(backbone.head.input_layer[0].in_features, 127).to("cuda")
    # linear_model_speed = LinearProbSpeed(backbone.head.input_layer[0].in_features, 127).to("cuda")
    
    # Optimizer
    action_optimizer = AdamW(linear_model_action.parameters(), lr=config.lr, eps=0.0001)
    # pos_optimizer = AdamW(linear_model_pos.parameters(), lr=config.lr, eps=0.0001)
    # angle_optimizer = AdamW(linear_model_angle.parameters(), lr=config.lr, eps=0.0001)
    # speed_optimizer = AdamW(linear_model_speed.parameters(), lr=config.lr, eps=0.0001)
    
    # DataLoaders
    expert_data_loader = get_dataloader(config.data_path, config.train_data, config)
    eval_expert_data_loader = get_dataloader(config.data_path, config.test_data, config, isshuffle=False)

    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="epoch"):
        linear_model_action.train()
        # linear_model_pos.train()
        # linear_model_angle.train()
        # linear_model_speed.train()
        
        action_losses = 0
        # pos_losses = 0
        # angle_losses = 0
        # speed_losses = 0
        
        for i, batch in enumerate(expert_data_loader):
            batch_size = batch[0].size(0)
            if len(batch) == 8:
                obs, collision_risk, expert_action, masks, ego_masks, partner_masks, road_masks, other_info = batch
            elif len(batch) == 7:
                obs, collision_risk, expert_action, masks, ego_masks, partner_masks, road_masks = batch 
            elif len(batch) == 4:
                obs, collision_risk, expert_action, masks = batch
            else:
                obs, expert_action = batch
            
            obs, expert_action = obs.to("cuda"), expert_action.to("cuda")
            masks = masks.to("cuda") if len(batch) > 2 else None
            ego_masks = ego_masks.to("cuda") if len(batch) > 3 else None
            partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
            road_masks = road_masks.to("cuda") if len(batch) > 3 else None
            other_info = other_info.to("cuda").transpose(1, 2).reshape(batch_size, 127, -1) if len(batch) > 6 else None
            all_masks= [ego_masks, partner_masks, road_masks]
            collision_risk = collision_risk.to("cuda")
            try:
                context, *_, = backbone.get_context(obs, all_masks, other_info=other_info)
            except TypeError:
                context, *_, = backbone.get_context(obs, all_masks)
                
            pred_action = linear_model_action(context)
            masked_action = pred_action[~partner_masks[:, -1]]
            maksed_collision_risk = collision_risk[~partner_masks[:, -1]]
            other_actions = other_info[..., 4:7]
            other_actions = other_actions.clone()
            dyaw_actions = other_actions[:, :, 2] / np.pi
            dxy_actions = other_actions[:, :, :2] / 6
            other_actions = torch.cat([dxy_actions, dyaw_actions.unsqueeze(-1)], dim=-1)
            masked_other_actions = other_actions[~partner_masks[:, -1]]
            # pred_pos = linear_model_pos(context)
            # pred_angle = linear_model_angle(context)
            # pred_speed = linear_model_speed(context)
            action_loss = linear_model_action.loss(masked_action, masked_other_actions)
            # pos_loss = linear_model_pos.loss(pred_pos, other_info[..., 1:3])
            # angle_loss = linear_model_angle.loss(pred_angle, other_info[..., 3])
            # speed_loss = linear_model_speed.loss(pred_speed, other_info[..., 0])
            
            total_loss = action_loss # + pos_loss + angle_loss + speed_loss
            
            action_optimizer.zero_grad()
            # pos_optimizer.zero_grad()
            # angle_optimizer.zero_grad()
            # speed_optimizer.zero_grad()
            
            total_loss.mean().backward()
            
            action_optimizer.step()
            # pos_optimizer.step()
            # angle_optimizer.step()
            # speed_optimizer.step()
            action_losses += action_loss.mean().item()
            # pos_losses += pos_loss.item()
            # angle_losses += angle_loss.item()
            # speed_losses += speed_loss.item()
        
        if config.use_wandb:
            wandb.log(
                {   
                    "train/action_loss": action_losses / (i + 1),
                    # "train/pos_loss": pos_losses / (i + 1),
                    # "train/angle_loss": angle_losses / (i + 1),
                    # "train/speed_loss": speed_losses / (i + 1),
                }, step=epoch
            )
        
        # Evaluation loop
        if epoch % 5 == 0:
            linear_model_action.eval()
            # linear_model_pos.eval()
            # linear_model_angle.eval()
            # linear_model_speed.eval()
            
            action_losses = 0
            # pos_losses = 0
            # angle_losses = 0
            # speed_losses = 0
            
            total_samples = 0
            for i, batch in enumerate(eval_expert_data_loader):
                batch_size = batch[0].size(0)
                if total_samples + batch_size > int(config.sample_per_epoch / 5): 
                    break
                total_samples += batch_size
                if len(batch) == 8:
                    obs, collision_risk, expert_action, masks, ego_masks, partner_masks, road_masks, other_info = batch
                elif len(batch) == 7:
                    obs, collision_risk, expert_action, masks, ego_masks, partner_masks, road_masks = batch 
                elif len(batch) == 4:
                    obs, collision_risk, expert_action, masks = batch
                else:
                    obs, expert_action = batch
                collision_risk = collision_risk.to("cuda")
                obs, expert_action = obs.to("cuda"), expert_action.to("cuda")
                masks = masks.to("cuda") if len(batch) > 2 else None
                ego_masks = ego_masks.to("cuda") if len(batch) > 3 else None
                partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
                road_masks = road_masks.to("cuda") if len(batch) > 3 else None
                other_info = other_info.to("cuda").transpose(1, 2).reshape(batch_size, 127, -1) if len(batch) > 6 else None
                all_masks= [ego_masks, partner_masks, road_masks]
                
                with torch.no_grad():
                    try:
                        context, *_, = backbone.get_context(obs, all_masks, other_info=other_info)
                    except TypeError:
                        context, *_, = backbone.get_context(obs, all_masks)

                    pred_action = linear_model_action(context)
                    # pred_pos = linear_model_pos(context)
                    # pred_angle = linear_model_angle(context)
                    # pred_speed = linear_model_speed(context)
                    masked_action = pred_action[~partner_masks[:, -1]]
                    maksed_collision_risk = collision_risk[~partner_masks[:, -1]]
                    other_actions = other_info[..., 4:7]  
                    dyaw_actions = other_actions[:, :, 2] / np.pi
                    dxy_actions = other_actions[:, :, :2] / 6
                    other_actions = torch.cat([dxy_actions, dyaw_actions.unsqueeze(-1)], dim=-1)
                    masked_other_actions = other_actions[~partner_masks[:, -1]]
                    action_loss = linear_model_action.loss(masked_action, masked_other_actions)
                    # pos_loss = linear_model_pos.loss(pred_pos, other_info[..., 1:3])
                    # angle_loss = linear_model_angle.loss(pred_angle, other_info[..., 3])
                    # speed_loss = linear_model_speed.loss(pred_speed, other_info[..., 0])

                    action_losses += action_loss.mean().item()
                    action_corr = action_loss.detach().mean(-1).cpu().numpy()
                    maksed_collision_risk = maksed_collision_risk.detach().cpu().numpy()
                    # pos_losses += pos_loss.item()
                    # angle_losses += angle_loss.item()
                    # speed_losses += speed_loss.item()
            corr, fig = compute_correlation_scatter(maksed_collision_risk[:500, 0], 
                            maksed_collision_risk[:500, 1],
                            action_corr[:500])
            print(corr)
            if config.use_wandb:
                wandb.log(
                    {
                        "eval/action_loss": action_losses / (i + 1) ,
                        # "eval/pos_loss": pos_losses / (i + 1),
                        # "eval/angle_loss": angle_losses / (i + 1),
                        # "eval/speed_loss": speed_losses / (i + 1),
                        "eval/loss_dist":wandb.Image(fig),
                        "eval/corr_table":wandb.Table(dataframe=corr),
                    }, step=epoch
                )
    
    # Save head
    os.makedirs(os.path.join(config.model_path, f"linear_prob/{config.model_name}"), exist_ok=True)
    torch.save(linear_model_action, os.path.join(config.model_path, f"linear_prob/{config.model_name}/action.pth"))
    # torch.save(linear_model_pos, os.path.join(config.model_path, f"linear_prob/{config.model_name}/pos.pth"))
    # torch.save(linear_model_angle, os.path.join(config.model_path, f"linear_prob/{config.model_name}/angle.pth"))
    # torch.save(linear_model_speed, os.path.join(config.model_path, f"linear_prob/{config.model_name}/speed.pth"))

if __name__ == "__main__":
    args = parse_args()
    if args.use_wandb:
        with open("algorithms/il/analyze/linear_probing/sweep.yaml") as f:
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
