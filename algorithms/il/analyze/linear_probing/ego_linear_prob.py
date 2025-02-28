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
# GPUDrive
from algorithms.il.analyze.linear_probing.dataloader import EgoFutureDataset
from algorithms.il.analyze.linear_probing.config import ExperimentConfig
from algorithms.il.analyze.linear_probing.model import *
from algorithms.il.utils import compute_correlation_scatter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--use-tom', '-ut', default=None, choices=[None, 'guide_weighted', 'no_guide_no_weighted',
                                                                   'no_guide_weighted', 'guide_no_weighted'])
    parser.add_argument('--ego-future-step', '-afs', type=int, default=30)
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
    dataset = EgoFutureDataset(expert_obs, expert_actions, expert_masks, partner_mask, road_mask, other_info,
        rollout_len=config.rollout_len, pred_len=config.pred_len, ego_future_step=config.ego_future_step)
    dataloader = DataLoader(
        dataset,
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
    # hidden_vector_dict = register_all_layers_forward_hook(backbone)
    linear_model_action = LinearProbAction(backbone.head.input_layer[0].in_features, 1).to("cuda")

    # Optimizer
    action_optimizer = AdamW(linear_model_action.parameters(), lr=config.lr, eps=0.0001)

    # DataLoaders
    expert_data_loader = get_dataloader(config.data_path, config.train_data, config)
    eval_expert_data_loader = get_dataloader(config.data_path, config.test_data, config, isshuffle=False)

    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="epoch"):
        linear_model_action.train()
        action_losses = 0
        for i, batch in enumerate(expert_data_loader):
            batch_size = batch[0].size(0)
            obs, future_actions, masks, ego_masks, partner_masks, road_masks = batch
            
            obs, future_actions = obs.to("cuda"), future_actions.to("cuda")
            masks = masks.to("cuda") if len(batch) > 2 else None
            ego_masks = ego_masks.to("cuda") if len(batch) > 3 else None
            partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
            road_masks = road_masks.to("cuda") if len(batch) > 3 else None
            all_masks= [ego_masks, partner_masks, road_masks]
            context, *_, = backbone.get_context(obs, all_masks)
                
            pred_action = linear_model_action(context)
            pred_action = pred_action.squeeze(1)
            masked_action = pred_action[ego_masks[:, -1]]
            future_actions = future_actions.clone()
            dyaw_actions = future_actions[:, :, 2] / np.pi
            dxy_actions = future_actions[:, :, :2] / 6
            future_actions = torch.cat([dxy_actions, dyaw_actions.unsqueeze(-1)], dim=-1)
            masked_other_actions = future_actions[ego_masks[:, -1]]
            action_loss = linear_model_action.loss(masked_action, masked_other_actions)
            
            total_loss = action_loss 
            
            action_optimizer.zero_grad()
            
            total_loss.mean().backward()
            
            action_optimizer.step()
            action_losses += action_loss.mean().item()
        
        if config.use_wandb:
            wandb.log(
                {   
                    "train/action_loss": action_losses / (i + 1),
                }, step=epoch
            )
        
        # Evaluation loop
        if epoch % 2 == 0:
            linear_model_action.eval()
            action_losses = 0
            
            total_samples = 0
            for i, batch in enumerate(eval_expert_data_loader):
                batch_size = batch[0].size(0)
                if total_samples + batch_size > int(config.sample_per_epoch / 5): 
                    break
                obs, future_actions = obs.to("cuda"), future_actions.to("cuda")
                masks = masks.to("cuda") if len(batch) > 2 else None
                ego_masks = ego_masks.to("cuda") if len(batch) > 3 else None
                partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
                road_masks = road_masks.to("cuda") if len(batch) > 3 else None
                all_masks= [ego_masks, partner_masks, road_masks]
                
                with torch.no_grad():
                    context, *_, = backbone.get_context(obs, all_masks)
                    pred_action = linear_model_action(context)
                    pred_action = pred_action.squeeze(1)
                    masked_action = pred_action[ego_masks[:, -1]]
                    future_actions = future_actions.clone()
                    dyaw_actions = future_actions[:, :, 2] / np.pi
                    dxy_actions = future_actions[:, :, :2] / 6
                    future_actions = torch.cat([dxy_actions, dyaw_actions.unsqueeze(-1)], dim=-1)
                    masked_other_actions = future_actions[ego_masks[:, -1]]
                    action_loss = linear_model_action.loss(masked_action, masked_other_actions)

                    action_losses += action_loss.mean().item()
                    action_corr = action_loss.detach().mean(-1).cpu().numpy()
                    action_corr = np.clip(action_corr, 0, 0.005)
            if config.use_wandb:
                wandb.log(
                    {
                        "eval/action_loss": action_losses / (i + 1) ,
                    }, step=epoch
                )
    
    # Save head
    os.makedirs(os.path.join(config.model_path, f"linear_prob/{config.model_name}"), exist_ok=True)
    torch.save(linear_model_action, os.path.join(config.model_path, f"linear_prob/{config.model_name}/action({current_time}).pth"))


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
