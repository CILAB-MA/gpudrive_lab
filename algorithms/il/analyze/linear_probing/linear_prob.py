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
from algorithms.il.analyze.linear_probing.dataloader import OtherFutureDataset
from algorithms.il.analyze.linear_probing.config import ExperimentConfig
from algorithms.il.analyze.linear_probing.model import *

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
    parser.add_argument('--aux-future-step', '-afs', type=int, default=30)
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
    dataset = OtherFutureDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask, other_info,
        rollout_len=config.rollout_len, pred_len=config.pred_len, aux_future_step=config.aux_future_step
    )
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
    backbone = torch.load(f"{config.model_path}/{config.model_name}.pth", weights_only=False)
    backbone.eval()
    print(backbone)
    ro_attn_layers = register_all_layers_forward_hook(backbone.ro_attn)
    pos_linear_model = LinearProbPosition(128, 64).to("cuda")
    action_linear_model = LinearProbAction(128, 12).to("cuda")

    # Optimizer
    pos_optimizer = AdamW(pos_linear_model.parameters(), lr=config.lr, eps=0.0001)
    action_optimizer = AdamW(action_linear_model.parameters(), lr=config.lr, eps=0.0001)

    # DataLoaders
    expert_data_loader = get_dataloader(config.data_path, config.train_data, config)
    eval_expert_data_loader = get_dataloader(config.data_path, config.test_data, config, isshuffle=False)

    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="epoch"):
        pos_linear_model.train()
        action_linear_model.train()
        
        pos_losses = 0
        action_losses = 0
        for i, batch in enumerate(expert_data_loader):
            batch_size = batch[0].size(0)

            obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info, aux_mask, other_pos, other_actions = batch
            
            obs, expert_action = obs.to("cuda"), expert_action.to("cuda")
            other_pos, other_actions = other_pos.to("cuda"), other_actions.to("cuda")
            masks = masks.to("cuda") if len(batch) > 2 else None
            ego_masks = ego_masks.to("cuda") if len(batch) > 3 else None
            partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
            road_masks = road_masks.to("cuda") if len(batch) > 3 else None
            other_info = other_info.to("cuda").transpose(1, 2).reshape(batch_size, 127, -1) if len(batch) > 6 else None
            all_masks= [ego_masks, partner_masks, road_masks]

            try:
                context, *_, = backbone.get_context(obs, all_masks, other_info=other_info)
            except TypeError:
                context, *_, = backbone.get_context(obs, all_masks)

            # get partner pred pos and action
            pred_pos = pos_linear_model(ro_attn_layers['0'][:,1:,:])
            masked_pos = pred_pos[~aux_mask]
            pred_action = action_linear_model(ro_attn_layers['0'][:,1:,:])
            masked_action = pred_action[~aux_mask]

            # get partner expert pos and action
            other_pos = other_pos.clone()
            other_actions = other_actions.clone()
            masked_other_pos = other_pos[~aux_mask]
            masked_other_actions = other_actions[~aux_mask]
            
            # get loss
            pos_loss = pos_linear_model.loss(masked_pos, masked_other_pos)
            action_loss = action_linear_model.loss(masked_action, masked_other_actions)
            total_loss = pos_loss + action_loss
            
            pos_optimizer.zero_grad()
            action_optimizer.zero_grad()
            total_loss.backward()
            pos_optimizer.step()
            action_optimizer.step()
            pos_losses += pos_loss.item()
            action_losses += action_loss.item()

        if config.use_wandb:
            wandb.log(
                {
                    "train/pos_loss": pos_losses / (i + 1),
                    "train/action_loss": action_losses / (i + 1),
                }, step=epoch
            )
        
        # Evaluation loop
        if epoch % 2 == 0:
            pos_linear_model.eval()
            action_linear_model.eval()
            
            pos_losses = 0
            action_losses = 0
            total_samples = 0
            for i, batch in enumerate(eval_expert_data_loader):
                batch_size = batch[0].size(0)
                if total_samples + batch_size > int(config.sample_per_epoch / 5): 
                    break
                total_samples += batch_size
                obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info, aux_mask, other_pos, other_actions = batch
                obs, expert_action = obs.to("cuda"), expert_action.to("cuda")
                other_pos, other_actions = other_pos.to("cuda"), other_actions.to("cuda")
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

                    # get partner pred pos and action
                    pred_pos = pos_linear_model(ro_attn_layers['0'][:,1:,:])
                    pred_action = action_linear_model(ro_attn_layers['0'][:,1:,:])
                    masked_pos = pred_pos[~aux_mask]
                    masked_action = pred_action[~aux_mask]

                    # get partner expert pos and action
                    other_pos = other_pos.clone()
                    other_actions = other_actions.clone()
                    masked_other_pos = other_pos[~aux_mask]
                    masked_other_actions = other_actions[~aux_mask]
                    pos_loss = pos_linear_model.loss(masked_pos, masked_other_pos)
                    action_loss = action_linear_model.loss(masked_action, masked_other_actions)
                    
                    pos_losses += pos_loss.item()
                    action_losses += action_loss.item()

            if config.use_wandb:
                wandb.log(
                    {
                        "eval/pos_loss": pos_losses / (i + 1),
                        "eval/action_loss": action_losses / (i + 1) ,
                    }, step=epoch
                )
    
    # Save head
    os.makedirs(os.path.join(config.model_path, f"linear_prob/{config.model_name}"), exist_ok=True)
    torch.save(pos_linear_model, os.path.join(config.model_path, f"linear_prob/{config.model_name}/pos({current_time}).pth"))
    torch.save(action_linear_model, os.path.join(config.model_path, f"linear_prob/{config.model_name}/action({current_time}).pth"))

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
