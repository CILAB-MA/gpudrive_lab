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
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-mask', action='store_true')
    parser.add_argument('--use-tom', '-ut', default=None, choices=[None, 'guide_weighted', 'no_guide_no_weighted',
                                                                   'no_guide_weighted', 'guide_no_weighted'])
    parser.add_argument('--aux-future-step', '-afs', type=int, default=30)
    parser.add_argument('--model', '-m', default='baseline', choices=['baseline', 'early_lp', 'final_lp'])
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
    with np.load(os.path.join(data_path, "linear_probing", "global_" + data_file)) as global_npz:
        ego_global_pos = global_npz['ego_global_pos']
        ego_global_rot = global_npz['ego_global_rot']
    dataset = OtherFutureDataset(
        expert_obs, expert_actions, ego_global_pos, ego_global_rot, expert_masks, partner_mask, road_mask, other_info,
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
        current_time = datetime.now().strftime("%m%d_%H%M%S")
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
    if config.model == 'baseline':
        hidden_dim = 50 + 30 # partner info + ego info
        backbone = None
    else:
        backbone = torch.load(f"{config.model_path}/{config.model_name}.pth", weights_only=False)
        backbone.eval()
        if config.model == 'early_lp':
            layers = register_all_layers_forward_hook(backbone.fusion_attn)
            nth_layer = '2'
        else:
            layers = register_all_layers_forward_hook(backbone.ro_attn)
            nth_layer = '1'
        hidden_dim = 128

    pos_linear_model = LinearProbPosition(hidden_dim, 64, future_step=config.aux_future_step).to("cuda")
    # head_linear_model = LinearProbAngle(hidden_dim, 64, future_step=config.aux_future_step).to("cuda")
    
    # Optimizer
    pos_optimizer = AdamW(pos_linear_model.parameters(), lr=config.lr, eps=0.0001)
    # head_optimizer = AdamW(pos_linear_model.parameters(), lr=config.lr, eps=0.0001)
    
    # DataLoaders
    expert_data_loader = get_dataloader(config.data_path, config.train_data, config)
    eval_expert_data_loader = get_dataloader(config.data_path, config.test_data, config, isshuffle=False)
    
    for epoch in tqdm(range(config.epochs), desc="Epochs", unit="epoch"):
        pos_linear_model.train()
        
        pos_accuracys = 0
        heading_accuracys = 0
        pos_losses = 0
        heading_losses = 0
        pos_f1_macros = 0
        heading_f1_macros = 0
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
            if config.model == 'baseline':
                B, T, _ = obs.shape
                ego_obs = obs[..., :6].unsqueeze(2).repeat(1, 1, 127, 1)
                partner_obs = obs[..., 6:1276].reshape(B, T, 127, 10)
                lp_input = torch.cat([ego_obs, partner_obs], dim=-1).permute(0, 2, 1, 3).reshape(B, 127, -1)
            else:
                try:
                    backbone.get_context(obs, all_masks, other_info=other_info)
                except TypeError:
                    backbone.get_context(obs, all_masks)
                if config.model == 'early_lp':
                    lp_input = layers[nth_layer][:,1:128,:]
                else:
                    lp_input = layers[nth_layer][:,1:,:]
            # get partner pred pos and action
            pred_pos = pos_linear_model(lp_input)
            masked_pos = pred_pos[~aux_mask]
            # pred_heading = head_linear_model(lp_input)
            # masked_heading = pred_heading[~aux_mask]

            # get partner expert pos and action
            other_pos = other_pos.clone()
            other_heading = other_heading.clone()
            masked_other_pos = other_pos[~aux_mask]
            masked_other_heading = other_heading[~aux_mask]
            
            # get loss
            pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_other_pos)
            # heading_loss, heading_acc, heading_class = head_linear_model.loss(masked_heading, masked_other_heading)
            total_loss = pos_loss# + heading_loss
            
            pos_optimizer.zero_grad()
            # head_optimizer.zero_grad()
            total_loss.backward()
            pos_optimizer.step()
            # head_optimizer.step()

            # get F1 scores
            pos_class = pos_class.detach().cpu().numpy()
            # heading_class = heading_class.detach().cpu().numpy()
            masked_other_pos = masked_other_pos.detach().cpu().numpy()
            # masked_other_heading = masked_other_heading.detach().cpu().numpy()
            pos_f1_macro = f1_score(pos_class, masked_other_pos, average='macro')
            # heading_f1_macro = f1_score(heading_class, masked_other_heading, average='macro')

            pos_accuracys += pos_acc
            # heading_accuracys += heading_acc
            pos_losses += pos_loss.item()
            # heading_losses += heading_loss.item()
            # heading_f1_macros += heading_f1_macro
            pos_f1_macros += pos_f1_macro

        if config.use_wandb:
            wandb.log(
                {
                    "train/pos_accuracy": pos_accuracys / (i + 1),
                    # "train/heading_accuracy": heading_accuracys / (i + 1),
                    "train/pos_loss": pos_losses / (i + 1),
                    # "train/heading_loss": heading_losses / (i + 1),
                    "train/pos_f1_macro": pos_f1_macros / (i + 1),
                    # "train/heading_f1_macro": heading_f1_macros / (i + 1)
                }, step=epoch
            )

        # Evaluation loop
        if epoch % 2 == 0:
            pos_linear_model.eval()
            # head_linear_model.eval()
            
            pos_accuracys = 0
            heading_accuracys = 0
            pos_losses = 0
            total_samples = 0
            heading_losses = 0
            pos_f1_macros = 0
            heading_f1_macros = 0
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
                    if config.baseline:
                        B, T, _ = obs.shape
                        ego_obs = obs[..., :6].unsqueeze(2).repeat(1, 1, 127, 1)
                        partner_obs = obs[..., 6:1276].reshape(B, T, 127, 10)
                        lp_input = torch.cat([ego_obs, partner_obs], dim=-1).permute(0, 2, 1, 3).reshape(B, 127, -1)

                    else:
                        try:
                            backbone.get_context(obs, all_masks, other_info=other_info)
                        except TypeError:
                            backbone.get_context(obs, all_masks)
                        lp_input = layers[nth_layer][:,1:,:]

                    # get partner pred pos and action
                    pred_pos = pos_linear_model(lp_input)
                    masked_pos = pred_pos[~aux_mask]
                    # pred_heading = head_linear_model(lp_input)
                    # masked_heading = pred_heading[~aux_mask]

                    # get partner expert pos and action
                    other_pos = other_pos.clone()
                    # other_heading = other_heading.clone()
                    masked_other_pos = other_pos[~aux_mask]
                    # masked_other_heading = other_heading[~aux_mask]

                    pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_other_pos)
                    # heading_loss, heading_acc, heading_class = head_linear_model.loss(masked_heading, masked_other_heading)
                    
                    # get F1 scores
                    pos_class = pos_class.detach().cpu().numpy()
                    heading_class = heading_class.detach().cpu().numpy()
                    masked_other_pos = masked_other_pos.detach().cpu().numpy()
                    masked_other_heading = masked_other_heading.detach().cpu().numpy()
                    pos_f1_macro = f1_score(pos_class, masked_other_pos, average='macro')
                    heading_f1_macro = f1_score(heading_class, masked_other_heading, average='macro')

                    pos_accuracys += pos_acc
                    # heading_accuracys += heading_acc
                    pos_losses += pos_loss.item()
                    # heading_losses += heading_loss.item()
                    heading_f1_macros += heading_f1_macro
                    pos_f1_macros += pos_f1_macro

            if config.use_wandb:
                wandb.log(
                    {
                        "eval/pos_accuracy": pos_accuracys / (i + 1),
                        # "eval/heading_accuracy": heading_accuracys / (i + 1),
                        "eval/pos_loss": pos_losses / (i + 1),
                        # "eval/heading_loss": heading_losses / (i + 1),
                        "eval/pos_f1_macro": pos_f1_macros / (i + 1),
                        # "eval/heading_f1_macro": heading_f1_macros / (i + 1)
                    }, step=epoch
                )
    # Save head
    os.makedirs(os.path.join(config.model_path, f"linear_prob/seed{config.seed}v2"), exist_ok=True)
    torch.save(pos_linear_model, os.path.join(config.model_path, f"linear_prob/seed{config.seed}v2/pos_{config.model}_{config.aux_future_step}.pth"))
    # torch.save(head_linear_model, os.path.join(config.model_path, f"linear_prob/seed{config.seed}v2/action_{config.model}_{config.aux_future_step}.pth"))
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
