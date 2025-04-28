"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
import torch.nn.functional as F
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
from gpudrive.integrations.il.linear_probing.dataloader import FutureDataset
from gpudrive.integrations.il.linear_probing.lp_model import *
from sklearn.metrics import f1_score
from box import Box
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--num-workers', '-nw', type=int, default=8)
    parser.add_argument('--exp', '-e', type=str, default='other', choices=['other', 'ego'])
    parser.add_argument('--future-step', '-fs', type=int, default=10)
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
    ego_global_pos = None
    ego_global_rot = None
    with np.load(os.path.join(data_path, "global_" + data_file)) as global_npz:
        ego_global_pos = global_npz['ego_global_pos']
        ego_global_rot = global_npz['ego_global_rot']
    dataset = FutureDataset(
        expert_obs, ego_global_pos, ego_global_rot, expert_masks, partner_mask, road_mask,
        rollout_len=config.rollout_len, pred_len=config.pred_len, future_step=config.future_step,
        exp=config.exp
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

def train(exp_config=None):
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
    # Backbone and heads
    if exp_config.model == 'baseline':
        hidden_dim = 30 if args.exp == 'ego' else 60 # ego info
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
        
    pos_linear_model = LinearProbPosition(hidden_dim, 64, future_step=exp_config.future_step).to("cuda")
    train_data_path = os.path.join(exp_config.base_path, exp_config.data_path)
    train_data_file = f"training_trajectory_{exp_config.num_scene}.npz"
    eval_data_path = os.path.join(exp_config.base_path, exp_config.data_path)
    eval_data_file =  f"validation_trajectory_2500.npz"
    # Optimizer
    pos_optimizer = AdamW(pos_linear_model.parameters(), lr=exp_config.lr, eps=0.0001)

    # DataLoaders
    expert_data_loader = get_dataloader(train_data_path, train_data_file, exp_config)
    eval_expert_data_loader = get_dataloader(eval_data_path, eval_data_file, exp_config,
                                            isshuffle=False)
    pbar = tqdm(total=exp_config.total_gradient_steps, desc="Gradient Steps", ncols=100)
    gradient_steps = 0
    best_loss = 9999999
    while gradient_steps < exp_config.total_gradient_steps:
        pos_linear_model.train()

        pos_accuracys = 0
        heading_accuracys = 0
        pos_losses = 0
        heading_losses = 0
        pos_f1_macros = 0
        heading_f1_macros = 0
        continue_num = 0
        
        for i, batch in enumerate(expert_data_loader):
            if gradient_steps >= exp_config.total_gradient_steps:
                break
            batch_size = batch[0].size(0)
            obs, mask, valid_mask, partner_mask, road_mask, future_mask, future_pos = batch
            
            obs = obs.to("cuda")
            future_pos = future_pos.to("cuda")
            valid_mask = valid_mask.to("cuda")
            future_mask = future_mask.to("cuda")
            partner_mask = partner_mask.to("cuda")
            road_mask = road_mask.to("cuda")
            all_masks= [valid_mask, partner_mask, road_mask]

            if exp_config.model == 'baseline':
                baseline_obs = obs[..., :6].reshape(-1, 30)
                if exp_config.exp == 'other':
                    B, T, _ = obs.shape
                    ego_obs = obs[..., :6].unsqueeze(2).repeat(1, 1, 127, 1)
                    partner_obs = obs[..., 6:6 * 128].reshape(B, T, 127, 6)
                    lp_input = torch.cat([ego_obs, partner_obs], dim=-1).permute(0, 2, 1, 3).reshape(B, 127, -1)
                else:
                    lp_input = baseline_obs
            else:
                with torch.no_grad():
                    context, *_, = backbone.get_context(obs, all_masks)
                if exp_config.exp == 'ego':
                    lp_input = layers[nth_layer][:,0,:]
                else:
                    lp_input = layers[nth_layer][:,1:,:]

            # get future pred pos and action
            if exp_config.exp == 'ego':
                future_mask = future_mask.squeeze(1)
            pred_pos = pos_linear_model(lp_input)
            future_mask = ~future_mask if exp_config.exp == 'other' else future_mask
            masked_pos = pred_pos[future_mask]
            
            # get future expert pos and action
            future_pos = future_pos.clone()
            if exp_config.exp == 'ego':
                future_pos = future_pos.squeeze(1)
            masked_pos_label = future_pos[future_mask]

            if future_mask.sum() == 0:
                continue_num += 1
                continue
            
            # compute loss
            pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_pos_label)
            
            pos_optimizer.zero_grad()
            pos_loss.backward()
            gradient_steps += 1
            pos_optimizer.step()
            pbar.update(1)
            # get F1 scores
            pos_class = pos_class.detach().cpu().numpy()
            masked_pos_label = masked_pos_label.detach().cpu().numpy()
            pos_f1_macro = f1_score(pos_class, masked_pos_label, average='macro')

            pos_accuracys += pos_acc
            pos_losses += pos_loss.item()
            pos_f1_macros += pos_f1_macro
            # Evaluation loop
            if gradient_steps % exp_config.eval_freq == 0:
                pos_linear_model.eval()
                pos_accuracys = 0
                pos_losses = 0
                total_samples = 0
                pos_f1_macros = 0
                continue_num = 0
                for i, batch in enumerate(eval_expert_data_loader):
                    obs, mask, valid_mask, partner_mask, road_mask, future_mask, future_pos = batch
                    
                    obs = obs.to("cuda")
                    future_pos = future_pos.to("cuda")
                    valid_mask = valid_mask.to("cuda")
                    future_mask = future_mask.to("cuda")
                    partner_mask = partner_mask.to("cuda")
                    road_mask = road_mask.to("cuda")
                    all_masks= [valid_mask, partner_mask, road_mask]
                    if exp_config.model == 'baseline':
                        baseline_obs = obs[..., :6].reshape(-1, 30)
                        if exp_config.exp == 'other':
                            B, T, _ = obs.shape
                            ego_obs = obs[..., :6].unsqueeze(2).repeat(1, 1, 127, 1)
                            partner_obs = obs[..., 6:6 * 128].reshape(B, T, 127, 6)
                            lp_input = torch.cat([ego_obs, partner_obs], dim=-1).permute(0, 2, 1, 3).reshape(B, 127, -1)
                        else:
                            lp_input = baseline_obs
                    else:
                        with torch.no_grad():
                            context, *_, = backbone.get_context(obs, all_masks)
                        if exp_config.exp == 'other':
                            lp_input = layers[nth_layer][:,0,:]
                        else:
                            lp_input = layers[nth_layer][:,1:,:]

                        # get future pred pos and action
                        pred_pos = pos_linear_model(lp_input)
                        if exp_config.exp == 'ego':
                            future_mask = future_mask.squeeze(1)
                        future_mask = ~future_mask if exp_config.exp == 'other' else future_mask
                        masked_pos = pred_pos[future_mask]
                        
                        # get future expert action
                        
                        future_pos = future_pos.clone()
                        future_pos = future_pos.squeeze(1)
                        masked_pos_label = future_pos[future_mask]
                        
                        if future_mask.sum() == 0:
                            continue_num += 1
                            continue
                        
                        # compute loss
                        pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_pos_label)

                        # get F1 scores
                        pos_class = pos_class.detach().cpu().numpy()
                        masked_pos_label = masked_pos_label.detach().cpu().numpy()
                        pos_f1_macro = f1_score(pos_class, masked_pos_label, average='macro')


                        pos_accuracys += pos_acc
                        pos_losses += pos_loss.item()
                        pos_f1_macros += pos_f1_macro
                if exp_config.use_wandb:
                    wandb.log(
                        {
                            "eval/pos_accuracy": pos_accuracys / (i + 1 - continue_num),
                            "eval/pos_loss": pos_losses / (i + 1 - continue_num),
                            "eval/pos_f1_macro": pos_f1_macros / (i + 1 - continue_num),
                        }, step=gradient_step
                    )
                if pos_losses < best_loss:
                    save_dir = os.path.join(exp_config.model_path, f"{args.exp}_linear_prob/{exp_config.model_name}/seed{exp_config.seed}/")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(pos_linear_model, os.path.join(save_dir, f"pos_{exp_config.model}_{exp_config.future_step}.pth"))
                    best_loss = pos_losses
                    print(f'STEP {gradient_steps} gets BEST!')
        if exp_config.use_wandb:
            wandb.log(
                {
                    "train/pos_accuracy": pos_accuracys / (i + 1 - continue_num),
                    "train/pos_loss": pos_losses / (i + 1 - continue_num),
                    "train/pos_f1_macro": pos_f1_macros / (i + 1 - continue_num),
                }, step=gradient_step
            )
        
    
if __name__ == "__main__":
    args = parse_args()
    with open('baselines/il/config/lp.yaml', "r") as f:
        exp_config = Box(yaml.safe_load(f))
    if args.use_wandb:
        with open("baselines/il/lp_sweep.yaml") as f:
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