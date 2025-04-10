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
from algorithms.il.analyze.linear_probing.dataloader import EgoFutureDataset
from algorithms.il.analyze.linear_probing.config import ExperimentConfig
from algorithms.il.analyze.linear_probing.model import *
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--exp-name', '-en', type=str, default='all_data')
    parser.add_argument('--sweep-id', type=str, default=None)
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--use-tom', '-ut', default=None, choices=[None, 'guide_weighted', 'no_guide_no_weighted',
                                                                   'no_guide_weighted', 'guide_no_weighted'])
    parser.add_argument('--ego-future-step', '-afs', type=int, default=30)
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
    dataset = EgoFutureDataset(
        expert_obs, expert_actions, ego_global_pos, ego_global_rot, expert_masks, partner_mask, road_mask, other_info,
        rollout_len=config.rollout_len, pred_len=config.pred_len, ego_future_step=config.ego_future_step
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=isshuffle,
        num_workers=config.num_workers,
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
        hidden_dim = 30 # ego info
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
        
    pos_linear_model = LinearProbPosition(hidden_dim, 64, future_step=config.ego_future_step).to("cuda")
    # head_linear_model = LinearProbAngle(hidden_dim, 64, future_step=config.ego_future_step).to("cuda")
    
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
        continue_num = 0
        
        for i, batch in enumerate(expert_data_loader):
            batch_size = batch[0].size(0)
           
            obs, actions, future_pos, future_actions, cur_valid_mask, future_valid_mask, partner_masks, road_masks = batch
            
            obs = obs.to("cuda")
            future_pos, future_actions = future_pos.to("cuda"), future_actions.to("cuda")
            actions = actions.to("cuda")
            cur_valid_mask = cur_valid_mask.to("cuda") if len(batch) > 3 else None
            future_valid_mask = future_valid_mask.to("cuda") if len(batch) > 3 else None
            partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
            road_masks = road_masks.to("cuda") if len(batch) > 3 else None
            all_masks= [cur_valid_mask, partner_masks, road_masks]

            if config.model == 'baseline':
                ego_obs = obs[..., :6].reshape(-1, 30)
                lp_input = ego_obs
            else:
                with torch.no_grad():
                    context, *_, = backbone.get_context(obs, all_masks)
                lp_input = layers[nth_layer][:,0,:]

            # get future pred pos and action
            future_valid_mask = future_valid_mask.squeeze(1)
            pred_pos = pos_linear_model(lp_input)
            masked_pos = pred_pos[future_valid_mask]
            
            # get future expert pos and action
            future_pos = future_pos.clone()
            future_pos = future_pos.squeeze(1)
            masked_other_pos = future_pos[future_valid_mask]
            future_actions = future_actions.clone()
            future_actions = future_actions.squeeze(1)
            masked_other_actions = future_actions[future_valid_mask]
            
            if future_valid_mask.sum() == 0:
                continue_num += 1
                continue
            
            # compute loss
            pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_other_pos)
            total_loss = pos_loss #+ action_loss
            
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
                    "train/pos_accuracy": pos_accuracys / (i + 1 - continue_num),
                    # "train/heading_accuracy": heading_accuracys / (i + 1),
                    "train/pos_loss": pos_losses / (i + 1 - continue_num),
                    # "train/heading_loss": heading_losses / (i + 1),
                    "train/pos_f1_macro": pos_f1_macros / (i + 1 - continue_num),
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
            continue_num = 0
            for i, batch in enumerate(eval_expert_data_loader):
                batch_size = batch[0].size(0)
                obs, actions, future_pos, future_actions, cur_valid_mask, future_valid_mask, partner_masks, road_masks = batch
                actions = actions.to("cuda")
                obs = obs.to("cuda")
                future_pos, future_actions = future_pos.to("cuda"), future_actions.to("cuda")
                cur_valid_mask = cur_valid_mask.to("cuda") if len(batch) > 2 else None
                future_valid_mask = future_valid_mask.to("cuda") if len(batch) > 3 else None
                partner_masks = partner_masks.to("cuda") if len(batch) > 3 else None
                road_masks = road_masks.to("cuda") if len(batch) > 3 else None
                all_masks= [cur_valid_mask, partner_masks, road_masks]
                
                with torch.no_grad():
                    if config.model == 'baseline':
                        ego_obs = obs[..., :6].reshape(-1, 30)
                        lp_input = ego_obs
                    else:
                        context, *_, = backbone.get_context(obs, all_masks)
                        lp_input = layers[nth_layer][:,0,:]
                    # get future pred pos and action
                    pred_pos = pos_linear_model(lp_input)
                    future_valid_mask = future_valid_mask.squeeze(1)
                    masked_pos = pred_pos[future_valid_mask]
                    
                    # get future expert action
                    
                    future_pos = future_pos.clone()
                    future_pos = future_pos.squeeze(1)
                    masked_other_pos = future_pos[future_valid_mask]
                    
                    if future_valid_mask.sum() == 0:
                        continue_num += 1
                        continue
                    
                    # compute loss
                    pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_other_pos)

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
                        "eval/pos_accuracy": pos_accuracys / (i + 1 - continue_num),
                        # "eval/heading_accuracy": heading_accuracys / (i + 1),
                        "eval/pos_loss": pos_losses / (i + 1 - continue_num),
                        # "eval/heading_loss": heading_losses / (i + 1),
                        "eval/pos_f1_macro": pos_f1_macros / (i + 1 - continue_num),
                        # "eval/heading_f1_macro": heading_f1_macros / (i + 1)
                    }, step=epoch
                )
    
    # Save head
    save_dir = os.path.join(config.model_path, f"ego_linear_prob/{config.model_name}/seed{config.seed}v2/")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(pos_linear_model, os.path.join(save_dir, f"pos_{config.model}_{config.ego_future_step}.pth"))

if __name__ == "__main__":
    args = parse_args()
    if args.use_wandb:
        with open("algorithms/il/analyze/linear_probing/sweep_ego.yaml") as f:
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
