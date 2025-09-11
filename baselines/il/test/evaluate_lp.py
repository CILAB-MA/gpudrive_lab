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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_dataloader(data_path, data_file, config, isshuffle=True):
    with np.load(os.path.join(data_path, data_file)) as npz:
        ego_labels = None
        partner_labels = None
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
        if config.exp == 'ego':
            ego_labels = npz['ego_labels'].astype('int') if 'ego_labels' in npz.keys() else None
        if config.exp == 'other':
            partner_labels = npz['partner_labels'].astype('int') if 'partner_labels' in npz.keys() else None
    ego_global_pos = None
    ego_global_rot = None
    if 'validation' in data_file:
        data_file = data_file[6:]
    with np.load(os.path.join(data_path, "global_" + data_file)) as global_npz:
        ego_global_pos = global_npz['ego_global_pos']
        ego_global_rot = global_npz['ego_global_rot']
    dataset = FutureDataset(
        expert_obs, expert_actions, ego_global_pos, ego_global_rot, expert_masks, partner_mask, road_mask,
        rollout_len=config.rollout_len, pred_len=config.pred_len, future_step=config.future_step,
        exp=config.exp, partner_labels=partner_labels, ego_labels=ego_labels
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

def evaluate(exp_config):
    # Backbone and heads
    if exp_config['model'] == 'baseline':
        backbone = None
    else:
        backbone = torch.load(backbone_path, weights_only=False)
        backbone.eval()
        if exp_config.model == 'early_lp':
            layers = register_all_layers_forward_hook(backbone.fusion_attn)
        else:
            layers = register_all_layers_forward_hook(backbone.ro_attn)
    ood_label_tensor = torch.tensor(ood_labels, device='cuda')
    pos_linear_model = torch.load(exp_config.lp_path, weights_only=False)
    eval_data_path ='/data/full_version/processed/final/'
    eval_data_file =  f"label/validation_trajectory_2500.npz"
    # DataLoaders
    eval_expert_data_loader = get_dataloader(eval_data_path, eval_data_file, exp_config,
                                            isshuffle=False)
    print(f'EXP CONFIG {exp_config}')

    pos_linear_model.eval()
    test_pos_accuracys = 0
    test_pos_losses = 0
    test_pos_f1_macros = 0
    test_continue_num = 0
    test_ood_accuracys = 0
    test_ood_losses = 0
    ood_classes, ood_labels = [], []
    labeled_acc = torch.zeros(5)
    labeled_sum = torch.zeros(5)
    num_oods = 0
    for j, batch in enumerate(eval_expert_data_loader):
        obs, actions, mask, valid_mask, partner_mask, road_mask, future_mask, future_pos, labels = batch
        with torch.no_grad():
            obs = obs.to("cuda")
            actions = actions.to("cuda")
            future_pos = future_pos.to("cuda")
            valid_mask = valid_mask.to("cuda")
            future_mask = future_mask.to("cuda")
            partner_mask = partner_mask.to("cuda")
            road_mask = road_mask.to("cuda")
            labels = labels.to("cuda")
            all_masks= [partner_mask, road_mask]
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
                    _ = backbone.get_context(obs, all_masks)
                nth_layer =list(layers.keys())[-1]
                if exp_config.exp == 'ego':
                    lp_input = layers[nth_layer][:,0,:]
                else:
                    lp_input = layers[nth_layer][:,1:128,:]
        with torch.no_grad():
            # get future pred pos and action
            pred_pos = pos_linear_model(lp_input)
            future_mask = ~future_mask if exp_config.exp == 'other' else future_mask
            masked_pos = pred_pos[future_mask]
            masked_label = labels[future_mask]
            # get future expert actionpartner_mask
            future_pos = future_pos.clone()
            masked_pos_label = future_pos[future_mask]
            ood_mask = (masked_pos_label[..., None] == ood_label_tensor).any(dim=-1)
            ood_pos_label = masked_pos_label[ood_mask]
            ood_pos_pred = masked_pos[ood_mask]
            if future_mask.sum() == 0:
                test_continue_num += 1
                continue
            
            # compute loss
            pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_pos_label)
            if len(ood_pos_pred) > 0:
                ood_loss, ood_acc, ood_class, num_ood = pos_linear_model.loss_no_reduction(ood_pos_pred, ood_pos_label)
                ood_pos_label = ood_pos_label.detach().cpu().numpy()
                ood_class = ood_class.detach().cpu().numpy()
                test_ood_accuracys += ood_acc
                test_ood_losses += ood_loss.sum()
                num_oods += num_ood
                ood_classes.append(ood_class)
                ood_labels.append(ood_pos_label)
            pred_classes = masked_pos.argmax(-1) 
            error_mask = masked_label == -1
            filtered_label = masked_label[~error_mask]
            filtered_pos_label = masked_pos_label[~error_mask]
            filtered_pred_classes = pred_classes[~error_mask]
            one_hot = torch.nn.functional.one_hot(filtered_label.long(), num_classes=5).bool()
            cls_totals = one_hot.sum(dim=0)
            correct_mask = filtered_pred_classes == filtered_pos_label
            correct_one_hot = one_hot & correct_mask.unsqueeze(-1)
            correct_per_class = correct_one_hot.sum(dim=0)
            labeled_acc += correct_per_class.cpu()
            labeled_sum += cls_totals.cpu()

        # get F1 scores
        
        pos_class = pos_class.detach().cpu().numpy()
        masked_pos_label = masked_pos_label.detach().cpu().numpy()
        pos_f1_macro = f1_score(pos_class, masked_pos_label, average='macro')

        test_pos_accuracys += pos_acc
        test_pos_losses += pos_loss.item()
        test_pos_f1_macros += pos_f1_macro

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--exp', type=str, default='other', choices=['other', 'ego'])
    parser.add_argument('--model', type=str, default='baseline', choices=['early_lp', 'final_lp', 'baseline'])
    parser.add_argument('--model-path', '-mp', type=str, default='exp_100')
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--future-step', '-f', type=int, default=10)
    args = parser.parse_args()
    base_path = '/data/full_version/model'
    exp_path = os.path.join(base_path, args.model_path)
    lp_base_path = os.path.join(exp_path,  f'{args.exp}_linear_prob')
    backbone_name = os.listdir(lp_base_path)[0]
    lp_path = os.path.join(lp_base_path, backbone_name, f'seed{args.seed}')
    backbone_path = f'{exp_path}/{backbone_name}.pth' 
    lp_path = f'{lp_path}/pos_{args.exp}_{args.future_step}.pth'
    exp_config = dict(
        lp_path=lp_path,
        backbone_path =backbone_path,
        model = args.model,
        exp = args.exp,

    )
    evaluate(exp_config)