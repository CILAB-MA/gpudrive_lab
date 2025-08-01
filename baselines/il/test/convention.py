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
from scipy.stats import mode
from collections import defaultdict
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/data_cut_add')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_seed_3_0523_204605.pth')
    parser.add_argument('--seed', type=int, default=3)
    args = parser.parse_args()
    
    return args

def get_dataloader(data_path, data_file, config, isshuffle=True):
    with np.load(os.path.join(data_path, data_file)) as npz:
        ego_labels = None
        partner_labels = None
        ego_id = None
        scene_id = None
        partner_id = None
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
    if 'validation' in data_file:
        with np.load(os.path.join(data_path, "id_" + data_file)) as global_npz:
            ego_id = global_npz['ego_id']
            scene_id = global_npz['scene_id']
            partner_id = global_npz['partner_id']
    dataset = FutureDataset(
        expert_obs, expert_actions, ego_global_pos, ego_global_rot, expert_masks, partner_mask, road_mask,
        rollout_len=config.rollout_len, pred_len=config.pred_len, future_step=config.future_step,
        exp=config.exp, partner_labels=partner_labels, ego_labels=ego_labels,
        ego_id=ego_id, scene_id=scene_id, partner_id=partner_id
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=isshuffle,
        prefetch_factor=2,
        pin_memory=True,
        num_workers=8
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

def validation(exp_config=None):
    print(f'model: {args.model_path}/{args.model_name}', )
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    # Backbone and heads
    if exp_config.model == 'baseline':
        hidden_dim = 30 if exp_config.exp == 'ego' else 60 # ego info
        backbone = None
    else:
        backbone = torch.load(f"{exp_config.model_path}/{exp_config.model_name}.pth", weights_only=False)
        backbone.eval()
        if exp_config.model == 'early_lp':
            layers = register_all_layers_forward_hook(backbone.fusion_attn)
        else:
            layers = register_all_layers_forward_hook(backbone.ro_attn)
        hidden_dim = backbone.hidden_dim
    # Load other_lp heads
    other_lp_heads = {}
    other_lp_path = os.path.join(args.model_path, "other_linear_prob", args.model_name[:-4], f"seed{args.seed}")
    for lp in os.listdir(other_lp_path):
        if 'pth' not in lp:
            continue
        file_name, file_type = lp.split('.')
        if file_type == 'pth' and "early" in file_name:
            other_lp_heads[file_name] = torch.load(os.path.join(other_lp_path, lp), weights_only=False).to("cuda")
            other_lp_heads[file_name].eval()

    eval_data_path = os.path.join(exp_config.base_path, exp_config.data_path)
    eval_data_file =  f"label/validation_trajectory_2500.npz"

    # DataLoaders
    eval_expert_data_loader = get_dataloader(eval_data_path, eval_data_file, exp_config,
                                            isshuffle=False)
    # Evaluation loop
    test_pos_accuracys = 0
    test_pos_losses = 0
    test_pos_f1_macros = 0
    test_continue_num = 0
    labeled_acc = torch.zeros(5)
    labeled_sum = torch.zeros(5)
    num_oods = 0
    all_masked_preds = []
    all_masked_scene_ids = []
    all_masked_partner_ids = []
    all_masked_timesteps = []
    for j, batch in enumerate(eval_expert_data_loader):
        obs, actions, mask, valid_mask, partner_mask, road_mask, future_mask, future_pos, labels, ego_id, scene_id, partner_id, timestep = batch
        with torch.no_grad():
            obs = obs.to("cuda")
            actions = actions.to("cuda")
            future_pos = future_pos.to("cuda")
            valid_mask = valid_mask.to("cuda")
            future_mask = future_mask.to("cuda")
            partner_mask = partner_mask.to("cuda")
            road_mask = road_mask.to("cuda")
            labels = labels.to("cuda")
            ego_id = ego_id.to("cuda")
            scene_id = scene_id.to("cuda").repeat(1, 127)
            partner_id = partner_id.to("cuda").int()
            timestep = timestep.to("cuda").unsqueeze(-1).repeat(1, 127).int()
            all_masks= [partner_mask, road_mask]
        if exp_config.model == 'baseline':
            B, T, _ = obs.shape
            ego_obs = obs[..., :6].unsqueeze(2).repeat(1, 1, 127, 1)
            partner_obs = obs[..., 6:6 * 128].reshape(B, T, 127, 6)
            lp_input = torch.cat([ego_obs, partner_obs], dim=-1).permute(0, 2, 1, 3).reshape(B, 127, -1)
        else:
            with torch.no_grad():
                context, *_, = backbone.get_context(obs, all_masks)
            nth_layer =list(layers.keys())[-1]
            lp_input = layers[nth_layer][:,1:128,:]

        with torch.no_grad():
            # get future pred pos and action
            pred_pos = other_lp_heads['pos_early_lp_10'](lp_input)
            future_mask = ~future_mask
            masked_pos = pred_pos[future_mask]
            masked_label = labels[future_mask]
            masked_partner_ids = partner_id[future_mask]
            masked_scene_ids = scene_id[future_mask]
            masked_timesteps = timestep[future_mask]

            masked_partner_ids = masked_partner_ids.cpu().int()
            masked_scene_ids = masked_scene_ids.cpu().int()
            masked_timesteps = masked_timesteps.cpu().int()
            all_masked_preds.append(masked_pos.argmax(-1).cpu())
            all_masked_scene_ids.append(masked_scene_ids)
            all_masked_partner_ids.append(masked_partner_ids)
            all_masked_timesteps.append(masked_timesteps)
            if future_mask.sum() == 0:
                test_continue_num += 1
                continue
    # Concatenate all batches
    masked_preds = torch.cat(all_masked_preds, dim=0).numpy()
    scene_ids = torch.cat(all_masked_scene_ids, dim=0)
    partner_ids = torch.cat(all_masked_partner_ids, dim=0)
    timesteps = torch.cat(all_masked_timesteps, dim=0)

    # Group key = (scene, partner, timestep)
    group_keys = torch.stack([scene_ids, partner_ids, timesteps], dim=1)
    _, unique_indices = torch.unique(group_keys, return_inverse=True, dim=0)

    grouped_indices = defaultdict(list)
    for idx, group_id in enumerate(unique_indices):
        grouped_indices[int(group_id)].append(idx)

    # Agreement 계산
    agreement_scores = []
    for idx_list in grouped_indices.values():
        group_size = len(idx_list)
        if group_size < 2:
            continue
        group_preds = masked_preds[idx_list]
        if group_size == 2:
            agreement_ratio = float(group_preds[0] == group_preds[1])
        else:
            mode_val, count = mode(group_preds, keepdims=True)
            agreement_ratio = count[0] / group_size
        agreement_scores.append(agreement_ratio)

    agreement_scores = np.array(agreement_scores)
    print(f"전체 그룹 수: {len(agreement_scores)}")
    print(f"평균 일치율: {agreement_scores.mean():.4f}")
    print(f"중앙값 일치율: {np.median(agreement_scores):.4f}")
    print(f"상위 10개 예시: {np.sort(agreement_scores)[-10:]}")
        
    
if __name__ == "__main__":
    args = parse_args()
    with open('baselines/il/config/lp.yaml', "r") as f:
        exp_config = Box(yaml.safe_load(f))

    validation(exp_config)