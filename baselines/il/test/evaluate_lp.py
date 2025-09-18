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
from gpudrive.integrations.il.linear_probing.lp_model import *
from sklearn.metrics import f1_score
from box import Box
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, linregress
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class FutureDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, ego_global_pos, ego_global_rot, masks=None, partner_mask=None, road_mask=None,
                 rollout_len=5, pred_len=1, future_step=1, exp='other', xy_range=None, 
                 partner_labels=None, ego_labels=None):
        # obs
        self.obs = obs
        B, T, F = obs.shape
        new_shape = (B, T + rollout_len - 1, F)
        new_obs = np.zeros(new_shape, dtype=self.obs.dtype)
        new_obs[:, rollout_len - 1:] = obs # This is more cheaper than concatenate
        self.obs = new_obs
        self.actions = actions

        # masks
        valid_masks = 1 - masks
        action_mask = (np.abs(actions[..., 1]) >  0.5) | (np.abs(actions[..., 0]) >  5) | (np.abs(actions[..., -1]) > 0.2)
        valid_masks[action_mask] = 0
        new_shape = (B, T + rollout_len - 1)
        new_valid_mask = np.zeros(new_shape, dtype=self.obs.dtype)
        new_valid_mask[:, rollout_len - 1:] = valid_masks
        self.valid_masks = new_valid_mask.astype('bool')
        #### 
        if partner_labels is not None:
            partner_labels_pad = np.zeros((partner_labels.shape[0], future_step, *partner_labels.shape[2:]), dtype=np.int16)
            partner_labels_new = np.concatenate([partner_labels, partner_labels_pad], axis=1)[:, future_step:]
            self.partner_labels = partner_labels_new
        if ego_labels is not None:
            self.ego_labels = ego_labels
        if exp == 'other':
            # future partner_mask
            partner_info = obs[..., 6:128 * 6].reshape(B, T, 127, 6)[..., :4]
            aux_info, aux_mask = self._make_aux_info(partner_mask, partner_info, future_timestep=future_step)
            self.aux_mask = aux_mask.astype('bool')
        new_shape = (B, T + rollout_len - 1, 127)
        new_partner_mask = np.full(new_shape, 2, dtype=np.float32)
        new_partner_mask[:, rollout_len - 1:] = partner_mask
        self.partner_mask = np.where(new_partner_mask == 2, 1, 0).astype('bool')

        if exp == 'ego':
            # future ego mask
            future_valid_mask_pad = np.zeros((self.valid_masks.shape[0], future_step, *self.valid_masks.shape[2:]), dtype=np.float32)
            future_valid_masks = np.concatenate([valid_masks, future_valid_mask_pad], axis=1).astype('bool')[:, future_step:]
            self.future_valid_mask = self.valid_masks[:,rollout_len - 1:] & future_valid_masks
        
        # road_mask
        self.road_mask = road_mask
        new_shape = (B, T + rollout_len - 1, 200)
        new_road_mask = np.ones(new_shape)
        new_road_mask[:, rollout_len - 1:] = road_mask
        self.road_mask = new_road_mask.astype('bool')
        if exp == 'other':
            # future other pos
            current_relative_other_pos = self._transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step=future_step)
            current_relative_other_pos[aux_mask] = 0
            self.other_pos = self._get_multi_class_pos(current_relative_other_pos)
        else:
            # future ego pos
            current_relative_ego_pos = self._transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step=future_step,
                                                                    )
            self.ego_pos = self._get_multi_class_pos(current_relative_ego_pos, xy_range)

            # ego? -> current_relative_pos[aux_mask] = 0
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'valid_masks', 'partner_mask', 'road_mask']
        if exp == 'other':
            self.full_var += ['aux_mask', 'other_pos']
        else:
            self.full_var += ['future_valid_mask', 'ego_pos']
        if partner_labels is not None:
            self.full_var += ['partner_labels']
        if ego_labels is not None:
            self.full_var += ['ego_labels']
        self.future_step = future_step
    def __len__(self):
        return len(self.valid_indices)

    def _make_aux_info(self, partner_mask, partner_info, future_timestep):
        partner_mask_bool = np.where(partner_mask == 0, 0, 1).astype(bool)
        action_valid_mask = np.where(partner_mask == 0, 1, 0).astype(bool)
        partner_info_pad = np.zeros((partner_info.shape[0], future_timestep, *partner_info.shape[2:]), dtype=np.float32)
        partner_mask_pad = np.full((partner_mask.shape[0], future_timestep, *partner_mask.shape[2:]), 2, dtype=np.float32)

        future_mask = np.concatenate([partner_mask, partner_mask_pad], axis=1)
        future_mask_bool = np.where(future_mask == 0, 0, 1).astype(bool)[:, future_timestep:]
        partner_info = np.concatenate([partner_info, partner_info_pad], axis=1)[:, future_timestep:]
        combined_mask = np.logical_or(future_mask_bool, partner_mask_bool).astype('bool')
        return partner_info, combined_mask
    
    @staticmethod
    def _transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step):
        """transform time t relative pos to current relative pos"""
        # 1. transform t-relative pos to t-global pos
        # get partner's relative pos and rot at time t
        t_partner_pos = aux_info[..., 1:3] * MAX_REL_AGENT_POS
        
        # get ego's global pos and rot at time t
        t_ego_global_pos = np.zeros_like(ego_global_pos)
        t_ego_global_rot = np.zeros_like(ego_global_rot)
        t_ego_global_pos[:, :-future_step] = ego_global_pos[:, future_step:]
        t_ego_global_rot[:, :-future_step] = ego_global_rot[:, future_step:]
        
        t_partner_global_pos_x = t_ego_global_pos[..., 0, None] + t_partner_pos[..., 0] * np.cos(t_ego_global_rot) - t_partner_pos[..., 1] * np.sin(t_ego_global_rot)
        t_partner_global_pos_y = t_ego_global_pos[..., 1, None] + t_partner_pos[..., 0] * np.sin(t_ego_global_rot) + t_partner_pos[..., 1] * np.cos(t_ego_global_rot)
        
        # 2. transform t-global pos to current relative pos
        delta_x = t_partner_global_pos_x - ego_global_pos[..., 0, None]
        delta_y = t_partner_global_pos_y - ego_global_pos[..., 1, None]
        
        cos_theta = np.cos(-ego_global_rot)
        sin_theta = np.sin(-ego_global_rot)
        
        current_relative_pos_x = delta_x * cos_theta + delta_y * sin_theta
        current_relative_pos_y = -delta_x * sin_theta + delta_y * cos_theta
        current_relative_pos_x = 2 * ((current_relative_pos_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((current_relative_pos_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
        
        return current_relative_pos

    @staticmethod
    def _transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step):
        """transform global pos to current relative pos"""
        current_relative_pos = np.zeros_like(ego_global_pos)
        ego_current_pos = ego_global_pos[:, :-future_step]
        ego_future_pos = ego_global_pos[:, future_step:]
        
        delta_x = ego_future_pos[..., 0] - ego_current_pos[..., 0]
        delta_y = ego_future_pos[..., 1] - ego_current_pos[..., 1]
        
        ego_current_rot = ego_global_rot[:, :-future_step]
        
        cos_theta = np.cos(ego_current_rot)
        sin_theta = np.sin(ego_current_rot)
        
        rel_x = delta_x * cos_theta.squeeze(-1) + delta_y * sin_theta.squeeze(-1)
        rel_y = -delta_x * sin_theta.squeeze(-1) + delta_y * cos_theta.squeeze(-1)
        
        current_relative_pos_x = 2 * ((rel_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((rel_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos[:, :-future_step, :] = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
        
        return current_relative_pos

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))

    @staticmethod
    def _get_multi_class_pos(pos, xy_range=None):
        """
        Convert continuous pos to multi-class discrete pos based on x, y.
        """
        x, y = pos[..., 0], pos[..., 1]
        
        # Define bins for discretization (-1 to 1 with 8 bins)
        if xy_range is not None:
            xrange = xy_range[0]
            yrange = xy_range[1]
            xbins = np.linspace(xrange[0], xrange[1], 9)
            ybins = np.linspace(yrange[0], yrange[1], 9)
        else:
            xbins = np.linspace(-0.05, 0.05, 9)
            ybins = np.linspace(-0.05, 0.05, 9)

        # Digitize x and y into 8 categories (0 to 7)
        x_bins = np.digitize(x, xbins) - 1
        y_bins = np.digitize(y, ybins) - 1
        
        # Ensure values are within valid range (0 to 7)
        x_bins = np.clip(x_bins, 0, 7)
        y_bins = np.clip(y_bins, 0, 7)
        
        discrete_pos = x_bins * 8 + y_bins
        return discrete_pos

    @staticmethod
    def _get_multi_class_actions(actions):
        """
        Convert continuous actions to multi-class discrete actions based on dyaw.
        """
        dx = actions[..., 0]
        dy = actions[..., 1]
        dyaw = actions[..., 2]
        
        dx_bins = np.linspace(-3, 3, 5)  # 4 bins for dx
        dy_bins = np.linspace(-3, 3, 5)  # 4 bins for dy
        dyaw_bins = np.linspace(-np.pi / 4, np.pi / 4, 5)  # 4 bins for dyaw
        
        dx_bin = np.digitize(dx, dx_bins) - 1
        dy_bin = np.digitize(dy, dy_bins) - 1
        dyaw_bin = np.digitize(dyaw, dyaw_bins) - 1

        # Ensure indices are within range
        dx_bin = np.clip(dx_bin, 0, 4 - 1)
        dy_bin = np.clip(dy_bin, 0, 4 - 1)
        dyaw_bin = np.clip(dyaw_bin, 0, 4 - 1)

        # Compute single discrete index
        discrete_action = (dx_bin * 4 * 4) + (dy_bin * 4) + dyaw_bin

        return discrete_action
        
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        idx1 = int(idx1)
        idx2 = int(idx2)
        # row, column -> 
        batch = ()
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name in ['actions']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
                    elif var_name == 'valid_masks':
                        data = self.__dict__[var_name][idx1 ,idx2 + self.rollout_len + self.pred_len - 2] # idx 0 -> (0, 10 + 5 - 2) -> (0, 13) & padding = 9 -> end with last action timestep
                    elif var_name in ['aux_mask', 'other_pos', 'future_valid_mask', 'ego_pos', 'partner_labels']:
                        data = self.__dict__[var_name][idx1, idx2]
                    elif var_name == 'ego_labels':
                        data = self.__dict__[var_name][idx1]
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
                    if var_name == 'valid_masks':
                        ego_mask_data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len]
                        batch = batch + (ego_mask_data, )
                    if var_name == 'obs':
                        if idx2 < 91 - self.future_step:
                            future_dist = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len + self.future_step][-1, 6: 128 * 6].reshape(127, -1)
                            future_dist = future_dist[:, 1:3]
                        else:
                            future_dist = np.zeros((127, 2)).astype(self.obs.dtype)
                        batch = batch + (future_dist, )
        else:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    data = self.__dict__[var_name][idx]
                    batch = batch + (data, )
        return batch
    
def get_dataloader(data_path, data_file, config, isshuffle=True):
    with np.load(os.path.join(data_path, data_file)) as npz:
        ego_labels = None
        partner_labels = None
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
        if config['exp'] == 'ego':
            ego_labels = npz['ego_labels'].astype('int') if 'ego_labels' in npz.keys() else None
        if config['exp'] == 'other':
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
        rollout_len=5, pred_len=1, future_step=config['future_step'],
        exp=config['exp'], partner_labels=partner_labels, ego_labels=ego_labels
    )
    dataloader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=isshuffle,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True
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
        if exp_config['model'] == 'early_lp':
            layers = register_all_layers_forward_hook(backbone.fusion_attn)
        else:
            layers = register_all_layers_forward_hook(backbone.ro_attn)
    pos_linear_model = torch.load(exp_config['lp_path'], weights_only=False)
    eval_data_path ='/data/full_version/processed/final/'
    eval_data_file =  f"label/validation_trajectory_2500.npz"
    # DataLoaders
    eval_expert_data_loader = get_dataloader(eval_data_path, eval_data_file, exp_config,
                                            isshuffle=False)
    print(f'EXP CONFIG {exp_config}')
    per_batch = 5
    all_dists, all_true_probs = [], []
    pos_linear_model.eval()
    test_pos_accuracys = 0
    test_pos_losses = 0
    test_pos_f1_macros = 0
    test_continue_num = 0
    labeled_acc = torch.zeros(5)
    labeled_sum = torch.zeros(5)
    for j, batch in enumerate(eval_expert_data_loader):
        obs, future_dist, actions, mask, valid_mask, partner_mask, road_mask, future_mask, future_pos, labels = batch
        with torch.no_grad():
            obs = obs.to("cuda")
            future_dist = future_dist.to("cuda")
            actions = actions.to("cuda")
            future_pos = future_pos.to("cuda")
            current_dist = obs[:, -1, 6:128 * 6].reshape(-1, 127, 6)[..., 1:3]
            valid_mask = valid_mask.to("cuda")
            future_mask = future_mask.to("cuda")
            partner_mask = partner_mask.to("cuda")
            road_mask = road_mask.to("cuda")
            labels = labels.to("cuda")
            all_masks= [partner_mask, road_mask]
            if exp_config['model'] == 'baseline':
                baseline_obs = obs[..., :6].reshape(-1, 30)
                if exp_config['exp'] == 'other':
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
                if exp_config['exp'] == 'ego':
                    lp_input = layers[nth_layer][:,0,:]
                else:
                    lp_input = layers[nth_layer][:,1:128,:]
        with torch.no_grad():
            # get future pred pos and action
            pred_pos = pos_linear_model(lp_input)
            future_mask = ~future_mask if exp_config['exp'] == 'other' else future_mask
            masked_pos = pred_pos[future_mask]
            masked_label = labels[future_mask]
            # get future expert actionpartner_mask
            future_pos = future_pos.clone()
            masked_pos_label = future_pos[future_mask]
            if future_mask.sum() == 0:
                test_continue_num += 1
                continue
            
            # compute loss
            pos_loss, pos_acc, pos_class = pos_linear_model.loss(masked_pos, masked_pos_label)
            probs = torch.softmax(masked_pos, dim=-1)   
            true_prob = probs[torch.arange(probs.size(0), device=probs.device),
                              masked_pos_label.long()] 
            future_dists = torch.linalg.norm(future_dist[future_mask], dim=-1)
            curr_dists = torch.linalg.norm(current_dist[future_mask], dim=-1)
            how_closer = curr_dists - future_dists    
            valid_num = torch.isfinite(how_closer) & torch.isfinite(true_prob) & (future_dists <= 0.015)
            if valid_num.any():
                valid_idx = torch.nonzero(valid_num, as_tuple=False).squeeze(1)
                num_pick = min(per_batch, valid_idx.numel())
                pick = valid_idx[torch.randperm(valid_idx.numel(), device=valid_idx.device)[:num_pick]]

                all_dists.append(how_closer[pick].detach().cpu())
                all_true_probs.append(true_prob[pick].detach().cpu())
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

    if len(all_dists) > 0:
        dists_all = torch.cat(all_dists).numpy()
        probs_all = torch.cat(all_true_probs).numpy()

        # 상관 + 선형회귀
        r, p = pearsonr(dists_all, probs_all)
        slope, intercept, r_lin, p_lin, stderr = linregress(dists_all, probs_all)

        # 회귀선용 x/y
        xs = np.linspace(dists_all.min(), dists_all.max(), 200)
        ys = slope * xs + intercept

        plt.figure(figsize=(6,5))
        sc = plt.scatter(dists_all, probs_all, s=8, alpha=0.35, label=f"Samples (n={len(dists_all)})")
        ln, = plt.plot(xs, ys, linewidth=2, label=f"OLS fit: y={slope:.3f}x+{intercept:.3f}")
        # legend에 상관계수 표기
        extra = plt.Line2D([], [], linestyle='None', label=f"Pearson r={r:.3f}, p={p:.1e}")
        plt.legend(handles=[sc, ln, extra], loc="best", frameon=True)

        plt.xlabel("Future distance (norm)")
        plt.ylabel("Prob. of Label")
        plt.title("Distance Difference (Current - Future) vs True Label Probability")
        plt.tight_layout()
        plt.savefig(f"{exp_config['model_path']}_prob_dist_correlation.png", dpi=300)
        print(f"[Correlation] r={r:.6f}, p={p:.3e}, n={len(dists_all)}")
        print("[Saved] prob_dist_correlation.png")

def set_seed(seed=42, deterministic=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Select the dynamics model that you use')
    parser.add_argument('--exp', type=str, default='other', choices=['other', 'ego'])
    parser.add_argument('--model', type=str, default='early_lp', choices=['early_lp', 'final_lp', 'baseline'])
    parser.add_argument('--model-path', '-mp', type=str, default='exp_100')
    parser.add_argument('--seed', '-s', type=int, default=3)
    parser.add_argument('--future-step', '-f', type=int, default=10)
    args = parser.parse_args()
    base_path = '/data/full_version/model'
    exp_path = os.path.join(base_path, args.model_path)
    lp_base_path = os.path.join(exp_path,  f'{args.exp}_linear_prob')
    backbone_name = os.listdir(lp_base_path)[-1]
    lp_path = os.path.join(lp_base_path, backbone_name, f'seed{args.seed}')
    backbone_path = f'{exp_path}/{backbone_name}' if 'pth' in backbone_name else f'{exp_path}/{backbone_name}.pth'
    lp_path = f'{lp_path}/pos_{args.model}_{args.future_step}.pth'
    set_seed(0)
    exp_config = dict(
        lp_path=lp_path,
        backbone_path =backbone_path,
        model = args.model,
        exp = args.exp,
        future_step=args.future_step,
        model_path=args.model_path

    )
    evaluate(exp_config)