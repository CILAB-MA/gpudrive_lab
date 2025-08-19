"""Obtain a policy using behavioral cloning."""
import logging
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os, sys, torch
torch.backends.cudnn.benchmark = True
sys.path.append(os.getcwd())
import wandb, yaml, argparse
import matplotlib
matplotlib.use('Agg')
from baselines.il.il_utils import *
# GPUDrive
from gpudrive.integrations.il.dataloader import ExpertDataset
# from algorithms.il.utils import *
from gpudrive.integrations.il.loss import gmm_loss
from collections import defaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def evaluate(eval_expert_data_loader, bc_policy):
    total_samples = 0
    losses = 0
    dx_losses = 0
    dy_losses = 0
    dyaw_losses = 0
    dx_std2_losses = 0
    dy_std2_losses = 0
    dyaw_std2_losses = 0
    dx_values = 0
    dy_values = 0
    dyaw_values = 0
    dx_std2_count = 0
    dy_std2_count = 0
    dyaw_std2_count = 0
    dxc95_losses = 0
    dyc95_losses = 0
    dyawc95_losses = 0
    dxc95_count = 0
    dyc95_count = 0
    dyawc95_count = 0
    step_dx_vals  = defaultdict(list)   # 각 t의 |dx| 샘플들
    step_dy_vals  = defaultdict(list)
    step_dyaw_vals= defaultdict(list)
    step_count    = defaultdict(int)    # 커버리지 확인용


    for i, batch in enumerate(eval_expert_data_loader):
        batch_size = batch[0].size(0)
        total_samples += batch_size
        if len(batch) == 7:
            obs, expert_action, partner_masks, road_masks, other_pos, aux_mask, data_idx = batch
            other_pos = other_pos.to('cuda')
        elif len(batch) == 5:
            obs, expert_action, partner_masks, road_masks, data_idx = batch 
        step_idx = data_idx[:, 1].detach().cpu().numpy()
        obs, expert_action = obs.to('cuda'), expert_action.to('cuda')
        partner_masks = partner_masks.to('cuda') if len(batch) > 3 else None
        road_masks = road_masks.to('cuda') if len(batch) > 3 else None
        all_masks= [partner_masks, road_masks]
        with torch.no_grad():
            context, other_embeds, other_weights, *_  = bc_policy.get_context(obs, all_masks)
            pred_loss, _ = gmm_loss(bc_policy, context, expert_action)
            # pred_loss, _ = focal_loss(bc_policy, context, expert_action)
            loss = pred_loss
            pred_actions = bc_policy.get_action(context, deterministic=True)
            action_loss = torch.abs(pred_actions - expert_action)
            dx_q95 = torch.quantile(action_loss[..., 0], 0.95)
            dy_q95 = torch.quantile(action_loss[..., 1], 0.95)
            dyaw_q95 = torch.quantile(action_loss[..., 2], 0.95)
            dx_vals = action_loss[..., 0]
            dy_vals = action_loss[..., 1]
            dyaw_vals = action_loss[..., 2]
            al_cpu = action_loss.detach().cpu().squeeze(1).numpy()
            for t in np.unique(step_idx):
                m = (step_idx == t)
                if not m.any(): continue
                step_dx_vals[t].append(al_cpu[m, 0])
                step_dy_vals[t].append(al_cpu[m, 1])
                step_dyaw_vals[t].append(al_cpu[m, 2])
                step_count[t] += int(m.sum())
            dx_mask = dx_vals >= dx_q95
            dy_mask = dy_vals >= dy_q95
            dyaw_mask = dyaw_vals >= dyaw_q95

            dx_c95 = dx_vals[dx_mask].sum().item()
            dy_c95 = dy_vals[dy_mask].sum().item()
            dyaw_c95 = dyaw_vals[dyaw_mask].sum().item()
            dxc95_count += dx_mask.sum()
            dyc95_count += dy_mask.sum()
            dyawc95_count += dyaw_mask.sum()
            dx_std2_mask = expert_action[..., 0].abs() > 2 
            dy_std2_mask = expert_action[..., 1].abs() > 0.035 
            dyaw_std2_mask = expert_action[..., 2].abs() > 0.023
            dx_std2_mask = dx_std2_mask.cpu().numpy()
            dy_std2_mask = dy_std2_mask.cpu().numpy()
            dyaw_std2_mask = dyaw_std2_mask.cpu().numpy()
            action_loss = action_loss.cpu().numpy()
            dx_loss = action_loss[..., 0].mean()
            dy_loss = action_loss[..., 1].mean()
            dyaw_loss = action_loss[..., 2].mean()

            dx_std2_loss = action_loss[..., 0][dx_std2_mask].sum() if dx_std2_mask.sum() > 0 else 0
            dy_std2_loss = action_loss[..., 1][dy_std2_mask].sum() if dy_std2_mask.sum() > 0 else 0
            dyaw_std2_loss = action_loss[..., 2][dyaw_std2_mask].sum() if dyaw_std2_mask.sum() > 0 else 0
            dxc95_losses += dx_c95
            dyc95_losses += dy_c95
            dyawc95_losses += dyaw_c95
            # bsae loss
            dx_losses += dx_loss
            dy_losses += dy_loss
            dyaw_losses += dyaw_loss

            # std2 loss
            dx_std2_losses += dx_std2_loss
            dy_std2_losses += dy_std2_loss
            dyaw_std2_losses += dyaw_std2_loss

            # action values
            dx_mean = pred_actions[..., 0].mean()
            dy_mean = pred_actions[..., 1].mean()
            dyaw_mean = pred_actions[..., 2].mean()

            dx_values += dx_mean
            dy_values += dy_mean
            dyaw_values += dyaw_mean

            dx_std2_count += dx_std2_mask.sum()
            dy_std2_count += dy_std2_mask.sum()
            dyaw_std2_count += dyaw_std2_mask.sum()

            losses += loss.mean().item()

    test_loss = losses / (i + 1) 
    dx_loss = dx_losses / (i + 1) 
    dy_loss = dy_losses / (i + 1) 
    dyaw_loss = dyaw_losses / (i + 1) 
    dx_loss = dx_losses / (i + 1) 
    dy_loss = dy_losses / (i + 1) 
    dyaw_loss = dyaw_losses / (i + 1) 

    dx_values = dx_values / (i + 1) 
    dy_values = dy_values / (i + 1) 
    dyaw_values = dyaw_values / (i + 1) 

    dx_std2_loss = dx_std2_losses / dx_std2_count
    dy_std2_loss = dy_std2_losses / dy_std2_count
    dyaw_std2_loss = dyaw_std2_losses / dyaw_std2_count

    dx_c95_loss = dxc95_losses / dxc95_count
    dy_c95_loss = dyc95_losses / dyc95_count
    dyaw_c95_loss = dyawc95_losses / dyawc95_count



    # ---- 루프 끝: per-step 평균 + CVaR 계산 ----
    def cvar95_from_lists(lst_list):
        a = np.concatenate(lst_list) if len(lst_list) else np.array([])
        if a.size == 0: return 0.0
        q = np.quantile(a, 0.95)
        tail = a[a >= q]
        return float(tail.mean()) if tail.size > 0 else 0.0

    step_dx_mean   = {int(t): float(np.concatenate(step_dx_vals[t]).mean())   for t in step_dx_vals}
    step_dy_mean   = {int(t): float(np.concatenate(step_dy_vals[t]).mean())   for t in step_dy_vals}
    step_dyaw_mean = {int(t): float(np.concatenate(step_dyaw_vals[t]).mean()) for t in step_dyaw_vals}

    step_dx_cvar95   = {int(t): cvar95_from_lists(step_dx_vals[t])   for t in step_dx_vals}
    step_dy_cvar95   = {int(t): cvar95_from_lists(step_dy_vals[t])   for t in step_dy_vals}
    step_dyaw_cvar95 = {int(t): cvar95_from_lists(step_dyaw_vals[t]) for t in step_dyaw_vals}

    per_step = {
        # 평균(이미 계산한 mean들과 동일하지만 여기선 일괄 dict로 반환)
        "eval/step_dx_loss_mean": step_dx_mean,
        "eval/step_dy_loss_mean": step_dy_mean,
        "eval/step_dyaw_loss_mean": step_dyaw_mean,
        # 꼬리 지표
        "eval/step_dx_cvar95": step_dx_cvar95,
        "eval/step_dy_cvar95": step_dy_cvar95,
        "eval/step_dyaw_cvar95": step_dyaw_cvar95,
        # 커버리지
        "eval/step_count": {int(t): int(step_count[t]) for t in step_count},
    }

    return (test_loss, dx_loss, dy_loss, dyaw_loss, 
            dx_std2_loss, dy_std2_loss, dyaw_std2_loss, 
            dx_c95_loss, dy_c95_loss, dyaw_c95_loss, 
            dx_values, dy_values, dyaw_values, 
            per_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Simulation experiment')
    parser.add_argument('--batch-size', type=int, default=512) # num_world
    # EXPERIMENT
    parser.add_argument('--model-path', '-mp', type=str, default='/data/full_version/model/exp_20000')
    parser.add_argument('--model-name', '-mn', type=str, default='early_attn_s42_0808_044406.pth') # early_attn_s11_0808_043910
    parser.add_argument('--dataset', '-d', type=str, default='validation', choices=['training', 'validation'])
    args = parser.parse_args()
    bc_policy = torch.load(f"{args.model_path}/{args.model_name}", weights_only=False).to("cuda")
    bc_policy.eval()
    eval_data_path = os.path.join('/data/full_version/processed', 'final')
    eval_data_file =  f"unseen_validation_trajectory_2500.npz"
    with np.load(os.path.join(eval_data_path, eval_data_file), mmap_mode='r') as npz:
        expert_obs = npz['obs']
        expert_actions = npz['actions']
        expert_masks = npz['dead_mask'] if 'dead_mask' in npz.keys() else None
        partner_mask = npz['partner_mask'] if 'partner_mask' in npz.keys() else None
        road_mask = npz['road_mask'] if 'road_mask' in npz.keys() else None
    dataset = ExpertDataset(
        expert_obs, expert_actions, expert_masks, partner_mask, road_mask,
        rollout_len=5, pred_len=1, aux_future_step=None,
        ego_global_pos=None, ego_global_rot=None
    )
    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        shuffle=False,
        num_workers=8,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True
    )
    del dataset
    test_losses = evaluate(dataloader, bc_policy)
    (test_loss, dx_loss, dy_loss, dyaw_loss,
    dx_std2_loss, dy_std2_loss, dyaw_std2_loss,
    dx_c95_loss, dy_c95_loss, dyaw_c95_loss,
    dx_vals, dy_vals, dyaw_vals,
    per_step) = test_losses

    log_dict = {
        "eval/loss": test_loss,
        "eval/dx_loss": dx_loss, "eval/dy_loss": dy_loss, "eval/dyaw_loss": dyaw_loss,
        "eval/dx_std2_loss": dx_std2_loss, "eval/dy_std2_loss": dy_std2_loss, "eval/dyaw_std2_loss": dyaw_std2_loss,
        "eval/dx_cval_95": dx_c95_loss, "eval/dy_cval_95": dy_c95_loss, "eval/dyaw_cval_95": dyaw_c95_loss,
        "eval/dx_values": dx_vals, "eval/dy_values": dy_vals, "eval/dyaw_values": dyaw_vals,
    }
    print(f'model {args.model_name}, log {log_dict}')
    print("per-step means:", per_step)