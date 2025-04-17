import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def aux_loss(model, context, expert_actions, masks=None, aux_info=None):
    '''
    compute the l1 loss between the predicted and expert actions
    '''
    if aux_info:
        aux_task, attn_weights, aux_style = aux_info
    if aux_task == 'action':
        partner_masks = masks
        pred_actions = model.aux_action_head(context, partner_masks)
        expert_actions[..., :2] /= 6 
        expert_actions[..., 2] /= np.pi 
    elif aux_task == 'pos':
        partner_masks = masks
        pred_actions = model.aux_head(context, partner_masks)
    elif aux_task == 'heading':
        partner_masks = masks
        pred_actions = model.aux_heading_head(context, partner_masks)
    elif aux_task == 'speed':
        partner_masks = masks
        pred_actions = model.aux_speed_head(context, partner_masks)
    else:
        pred_actions = model.get_action(context)

    # if pred_actions.shape[-1] == 1:
    #     expert_actions = expert_actions.unsqueeze(-1)
    
    pred_actions = pred_actions[~partner_masks]
    expert_actions = expert_actions[~partner_masks]
    loss = F.cross_entropy(pred_actions, expert_actions, reduction='none')
    if 'no_weight' not in aux_style:
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        count_pos = (attn_weights > 0).sum(dim=-1, keepdim=True).float()
        count_pos_safe = count_pos + 1e-6
        attn_weights_scaled = attn_weights * count_pos_safe
        masked_weights = attn_weights_scaled[~partner_masks]
        weighted_mse = loss * masked_weights.unsqueeze(-1)
    else:
        weighted_mse = loss
    loss = weighted_mse.sum() / len(pred_actions)
    return loss


def l1_loss(model, context, expert_actions, masks=None, aux_info=None):
    '''
    compute the l1 loss between the predicted and expert actions
    '''
    if aux_info:
        aux_task, attn_weights, aux_style = aux_info
    if aux_task == 'action':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_action_head(context, partner_masks)
        expert_actions[..., :2] /= 6 
        expert_actions[..., 2] /= np.pi 
    elif aux_task == 'pos':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_pos_head(context, partner_masks)
    elif aux_task == 'heading':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_heading_head(context, partner_masks)
    elif aux_task == 'speed':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_speed_head(context, partner_masks)
    else:
        pred_actions = model.get_action(context)

    if pred_actions.shape[-1] == 1:
        expert_actions = expert_actions.unsqueeze(-1)
    
    pred_actions = pred_actions[~partner_masks]
    expert_actions = expert_actions[~partner_masks]
    loss = F.smooth_l1_loss(pred_actions, expert_actions, reduction='none')
    if 'no_weight' not in aux_style:
        attn_weights_scaled = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        masked_weights = attn_weights_scaled[~partner_masks]
        weighted_mse = loss * masked_weights.unsqueeze(-1)
    else:
        weighted_mse = loss
    loss = weighted_mse.sum() / len(pred_actions)
    return loss

def mse_loss(model, context, expert_actions, masks=None, aux_info=None):
    '''
    Compute the mean squared error loss between the predicted and expert actions
    '''
    if aux_info:
        aux_task, attn_weights, aux_style = aux_info
    if aux_task == 'action':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_action_head(context, partner_masks)
        expert_actions[..., :2] /= 6 
        expert_actions[..., 2] /= np.pi 
    elif aux_task == 'pos':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_pos_head(context, partner_masks)
    elif aux_task == 'heading':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_heading_head(context, partner_masks)
    elif aux_task == 'speed':
        partner_masks = masks[:, -1]
        pred_actions = model.aux_speed_head(context, partner_masks)
    else:
        pred_actions = model.get_action(context)

    if pred_actions.shape[-1] == 1:
        expert_actions = expert_actions.unsqueeze(-1)
    
    pred_actions = pred_actions[~partner_masks]
    if expert_actions.size(1) == 1:
        expert_actions = expert_actions.squeeze(1)
    expert_actions = expert_actions[~partner_masks]
    loss = F.smooth_l1_loss(pred_actions, expert_actions, reduction='none')
    if 'no_weight' not in aux_style:
        attn_weights_scaled = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        masked_weights = attn_weights_scaled[~partner_masks]
        weighted_mse = loss * masked_weights.unsqueeze(-1)
    else:
        weighted_mse = loss
    loss = weighted_mse.sum() / len(pred_actions)
    return loss

def two_hot_loss(model, context, expert_actions, masks=None):
    '''
    Compute the two hot loss between the predicted and expert actions
    '''
    def two_hot_encoding(value, bins):
        idx_upper = torch.searchsorted(bins, value, right=True).clamp(max=len(bins) - 1)
        idx_lower = torch.clamp(idx_upper - 1, min=0)
        
        lower_weight = (value - bins[idx_lower]) / (bins[idx_upper] - bins[idx_lower])
        upper_weight =  (bins[idx_upper] - value) / (bins[idx_upper] - bins[idx_lower])
        batch_indices = torch.arange(len(value), device=value.device)
        two_hot = torch.zeros(len(value), len(bins), device=value.device)
        two_hot[batch_indices, idx_lower] = lower_weight
        two_hot[batch_indices, idx_upper] = upper_weight
        
        return two_hot
    
    pred = model.get_action(context)
    targ = expert_actions
    dx_bins = model.config.dx
    dy_bins = model.config.dy
    dyaw_bins = model.config.dyaw
    
    pred_dist = torch.zeros(len(pred), len(dx_bins), 3,  device=pred.device)
    targ_dist = torch.zeros(len(targ), len(dx_bins), 3, device=pred.device)
    pred_dist[..., 0] = two_hot_encoding(bins=dx_bins, value=pred[:, 0] )
    pred_dist[..., 1] = two_hot_encoding(bins=dy_bins, value=pred[:, 1] )
    pred_dist[..., 2] = two_hot_encoding(bins=dyaw_bins, value=pred[:, 2] )

    targ_dist[..., 0] = two_hot_encoding(bins=dx_bins, value=targ[:, 0] )
    targ_dist[...,1] = two_hot_encoding(bins=dy_bins, value=targ[:, 1] )
    targ_dist[...,2] = two_hot_encoding(bins=dyaw_bins, value=targ[:, 2] )
    epsilon = 1e-8
    log_targ_dist = torch.log(targ_dist + epsilon)

    loss_dx = (pred_dist[..., 0] * log_targ_dist[..., 0]).sum(dim=-1).mean()
    loss_dy = (pred_dist[..., 1] * log_targ_dist[..., 1]).sum(dim=-1).mean()
    loss_dyaw = (pred_dist[..., 2] * log_targ_dist[..., 2]).sum(dim=-1).mean()

    total_loss = (loss_dx + loss_dy + loss_dyaw) / 3

    return total_loss

def nll_loss(model, context, expert_actions, masks=None):
    means, log_std = model.head.get_dist_params(context)
    stds = torch.exp(log_std)

    gaussian = Normal(means, stds)
    log_probs = gaussian.log_prob(expert_actions)
    
    loss = -log_probs.sum(dim=-1)
    return loss.mean()

def gmm_loss(model, context, expert_actions, masks=None, aux_head=None):
    '''
    compute the gmm loss between the predicted and expert actions
    '''
    if aux_head == 'action':
        means, covariances, weights, components = model.aux_action_head.get_gmm_params(context)
    elif aux_head == 'goal':
        means, covariances, weights, components = model.aux_goal_head.get_gmm_params(context)
    else:
        means, covariances, weights, components = model.head.get_gmm_params(context)    
    
    expert_actions = expert_actions.unsqueeze(1) if expert_actions.dim() == 2 else expert_actions    
    
    log_probs = []
    for i in range(components):
        mean = means[..., i, :]
        cov_diag = covariances[..., i, :]
        gaussian = MultivariateNormal(mean, torch.diag_embed(cov_diag))
        log_probs.append(gaussian.log_prob(expert_actions))

    log_probs = torch.stack(log_probs, dim=-1)
    weighted_log_probs = log_probs + torch.log(weights + 1e-8)
    loss = -torch.logsumexp(weighted_log_probs, dim=-1)

    mask, _, partner_masks, _ = masks
    if aux_head != None:
        mask = partner_masks[:, -1]
    else:
        mask = mask.unsqueeze(-1)
    loss = loss[mask > 0] 
    return loss.mean(), loss.clone()

def new_gmm_loss(model, context, expert_actions, masks=None, aux_head=None):
    '''
    compute the gmm loss between the predicted and expert actions
    '''
    if aux_head == 'action':
        means, log_std, rho, pred_scores = model.aux_action_head.get_gmm_params(context)
    elif aux_head == 'goal':
        means, log_std, rho, pred_scores = model.aux_goal_head.get_gmm_params(context)
    else:
        means, log_std, rho, pred_scores = model.head.get_gmm_params(context) # (B, T, C, 3), (B, T, C, 3), (B, T, C, 3), (B, C)
    
    # Rescaling actions and resquash
    expert_actions = expert_actions.unsqueeze(1) if expert_actions.dim() == 2 else expert_actions
    
    mask, _, partner_masks, _ = masks
    distance = (means - expert_actions[:, :, None, :]).norm(dim=-1) # (B, T, C, 3) - (B, T, 1, 3) -> (B, T, C)
    distance = distance[mask].sum(dim=1) # (B, C)
    nearest_component_idxs = distance.argmin(dim=-1) # (B)
    nearest_component_bs_idxs = torch.arange(len(nearest_component_idxs)).type_as(nearest_component_idxs) # (B)
    
    nearest_trajs = means[nearest_component_bs_idxs, :, nearest_component_idxs] # (B, T, 3)
    ddx, ddy, ddyaw = (expert_actions - nearest_trajs).unbind(dim=-1) # (B, T, 3)
    log_std_dx, log_std_dy, log_std_dyaw = log_std[nearest_component_bs_idxs, :, nearest_component_idxs].unbind(dim=-1) # (B, T, 3)
    std_dx, std_dy, std_dyaw = torch.exp(log_std_dx), torch.exp(log_std_dy), torch.exp(log_std_dyaw)
    rho_dxdy, rho_dxdyaw, rho_dydyaw = rho[nearest_component_bs_idxs, :, nearest_component_idxs].unbind(dim=-1) # (B, T, 3)
    
    # Compute the gaussian mixture model loss
    gmm_log_coefficient = (
        log_std_dx + log_std_dy + log_std_dyaw +
        0.5 * torch.log(torch.clamp(1 - rho_dxdy**2 - rho_dxdyaw**2 - rho_dydyaw**2 + 2 * rho_dxdy * rho_dxdyaw * rho_dydyaw, min=1e-3))
    )
    
    gmm_exp = (
        0.5 / (1 - rho_dxdy**2 - rho_dxdyaw**2 - rho_dydyaw**2) * (
            ddx**2 / std_dx**2 + ddy**2 / std_dy**2 + ddyaw**2 / std_dyaw**2 -
            2 * rho_dxdy * ddx * ddy / (std_dx * std_dy) -
            2 * rho_dxdyaw * ddx * ddyaw / (std_dx * std_dyaw) -
            2 * rho_dydyaw * ddy * ddyaw / (std_dy * std_dyaw)
        )
    )
    
    reg_loss = (gmm_log_coefficient + gmm_exp)[mask].sum(dim=-1)
    cls_loss = F.cross_entropy(pred_scores, nearest_component_idxs, reduction='none')[mask]
    
    return (reg_loss + cls_loss).mean()