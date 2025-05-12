import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def aux_loss(model, context, expert_actions, masks=None, aux_info=None):
    '''
    compute the l1 loss between the predicted and expert actions
    TODO: should be fixed with new version
    '''
    aux_task, attn_weights, aux_style = aux_info
    partner_masks = masks
    pred_actions = model.aux_head(context, partner_masks)

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

def gmm_loss(model, context, expert_actions):
    '''
    compute the gmm loss between the predicted and expert actions
    '''
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
    return loss.mean(), loss.clone()

def l1_loss(model, context, expert_actions):
    pred_actions = model.get_action(context, deterministic=True)
    loss = F.smooth_l1_loss(pred_actions, expert_actions)
    return loss, loss.clone()

def focal_loss(model, context, expert_actions, gamma=2.0):
    pred_actions = model.get_action(context, deterministic=True)
    diff = torch.abs(pred_actions - expert_actions)
    weight = (1 - torch.exp(-diff)) ** gamma
    return (weight * diff ** 2).mean()