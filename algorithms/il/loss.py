import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

def l1_loss(model, obs, expert_actions):
    '''
    compute the l1 loss between the predicted and expert actions
    '''
    pred_actions = model(obs)
    loss = F.smooth_l1_loss(pred_actions, expert_actions)
    return loss

def mse_loss(model, obs, expert_actions):
    '''
    Compute the mean squared error loss between the predicted and expert actions
    '''
    pred_actions = model(obs)
    loss = F.mse_loss(pred_actions, expert_actions)
    return loss

def two_hot_loss(model, obs, expert_actions):
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
    
    pred = model(obs)
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

def nll_loss(model, obs, expert_actions):
    embedding_vector = model.get_embedded_obs(obs)
    means, log_std = model.head.get_dist_params(embedding_vector)
    stds = torch.exp(log_std)

    gaussian = Normal(means, stds)
    
    scale_factor = torch.tensor([6.0, 6.0, np.pi], device=expert_actions.device)
    squash_expert_actions = expert_actions / scale_factor
    squash_expert_actions = torch.clamp(squash_expert_actions, -1 + 1e-6, 1 - 1e-6)
    
    unsquash_expert_actions = torch.atanh(squash_expert_actions)
    
    log_probs = gaussian.log_prob(unsquash_expert_actions)
    log_probs -= torch.log(1 - squash_expert_actions.pow(2) + 1e-6)
    
    loss = -log_probs.sum(dim=-1)

    return loss.mean()

def gmm_loss(model, obs, expert_actions):
    '''
    compute the gmm loss between the predicted and expert actions
    '''
    embedding_vector = model.get_embedded_obs(obs)
    means, covariances, weights, components = model.head.get_gmm_params(embedding_vector)
    
    # Rescaling actions and resquash
    scale_factor = torch.tensor([6.0, 6.0, np.pi], device=expert_actions.device)
    squash_expert_actions = expert_actions / scale_factor
    squash_expert_actions = torch.clamp(squash_expert_actions, -1 + 1e-6, 1 - 1e-6)
    
    unsquash_expert_actions = torch.atanh(squash_expert_actions)
    
    log_probs = []

    for i in range(components):
        mean = means[:, i, :]
        cov_diag = covariances[:, i, :]
        gaussian = MultivariateNormal(mean, torch.diag_embed(cov_diag))
        log_probs.append(gaussian.log_prob(unsquash_expert_actions))

    log_probs = torch.stack(log_probs, dim=1)
    weighted_log_probs = log_probs + torch.log(weights + 1e-8) + torch.log(1 - squash_expert_actions**2 + 1e-6).sum(dim=1, keepdim=True)
    loss = -torch.logsumexp(weighted_log_probs, dim=1)
    return loss.mean()
