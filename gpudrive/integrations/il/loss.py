import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np

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
    return loss.mean(), loss.detach()

def l1_loss(model, context, expert_actions):
    pred_actions = model.get_action(context, deterministic=True)
    loss = F.smooth_l1_loss(pred_actions, expert_actions)
    return loss, loss.detach()

def focal_loss(model, context, expert_actions, alpha=1.0, gamma=2.0, eps=1e-6):
    pred_actions = model.get_action(context, deterministic=True)
    diff = torch.abs(pred_actions - expert_actions)
    weight = (diff + eps) ** gamma
    loss = (alpha * weight * diff ** 2).mean()
    return loss, loss.detach()