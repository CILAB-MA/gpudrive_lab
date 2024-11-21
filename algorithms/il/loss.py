import torch


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

def two_hot_loss(pred, targ, dx_bins, dy_bins, dyaw_bins):
    '''
    pred: real value of model output
    targ: real value of label
    dx_bins: 
    '''
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

def gmm_loss(self, obs, expert_actions):
        # todo: move gmm loss to here
        # means, covariances, weights = self.forward(obs)
        log_probs = []

        for i in range(self.gmm.n_components):
            mean = means[:, i, :]
            cov_diag = covariances[:, i, :]
            gaussian = dist.MultivariateNormal(mean, torch.diag_embed(cov_diag))
            log_probs.append(gaussian.log_prob(expert_actions))

        log_probs = torch.stack(log_probs, dim=1)
        weighted_log_probs = log_probs + torch.log(weights + 1e-8)
        loss = -torch.logsumexp(weighted_log_probs, dim=1)
        return loss.mean()
