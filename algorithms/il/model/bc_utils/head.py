import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from typing import List


class ContHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_num):
        super(ContHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.dx_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(hidden_num)
        ])
        self.dy_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(hidden_num)
        ])
        self.dyaw_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(hidden_num)
        ])
    
    def forward(self, x, deterministic=None):
        # TODO: residual dx, dy, dyaw block (to do fair comparison with GMM, Dist)
        x = self.input_layer(x)
        dx = self.dx_head(x)
        dy = self.dy_head(x)
        dyaw = self.dyaw_head(x)
        actions = torch.cat([dx, dy, dyaw], dim=-1)
        return actions
    
class DistHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, hidden_num = 4, action_dim=3):
        super(DistHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(hidden_num)
        ])
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def get_dist_params(self, x):
        """
        Get the means, stds of the Dist Head
        """
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x = x + residual
        
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std
        
    def forward(self, x, deterministic=None):
        means, log_std = self.get_dist_params(x)
        stds = torch.exp(log_std)
        
        if deterministic:
            actions = means
        else:
            dist = torch.distributions.Normal(means, stds)
            actions = dist.rsample()

        squashed_actions = torch.tanh(actions)

        scaled_factor = torch.tensor([6.0, 6.0, np.pi], device=x.device)

        scaled_actions = scaled_factor * squashed_actions
        return scaled_actions

class GMM(nn.Module):
    def __init__(self, network_type, input_dim, hidden_dim=128, hidden_num=4, action_dim=3, n_components=10, time_dim=1, device='cuda'):
        super(GMM, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            ) for _ in range(hidden_num)
        ])
        self.relu = nn.ReLU()
        self.head = nn.Linear(hidden_dim, n_components * (2 * action_dim + 1))
        self.n_components = n_components
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.network_type = network_type
        if action_dim == 3:
            self.scale_factor = torch.tensor([6.0, 6.0, np.pi]).to(device)
        elif action_dim == 2:
            self.scale_factor = torch.tensor([1.0, 1.0]).to(device)

    def get_gmm_params(self, x):
        """
        Get the parameters of the Gaussian Mixture Model
        """
        x = x.reshape(x.size(0), self.time_dim, x.size(-1))
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x = self.relu(x + residual)
        
        params = self.head(x)
        
        means = params[..., :self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        covariances = params[..., self.n_components * self.action_dim:2 * self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        weights = params[..., -self.n_components:].view(-1, self.time_dim, self.n_components)
        
        covariances = torch.clamp(covariances, -20, 2)
        covariances = torch.exp(covariances)
        weights = torch.softmax(weights, dim=-1)
        
        return means, covariances, weights, self.n_components

    def forward(self, x, deterministic=None):
        """
        Sample actions from the Gaussian Mixture Model
        """
        means, covariances, weights, components = self.get_gmm_params(x)

        component_indices = torch.argmax(weights, dim=-1) if deterministic else dist.Categorical(weights).sample()
        component_indices = component_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.action_dim)
        
        sampled_means = torch.gather(means, 2, component_indices)
        sampled_covariances = torch.gather(covariances, 2, component_indices)
        
        actions = sampled_means if deterministic else dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        actions = actions.squeeze(2)
        
        # Squash actions and scaling
        actions = torch.tanh(actions)
        actions = self.scale_factor * actions

        return actions
