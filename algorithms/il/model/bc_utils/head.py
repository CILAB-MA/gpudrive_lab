import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from typing import List


class ContHead(nn.Module):
    def __init__(self, hidden_size, net_arch):
        super(ContHead, self).__init__()
        self.dx_head = self._build_out_network(hidden_size, 1, net_arch)
        self.dy_head = self._build_out_network(hidden_size, 1, net_arch)
        self.dyaw_head = self._build_out_network(hidden_size, 1, net_arch)

    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ):
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(0.0))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)
    
    def forward(self, x, deterministic=None):
        dx = self.dx_head(x)
        dy = self.dy_head(x)
        dyaw = self.dyaw_head(x)
        actions = torch.cat([dx, dy, dyaw], dim=-1)
        return actions
    
class DistHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, action_dim=3):
        super(DistHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(4)
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
            x += residual
        
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
    def __init__(self, network_type, input_dim, hidden_dim=128, action_dim=3, n_components=10, time_dim=1):
        super(GMM, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(4)
        ])
        
        self.head = nn.Linear(hidden_dim, n_components * (2 * action_dim + 1))
        self.n_components = n_components
        self.action_dim = action_dim
        self.time_dim = time_dim
        self.network_type = network_type

    def get_gmm_params(self, x):
        """
        Get the parameters of the Gaussian Mixture Model
        """
        x = x.reshape(x.size(0), self.time_dim, x.size(-1))
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x += residual
        
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
        actions = actions.squeeze(1) if self.network_type != 'WayformerEncoder' else actions
        
        # Squash actions and scaling
        actions = torch.tanh(actions)
        scale_factor = torch.tensor([6.0, 6.0, np.pi], device=actions.device)
        actions = scale_factor * actions

        return actions
