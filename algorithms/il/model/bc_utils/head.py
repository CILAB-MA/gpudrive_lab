import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

from typing import List


class ContHead(nn.Module):
    def __init__(self, input_dim, head_config):
        super(ContHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, head_config.head_dim),
            nn.ReLU()
        )
        self.dx_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_config.head_dim, head_config.head_dim),
                nn.ReLU(),
            ) for _ in range(head_config.head_num_layers)
        ])
        self.dy_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_config.head_dim, head_config.head_dim),
                nn.ReLU(),
            ) for _ in range(head_config.head_num_layers)
        ])
        self.dyaw_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_config.head_dim, head_config.head_dim),
                nn.ReLU(),
            ) for _ in range(head_config.head_num_layers)
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
    def __init__(self, input_dim, head_config):
        super(DistHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, head_config.head_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_config.head_dim, head_config.head_dim),
                nn.ReLU(),
                nn.Linear(head_config.head_dim, head_config.head_dim),
            ) for _ in range(head_config.head_num_layers)
        ])
        
        self.relu = nn.ReLU()
        self.mean = nn.Linear(head_config.head_dim, head_config.action_dim)
        self.log_std = nn.Linear(head_config.head_dim, head_config.action_dim)
    
    def get_dist_params(self, x):
        """
        Get the means, stds of the Dist Head
        """
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x = self.relu(x + residual)
        
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

        return actions

class GMM(nn.Module):
    def __init__(self, network_type, input_dim, head_config, time_dim=1):
        super(GMM, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, head_config.head_dim),
            nn.ReLU(),
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_config.head_dim, head_config.head_dim),
                nn.ReLU(),
            ) for _ in range(head_config.head_num_layers)
        ])
        self.relu = nn.ReLU()
        self.head = nn.Linear(head_config.head_dim, head_config.n_components * (2 * head_config.action_dim + 1))
        self.n_components = head_config.n_components
        self.action_dim = head_config.action_dim
        self.time_dim = time_dim
        self.clip_value = head_config.clip_value
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
            x = x + residual
        
        params = self.head(x)
        
        means = params[..., :self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        covariances = params[..., self.n_components * self.action_dim:2 * self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        weights = params[..., -self.n_components:].view(-1, self.time_dim, self.n_components)
        
        covariances = torch.clamp(covariances, self.clip_value, 3.58352)
        covariances = torch.exp(covariances)
        weights = torch.softmax(weights, dim=-1)
        self.component_probs = weights[0,0].detach() # To wandb log
        
        return means, covariances, weights, self.n_components

    def get_component_probs(self):
        return self.component_probs

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

        return actions

class NewGMM(nn.Module):
    def __init__(self, network_type, input_dim, head_config, time_dim=1):
        super(NewGMM, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, head_config.head_dim),
            nn.ReLU(),
            nn.LayerNorm(head_config.head_dim)
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(head_config.head_dim, head_config.head_dim),
                nn.ReLU(),
                nn.Linear(head_config.head_dim, head_config.head_dim),
            ) for _ in range(head_config.head_num_layers)
        ])
        self.relu = nn.ReLU()
        self.head = nn.Linear(head_config.head_dim, head_config.n_components * (3 * head_config.action_dim))
        def init(module, weight_init, bias_init, gain=1):
            '''
            This function provides weight and bias initializations for linear layers.
            '''
            weight_init(module.weight.data, gain=gain)
            bias_init(module.bias.data)
            return module
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        self.prob_predictor = nn.Sequential(init_(nn.Linear(head_config.head_dim, head_config.n_components))) # TODO: unitraj code is (head_dim, 1)
        self.n_components = head_config.n_components
        self.action_dim = head_config.action_dim
        self.time_dim = time_dim
        self.clip_value = head_config.clip_value
        self.network_type = network_type

    def get_gmm_params(self, x):
        """
        Get the parameters of the Gaussian Mixture Model
        """
        # TODO: x shape is (B, query_len, channel) and use (B, C, channel) in unitraj
        x = x.reshape(x.size(0), self.time_dim, x.size(-1))
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x = self.relu(x + residual)
        
        params = self.head(x)
        component_probs = self.prob_predictor(x).reshape(-1, self.n_components)
        self.component_probs = torch.softmax(component_probs[0].detach(), dim=-1) # To wandb log
        
        means = params[..., :self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        log_std = params[..., self.n_components * self.action_dim:2 * self.n_components * self.action_dim].view(-1, self.time_dim, self.n_components, self.action_dim)
        rho = params[..., 2 * self.n_components * self.action_dim:].view(-1, self.time_dim, self.n_components, self.action_dim)
        
        log_std = torch.clamp(log_std, self.clip_value, 5.0)
        rho = torch.clamp(rho, -0.5, 0.5)
        
        return means, log_std, rho, component_probs

    def get_component_probs(self):
        return self.component_probs

    def forward(self, x, deterministic=None):
        """
        Sample actions from the Gaussian Mixture Model
        """
        means, _, _, pred_scores = self.get_gmm_params(x) # (B, T, C, 3), (B, C)
        
        best_component_idx = torch.argmax(pred_scores, dim=-1)
        actions = means[torch.arange(means.size(0)), :, best_component_idx]
        
        return actions
