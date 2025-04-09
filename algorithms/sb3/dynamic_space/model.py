import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from abc import ABC, abstractmethod
import torch.distributions as dist

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class ContHead(nn.Module):
    def __init__(self, input_dim, head_config):
        super(ContHead, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, head_config.head_dim),
            nn.ReLU()
        )
        self.dx_head =nn.Sequential(
                nn.Linear(head_config.head_dim, 1),
            )
        self.dy_head =nn.Sequential(
                nn.Linear(head_config.head_dim, 1),
            )
        self.dyaw_head =nn.Sequential(
                nn.Linear(head_config.head_dim, 1),
            )

    def forward(self, x, deterministic=None):
        # TODO: residual dx, dy, dyaw block (to do fair comparison with GMM, Dist)
        x = self.input_layer(x)
        dx = self.dx_head(x)
        dy = self.dy_head(x)
        dyaw = self.dyaw_head(x)
        mu = torch.cat([dx, dy, dyaw], dim=-1)
        return mu

class LateFusionBCNet(nn.Module):
    def __init__(self, env_config, exp_config, head_config):
        super(LateFusionBCNet, self).__init__()
        self.num_stack = exp_config.num_stack

        self.ego_input_dim = 6
        self.ro_input_dim = 10
        self.rg_input_dim = 13
        self.hidden_dim = 128
        self.hidden_num = 2
        self.dropout = 0.0
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim
        )

        self.mu_head = ContHead(
            input_dim=128 * 3,
            head_config=head_config,
        )
        self.std_head = ContHead(
            input_dim=128 * 3,
            head_config=head_config,
        )
        # self.mu_head = GMM(
        #     network_type=self.__class__.__name__,
        #     input_dim=128 * 3,
        #     head_config=head_config,
        #     time_dim=1,
        # )

        # self.std_head = GMM(
        #     network_type=self.__class__.__name__,
        #     input_dim=128 * 3,
        #     head_config=head_config,
        #     time_dim=1,
        # )
        self.ro_max = 127
        self.rg_max = 200
    @abstractmethod
    def _build_network(self, input_dim: int) -> nn.Module:
        """Build a network with the specified architecture."""
        init_ = lambda m: init(m, nn.init.xavier_normal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        layers = []
        prev_dim = input_dim
        net_arch = [self.hidden_dim] * self.hidden_num
        for layer_dim in net_arch:
            layers.append(init_(nn.Linear(prev_dim, layer_dim)))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(nn.ReLU())
            prev_dim = layer_dim
        
        network = nn.Sequential(*layers)
        network.last_layer_dim = prev_dim

        return network



    def _unpack_obs(self, obs_flat, num_stack=1):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_staye, road_objects, stop_signs, road_graph (torch.Tensor).
            ego_stayawe, road_objects, stop_signs, road_graph (torch.Tensor).
            ego_state, rodx, dy, dyawobjects, stop_signs, road_graph (torch.Tensor).
        """
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, num_stack,  ego_size + ro_size + rg_size)
        ego_stack = obs_flat_unstack[..., :ego_size].view(-1, num_stack, self.ego_input_dim).reshape(-1, num_stack * self.ego_input_dim)
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, num_stack, self.ro_max, self.ro_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, ro_max, num_stack, ro_input_dim)
            .reshape(-1, self.ro_max, num_stack * self.ro_input_dim)
        )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, num_stack, self.rg_max, self.rg_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, rg_max, num_stack, rg_input_dim)
            .reshape(-1, self.rg_max, num_stack * self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack

    def forward(self, obs, deterministic=False):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obs)
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        context = torch.cat((ego_state, road_objects, road_graph), dim=1)
        
        mu = self.mu_head(context, deterministic)
        std = self.std_head(context, deterministic)
        return mu, std

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