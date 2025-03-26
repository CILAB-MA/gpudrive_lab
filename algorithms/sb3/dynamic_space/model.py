import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod
from pygpudrive.env import constants

def init(module, weight_init, bias_init, gain=1):
    '''
    This function provides weight and bias initializations for linear layers.
    '''
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class LateFusionBCNet(nn.Module):
    def __init__(self, env_config, exp_config):
        super(LateFusionBCNet, self).__init__()

        # Unpack feature dimensions
        self.ego_input_dim = constants.EGO_FEAT_DIM
        self.ro_input_dim = constants.PARTNER_FEAT_DIM
        self.rg_input_dim = constants.ROAD_GRAPH_FEAT_DIM
        self.hidden_dim = 128
        self.hidden_num = 2
        self.ro_max = 127
        self.rg_max = 200
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

        self.mu_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )

        self.std_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )

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
        out = self.actor_out_net(
            torch.cat((ego_state, road_objects, road_graph), dim=1)
        )
        mu = self.mu_head(out)
        std = self.std_head(out)
        return mu, std