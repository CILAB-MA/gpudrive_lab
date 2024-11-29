# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from networks.perm_eq_late_fusion import LateFusionNet
from algorithms.il.model.bc_utils.wayformer import MultiHeadAttention, PerceiverEncoder
from algorithms.il.model.bc_utils.gmm import GMM

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

class ContFeedForward(LateFusionNet):
    def __init__(self,  env_config, exp_config, loss='l1', num_stack=5):
        super(ContFeedForward, self).__init__(None, env_config, exp_config)
        self.num_stack = num_stack

        self.nn = self._build_network(
            input_dim=(self.ego_input_dim + self.ro_input_dim * self.ro_max + self.rg_input_dim * self.rg_max) * num_stack,
            net_arch=exp_config.feedforward.hidden_size,
        )

        self.loss_func = loss
        
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.arch_shared_net[0] = exp_config.feedforward.hidden_size[-1]
            self.head = ContHead(
                hidden_size=exp_config.feedforward.hidden_size[-1],
                net_arch=self.arch_shared_net
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=exp_config.feedforward.hidden_size[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
            )
        elif loss == 'gmm':
            self.head = GMM(
                input_dim=exp_config.feedforward.hidden_size[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
                n_components=exp_config.gmm.n_components
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")

    def get_embedded_obs(self, obs):
        """Get the embedded observation."""
        return self.nn(obs)
    
    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        out = self.get_embedded_obs(obs)
        actions = self.head(out, deterministic)
        return actions

class LateFusionBCNet(LateFusionNet):
    def __init__(self, env_config, exp_config, loss='l1', num_stack=5):
        super(LateFusionBCNet, self).__init__(None, env_config, exp_config)
        self.num_stack = num_stack
        
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
            net_arch=self.arch_ego_state,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * num_stack,
            net_arch=self.arch_road_objects,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
            net_arch=self.arch_road_graph,
        )

        self.loss_func = loss
        
        # Action head
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(
                hidden_size=self.shared_net_input_dim,
                net_arch=self.arch_shared_net
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=exp_config.feedforward.hidden_size[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
            )
        elif loss == 'gmm':
            self.head = GMM(
                input_dim=self.shared_net_input_dim,
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
                n_components=exp_config.gmm.n_components
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")
   
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
    
    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ):
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def get_embedded_obs(self, obs):
        """Get the embedded observation."""
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=5)
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        embedding_vector = torch.cat((ego_state, road_objects, road_graph), dim=1)
        return embedding_vector

    def forward(self, obss):
        # Unpack observation
        embedding_vector = self.get_embedded_obs(obss)
        actions = self.head(embedding_vector)
        
        return actions
    
class LateFusionAttnBCNet(LateFusionNet):
    def __init__(self, env_config, exp_config, loss='l1', num_stack=5):
        super(LateFusionAttnBCNet, self).__init__(None, env_config, exp_config)
        self.num_stack = num_stack
        
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
            net_arch=self.arch_ego_state,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * num_stack,
            net_arch=self.arch_road_objects,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
            net_arch=self.arch_road_graph,
        )
        
        # Attention
        self.ro_attn = MultiHeadAttention(
            num_heads=4,
            num_q_input_channels=self.arch_road_objects[-1],
            num_kv_input_channels=self.arch_road_objects[-1],
        )
        self.rg_attn = MultiHeadAttention(
            num_heads=4,
            num_q_input_channels=self.arch_road_graph[-1],
            num_kv_input_channels=self.arch_road_graph[-1],
        )
        
        self.loss_func = loss
        
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.arch_shared_net[0] = exp_config.feedforward.hidden_size[-1]
            self.head = ContHead(
                hidden_size=exp_config.feedforward.hidden_size[-1],
                net_arch=self.arch_shared_net
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=exp_config.feedforward.hidden_size[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
            )
        elif loss == 'gmm':
            self.head = GMM(
                input_dim=self.shared_net_input_dim,
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
                n_components=exp_config.gmm.n_components
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")
    
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
    
    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ):
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def get_embedded_obs(self, obs):
        """Get the embedded observation."""
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=5)
        
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_objects_attn = self.ro_attn(road_objects, road_objects)
        
        road_graph = self.road_graph_net(road_graph)
        road_graph_attn = self.ro_attn(road_graph, road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.avg_pool1d(
            road_objects_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.avg_pool1d(
            road_graph_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        embedding_vector = torch.cat((ego_state, road_objects, road_graph), dim=1)
        return embedding_vector

    def forward(self, obss, attn_weights=False):
        embedding_vector = self.get_embedded_obs(obss)
        actions = self.head(embedding_vector)
        
        return actions

class WayformerEncoder(LateFusionBCNet):
    def __init__(self, env_config, exp_config, loss='l1', num_stack=5):
        super(WayformerEncoder, self).__init__(None, env_config, exp_config, num_stack)
        self.num_stack = num_stack
        
         # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
            net_arch=self.arch_ego_state,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * num_stack,
            net_arch=self.arch_road_objects,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
            net_arch=self.arch_road_graph,
        )
        
        self.encoder = PerceiverEncoder(64, 64)
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self.ro_max + 1), 64)),
            requires_grad=True
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 91, 1, 64)),
            requires_grad=True
        )

        self.loss_func = loss
        
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.arch_shared_net[0] = exp_config.feedforward.hidden_size[-1]
            self.head = ContHead(
                hidden_size=exp_config.feedforward.hidden_size[-1],
                net_arch=self.arch_shared_net
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=exp_config.feedforward.hidden_size[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
            )
        elif loss == 'gmm':
            self.head = GMM(
                input_dim=self.shared_net_input_dim,
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
                n_components=exp_config.gmm.n_components
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")
        
    def _unpack_obs(self, obs_flat, timestep=91):
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, timestep,  ego_size + ro_size + rg_size)
        ego_stack = obs_flat_unstack[..., :ego_size].view(-1, timestep, self.ego_input_dim).reshape(-1, self.ego_input_dim)
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, timestep, self.ro_max, self.ro_input_dim)
            .reshape(-1, self.ro_max, self.ro_input_dim)
         )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, timestep, self.rg_max, self.rg_input_dim)
            .reshape(-1, self.rg_max, self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack
    
    def get_embedded_obs(self, obs, mask):
        # TODO: Implement function using mask
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obs)
        batch_size = obs.shape[0]
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        ego_state = ego_state.reshape(batch_size, -1, 64)
        road_objects = road_objects.reshape(batch_size, -1, 64)
        road_graph = road_graph.reshape(batch_size, -1, 64)

        embedding_vector = torch.cat((ego_state, road_objects, road_graph), dim=1)
        
        context = self.encoder(embedding_vector) 
        context = context.reshape(batch_size, -1)
        
        return context    

    def forward(self, obss, mask):
        context = self.get_embedded_obs(obss, mask)
        actions = self.head(context)
        
        return actions
