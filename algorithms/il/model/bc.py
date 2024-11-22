# Define network
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F
from typing import List

from networks.perm_eq_late_fusion import LateFusionNet
from algorithms.il.model.bc_utils.wayformer import MultiHeadAttention, PerceiverEncoder
from algorithms.il.model.bc_utils.gmm import GMM

class ContHead(nn.Module):
    def __init__(self, hidden_size, net_arch):
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
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
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
    
class ContFeedForward(LateFusionNet):
    def __init__(self, observation_space, env_config, exp_config, num_stack=5,
                 loss='l1'):
        super(ContFeedForward, self).__init__()

        self.nn = self._build_network(
            input_dim=(self.ego_input_dim + self.ro_input_dim + self.rg_input_dim) * num_stack,
            net_arch=exp_config.hidden_size,
        )

        self.loss_func = loss
        self.arch_shared_net[0] = exp_config.hidden_size[-1]
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(exp_config.hidden_size[-1], self.arch_shared_net)
        elif loss == 'gmm':
            pass
        elif loss == 'dist':
            pass
        else:
            raise ValueError(f"Loss name {loss} is not supported")

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        out = self.nn(obs)
        actions = self.head(obs, deterministic)
        return actions

class LateFusionBCNet(LateFusionNet):
    def __init__(self, env_config, net_config, head):
        super(LateFusionBCNet, self).__init__(None, env_config, net_config)
        num_stack = env_config.num_stack
        
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

        # Action head
        self.head = head

        
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
    
    def forward(self, obss):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss, num_stack=5)
        
        # Embed features
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

        actions = self.head(embedding_vector)
        
        return actions

class LateFusionAttnBCNet(LateFusionNet):
    def __init__(self,  observation_space, env_config, exp_config, num_stack=5):
        super(LateFusionAttnBCNet, self).__init__(observation_space, env_config, exp_config)
        
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
        
        # Action head
        self.dx_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )
        self.dy_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )
        self.dyaw_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )
    
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
    
    def forward(self, obss, attn_weights=False):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss, num_stack=5)
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

        dx = self.dx_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        dy = self.dy_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        dyaw = self.dyaw_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        actions = torch.cat([dx, dy, dyaw], dim=-1)
        return actions

class LateFusionGmmBCNet(LateFusionNet):
    def __init__(self,  observation_space, env_config, exp_config, num_stack=5):
        super().__init__(observation_space, env_config, exp_config)

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

        # Action head
        self.gmm = GMM(
            input_dim=self.shared_net_input_dim,
            hidden_dim=self.net_config.hidden_dim,
            action_dim=self.net_config.action_dim,
            n_components=self.net_config.n_components
        )
    
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
    
    def forward(self, obss):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss, num_stack=5)
        
        # Embed features
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

        means, covariances, weights = self.gmm(embedding_vector)
        n_components = self.net_config.n_components
        return means, covariances, weights, n_components
    
    def sample(self, obs):
        means, covariances, weights = self.forward(obs)
        
        # Sample component indices based on weights
        component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        
        sampled_means = means[torch.arange(obs.size(0)), component_indices]
        sampled_covariances = covariances[torch.arange(obs.size(0)), component_indices]
        
        # Sample actions from the chosen component's Gaussian
        actions = dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        return actions

class LateFusionAttnGmmBCNet(LateFusionNet):
    def __init__(self,  observation_space, env_config, exp_config, num_stack=5):
        super().__init__(observation_space, env_config, exp_config)
        
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

        # GMM
        self.gmm = GMM(
            input_dim=self.shared_net_input_dim,
            hidden_dim=self.net_config.hidden_dim,
            action_dim=self.net_config.action_dim,
            n_components=self.net_config.n_components
        )

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
    
    def forward(self, obss):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss, num_stack=5)
        
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        
        road_objects = self.road_object_net(road_objects)
        road_objects_attn = self.ro_attn(road_objects, road_objects)
        
        road_graph = self.road_graph_net(road_graph)
        road_graph_attn = self.rg_attn(road_graph, road_graph)

        # Average pooling across the object dimension
        # (M, E) -> (1, E) (avg pool across features)
        road_objects = F.avg_pool1d(
            road_objects_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.avg_pool1d(
            road_graph_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        embedding_vector = torch.cat((ego_state, road_objects, road_graph), dim=1)

        means, covariances, weights = self.gmm(embedding_vector)
        n_components = self.net_config.n_components
        
        return means, covariances, weights, n_components
    
    def sample(self, obs):
        means, covariances, weights = self.forward(obs)
        
        # Sample component indices based on weights
        component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        
        sampled_means = means[torch.arange(obs.size(0)), component_indices]
        sampled_covariances = covariances[torch.arange(obs.size(0)), component_indices]
        
        # Sample actions from the chosen component's Gaussian
        actions = dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        return actions

class WayformerEncoder(LateFusionBCNet):
    def __init__(self,  observation_space, env_config, exp_config,
                 num_stack=1):
        super(WayformerEncoder, self).__init__(observation_space, env_config, exp_config)

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
        self.gmm = GMM(
            input_dim=64,
            hidden_dim=self.net_config.hidden_dim,
            action_dim=self.net_config.action_dim,
            n_components=self.net_config.n_components
        )
    
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
    
    def forward(self, obss, mask):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss)
        batch_size = obss.shape[0]
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
        means, covariances, weights = self.gmm(context)
        return means, covariances, weights

    def compute_loss(self, obs, expert_actions):
        means, covariances, weights = self.forward(obs)
       
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
    
    def sample(self, obs):
        means, covariances, weights = self.forward(obs)
        
        # Sample component indices based on weights
        component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        
        sampled_means = means[torch.arange(obs.size(0)), component_indices]
        sampled_covariances = covariances[torch.arange(obs.size(0)), component_indices]
        
        # Sample actions from the chosen component's Gaussian
        actions = dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        return actions


if __name__ == "__main__":
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    from baselines.ippo.config import ExperimentConfig
    from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
    import pyrallis
    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_CONTROLLED_AGENTS = 128
    NUM_WORLDS = 1

    env_config = EnvConfig(dynamics_model="delta_local")
    render_config = RenderConfig()
    scene_config = SceneConfig("data/processed/examples", NUM_WORLDS)

    # MAKE ENVIRONMENT
    kwargs = {
        "config": env_config,
        "scene_config": scene_config,
        "max_cont_agents": MAX_CONTROLLED_AGENTS,
        "device": "cuda", 
        "num_stack": 5
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)
    obs = env.reset()
    exp_config = pyrallis.parse(config_class=ExperimentConfig)
    bc_policy = LateFusionAttnBCNet(
        observation_space=None,
        exp_config=exp_config,
        env_config=env_config
    ).to("cuda")
    actions = bc_policy(obs.squeeze(0))
    dead_agent_mask = ~env.cont_agent_mask.clone()
    actions = actions.unsqueeze(0)
    print(actions.shape)
    for _ in range(5):
        env.step_dynamics(actions, use_indices=False)
        obs = env.get_obs()
