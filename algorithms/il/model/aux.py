# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from networks.perm_eq_late_fusion import LateFusionNet
from algorithms.il.model.bc_utils.wayformer import SelfAttentionBlock, PerceiverEncoder
from algorithms.il.model.bc_utils.head import *


class LateFusionAuxNet(LateFusionNet):
    def __init__(self, env_config, exp_config, loss='l1', num_stack=5, use_tom=None):
        super(LateFusionAuxNet, self).__init__(None, env_config, exp_config)
        self.num_stack = num_stack
        other_input_dim = self.ro_input_dim * num_stack
        # Aux head
        self.use_tom = use_tom
        if use_tom == 'aux_head':
            self.aux_action_head = GMM(
                network_type=self.__class__.__name__,
                input_dim=self.arch_road_graph[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=exp_config.gmm.action_dim,
                n_components=exp_config.gmm.n_components,
                time_dim=self.ro_max
            )
            self.aux_goal_head = GMM(
                network_type=self.__class__.__name__,
                input_dim=self.arch_road_graph[-1],
                hidden_dim=exp_config.gmm.hidden_dim,
                action_dim=2,
                n_components=exp_config.gmm.n_components,
                time_dim=self.ro_max
            )
        elif use_tom == 'oracle':
            other_input_dim += 5
        else:
            raise ValueError(f'ToM method "{use_tom}" is not implemented yet!!')
        
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
            net_arch=self.arch_ego_state,
        )
        self.road_object_net = self._build_network(
            input_dim=other_input_dim,
            net_arch=self.arch_road_objects,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
            net_arch=self.arch_road_graph,
        )

        self.loss_func = loss

        self.head = GMM(
            network_type=self.__class__.__name__,
            input_dim=self.shared_net_input_dim,
            hidden_dim=exp_config.gmm.hidden_dim,
            action_dim=exp_config.gmm.action_dim,
            n_components=exp_config.gmm.n_components,
            time_dim=1
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

    def get_embedded_obs(self, obs, masks=None, other_info=None):
        """Get the embedded observation."""
        batch = obs.shape[0]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=5)
        if other_info != None:
            other_info = other_info.transpose(1, 2).reshape(batch, self.ro_max, -1)
            road_objects = torch.cat([road_objects, other_info], dim=-1)
            
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects_max = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        context = torch.cat((ego_state, road_objects_max, road_graph), dim=1)
        if self.use_tom == 'aux_head':
            return context, road_objects
        else:
            return context

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def get_tom(self, road_objects, deterministic=False):
        """Get the tom info from the context."""
        other_actions = self.aux_action_head(road_objects, deterministic)
        other_goals = self.aux_goal_head(road_objects, deterministic)
        return other_actions, other_goals
    
    def forward(self, obs, masks=None, deterministic=False, other_info=None):
        """Generate an actions by end-to-end network."""
        context, road_objects = self.get_embedded_obs(obs, other_info=other_info)
        actions = self.get_action(context, deterministic)
        return actions