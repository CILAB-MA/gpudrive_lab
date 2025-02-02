# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from networks.perm_eq_late_fusion import CustomLateFusionNet
from networks.norms import *
from algorithms.il.model.bc_utils.head import *
from algorithms.il.model.bc_utils.wayformer import SelfAttentionBlock

class LateFusionAuxNet(CustomLateFusionNet):

    def __init__(self, env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(LateFusionAuxNet, self).__init__(env_config, net_config)
        self.num_stack = num_stack
        other_input_dim = self.ro_input_dim * num_stack
        # Aux head
        self.use_tom = use_tom
        if use_tom == 'aux_head':
            self.aux_speed_head = AuxHead(
                input_dim=self.hidden_dim,
                head_config=head_config,
                num_ro=self.ro_max,
                aux_action_dim=1
            )
            
            self.aux_pos_head = AuxHead(
                input_dim=self.hidden_dim,
                head_config=head_config,
                num_ro=self.ro_max,
                aux_action_dim=2
            )
            
            self.aux_heading_head = AuxHead(
                input_dim=self.hidden_dim,
                head_config=head_config,
                num_ro=self.ro_max,
                aux_action_dim=1
            )
            
            self.aux_action_head = AuxHead(
                input_dim=self.hidden_dim,
                head_config=head_config,
                num_ro=self.ro_max,
                aux_action_dim=3
            )

        elif use_tom == 'oracle':
            other_input_dim += 5 * 5
        else:
            raise ValueError(f'ToM method "{use_tom}" is not implemented yet!!')
        
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack
        )
        self.road_object_net = self._build_partner_network(
            input_dim=self.ro_input_dim * num_stack,
        )
        self.road_graph_net = self._build_partner_network(
            input_dim=self.rg_input_dim * num_stack, is_ro=False
        )

        self.head = GMM(
            network_type=self.__class__.__name__,
            input_dim= 2 * net_config.network_dim + net_config.network_dim,
            head_config=head_config,
            time_dim=1
        )
   
    def _unpack_obs(self, obs_flat, num_stack):
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

    def get_tsne(self, obs, mask):
        obs = obs.unsqueeze(0)
        mask = mask.unsqueeze(0).bool()
        _, road_objects, _ = self._unpack_obs(obs, self.num_stack)
        [norm_layer.__setattr__('mask', mask) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        road_objects = self.road_object_net(road_objects)
        masked_road_objects = road_objects[~mask.unsqueeze(-1).expand_as(road_objects)].view(-1, road_objects.size(-1))
        return masked_road_objects
    
    def get_context(self, obs, masks=None):
        """Get the embedded observation."""
        # Set mask for road_object_net (for SetNorm)
        partner_mask = masks[1][:,-1,:]
        road_mask = masks[2][:,-1,:]
        [norm_layer.__setattr__('mask', partner_mask) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        [norm_layer.__setattr__('mask', road_mask) for norm_layer in self.road_graph_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, self.num_stack)

        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        max_indices_ro = torch.argmax(road_objects.permute(0, 2, 1), dim=-1)
        selected_mask_ro = torch.gather(partner_mask.squeeze(-1), 1, max_indices_ro)  # (B, D)
        mask_zero_ratio_ro = (selected_mask_ro == 0).sum().item() / selected_mask_ro.numel()
        
        max_indices_rg = torch.argmax(road_graph.permute(0, 2, 1), dim=-1)
        selected_mask_rg = torch.gather(road_mask.squeeze(-1), 1, max_indices_rg)  # (B, D)
        mask_zero_ratio_rg = (selected_mask_rg == 0).sum().item() / selected_mask_rg.numel()
        mask_zero_ratio = [mask_zero_ratio_ro, mask_zero_ratio_rg]

        road_objects.masked_fill_(partner_mask.unsqueeze(-1), 0)
        road_graph.masked_fill_(road_mask.unsqueeze(-1), 0)

        max_road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        context = torch.cat((ego_state, max_road_objects, road_graph), dim=1)
        return context, mask_zero_ratio, road_objects

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _, _ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)

        return actions

class LateFusionAttnAuxNet(CustomLateFusionNet):
    def __init__(self, env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(LateFusionAttnAuxNet, self).__init__(env_config, net_config)
        self.num_stack = num_stack 
        other_input_dim = self.ro_input_dim * num_stack
        # Aux head
        self.use_tom = use_tom
        if use_tom == 'aux_head':
            self.aux_action_head = GMM(
                network_type=self.__class__.__name__,
                input_dim=self.hidden_dim,
                hidden_dim=head_config.head_dim,
                hidden_num=head_config.head_num_layers,
                action_dim=head_config.action_dim,
                n_components=head_config.n_components,
                time_dim=self.ro_max
            )
            self.aux_goal_head = GMM(
                network_type=self.__class__.__name__,
                input_dim=self.hidden_dim,
                hidden_dim=head_config.head_dim,
                hidden_num=head_config.head_num_layers,
                action_dim=2,
                n_components=head_config.n_components,
                time_dim=self.ro_max
            )
        elif use_tom == 'oracle':
            other_input_dim += 5 * 5
        else:
            raise ValueError(f'ToM method "{use_tom}" is not implemented yet!!')
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
        )
        self.road_object_net = self._build_network(
            input_dim=other_input_dim,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
        )
        
        # Attention
        self.ro_attn = SelfAttentionBlock(
            num_layers=3,
            num_heads=4,
            num_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
        )

        self.rg_attn = SelfAttentionBlock(
            num_layers=3,
            num_heads=4,
            num_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
        )
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(
                input_dim=self.shared_net_input_dim,
                hidden_dim=head_config.head_dim,
                hidden_num=head_config.head_num_layers
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=self.shared_net_input_dim,
                hidden_dim=head_config.head_dim,
                hidden_num=head_config.head_num_layers,
                action_dim=head_config.action_dim,
            )
        elif loss == 'gmm':
            self.head = GMM(
                network_type=self.__class__.__name__,
                input_dim= 2 * 4 * net_config.network_dim + net_config.network_dim,
                hidden_dim=head_config.head_dim,
                hidden_num=head_config.head_num_layers,
                action_dim=head_config.action_dim,
                n_components=head_config.n_components,
                time_dim=1
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")

    def _unpack_obs(self, obs_flat, num_stack):
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
    
    def get_tom(self, road_objects, deterministic=False):
        """Get the tom info from the context."""
        other_actions = self.aux_action_head(road_objects, deterministic)
        other_goals = self.aux_goal_head(road_objects, deterministic)
        return other_actions, other_goals
    
    def get_context(self, obs, masks=None, other_info=None):
        """Get the embedded observation."""
        batch = obs.shape[0]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=self.num_stack)
        if other_info != None:
            other_info = other_info.transpose(1, 2).reshape(batch, self.ro_max, -1)
            road_objects = torch.cat([road_objects, other_info], dim=-1)
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        ego_masks = masks[0][:, -1]
        ro_masks = masks[1][:, -1]
        rg_masks = masks[2][:, -1]
        all_objects = torch.cat([ego_state.unsqueeze(1), road_objects], dim=1)
        obj_masks = torch.cat([ego_masks.unsqueeze(1), ro_masks], dim=-1)
        objects_attn = self.ro_attn(all_objects, pad_mask=obj_masks)
        
        road_graph = self.road_graph_net(road_graph)
        road_graph_attn = self.rg_attn(road_graph, pad_mask=rg_masks)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)

        ro_pool_dim = int(self.ro_max / 4)
        rg_pool_dim = int(self.rg_max / 4)
        road_objects_avg = F.avg_pool1d(
            objects_attn['last_hidden_state'][:, 1:].permute(0, 2, 1), kernel_size=ro_pool_dim
        ).squeeze(-1)
        road_graph = F.avg_pool1d(
            road_graph_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=rg_pool_dim
        ).squeeze(-1)
        road_objects_avg = road_objects_avg.reshape(batch, -1)
        road_graph = road_graph.reshape(batch, -1)
        embedding_vector = torch.cat((objects_attn['last_hidden_state'][:, 0], road_objects_avg, road_graph), dim=1)
        if self.use_tom == 'aux_head':
            return embedding_vector, road_objects
        else:
            return embedding_vector, None

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, attn_weights=False, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _  = self.get_context(obs, masks, other_info=other_info)
        actions = self.get_action(context, deterministic)
        return actions