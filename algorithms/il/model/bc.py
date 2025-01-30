# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from networks.perm_eq_late_fusion import CustomLateFusionNet
from networks.norms import *
from algorithms.il.model.bc_utils.wayformer import SelfAttentionBlock, PerceiverEncoder
from algorithms.il.model.bc_utils.head import *


class ContFeedForward(CustomLateFusionNet):
    def __init__(self,  env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(ContFeedForward, self).__init__(env_config, net_config)
        self.nn = self._build_network(
            input_dim=(self.ego_input_dim + self.ro_input_dim * self.ro_max + self.rg_input_dim * self.rg_max) * num_stack,
        )
        
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(
                input_dim=self.nn.last_layer_dim,
                head_config=head_config
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=self.nn.last_layer_dim,
                head_config=head_config
            )
        elif loss == 'gmm':
            self.head = GMM(
                network_type=self.__class__.__name__,
                input_dim=self.nn.last_layer_dim,
                head_config=head_config,
                time_dim=1
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")

    def get_context(self, obs, masks=None):
        """Get the embedded observation."""
        if obs.dim() == 3:
            obs = obs.reshape(-1, obs.shape[1]*obs.shape[2])
        return self.nn(obs), 0

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _ = self.get_context(obs)
        actions = self.get_action(context, deterministic)
        return actions

class LateFusionBCNet(CustomLateFusionNet):
    def __init__(self, env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(LateFusionBCNet, self).__init__(env_config, net_config)
        self.num_stack = num_stack
        
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
        )
        self.road_object_net = self._build_partner_network(
            input_dim=self.ro_input_dim * num_stack,
        )
        self.road_graph_net = self._build_partner_network(
            input_dim=self.rg_input_dim * num_stack, is_ro=False
        )

        # Action head
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(
                input_dim=self.shared_net_input_dim,
                head_config=head_config
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=self.shared_net_input_dim,
                head_config=head_config
            )
        elif loss == 'gmm':
            self.head = GMM(
                network_type=self.__class__.__name__,
                input_dim=self.shared_net_input_dim,
                head_config=head_config,
                time_dim=1,
            )
        elif loss == 'new_gmm':
            self.head = NewGMM(
                network_type=self.__class__.__name__,
                input_dim=self.shared_net_input_dim,
                head_config=head_config,
                time_dim=1,
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

    def get_tsne(self, obs, mask, road_mask=None):
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

        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        context = torch.cat((ego_state, road_objects, road_graph), dim=1)
        return context, mask_zero_ratio

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)

        return actions

class LateFusionAttnBCNet(CustomLateFusionNet):
    def __init__(self, env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(LateFusionAttnBCNet, self).__init__(env_config, net_config)
        self.num_stack = num_stack 
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
        )
        self.road_object_net = self._build_partner_network(
            input_dim=self.ro_input_dim * num_stack,
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
            norm=net_config.norm,
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
                head_config=head_config
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=self.shared_net_input_dim,
                head_config=head_config
            )
        elif loss == 'gmm':
            self.head = GMM(
                network_type=self.__class__.__name__,
                input_dim= 2 * net_config.network_dim + net_config.network_dim,
                head_config=head_config,
                time_dim=1
            )
        elif loss == 'new_gmm':
            self.head = NewGMM(
                network_type=self.__class__.__name__,
                input_dim= 2 * net_config.network_dim + net_config.network_dim,
                head_config=head_config,
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

    def get_tsne(self, obs, mask, road_mask=None):
        obs = obs.unsqueeze(0)
        mask = mask.unsqueeze(0).bool()
        [norm_layer.__setattr__('mask', mask) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]

        ego_state, road_objects, _ = self._unpack_obs(obs, self.num_stack)
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        
        ego_mask = torch.zeros(1, 1, dtype=torch.bool).to(mask.device)
        all_mask = torch.cat([ego_mask, mask], dim=-1)
        for norm_layer in self.ro_attn.modules():
            if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', all_mask)
        all_objects = torch.cat([ego_state.unsqueeze(1), road_objects], dim=1)
        objects_attn = self.ro_attn(all_objects, pad_mask=all_mask)

        masked_road_objects = objects_attn["last_hidden_state"][:,1:][~mask.unsqueeze(-1).expand_as(road_objects)].view(-1, road_objects.size(-1))
        return masked_road_objects

    def get_context(self, obs, masks=None):
        """Get the embedded observation."""
        batch = obs.shape[0]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=self.num_stack)
        ego_masks = masks[0][:, -1]
        ro_masks = masks[1][:, -1]
        rg_masks = masks[2][:, -1]

        # Ego state and road object encoding
        [norm_layer.__setattr__('mask', ro_masks) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        [norm_layer.__setattr__('mask', rg_masks) for norm_layer in self.road_graph_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        
        # Road object attention
        all_objects = torch.cat([ego_state.unsqueeze(1), road_objects], dim=1)
        obj_masks = torch.cat([ego_masks.unsqueeze(1), ro_masks], dim=-1)
        for norm_layer in self.ro_attn.modules():
            if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', obj_masks)
        for norm_layer in self.rg_attn.modules():
            if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', rg_masks)
        objects_attn = self.ro_attn(all_objects, pad_mask=obj_masks)
        
        # Road graph attention
        road_graph = self.road_graph_net(road_graph)
        road_graph_attn = self.rg_attn(road_graph, pad_mask=rg_masks)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        max_indices_ro = torch.argmax(objects_attn['last_hidden_state'][:, 1:].permute(0, 2, 1), dim=-1)
        selected_mask_ro = torch.gather(ro_masks.squeeze(-1), 1, max_indices_ro)  # (B, D)
        mask_zero_ratio_ro = (selected_mask_ro == 0).sum().item() / selected_mask_ro.numel()
        
        max_indices_rg = torch.argmax(road_graph_attn['last_hidden_state'].permute(0, 2, 1), dim=-1)
        selected_mask_rg = torch.gather(rg_masks.squeeze(-1), 1, max_indices_rg)  # (B, D)
        mask_zero_ratio_rg = (selected_mask_rg == 0).sum().item() / selected_mask_rg.numel()
        mask_zero_ratio = [mask_zero_ratio_ro, mask_zero_ratio_rg]

        objects_attn['last_hidden_state'][:, 1:].masked_fill_(ro_masks.unsqueeze(-1), 0)
        road_graph_attn['last_hidden_state'].masked_fill_(rg_masks.unsqueeze(-1), 0)

        road_objects = F.max_pool1d(
            objects_attn['last_hidden_state'][:, 1:].permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)
        road_objects = road_objects.reshape(batch, -1)
        road_graph = road_graph.reshape(batch, -1)
        context = torch.cat((objects_attn['last_hidden_state'][:, 0], road_objects, road_graph), dim=1)
        return context, mask_zero_ratio

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, attn_weights=False, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)
        return actions

class EarltFusionAttnBCNet(CustomLateFusionNet):
    def __init__(self, env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(EarltFusionAttnBCNet, self).__init__(env_config, net_config)
        self.num_stack = num_stack 
        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
        )
        self.road_object_net = self._build_partner_network(
            input_dim=self.ro_input_dim * num_stack,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
        )
        
        # Attention
        self.fusion_attn = SelfAttentionBlock(
            num_layers=3,
            num_heads=4,
            num_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
            norm=net_config.norm,
        )

        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(
                input_dim=self.shared_net_input_dim,
                head_config=head_config
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=self.shared_net_input_dim,
                head_config=head_config
            )
        elif loss == 'gmm':
            self.head = GMM(
                network_type=self.__class__.__name__,
                input_dim= 2 * net_config.network_dim + net_config.network_dim,
                head_config=head_config,
                time_dim=1
            )
        elif loss == 'new_gmm':
            self.head = NewGMM(
                network_type=self.__class__.__name__,
                input_dim= 2 * net_config.network_dim + net_config.network_dim,
                head_config=head_config,
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

    def get_tsne(self, obs, mask, road_mask=None):
        obs = obs.unsqueeze(0)
        mask = mask.unsqueeze(0).bool()
        road_mask = road_mask.unsqueeze(0).bool()
        [norm_layer.__setattr__('mask', mask) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]

        ego_state, road_objects, road_graph = self._unpack_obs(obs, self.num_stack)
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)
        
        ego_mask = torch.zeros(1, 1, dtype=torch.bool).to(mask.device)
        all_mask = torch.cat([ego_mask, mask, road_mask], dim=-1)
        for norm_layer in self.fusion_attn.modules():
            if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', all_mask)
        all_objects = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
        all_attn = self.fusion_attn(all_objects, pad_mask=all_mask)

        masked_road_objects = all_attn["last_hidden_state"][:,1: 1 + self.ro_max][~mask.unsqueeze(-1).expand_as(road_objects)].view(-1, road_objects.size(-1))
        return masked_road_objects

    def get_context(self, obs, masks=None):
        """Get the embedded observation."""
        batch = obs.shape[0]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=self.num_stack)
        ego_masks = masks[0][:, -1]
        ro_masks = masks[1][:, -1]
        rg_masks = masks[2][:, -1]

        # Ego state and road object encoding
        [norm_layer.__setattr__('mask', ro_masks) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        [norm_layer.__setattr__('mask', rg_masks) for norm_layer in self.road_graph_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        # Road graph attention
        road_graph = self.road_graph_net(road_graph)
        # Road object-map attention
        all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
        all_masks = torch.cat([ego_masks.unsqueeze(1), ro_masks, rg_masks], dim=-1)
        for norm_layer in self.fusion_attn.modules():
            if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', all_masks)
        all_attn = self.fusion_attn(all_objs_map, pad_mask=all_masks)


        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        ego_attn = all_attn['last_hidden_state'][:, 0]
        objects_attn = all_attn['last_hidden_state'][:, 1:self.ro_max + 1]
        road_graph_attn = all_attn['last_hidden_state'][:, self.ro_max + 1:]

        max_indices_ro = torch.argmax(objects_attn.permute(0, 2, 1), dim=-1)
        selected_mask_ro = torch.gather(ro_masks.squeeze(-1), 1, max_indices_ro)  # (B, D)
        mask_zero_ratio_ro = (selected_mask_ro == 0).sum().item() / selected_mask_ro.numel()
        
        max_indices_rg = torch.argmax(road_graph_attn.permute(0, 2, 1), dim=-1)
        selected_mask_rg = torch.gather(rg_masks.squeeze(-1), 1, max_indices_rg)  # (B, D)
        mask_zero_ratio_rg = (selected_mask_rg == 0).sum().item() / selected_mask_rg.numel()
        mask_zero_ratio = [mask_zero_ratio_ro, mask_zero_ratio_rg]

        objects_attn.masked_fill_(ro_masks.unsqueeze(-1), 0)
        road_graph_attn.masked_fill_(rg_masks.unsqueeze(-1), 0)

        road_objects = F.max_pool1d(
            objects_attn.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph_attn.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)
        road_objects = road_objects.reshape(batch, -1)
        road_graph = road_graph.reshape(batch, -1)
        context = torch.cat((ego_attn, road_objects, road_graph), dim=1)
        return context, mask_zero_ratio

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, attn_weights=False, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)
        return actions
    
class WayformerEncoder(CustomLateFusionNet):
    def __init__(self, env_config, net_config, head_config, loss, num_stack=1, rollout_len=10, pred_len=5):
        super(WayformerEncoder, self).__init__(env_config, net_config)        
         # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim,
        )
        self.rollout_len = rollout_len
        self.encoder = PerceiverEncoder(64, 64)
        self.agents_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, 1, (self.ro_max + 1), 64)),
            requires_grad=True
        )
        self.temporal_positional_embedding = nn.parameter.Parameter(
            torch.zeros((1, rollout_len, 1, 64)),
            requires_grad=True
        )
        self.timestep_linear = nn.Linear(64, pred_len)
        
        if loss in ['l1', 'mse', 'twohot']: # make head module
            self.head = ContHead(
                input_dim=64,
                hidden_dim=head_config.head_dim,
                hidden_num=head_config.head_num_layers
            )
        elif loss == 'nll':
            self.head = DistHead(
                input_dim=64,
                head_config=head_config
            )
        elif loss == 'gmm':
            self.head = GMM(
                network_type=self.__class__.__name__,
                input_dim=64,
                head_config=head_config,
                time_dim=pred_len
            )
        elif loss == 'new_gmm':
            self.head = NewGMM(
                network_type=self.__class__.__name__,
                input_dim=64,
                head_config=head_config,
                time_dim=pred_len
            )
        else:
            raise ValueError(f"Loss name {loss} is not supported")

    def _unpack_obs(self, obs_flat, timestep=91):
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, timestep,  ego_size + ro_size + rg_size)
        ego_stack = obs_flat_unstack[..., :ego_size].view(-1, timestep, self.ego_input_dim)
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, timestep, self.ro_max, self.ro_input_dim)
         )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, timestep, self.rg_max, self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack

    def get_context(self, obs, masks=None):
        # Unpack observation
        ego_mask, partner_mask, road_mask = masks
        ego_state, road_objects, road_graph = self._unpack_obs(obs, timestep=self.rollout_len)
        batch_size = obs.shape[0]
        
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)
        ego_state = ego_state.unsqueeze(2)
        agent_state = torch.cat([ego_state, road_objects], dim=2)
        agent_state = agent_state + self.agents_positional_embedding + self.temporal_positional_embedding
        
        ego_mask = ego_mask.unsqueeze(-1)
        agent_mask = torch.cat([ego_mask, partner_mask], dim=-1)
        road_mask = road_mask.reshape(batch_size, -1)
        agent_mask = agent_mask.reshape(batch_size, -1)

        agent_state = agent_state.reshape(batch_size, -1, 64)
        road_graph = road_graph.reshape(batch_size, -1, 64)

        embedding_vector = torch.cat((agent_state, road_graph), dim=1)
        embedding_mask = torch.cat((agent_mask, road_mask), dim=1)
        context = self.encoder(embedding_vector, embedding_mask) 
        context = context.transpose(1, 2)
        context = self.timestep_linear(context)
        context = context.transpose(1, 2)

        return context, 0

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, _ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)
        return actions
