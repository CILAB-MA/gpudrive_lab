# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from networks.perm_eq_late_fusion import CustomLateFusionNet
from networks.norms import *
from algorithms.il.model.bc_utils.head import *
from algorithms.il.model.bc_utils.wayformer import SelfAttentionBlock, CrossAttentionLayer

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
        self.road_object_net = self._build_network_v2(
            input_dim=self.ro_input_dim * num_stack,
        )
        self.road_graph_net = self._build_network_v2(
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

        road_objects.masked_fill(partner_mask.unsqueeze(-1), 0)
        road_graph.masked_fill(road_mask.unsqueeze(-1), 0)

        max_road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        context = torch.cat((ego_state, max_road_objects, road_graph), dim=1)
        return context, mask_zero_ratio, road_objects, max_indices_ro, max_indices_rg

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, *_ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)

        return actions

class LateFusionAttnAuxNet(CustomLateFusionNet):
    def __init__(self, env_config, net_config, head_config, loss, num_stack=5, use_tom=None):
        super(LateFusionAttnAuxNet, self).__init__(env_config, net_config)
        self.num_stack = num_stack 
        other_input_dim = self.ro_input_dim * num_stack
        # Aux head
        self.use_tom = use_tom
        aux_input_dim = self.hidden_dim if 'no_guide' in use_tom else int(self.hidden_dim / 4)
        self.aux_speed_head = AuxHead(
            input_dim=aux_input_dim,
            head_config=head_config,
            num_ro=self.ro_max,
            aux_action_dim=1
        )
        
        self.aux_pos_head = AuxHead(
            input_dim=aux_input_dim,
            head_config=head_config,
            num_ro=self.ro_max,
            aux_action_dim=2
        )
        
        self.aux_heading_head = AuxHead(
            input_dim=aux_input_dim,
            head_config=head_config,
            num_ro=self.ro_max,
            aux_action_dim=1
        )
        
        self.aux_action_head = AuxHead(
            input_dim=aux_input_dim,
            head_config=head_config,
            num_ro=self.ro_max,
            aux_action_dim=3
        )

        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
        )
        self.road_object_net = self._build_network_v2(
            input_dim=other_input_dim,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack
        )
        
        # Attention
        self.fusion_attn = SelfAttentionBlock(
            num_layers=1,
            num_heads=4,
            num_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
            norm=net_config.norm,
            separate_attn_weights=False
        )
        self.ro_attn = SelfAttentionBlock(
            num_layers=1,
            num_heads=4,
            num_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
            norm=net_config.norm,
            separate_attn_weights=False
        )
        self.rg_attn = SelfAttentionBlock(
            num_layers=1,
            num_heads=4,
            num_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
            norm=net_config.norm,
            separate_attn_weights=False
        )
        self.ego_ro_attn = CrossAttentionLayer(
            num_heads=4,
            num_q_input_channels=net_config.network_dim,
            num_kv_input_channels=net_config.network_dim,
            num_qk_channels=net_config.network_dim,
            num_v_channels=net_config.network_dim,
        )

        self.ego_rg_attn = CrossAttentionLayer(
            num_heads=4,
            num_q_input_channels=net_config.network_dim,
            num_kv_input_channels=net_config.network_dim,
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
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0)
            mask = mask.unsqueeze(0)
            road_mask = road_mask.unsqueeze(0)
        mask = mask.bool()
        road_mask = road_mask.bool()
        [norm_layer.__setattr__('mask', mask) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        [norm_layer.__setattr__('mask', road_mask) for norm_layer in self.road_graph_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, self.num_stack)
        masked_positions = road_objects[..., 1:3]
        masked_speed = road_objects[..., 0]
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)
        
        ego_mask = torch.zeros(len(obs), 1, dtype=torch.bool).to(mask.device)
        # Road object-map attention
        all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
        all_masks = torch.cat([ego_mask, mask, road_mask], dim=-1)
        obj_masks = torch.cat([ego_mask, mask], dim=-1)
        for norm_layer in self.fusion_attn.modules():
            if isinstance(norm_layer, CrossSetNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', all_masks)
        all_attn = self.fusion_attn(all_objs_map, pad_mask=all_masks)
        ego_attn = all_attn['last_hidden_state'][:, 0].unsqueeze(1)
        objects_attn = all_attn['last_hidden_state'][:, :self.ro_max + 1]

        all_objects_attn = self.ro_attn(objects_attn, pad_mask=obj_masks)
        objects_attn = all_objects_attn['last_hidden_state'][:, 1:self.ro_max + 1]

        masked_road_objects = objects_attn[~mask.unsqueeze(-1).expand_as(road_objects)].view(-1, road_objects.size(-1))
        masked_positions = masked_positions[~mask.unsqueeze(-1).expand_as(masked_positions)].view(-1, 2)
        masked_speed = masked_speed[~mask].view(-1, 1)
        masked_distances = masked_positions.norm(dim=-1)

        objects_attn = self.ego_ro_attn(ego_attn, objects_attn, pad_mask=mask) 
        ego_attn_score = objects_attn['ego_attn'].clone()
        ego_attn_score = ego_attn_score / ego_attn_score.sum(dim=-1, keepdim=True)
        attn_score0 = objects_attn['ego_attn'][:, 0][~mask].unsqueeze(-1)
        attn_score1 = objects_attn['ego_attn'][:, 1][~mask].unsqueeze(-1)
        attn_score2 = objects_attn['ego_attn'][:, 2][~mask].unsqueeze(-1)
        attn_score3 = objects_attn['ego_attn'][:, 3][~mask].unsqueeze(-1)
        ego_attn_score = torch.cat([attn_score0, attn_score1, attn_score2, attn_score3], dim=-1)
        dist_min = masked_distances.min()
        dist_max = masked_distances.max()
        dist_range = dist_max - dist_min

        speed_min = masked_speed.min()
        speed_max = masked_speed.max()
        speed_range = speed_max - speed_min
        if dist_range == 0:
            normalized_distances = torch.zeros_like(masked_distances)
            normalized_speed = torch.zeros_like(masked_speed)
        else:
            normalized_distances = (masked_distances - dist_min) / dist_range
            normalized_speed = (masked_speed - dist_min) / speed_range
        return masked_road_objects.detach().cpu().numpy(), normalized_distances.detach().cpu().numpy(), normalized_speed.detach().cpu().numpy(), ego_attn_score.cpu().numpy()

    def get_context(self, obs, masks=None):
        """Get the embedded observation."""
        batch = obs.shape[0]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=self.num_stack)
        ego_masks = masks[0][:, -1]
        ro_masks = masks[1][:, -1]
        rg_masks = masks[2][:, -1]
        [norm_layer.__setattr__('mask', ro_masks) for norm_layer in self.road_object_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        [norm_layer.__setattr__('mask', rg_masks) for norm_layer in self.road_graph_net if isinstance(norm_layer, SetBatchNorm) or isinstance(norm_layer, MaskedBatchNorm1d)]
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.road_graph_net(road_graph)

        # Road object-map attention
        all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
        all_masks = torch.cat([ego_masks.unsqueeze(1), ro_masks, rg_masks], dim=-1)
        obj_masks = torch.cat([ego_masks.unsqueeze(1), ro_masks], dim=-1)
        for norm_layer in self.fusion_attn.modules():
            if isinstance(norm_layer, CrossSetNorm) or isinstance(norm_layer, MaskedBatchNorm1d):
                setattr(norm_layer, 'mask', all_masks)
        all_attn = self.fusion_attn(all_objs_map, pad_mask=all_masks)
        objects_attn = all_attn['last_hidden_state'][:, :self.ro_max + 1]
        road_graph_attn = all_attn['last_hidden_state'][:, self.ro_max + 1:]

        all_objects_attn = self.ro_attn(objects_attn, pad_mask=obj_masks)
        ego_attn = all_objects_attn['last_hidden_state'][:, 0].unsqueeze(1)
        objects_attn = all_objects_attn['last_hidden_state'][:, 1:self.ro_max + 1]
        other_attn = objects_attn.clone()
        road_graph_attn = self.rg_attn(road_graph_attn, pad_mask=rg_masks)
        road_graph_attn = road_graph_attn['last_hidden_state']

        objects_attn = self.ego_ro_attn(ego_attn, objects_attn, pad_mask=ro_masks)     
        road_graph_attn = self.ego_rg_attn(ego_attn, road_graph_attn, pad_mask=rg_masks)   

        road_objects_attn = objects_attn['last_hidden_state']
        road_graph_attn = road_graph_attn['last_hidden_state']

        # Max pooling across the object dimensions
        # (M, E) -> (1, E) (max pool across features)
        mask_zero_ratio = [0, 0]
        road_objects = road_objects_attn.reshape(batch, -1)
        road_graph = road_graph_attn.reshape(batch, -1)
        context = torch.cat((ego_attn.squeeze(1), road_objects, road_graph), dim=1)
        
        ego_attn_score = objects_attn['ego_attn'].clone()
        ego_attn_score = ego_attn_score / ego_attn_score.sum(dim=-1, keepdim=True)
        return context, mask_zero_ratio, other_attn, objects_attn['ego_attn'], ego_attn_score, None

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, *_  = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)
        return actions