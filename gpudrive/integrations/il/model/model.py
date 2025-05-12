# Define network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from gpudrive.integrations.il.model.networks import *

class EarlyFusionAttnBCNet(CustomLateFusionNet):
    def __init__(self, env_config, exp_config, num_stack=5, use_tom=None):
        super(EarlyFusionAttnBCNet, self).__init__(env_config, exp_config)
        self.num_stack = num_stack 

        # Scene encoder
        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * num_stack,
        )
        self.road_graph_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
        )
        if use_tom:
            self.aux_head = nn.Sequential(
                nn.Linear(exp_config.network_dim, exp_config.network_dim),
                nn.ReLU(),
                nn.Linear(exp_config.network_dim, 64)
            )
        # Attention
        self.fusion_attn = SelfAttentionBlock(
            num_layers=exp_config.num_layer[0],
            num_heads=exp_config.num_head,
            num_channels=exp_config.network_dim,
            num_qk_channels=exp_config.network_dim,
            num_v_channels=exp_config.network_dim,
            separate_attn_weights=False
        )
        self.ro_attn = SelfAttentionBlock(
            num_layers=exp_config.num_layer[1],
            num_heads=exp_config.num_head,
            num_channels=exp_config.network_dim,
            num_qk_channels=exp_config.network_dim,
            num_v_channels=exp_config.network_dim,
            separate_attn_weights=False
        )
        self.rg_attn = SelfAttentionBlock(
            num_layers=exp_config.num_layer[1],
            num_heads=exp_config.num_head,
            num_channels=exp_config.network_dim,
            num_qk_channels=exp_config.network_dim,
            num_v_channels=exp_config.network_dim,
            separate_attn_weights=False
        )
        self.ego_ro_attn = CrossAttentionLayer(
            num_heads=exp_config.num_head,
            num_q_input_channels=exp_config.network_dim,
            num_kv_input_channels=exp_config.network_dim,
            num_qk_channels=exp_config.network_dim,
            num_v_channels=exp_config.network_dim,
        )

        self.ego_rg_attn = CrossAttentionLayer(
            num_heads=exp_config.num_head,
            num_q_input_channels=exp_config.network_dim,
            num_kv_input_channels=exp_config.network_dim,
            num_qk_channels=exp_config.network_dim,
            num_v_channels=exp_config.network_dim,
        )

        # self.head = GMM(
        #     network_type=self.__class__.__name__,
        #     input_dim= 2 * exp_config.network_dim + exp_config.network_dim,
        #     head_config=exp_config,
        #     time_dim=1
        # )
        self.head = ContHead(
            input_dim= 2 * exp_config.network_dim + exp_config.network_dim,
            head_config=exp_config,)
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

    def get_context(self, obs, masks=None):
        """Get the embedded observation."""
        batch = obs.shape[0]
        ego_state, road_objects, road_graph = self._unpack_obs(obs, num_stack=self.num_stack)
        ro_masks = masks[0][:, -1]
        rg_masks = masks[1][:, -1]

        # Ego state and road object encoding
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        # Road graph attention
        road_graph = self.road_graph_net(road_graph)
        # Road object-map attention
        ego_mask = torch.zeros(len(obs), 1, dtype=torch.bool).to(ego_state.device)
        all_objs_map = torch.cat([ego_state.unsqueeze(1), road_objects, road_graph], dim=1)
        all_masks = torch.cat([ego_mask, ro_masks, rg_masks], dim=-1)
        obj_masks = torch.cat([ego_mask, ro_masks], dim=-1)

        all_attn = self.fusion_attn(all_objs_map, pad_mask=all_masks)
        objects_attn = all_attn['last_hidden_state'][:, :self.ro_max + 1]
        road_graph_attn = all_attn['last_hidden_state'][:, self.ro_max + 1:]

        all_objects_attn = self.ro_attn(objects_attn, pad_mask=obj_masks)
        ego_attn = all_objects_attn['last_hidden_state'][:, 0].unsqueeze(1)
        objects_attn = all_objects_attn['last_hidden_state'][:, 1:self.ro_max + 1]
        road_graph_attn = self.rg_attn(road_graph_attn, pad_mask=rg_masks)
        road_graph_attn = road_graph_attn['last_hidden_state']

        objects_attn = self.ego_ro_attn(ego_attn, objects_attn, pad_mask=ro_masks)     
        road_graph_attn = self.ego_rg_attn(ego_attn, road_graph_attn, pad_mask=rg_masks)   

        road_objects_attn = objects_attn['last_hidden_state']
        road_graph_attn = road_graph_attn['last_hidden_state'] 

        road_objects = road_objects_attn.reshape(batch, -1)
        road_graph = road_graph_attn.reshape(batch, -1)
        context = torch.cat((ego_attn.squeeze(1), road_objects, road_graph), dim=1)

        ego_attn_score = objects_attn['ego_attn'].clone()
        ego_attn_score = ego_attn_score / ego_attn_score.sum(dim=-1, keepdim=True)

        return context, ego_attn_score, None

    def get_action(self, context, deterministic=False):
        """Get the action from the context."""
        return self.head(context, deterministic)

    def forward(self, obs, masks=None, other_info=None, attn_weights=False, deterministic=False):
        """Generate an actions by end-to-end network."""
        context, *_ = self.get_context(obs, masks)
        actions = self.get_action(context, deterministic)
        return actions