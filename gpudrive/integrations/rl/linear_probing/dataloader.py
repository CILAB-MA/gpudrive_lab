import torch
import numpy as np
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS


class FutureDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, ego_global_pos, ego_global_rot, masks=None, partner_mask=None, future_step=1, exp='other', xy_range=None, 
                 partner_labels=None, ego_labels=None):
        # obs
        self.obs = obs
        B, T, F = obs.shape
        self.actions = actions

        # masks
        valid_masks = 1 - masks
        action_mask = (np.abs(actions[..., 1]) >  0.5) | (np.abs(actions[..., 0]) >  5) | (np.abs(actions[..., -1]) > 0.2)
        valid_masks[action_mask] = 0
        self.valid_masks = valid_masks.astype('bool')
        self.future_step = future_step
        print(self.obs.shape, self.actions.shape, self.valid_masks.shape)
        #### 
        if partner_labels is not None:
            partner_labels_pad = np.zeros((partner_labels.shape[0], future_step, *partner_labels.shape[2:]), dtype=np.int16)
            partner_labels_new = np.concatenate([partner_labels, partner_labels_pad], axis=1)[:, future_step:]
            self.partner_labels = partner_labels_new
            print(f"future partner labels shape: {self.partner_labels.shape}")
        if ego_labels is not None:
            self.ego_labels = ego_labels
            print(f"ego_labels shape: {self.ego_labels.shape}")
        if exp == 'other':
            # future partner_mask
            partner_info = obs[..., 6:128 * 6].reshape(B, T, 127, 6)[..., :4]
            aux_info, aux_mask = self._make_aux_info(partner_mask, partner_info, future_timestep=future_step)
            self.aux_mask = aux_mask.astype('bool')
            print(f"future aux mask shape: {self.aux_mask.shape}")

        if exp == 'ego':
            # future ego mask
            future_valid_mask_pad = np.zeros((self.valid_masks.shape[0], future_step, *self.valid_masks.shape[2:]), dtype=np.float32)
            future_valid_masks = np.concatenate([valid_masks, future_valid_mask_pad], axis=1).astype('bool')[:, future_step:]
            self.future_valid_mask = self.valid_masks & future_valid_masks
            print(f"future_valid_mask shape: {self.future_valid_mask.shape}")
        # road_mask
        if exp == 'other':
            # future other pos
            current_relative_other_pos = self._transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step=future_step)
            current_relative_other_pos[aux_mask] = 0
            self.other_pos = self._get_multi_class_pos(current_relative_other_pos)
            print(f"future other pos shape: {self.other_pos.shape}")
        else:
            # future ego pos
            current_relative_ego_pos = self._transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step=future_step,
                                                                    )
            self.ego_pos = self._get_multi_class_pos(current_relative_ego_pos, xy_range)

            # ego? -> current_relative_pos[aux_mask] = 0
            print("future ego pos shape: ", self.ego_pos.shape)
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'valid_masks']
        if exp == 'other':
            self.full_var += ['aux_mask', 'other_pos']
        else:
            self.full_var += ['future_valid_mask', 'ego_pos']
        if partner_labels is not None:
            self.full_var += ['partner_labels']
        if ego_labels is not None:
            self.full_var += ['ego_labels']

    def __len__(self):
        return len(self.valid_indices)

    def _make_aux_info(self, partner_mask, partner_info, future_timestep):
        partner_mask_bool = np.where(partner_mask == 0, 0, 1).astype(bool)
        partner_info_pad = np.zeros((partner_info.shape[0], future_timestep, *partner_info.shape[2:]), dtype=np.float32)
        partner_mask_pad = np.full((partner_mask.shape[0], future_timestep, *partner_mask.shape[2:]), 2, dtype=np.float32)

        future_mask = np.concatenate([partner_mask, partner_mask_pad], axis=1)
        future_mask_bool = np.where(future_mask == 0, 0, 1).astype(bool)[:, future_timestep:]
        partner_info = np.concatenate([partner_info, partner_info_pad], axis=1)[:, future_timestep:]
        combined_mask = np.logical_or(future_mask_bool, partner_mask_bool).astype('bool')
        return partner_info, combined_mask
    
    @staticmethod
    def _transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step):
        """transform time t relative pos to current relative pos"""
        # 1. transform t-relative pos to t-global pos
        # get partner's relative pos and rot at time t
        t_partner_pos = aux_info[..., 1:3] * MAX_REL_AGENT_POS
        
        # get ego's global pos and rot at time t
        t_ego_global_pos = np.zeros_like(ego_global_pos)
        t_ego_global_rot = np.zeros_like(ego_global_rot)
        t_ego_global_pos[:, :-future_step] = ego_global_pos[:, future_step:]
        t_ego_global_rot[:, :-future_step] = ego_global_rot[:, future_step:]
        
        t_partner_global_pos_x = t_ego_global_pos[..., 0, None] + t_partner_pos[..., 0] * np.cos(t_ego_global_rot) - t_partner_pos[..., 1] * np.sin(t_ego_global_rot)
        t_partner_global_pos_y = t_ego_global_pos[..., 1, None] + t_partner_pos[..., 0] * np.sin(t_ego_global_rot) + t_partner_pos[..., 1] * np.cos(t_ego_global_rot)
        
        # 2. transform t-global pos to current relative pos
        delta_x = t_partner_global_pos_x - ego_global_pos[..., 0, None]
        delta_y = t_partner_global_pos_y - ego_global_pos[..., 1, None]
        
        cos_theta = np.cos(-ego_global_rot)
        sin_theta = np.sin(-ego_global_rot)
        
        current_relative_pos_x = delta_x * cos_theta + delta_y * sin_theta
        current_relative_pos_y = -delta_x * sin_theta + delta_y * cos_theta
        current_relative_pos_x = 2 * ((current_relative_pos_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((current_relative_pos_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
        
        return current_relative_pos

    @staticmethod
    def _transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step):
        """transform global pos to current relative pos"""
        current_relative_pos = np.zeros_like(ego_global_pos)
        ego_current_pos = ego_global_pos[:, :-future_step]
        ego_future_pos = ego_global_pos[:, future_step:]
        
        delta_x = ego_future_pos[..., 0] - ego_current_pos[..., 0]
        delta_y = ego_future_pos[..., 1] - ego_current_pos[..., 1]
        
        ego_current_rot = ego_global_rot[:, :-future_step]
        
        cos_theta = np.cos(ego_current_rot)
        sin_theta = np.sin(ego_current_rot)
        
        rel_x = delta_x * cos_theta.squeeze(-1) + delta_y * sin_theta.squeeze(-1)
        rel_y = -delta_x * sin_theta.squeeze(-1) + delta_y * cos_theta.squeeze(-1)
        
        current_relative_pos_x = 2 * ((rel_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((rel_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos[:, :-future_step, :] = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
        
        return current_relative_pos

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T)
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))

    @staticmethod
    def _get_multi_class_pos(pos, xy_range=None):
        """
        Convert continuous pos to multi-class discrete pos based on x, y.
        """
        x, y = pos[..., 0], pos[..., 1]
        
        # Define bins for discretization (-1 to 1 with 8 bins)
        if xy_range is not None:
            xrange = xy_range[0]
            yrange = xy_range[1]
            xbins = np.linspace(xrange[0], xrange[1], 9)
            ybins = np.linspace(yrange[0], yrange[1], 9)
        else:
            xbins = np.linspace(-0.05, 0.05, 9)
            ybins = np.linspace(-0.05, 0.05, 9)

        # Digitize x and y into 8 categories (0 to 7)
        x_bins = np.digitize(x, xbins) - 1
        y_bins = np.digitize(y, ybins) - 1
        
        # Ensure values are within valid range (0 to 7)
        x_bins = np.clip(x_bins, 0, 7)
        y_bins = np.clip(y_bins, 0, 7)
        
        discrete_pos = x_bins * 8 + y_bins
        return discrete_pos
        
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        idx1 = int(idx1)
        idx2 = int(idx2)
        # row, column -> 
        batch = ()
        for var_name in self.full_var:
            if self.__dict__[var_name] is not None:
                if var_name in ['ego_labels']:
                    data = self.__dict__[var_name][idx1]
                else:
                    data = self.__dict__[var_name][idx1, idx2]
                batch = batch + (data, )

        return batch
