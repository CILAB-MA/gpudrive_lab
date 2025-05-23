import torch
import numpy as np
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, partner_mask=None, road_mask=None,
                 rollout_len=5, pred_len=1, aux_future_step=None, ego_global_pos=None, ego_global_rot=None,
                 use_tom=False):
        # obs
        self.obs = np.pad(obs, ((0, 0), (rollout_len - 1, 0), (0, 0)))

        # actions
        self.actions = actions
        
        # masks
        valid_masks = 1 - masks
        action_mask = (np.abs(actions[..., 1]) >  0.5) | (np.abs(actions[..., 0]) >  5) | (np.abs(actions[..., -1]) > 0.2)
        valid_masks[action_mask] = 0
        B, T, _ = obs.shape
        new_shape = (B, T + rollout_len - 1)
        new_valid_mask = np.zeros(new_shape, dtype=self.obs.dtype)
        new_valid_mask[:, rollout_len - 1:] = valid_masks
        self.valid_masks = new_valid_mask.astype('bool')
        self.use_mask = True if self.valid_masks is not None else False

        # partner_mask
        self.aux_valid_mask = None
        partner_info = obs[..., 6:128 * 6].reshape(B, T, 127, 6)[..., :4]
        self.aux_mask = None
        self.other_info = None
        self.other_pos = None
        if use_tom:
            # todo: concat remove need
            aux_info, aux_mask = self._make_aux_info(partner_mask, partner_info, 
                                                     future_timestep=aux_future_step)
            max_aux_actions = np.max(aux_info[..., -4:], axis=-1)
            min_aux_actions = np.min(aux_info[..., -4:], axis=-1)
            aux_action_mask = (max_aux_actions == 6) | (min_aux_actions == -6) | (aux_info[..., -1] >= 3.14)  | (aux_info[..., -1] <= -3.14)
            aux_mask[aux_action_mask] = True
            self.aux_mask = aux_mask.astype('bool')
            self.other_info = aux_info
            current_relative_pos = self._transform_relative_pos(aux_info, ego_global_pos, ego_global_rot, future_step=aux_future_step)
            current_relative_pos[self.aux_mask] = 0
            self.other_pos = self._get_multi_class_pos(current_relative_pos)

        self.partner_mask = np.pad(partner_mask, ((0, 0), (rollout_len - 1, 0), (0, 0)), constant_values=2)
        self.partner_mask = (self.partner_mask != 2)
        road_mask = road_mask.astype(bool)
        self.road_mask = np.pad(
            road_mask,
            pad_width=((0, 0), (rollout_len - 1, 0), (0, 0)),
            mode='constant',
            constant_values=True
        )
          
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'partner_mask', 'road_mask',
                         'other_pos', 'aux_mask']

    def __len__(self):
        return len(self.valid_indices)

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
    
    def _make_aux_info(self, partner_mask, info, partner_info, future_timestep):
        partner_mask_bool = np.where(partner_mask == 0, 0, 1).astype(bool)
        action_valid_mask = np.where(partner_mask == 0, 1, 0).astype(bool)
        info[..., :-1] *= action_valid_mask[..., np.newaxis]
        current_info_id = info[:, :, :, -1]
        all_infos = np.concatenate([partner_info, info], axis=-1)
        other_info_pad = np.zeros((all_infos.shape[0], future_timestep, *all_infos.shape[2:]), dtype=np.float32)
        partner_mask_pad = np.full((partner_mask.shape[0], future_timestep, *partner_mask.shape[2:]), 2, dtype=np.float32)

        future_mask = np.concatenate([partner_mask, partner_mask_pad], axis=1)
        future_mask_bool = np.where(future_mask == 0, 0, 1).astype(bool)[:, future_timestep:]
        other_info = np.concatenate([all_infos, other_info_pad], axis=1)[:, future_timestep:]
        future_info_id = other_info[:, :, :, -1]
        future_acton_sum = other_info[:, :, :, :-1]

        future_info_id_masked = future_info_id * ~future_mask_bool - future_mask_bool
        current_info_id_masked = current_info_id * ~partner_mask_bool - partner_mask_bool
        future_info_id_masked = future_info_id_masked.astype(np.int64)
        current_info_id_masked = current_info_id_masked.astype(np.int64)

        aligned_future_acton_sum = np.zeros_like(future_acton_sum)

        aligned_future_mask_bool = np.zeros_like(future_mask_bool, dtype=bool)

        B, T, _ = future_info_id_masked.shape
        for b in range(B):
            for t in range(T):
                future_ids_1d = future_info_id_masked[b, t]       
                current_ids_1d = current_info_id_masked[b, t]    
                future_acts_2d = future_acton_sum[b, t]           
                future_mask_1d = future_mask_bool[b, t]
                valid_mask = (future_ids_1d != -1)
                valid_future_ids = future_ids_1d[valid_mask]
                valid_future_acts = future_acts_2d[valid_mask]
                valid_future_mask = future_mask_1d[valid_mask]
                match_idx = np.searchsorted(valid_future_ids, current_ids_1d)

                reordered_acts = np.zeros_like(future_acts_2d)
                reordered_mask = np.ones_like(future_mask_1d, dtype=bool)
                in_bounds = (match_idx >= 0) & (match_idx < len(valid_future_ids))
                valid_positions = np.where(in_bounds)[0]

                if len(valid_positions) == 0:
                    aligned_future_acton_sum[b, t] = reordered_acts
                    aligned_future_mask_bool[b, t] = reordered_mask
                    continue
                exact_match_array = (
                    valid_future_ids[ match_idx[valid_positions] ] == current_ids_1d[valid_positions]
                )
                exact_match = np.zeros_like(in_bounds, dtype=bool)
                exact_match[valid_positions] = exact_match_array
                reordered_acts[exact_match] = valid_future_acts[ match_idx[exact_match] ]
                reordered_mask[exact_match] = valid_future_mask[ match_idx[exact_match] ]
                aligned_future_acton_sum[b, t] = reordered_acts
                aligned_future_mask_bool[b, t] = reordered_mask
        combined_mask_bool = partner_mask_bool | aligned_future_mask_bool

        return aligned_future_acton_sum, combined_mask_bool
    
    @staticmethod
    def _get_multi_class_pos(pos):
        """
        Convert continuous pos to multi-class discrete pos based on x, y.
        """
        x, y = pos[..., 0], pos[..., 1]
        
        # Define bins for discretization (-1 to 1 with 8 bins)
        bins = np.linspace(-0.1, 0.1, 9)
        
        # Digitize x and y into 8 categories (0 to 7)
        x_bins = np.digitize(x, bins) - 1
        y_bins = np.digitize(y, bins) - 1
        
        # Ensure values are within valid range (0 to 7)
        x_bins = np.clip(x_bins, 0, 7)
        y_bins = np.clip(y_bins, 0, 7)
        
        discrete_pos = x_bins * 8 + y_bins
        return discrete_pos
    
    def _transform_relative_pos(self, aux_info, ego_global_pos, ego_global_rot, future_step):
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
    
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        idx1 = int(idx1)
        idx2 = int(idx2)
        # row, column -> 
        batch = ()
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name in ['actions']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
                    elif var_name in ['other_pos', 'aux_mask']:
                        data = self.__dict__[var_name][idx1, idx2]
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
                    if var_name == 'valid_masks':
                        ego_mask_data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len]
                        if ego_mask_data != True:
                            print('Not valid data!!!')
            batch = batch + (torch.tensor([idx1, idx2]),)
        else:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    data = self.__dict__[var_name][idx]
                    batch = batch + (data, )
        return batch
    

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    data_name = [100, 500, 1000, 5000]
    data = np.load(f"/data/full_version/processed/final/training_trajectory_1000.npz")
    actions =data['actions'].reshape(-1, 3)
    valid_masks = data['dead_mask'].reshape(-1)
    valid_masks = 1 - valid_masks
    action_mask = (np.abs(actions[:, 1]) >  0.5) | (np.abs(actions[:, 0]) >  5) | (np.abs(actions[:, -1]) > 0.2)
    action_mask = action_mask.reshape(-1)
    valid_masks[action_mask] = 0
    valid_masks = valid_masks.astype('bool')
    filtered_actions = actions[valid_masks]
    dx = filtered_actions[:, 0]
    dy = filtered_actions[:, 1]
    dyaw = filtered_actions[:, 2]
    print(f'dx max {dx.max():.3f} min {dx.min():.3f} dy max {dy.max():.3f} min {dy.max():.3f} dyaw max {dyaw.max():.3f} min {dyaw.max():.3f} ')
    print(f'dx mean {dx.mean():.3f} std {dx.std():.3f} dy mean {dy.mean():.3f} std {dy.std():.3f} dyaw max {dyaw.mean():.3f} std {dyaw.std():.3f} ')