import torch
import numpy as np
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS


class FutureDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, ego_global_pos, ego_global_rot, masks=None, partner_mask=None, road_mask=None,
                 rollout_len=5, pred_len=1, future_step=1, exp='other', xy_range=None):
        # obs
        self.obs = obs
        B, T, F = obs.shape
        new_shape = (B, T + rollout_len - 1, F)
        new_obs = np.zeros(new_shape, dtype=self.obs.dtype)
        new_obs[:, rollout_len - 1:] = obs # This is more cheaper than concatenate
        self.obs = new_obs
        self.actions = actions

        # masks
        valid_masks = 1 - masks
        action_mask = (np.abs(actions[..., 1]) >  0.5) | (np.abs(actions[..., 0]) >  5) | (np.abs(actions[..., -1]) > 0.2)
        valid_masks[action_mask] = 0
        new_shape = (B, T + rollout_len - 1)
        new_valid_mask = np.zeros(new_shape, dtype=self.obs.dtype)
        new_valid_mask[:, rollout_len - 1:] = valid_masks
        self.valid_masks = new_valid_mask.astype('bool')

        if exp == 'other':
            # future partner_mask
            partner_info = obs[..., 6:128 * 6].reshape(B, T, 127, 6)[..., :4]
            aux_info, aux_mask = self._make_aux_info(partner_mask, partner_info, future_timestep=future_step)
            self.aux_mask = aux_mask.astype('bool')
        new_shape = (B, T + rollout_len - 1, 127)
        new_partner_mask = np.full(new_shape, 2, dtype=np.float32)
        new_partner_mask[:, rollout_len - 1:] = partner_mask
        self.partner_mask = np.where(new_partner_mask == 2, 1, 0).astype('bool')

        if exp == 'ego':
            # future ego mask
            future_valid_mask_pad = np.zeros((self.valid_masks.shape[0], future_step, *self.valid_masks.shape[2:]), dtype=np.float32)
            future_valid_masks = np.concatenate([valid_masks, future_valid_mask_pad], axis=1).astype('bool')[:, future_step:]
            self.future_valid_mask = self.valid_masks[:,rollout_len - 1:] & future_valid_masks
        
        # road_mask
        self.road_mask = road_mask
        new_shape = (B, T + rollout_len - 1, 200)
        new_road_mask = np.ones(new_shape)
        new_road_mask[:, rollout_len - 1:] = road_mask
        self.road_mask = new_road_mask.astype('bool')
        if exp == 'other':
            # future other pos
            current_relative_other_pos = self._transform_relative_other_pos(aux_info, ego_global_pos, ego_global_rot, future_step=future_step)
            current_relative_other_pos[aux_mask] = 0
            self.other_pos = self._get_multi_class_pos(current_relative_other_pos)
        else:
            # future ego pos
            current_relative_ego_pos = self._transform_relative_ego_pos(ego_global_pos, ego_global_rot, future_step=future_step,
                                                                    )
            self.ego_pos = self._get_multi_class_pos(current_relative_ego_pos, xy_range)

            # ego? -> current_relative_pos[aux_mask] = 0
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'valid_masks', 'partner_mask', 'road_mask']
        if exp == 'other':
            self.full_var += ['aux_mask', 'other_pos']
        else:
            self.full_var += ['future_valid_mask', 'ego_pos']
    def __len__(self):
        return len(self.valid_indices)

    def _make_aux_info(self, partner_mask, partner_info, future_timestep):
        partner_mask_bool = np.where(partner_mask == 0, 0, 1).astype(bool)
        action_valid_mask = np.where(partner_mask == 0, 1, 0).astype(bool)
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
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
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

    @staticmethod
    def _get_multi_class_actions(actions):
        """
        Convert continuous actions to multi-class discrete actions based on dyaw.
        """
        dx = actions[..., 0]
        dy = actions[..., 1]
        dyaw = actions[..., 2]
        
        dx_bins = np.linspace(-3, 3, 5)  # 4 bins for dx
        dy_bins = np.linspace(-3, 3, 5)  # 4 bins for dy
        dyaw_bins = np.linspace(-np.pi / 4, np.pi / 4, 5)  # 4 bins for dyaw
        
        dx_bin = np.digitize(dx, dx_bins) - 1
        dy_bin = np.digitize(dy, dy_bins) - 1
        dyaw_bin = np.digitize(dyaw, dyaw_bins) - 1

        # Ensure indices are within range
        dx_bin = np.clip(dx_bin, 0, 4 - 1)
        dy_bin = np.clip(dy_bin, 0, 4 - 1)
        dyaw_bin = np.clip(dyaw_bin, 0, 4 - 1)

        # Compute single discrete index
        discrete_action = (dx_bin * 4 * 4) + (dy_bin * 4) + dyaw_bin

        return discrete_action
        
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
                    elif var_name == 'valid_masks':
                        data = self.__dict__[var_name][idx1 ,idx2 + self.rollout_len + self.pred_len - 2] # idx 0 -> (0, 10 + 5 - 2) -> (0, 13) & padding = 9 -> end with last action timestep
                    elif var_name in ['aux_mask', 'other_pos', 'future_valid_mask', 'ego_pos']:
                        data = self.__dict__[var_name][idx1, idx2]
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
                    if var_name == 'valid_masks':
                        ego_mask_data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len]
                        batch = batch + (ego_mask_data, )
        else:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    data = self.__dict__[var_name][idx]
                    batch = batch + (data, )
        return batch


if __name__ == "__main__":
    import numpy as np
    import os
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    exp = 'ego'
    future_step = 35
    # xy_range_dict={5: [(-0.0125, 0.0125), (-0.00625, 0.00625)],
    #              15: [(-0.0125, 0.025), (-0.00625, 0.00625)],
    #              25: [(-0.0175, 0.0375), (-0.0125, 0.0125)],
    #              35: [(-0.025, 0.05), (-0.025, 0.025)]} 
    data = np.load("/data/full_version/processed/final/training_trajectory_1000.npz")
    # data = np.load("/data/ICRA_Workshop/tom_v5/train_trajectory_1000.npz")
    global_data = np.load("/data/full_version/processed/final/global_training_trajectory_1000.npz")
    # global_data = np.load("/data/ICRA_Workshop/tom_v5/linear_probing/global_train_trajectory_1000.npz")
    expert_data_loader = DataLoader(
        FutureDataset(
            data['obs'], data['actions'], global_data['ego_global_pos'], global_data['ego_global_rot'],
            data['dead_mask'], data['partner_mask'], data['road_mask'], 
            rollout_len=5, pred_len=1, future_step=future_step, exp=exp, 
            # xy_range=xy_range_dict[future_step]
        ),
        batch_size=256,
        shuffle=True,

        num_workers=4,
    )
    del data
    del global_data

    total_pos_counts = np.zeros(64, dtype=int)      # other_pos는 0~63
    total_action_counts = np.zeros(64, dtype=int)   # other_actions는 0~63

    
    for batch in expert_data_loader:
        obs, mask, valid_mask, partner_mask, road_mask, aux_mask, other_pos = batch
        aux_mask = ~aux_mask if exp == 'other' else aux_mask
        pos_vals = other_pos[aux_mask].cpu().numpy().astype(int)

        total_pos_counts += np.bincount(pos_vals, minlength=64)
    
    # # ego_future_pos 분포
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(64), total_pos_counts, color='blue', alpha=0.7)
    # plt.xlabel(f"{exp}_future_pos (0~63)")
    # plt.ylabel("count")
    # plt.title(f"{exp}_future_pos distribution (future: {future_step})")
    # plt.xticks(range(0, 64, 4))
    # plt.grid(axis="y", linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.savefig(f'{exp} future pos dist.png', dpi=300)

    import seaborn as sns

    heatmap_data = total_pos_counts.reshape(8, 8)

    heatmap_data = np.flipud(np.fliplr(heatmap_data))

    index_matrix = np.arange(64).reshape(8, 8)
    index_matrix = np.flipud(np.fliplr(index_matrix))  
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        heatmap_data, 
        cmap="YlGnBu", 
        annot=index_matrix, 
        fmt='d', 
        square=True,
        cbar_kws={"label": "Count"}, 
        linewidths=0.5
    )
    plt.title(f"{exp} Heatmap (Index Annoted, future: {future_step})")
    plt.ylabel(f"{xy_range_dict[future_step][0][0]} ~ {xy_range_dict[future_step][0][1]}")
    plt.xlabel(f"{xy_range_dict[future_step][1][0]} ~ {xy_range_dict[future_step][1][1]}")
    plt.tight_layout()
    plt.savefig(f'{exp}_future_pos_heatmap_indexed.png', dpi=300)