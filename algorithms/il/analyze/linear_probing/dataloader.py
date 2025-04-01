import torch
import numpy as np

from pygpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS


class OtherFutureDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, ego_global_pos, ego_global_rot, masks=None, partner_mask=None, road_mask=None, other_info=None,
                 rollout_len=5, pred_len=1, aux_future_step=1):
        # obs
        self.obs = obs
        B, T, F = obs.shape
        new_shape = (B, T + rollout_len - 1, F)
        new_obs = np.zeros(new_shape, dtype=self.obs.dtype)
        new_obs[:, rollout_len - 1:] = obs # This is more cheaper than concatenate
        self.obs = new_obs

        # actions, 
        self.actions = actions
        
        # masks
        valid_masks = 1 - masks
        max_actions = np.max(actions, axis=-1)
        min_actions = np.min(actions, axis=-1)
        action_mask = (max_actions == 6) | (min_actions == -6) | (actions[..., -1] >= 3.14)  | (actions[..., -1] <= -3.14)
        valid_masks[action_mask] = 0
        new_shape = (B, T + rollout_len - 1)
        new_valid_mask = np.zeros(new_shape, dtype=self.obs.dtype)
        new_valid_mask[:, rollout_len - 1:] = valid_masks
        self.valid_masks = new_valid_mask.astype('bool')

        # partner_mask
        partner_info = obs[..., 6:1276].reshape(B, T, 127, 10)[..., :4]
        aux_info, aux_mask = self._make_aux_info(partner_mask, other_info, partner_info, future_timestep=aux_future_step)
        self.aux_mask = aux_mask.astype('bool')
        new_shape = (B, T + rollout_len - 1, 127)
        new_partner_mask = np.full(new_shape, 2, dtype=np.float32)
        new_partner_mask[:, rollout_len - 1:] = partner_mask
        self.partner_mask = np.where(new_partner_mask == 2, 1, 0).astype('bool')

        # road_mask
        self.road_mask = road_mask
        new_shape = (B, T + rollout_len - 1, 200)
        new_road_mask = np.ones(new_shape)
        new_road_mask[:, rollout_len - 1:] = road_mask
        self.road_mask = new_road_mask.astype('bool')
        
        # linear probing
        current_relative_pos = self._transform_relative_pos(aux_info, partner_info[..., -1:], ego_global_pos, ego_global_rot, future_step=aux_future_step)
        current_relative_pos[aux_mask] = 0
        self.other_pos = self._get_multi_class_pos(current_relative_pos)
        self.other_actions = self._get_multi_class_actions(aux_info[..., 4:7])

        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.other_info = aux_info
        self.full_var = ['obs', 'actions', 'valid_masks', 'partner_mask', 'road_mask',
                         'other_info', 'aux_mask', 'other_pos', 'other_actions']
    def __len__(self):
        return len(self.valid_indices)

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
    
    def _transform_relative_pos(self, aux_info, partner_rot, ego_global_pos, ego_global_rot, future_step):
        """transform time t relative pos to current relative pos"""
        # 1. transform t-relative pos to t-global pos
        # get partner's relative pos and rot at time t
        t_partner_pos = aux_info[..., 1:3] * MAX_REL_AGENT_POS
        t_partner_rot = np.zeros_like(partner_rot)
        t_partner_rot[:, :-future_step] = partner_rot[:, future_step:]
        
        # get ego's global pos and rot at time t
        t_ego_global_pos = np.zeros_like(ego_global_pos)
        t_ego_global_rot = np.zeros_like(ego_global_rot)
        t_ego_global_pos[:, :-future_step] = ego_global_pos[:, future_step:]
        t_ego_global_rot[:, :-future_step] = ego_global_rot[:, future_step:]
        
        theta = t_ego_global_rot + t_partner_rot.squeeze(-1)
        t_partner_global_pos_x = t_ego_global_pos[..., 0, None] + t_partner_pos[..., 0] * np.cos(theta) - t_partner_pos[..., 1] * np.sin(theta)
        t_partner_global_pos_y = t_ego_global_pos[..., 1, None] + t_partner_pos[..., 0] * np.sin(theta) + t_partner_pos[..., 1] * np.cos(theta)
        
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

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))

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
                    elif var_name in ['other_info', 'aux_mask', 'other_pos', 'other_actions']:
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

class EgoFutureDataset(OtherFutureDataset):
    def __init__(self, obs, actions, ego_global_pos, masks=None, partner_mask=None, road_mask=None, other_info=None,
                 rollout_len=5, pred_len=1, ego_future_step=1):
        # obs
        obs_pad = np.zeros((obs.shape[0], rollout_len - 1, *obs.shape[2:]), dtype=np.float32)
        self.obs = np.concatenate([obs_pad, obs], axis=1)
        
        # actions
        self.actions = actions
        
        # current ego mask        
        valid_masks = 1 - masks
        valid_masks_pad = np.zeros((valid_masks.shape[0], rollout_len - 1, *valid_masks.shape[2:]), dtype=np.float32).astype('bool')
        self.cur_valid_mask = np.concatenate([valid_masks_pad, valid_masks], axis=1).astype('bool')
        
        # future ego mask
        future_valid_mask_pad = np.zeros((valid_masks.shape[0], ego_future_step, *valid_masks.shape[2:]), dtype=np.float32)
        future_valid_masks = np.concatenate([valid_masks, future_valid_mask_pad], axis=1).astype('bool')[:, ego_future_step:]
        self.future_valid_mask = self.cur_valid_mask[:,rollout_len - 1:] & future_valid_masks

        # partner_mask
        partner_mask_pad = np.full((partner_mask.shape[0], rollout_len - 1, *partner_mask.shape[2:]), 2, dtype=np.float32)
        partner_mask = np.concatenate([partner_mask_pad, partner_mask], axis=1)
        self.partner_mask = np.where(partner_mask == 2, 1, 0).astype('bool')
        
        # road_mask
        self.road_mask = road_mask
        road_mask_pad = np.ones((road_mask.shape[0], rollout_len - 1, *road_mask.shape[2:]), dtype=np.float32).astype('bool')
        self.road_mask = np.concatenate([road_mask_pad, self.road_mask], axis=1).astype('bool')
        
        # linear probing
        future_pos = self._transform_relative_pos(ego_global_pos, future_step=ego_future_step)
        self.future_pos = self._get_multi_class_pos(future_pos)
        future_actions_pad = np.zeros((actions.shape[0], ego_future_step, *actions.shape[2:]), dtype=np.float32)
        future_actions = np.concatenate([actions, future_actions_pad], axis=1)[:, ego_future_step:]
        self.future_actions = self._get_multi_class_actions(future_actions)
          
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'future_pos', 'future_actions', 'cur_valid_mask', 'future_valid_mask', 'partner_mask', 'road_mask']

    def __len__(self):
        return len(self.valid_indices)
    
    def _compute_valid_indices(self):
        N, T = self.cur_valid_mask.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.cur_valid_mask[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
    
    @staticmethod
    def _transform_relative_pos(ego_global_pos, future_step):
        """transform global pos to current relative pos"""
        current_relative_pos = np.zeros_like(ego_global_pos)
        ego_current_pos = ego_global_pos[:, :-future_step]
        ego_future_pos = ego_global_pos[:, future_step:]
        
        delta_x = ego_future_pos[..., 0] - ego_current_pos[..., 0]
        delta_y = ego_future_pos[..., 1] - ego_current_pos[..., 1]
        
        current_relative_pos_x = 2 * ((delta_x - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos_y = 2 * ((delta_y - MIN_REL_AGENT_POS) / (MAX_REL_AGENT_POS - MIN_REL_AGENT_POS)) - 1
        current_relative_pos[:, :-future_step, :] = np.stack([current_relative_pos_x, current_relative_pos_y], axis=-1)
        
        return current_relative_pos
        
    
    @staticmethod
    def _get_multi_class_pos(pos):
        """
        Convert continuous pos to multi-class discrete pos based on x, y.
        """
        x, y = pos[..., 0], pos[..., 1]
        
        # Define bins for discretization (-1 to 1 with 8 bins)
        bins = np.linspace(-0.05, 0.05, 9)
        
        # Digitize x and y into 8 categories (0 to 7)
        x_bins = np.digitize(x, bins) - 1
        y_bins = np.digitize(y, bins) - 1
        
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
                    if var_name in ['obs', 'cur_valid_mask','road_mask', 'partner_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name in ['actions', 'future_pos', 'future_actions', 'future_valid_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
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
    from algorithms.il.analyze.linear_probing.dataloader import OtherFutureDataset

    data = np.load("/data/tom_v5/train_trajectory_100.npz")
    global_data = np.load("/data/tom_v5/linear_probing/global_train_trajectory_100.npz")

    # expert_data_loader = DataLoader(
    #     OtherFutureDataset(
    #         data['obs'], data['actions'], global_data['ego_global_pos'], global_data['ego_global_rot'],
    #         data['dead_mask'], data['partner_mask'], data['road_mask'], data['other_info'], 
    #         rollout_len=5, pred_len=1, aux_future_step=1
    #     ),
    #     batch_size=256,
    #     shuffle=True,
    #     num_workers=4,
    # )

    # total_pos_counts = np.zeros(64, dtype=int)      # other_pos는 0~63
    # total_action_counts = np.zeros(64, dtype=int)   # other_actions는 0~63

    # for batch in expert_data_loader:
    #     _, _, _, _, _, _, _, aux_mask, other_pos, other_actions = batch

    #     # 텐서를 CPU로 옮기고 numpy로 변환
    #     pos_vals = other_pos[~aux_mask].cpu().numpy().astype(int)
    #     action_vals = other_actions[~aux_mask].cpu().numpy().astype(int)

    #     # np.bincount로 개수 세기
    #     total_pos_counts += np.bincount(pos_vals, minlength=64)
    #     total_action_counts += np.bincount(action_vals, minlength=64)

    # # 누적된 값으로 바 그래프 그리기

    # # other_pos 분포
    # plt.figure(figsize=(10, 4))
    # plt.bar(range(64), total_pos_counts, color='blue', alpha=0.7)
    # plt.xlabel("other_pos (0~63)")
    # plt.ylabel("count")
    # plt.title("other_pos distribution")
    # plt.xticks(range(0, 64, 4))
    # plt.grid(axis="y", linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.savefig('Other pos dist.png', dpi=300)

    # # other_actions 분포
    # plt.figure(figsize=(16, 4))
    # plt.bar(range(64), total_action_counts, color='red', alpha=0.7)
    # plt.xlabel("other_actions (0~63)")
    # plt.ylabel("count")
    # plt.title("other_actions distribution")
    # plt.xticks(range(64))
    # plt.grid(axis="y", linestyle="--", alpha=0.5)
    # plt.tight_layout()
    # plt.savefig('Other action dist.png', dpi=300)
    
    expert_data_loader = DataLoader(
        EgoFutureDataset(
            data['obs'], data['actions'], global_data['ego_global_pos'],
            data['dead_mask'], data['partner_mask'], data['road_mask'], data['other_info'], 
            rollout_len=5, pred_len=1, ego_future_step=40
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
        _, _, future_pos, future_action, _, future_valid_mask, *_ = batch

        # 텐서를 CPU로 옮기고 numpy로 변환
        pos_vals = future_pos[future_valid_mask].cpu().numpy().astype(int)
        action_vals = future_action[future_valid_mask].cpu().numpy().astype(int)

        # np.bincount로 개수 세기
        total_pos_counts += np.bincount(pos_vals, minlength=64)
        total_action_counts += np.bincount(action_vals, minlength=64)

    # 누적된 값으로 바 그래프 그리기

    # ego_future_pos 분포
    plt.figure(figsize=(10, 4))
    plt.bar(range(64), total_pos_counts, color='blue', alpha=0.7)
    plt.xlabel("ego_future_pos (0~63)")
    plt.ylabel("count")
    plt.title("ego_future_pos distribution")
    plt.xticks(range(0, 64, 4))
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('ego future pos dist.png', dpi=300)

    # ego_future_actions 분포
    plt.figure(figsize=(16, 4))
    plt.bar(range(64), total_action_counts, color='red', alpha=0.7)
    plt.xlabel("ego_future_actions (0~63)")
    plt.ylabel("count")
    plt.title("ego_future_actions distribution")
    plt.xticks(range(64))
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('ego future action dist.png', dpi=300)