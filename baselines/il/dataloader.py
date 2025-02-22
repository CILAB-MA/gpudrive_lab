import torch
import numpy as np

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, partner_mask=None, road_mask=None, other_info=None,
                 rollout_len=5, pred_len=1, aux_future_step=None):
        # obs
        self.obs = obs
        B, T, *_ = obs.shape
        obs_pad = np.zeros((obs.shape[0], rollout_len - 1, *obs.shape[2:]), dtype=np.float32)
        self.obs = np.concatenate([obs_pad, self.obs], axis=1)

        # actions
        self.actions = actions
        
        # masks
        self.valid_masks = 1 - masks
        valid_masks_pad = np.zeros((self.valid_masks.shape[0], rollout_len - 1, *self.valid_masks.shape[2:]), dtype=np.float32).astype('bool')
        self.valid_masks = np.concatenate([valid_masks_pad, self.valid_masks], axis=1).astype('bool')
        self.use_mask = True if self.valid_masks is not None else False

        # partner_mask
        partner_mask_pad = np.full((partner_mask.shape[0], rollout_len - 1, *partner_mask.shape[2:]), 2, dtype=np.float32)
        self.aux_valid_mask = None
        partner_info = obs[..., 6:1276].reshape(B, T, 127, 10)[..., :4]
        self.aux_mask = None
        if other_info is not None:
            aux_info, aux_mask = self._make_aux_info(partner_mask, other_info, partner_info, 
                                                     future_timestep=aux_future_step)
            self.aux_mask = aux_mask.astype('bool')
        partner_mask = np.concatenate([partner_mask_pad, partner_mask], axis=1)
        self.partner_mask = np.where(partner_mask == 2, 1, 0).astype('bool')
        # road_mask
        self.road_mask = road_mask
        road_mask_pad = np.ones((road_mask.shape[0], rollout_len - 1, *road_mask.shape[2:]), dtype=np.float32).astype('bool')
        self.road_mask = np.concatenate([road_mask_pad, self.road_mask], axis=1).astype('bool')
          
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.other_info = aux_info
        self.full_var = ['obs', 'actions', 'valid_masks', 'partner_mask', 'road_mask',
                         'other_info', 'aux_mask']

    def __len__(self):
        return len(self.valid_indices)

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
    
    def _make_aux_info(self, partner_mask, info, partner_info, future_timestep):
        partner_mask_bool = np.where(partner_mask == 2, 1, 0).astype(bool)
        action_valid_mask = np.where(partner_mask == 0, 1, 0).astype(bool)
        info[..., :-1] *= action_valid_mask[..., np.newaxis]
        current_info_id = info[:, :, :, -1]
        all_infos = np.concatenate([partner_info, info], axis=-1)
        other_info_pad = np.zeros((all_infos.shape[0], future_timestep, *all_infos.shape[2:]), dtype=np.float32)
        partner_mask_pad = np.full((partner_mask.shape[0], future_timestep, *partner_mask.shape[2:]), 2, dtype=np.float32)

        future_mask = np.concatenate([partner_mask, partner_mask_pad], axis=1)
        future_mask_bool = np.where(future_mask == 2, 1, 0).astype(bool)[:, future_timestep:]
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
                    elif var_name in ['other_info', 'aux_mask']:
                        data = self.__dict__[var_name][idx1, idx2]
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
                    if var_name == 'valid_masks':
                        ego_mask_data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len]
                        batch = batch + (ego_mask_data, )
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
    
    data = np.load("/data/tom_v4/train_subset/trajectory_200.npz")
    
    expert_data_loader = DataLoader(
        ExpertDataset(
            data['obs'], data['actions'], 
            data['dead_mask'], data['partner_mask'], data['road_mask'], data['other_info'], 
            rollout_len=5, pred_len=1, aux_future_step=1
        ),
        batch_size=256,
        shuffle=True,
        num_workers=os.cpu_count(),
        prefetch_factor=4,
        pin_memory=True
    )

    for i, batch in enumerate(expert_data_loader):
        batch_size = batch[0].size(0)

        if len(batch) == 9:
            obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info, aux_mask, data_idx = batch
        elif len(batch) == 7:
            obs, expert_action, masks, ego_masks, partner_masks, road_masks, data_idx = batch 
        elif len(batch) == 4:
            obs, expert_action, masks, data_idx = batch
        else:
            obs, expert_action, data_idx = batch
