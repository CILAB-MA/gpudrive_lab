import torch
import numpy as np

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, partner_mask=None, road_mask=None, other_info=None,
                 rollout_len=5, pred_len=1, tom_time='only_pred'):
        # obs
        self.obs = obs
        obs_pad = np.zeros((obs.shape[0], rollout_len - 1, *obs.shape[2:]), dtype=np.float32)
        self.obs = np.concatenate([obs_pad, self.obs], axis=1)
        
        # actions
        self.actions = actions
        
        # masks
        self.valid_masks = 1 - masks
        dead_masks_pad = np.zeros((self.valid_masks.shape[0], rollout_len - 1, *self.valid_masks.shape[2:]), dtype=np.float32)
        self.valid_masks = np.concatenate([dead_masks_pad, self.valid_masks], axis=1).astype('bool')
        self.use_mask = True if self.valid_masks is not None else False

        # partner_mask
        partner_mask_pad = np.full((partner_mask.shape[0], rollout_len - 1, *partner_mask.shape[2:]), 2, dtype=np.float32)
        partner_mask = np.concatenate([partner_mask_pad, partner_mask], axis=1)
        self.partner_mask = np.where(partner_mask == 2, 1, 0).astype('bool')
        
        # road_mask
        self.road_mask = road_mask
        road_mask_pad = np.ones((road_mask.shape[0], rollout_len - 1, *road_mask.shape[2:]), dtype=np.float32)
        self.road_mask = np.concatenate([road_mask_pad, self.road_mask], axis=1).astype('bool')
        
        # other_info
        self.other_info = other_info
        self.aux_valid_mask  = None
        if other_info != None:
            other_info_pad = np.zeros((other_info.shape[0], rollout_len - 1, *self.other_info.shape[2:]), dtype=np.float32)
            self.other_info = np.concatenate([other_info_pad, self.other_info], axis=1)
            # ToM
            self.aux_valid_mask = np.where(partner_mask == 0, 1, 0).astype('bool')

        if tom_time == 'only_pred':
            self.tom_timestep = pred_len
        elif tom_time == 'understand_pred':
            self.tom_timestep = rollout_len
        
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'valid_masks', 'partner_mask', 'road_mask',
                         'other_info', 'aux_valid_mask']

    def __len__(self):
        return len(self.valid_indices)

    def _compute_valid_indices(self):
        N, T = self.valid_masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
    
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        # row, column -> 
        batch = ()
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask', 'aux_valid_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name in ['actions']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
                    elif var_name in ['other_info']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.tom_timestep] # idx 0 -> (0, 0:6) -> start with first timestep
                    elif var_name == 'valid_masks':
                        data = self.__dict__[var_name][idx1 ,idx2 + self.rollout_len + self.pred_len - 2] # idx 0 -> (0, 10 + 5 - 2) -> (0, 13) & padding = 9 -> end with last action timestep
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