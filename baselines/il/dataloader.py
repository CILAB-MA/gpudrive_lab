import torch
import numpy as np

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, other_info=None, road_mask=None,
                 rollout_len=1, pred_len=1):
        # obs
        self.obs = obs
        obs_pad = np.zeros((obs.shape[0], rollout_len - 1, *obs.shape[2:]), dtype=np.float32)
        self.obs = np.concatenate([obs_pad, self.obs], axis=1)
        self.masks = 1 - masks
        dead_masks_pad = np.ones((self.masks.shape[0], rollout_len - 1, *self.masks.shape[2:]), dtype=np.float32)
        self.masks = np.concatenate([dead_masks_pad, self.masks], axis=1).astype('bool')

        self.road_mask = road_mask
        road_mask_pad = np.zeros((road_mask.shape[0], rollout_len - 1, *road_mask.shape[2:]), dtype=np.float32)
        self.road_mask = np.concatenate([road_mask_pad, self.road_mask], axis=1).astype('bool')
        
        self.actions = actions
        self.other_info = other_info
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.use_mask = False

        self.partner_mask = other_info[..., -1]
        partner_mask_pad = np.zeros((self.partner_mask.shape[0], rollout_len - 1, *self.partner_mask.shape[2:]), dtype=np.float32)
        self.partner_mask = np.concatenate([partner_mask_pad, self.partner_mask], axis=1).astype('bool')
        if self.masks is not None:
            self.use_mask = True
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'actions', 'masks', 'partner_mask', 'road_mask'] # todo: add other info

    def __len__(self):
        return len(self.valid_indices)

    def _compute_valid_indices(self):
        N, T = self.masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
    
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        # row, column -> 
        batch = ()
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name == 'actions':
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
                    elif var_name == 'masks':
                        data = self.__dict__[var_name][idx1 ,idx2 + self.rollout_len + self.pred_len - 2] # idx 0 -> (0, 10 + 5 - 2) -> (0, 13) & padding = 9 -> end with last action timestep
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    batch = batch + (data, )
                    if var_name == 'masks':
                        ego_mask_data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len]
                        batch = batch + (ego_mask_data, )
        else:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    data = self.__dict__[var_name][idx]
                    batch = batch + (data, )   
        return batch