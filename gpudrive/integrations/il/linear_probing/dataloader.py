import torch
import numpy as np
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS, LOG_TRAJECTORY_LEN


class FutureDataset(torch.utils.data.Dataset):
    def __init__(self, obs, masks, partner_mask, road_mask, future_pos, future_valid_mask, trajectory_type,
                 rollout_len=5, pred_len=1):
        # obs
        self.obs = obs

        # masks
        self.valid_masks = masks
        self.partner_mask = partner_mask
        self.road_mask = road_mask
        
        # linear probing
        self.future_pos = future_pos
        self.future_valid_mask = future_valid_mask
        
        # trajectory type (straight, turn, reverse, normal)
        self.trajectory_type = trajectory_type

        # etc.
        self.num_timestep = LOG_TRAJECTORY_LEN - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
        self.full_var = ['obs', 'partner_mask', 'road_mask', 'future_pos', 'future_valid_mask', 'trajectory_type']

    def __len__(self):
        return len(self.valid_indices)

    def _compute_valid_indices(self):
        _, T = self.valid_masks.shape
        valid_time = np.arange(T - (self.rollout_len + self.pred_len - 2))
        valid_idx1, valid_idx2 = np.where(self.valid_masks[:, valid_time + self.rollout_len + self.pred_len - 2] == 1)
        valid_idx2 = valid_time[valid_idx2]
        return list(zip(valid_idx1, valid_idx2))
        
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
                    elif var_name in ['future_pos', 'future_valid_mask']:
                        data = self.__dict__[var_name][idx1, idx2]
                    elif var_name in ['trajectory_type']:
                        if self.__dict__[var_name].ndim == 2:
                            # ego trajectory type
                            data = self.__dict__[var_name][idx1]
                        elif self.__dict__[var_name].ndim == 3:
                            # other trajectory type
                            data = self.__dict__[var_name][idx1, idx2]
                        else:
                            raise ValueError(f"Not in data {self.__dict__[var_name].ndim}. Your input is {var_name}")
                    else:
                        raise ValueError(f"Not in data {self.full_var}. Your input is {var_name}")
                    if isinstance(data, np.ndarray):
                        data = torch.tensor(data)
                    batch = batch + (data, )
        else:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    data = self.__dict__[var_name][idx]
                    batch = batch + (data, )
        return batch
