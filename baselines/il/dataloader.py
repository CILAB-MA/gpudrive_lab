import torch
import numpy as np

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, partner_mask=None, road_mask=None, other_info=None,
                 rollout_len=5, pred_len=1, aux_future_step=None):
        # obs
        self.obs = obs
        obs_pad = np.zeros((obs.shape[0], rollout_len - 1, *obs.shape[2:]), dtype=np.float32)
        self.obs = np.concatenate([obs_pad, self.obs], axis=1)

        # actions
        self.actions = actions
        
        # masks
        self.valid_masks = 1 - masks
        dead_masks_pad = np.zeros((self.valid_masks.shape[0], rollout_len - 1, *self.valid_masks.shape[2:]), dtype=np.float32).astype('bool')
        self.valid_masks = np.concatenate([dead_masks_pad, self.valid_masks], axis=1).astype('bool')
        self.use_mask = True if self.valid_masks is not None else False

        # partner_mask
        partner_mask_pad = np.full((partner_mask.shape[0], rollout_len - 1, *partner_mask.shape[2:]), 2, dtype=np.float32)
        self.aux_valid_mask = None
        
        aux_mask = np.empty_like(partner_mask) if aux_future_step else None
        if aux_future_step:
            # Aux Mask
            aux_mask[:, :-aux_future_step - 1, :] = partner_mask[:, aux_future_step + 1:, :].copy()
            aux_mask[:, -aux_future_step - 1:, :] = True
            aux_mask_pad = np.ones_like(partner_mask_pad)
            aux_mask = np.concatenate([aux_mask_pad, aux_mask], axis=1).astype('bool')
            self.aux_mask = aux_mask

        partner_mask = np.concatenate([partner_mask_pad, partner_mask], axis=1)
        self.partner_mask = np.where(partner_mask == 2, 1, 0).astype('bool')
        # road_mask
        self.road_mask = road_mask
        road_mask_pad = np.ones((road_mask.shape[0], rollout_len - 1, *road_mask.shape[2:]), dtype=np.float32).astype('bool')
        self.road_mask = np.concatenate([road_mask_pad, self.road_mask], axis=1).astype('bool')
        
        if other_info is not None:
            # other_info
            other_info_pad = np.zeros((other_info.shape[0], rollout_len - 1, *other_info.shape[2:]), dtype=np.float32)
            other_info[:, :-aux_future_step, ...] = other_info[:, aux_future_step:, ...]
            other_info[:, -aux_future_step:, ...] = 0
            other_info = np.concatenate([other_info_pad, other_info], axis=1)
            self.other_info = other_info
            self.aux_future_step = aux_future_step
        
        
        self.num_timestep = 1 if len(obs.shape) == 2 else obs.shape[1] - rollout_len - pred_len + 2
        self.rollout_len = rollout_len
        self.pred_len = pred_len
        self.valid_indices = self._compute_valid_indices()
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
    
    def __getitem__(self, idx):
        idx1, idx2 = self.valid_indices[idx]
        idx1 = int(idx1)
        idx2 = int(idx2)
        # row, column -> 
        batch = ()
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask', 'other_info', 'aux_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name in ['actions']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.pred_len] # idx 0 -> (0, 0:5) -> start with first timestep
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
    
if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    
    data = np.load("/data/tom_v3/train_subset/trajectory_200.npz")
    
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

        if len(batch) == 8:
            obs, expert_action, masks, ego_masks, partner_masks, road_masks, other_info, aux_mask = batch
        elif len(batch) == 6:
            obs, expert_action, masks, ego_masks, partner_masks, road_masks = batch 
        elif len(batch) == 3:
            obs, expert_action, masks = batch
        else:
            obs, expert_action = batch
