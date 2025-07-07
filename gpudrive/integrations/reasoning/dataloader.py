import torch
import numpy as np
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS

class ReasoningDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, masks=None, partner_mask=None, road_mask=None,
                 rollout_len=5, pred_len=1, questions=None, answers=None, qa_masks=None,
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
        self.other_info = None
        self.other_pos = None
        if use_tom:
            self.questions = questions
            self.answers = answers
            self.qa_masks = qa_masks
            self.qa_len = self.questions.shape[1]
            self.qa_num_sample = 50
        self.partner_mask = np.pad(partner_mask, ((0, 0), (rollout_len - 1, 0), (0, 0)), constant_values=2)
        self.partner_mask = (self.partner_mask == 2)
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
        self.full_var = ['obs', 'actions', 'partner_mask', 'road_mask']
        self.use_tom = use_tom
        if use_tom:
            self.full_var += ['questions', 'answers', 'qa_masks']

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
        valid_qa_indices = np.where(~self.qa_masks[idx1])[0]
        if len(valid_qa_indices) >= self.qa_num_sample:
            sample_qa = np.random.choice(valid_qa_indices, size=self.qa_num_sample, replace=False)
        else:
            sample_qa = np.random.choice(valid_qa_indices, size=self.qa_num_sample, replace=True)
        if self.num_timestep > 1:
            for var_name in self.full_var:
                if self.__dict__[var_name] is not None:
                    if var_name in ['obs', 'road_mask', 'partner_mask']:
                        data = self.__dict__[var_name][idx1, idx2:idx2 + self.rollout_len] # idx 0 -> (0, 0:10) -> (0, 9) end with first timestep
                    elif var_name in ['questions', 'answers', 'qa_masks']:
                        data = self.__dict__[var_name][idx1, sample_qa]
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