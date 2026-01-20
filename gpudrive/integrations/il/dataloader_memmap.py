import torch
import numpy as np
from gpudrive.env.constants import MIN_REL_AGENT_POS, MAX_REL_AGENT_POS

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, actions, partner_mask, road_mask, rollout_len=5, pred_len=1):
        # data
        self.obs = obs # (B, 91, F)
        self.actions = actions # (B, 91, 3)
        self.partner_mask = partner_mask # (B, 91, 127)
        self.road_mask = road_mask # (B, 91, 200)

        # etc
        self.rollout_len = rollout_len
        self.pred_len = pred_len

    def __len__(self):
        return self.obs.shape[0]
    
    def __getitem__(self, idx):
        return (
            self.obs[idx],
            self.actions[idx],
            self.partner_mask[idx],
            self.road_mask[idx],
        )
    