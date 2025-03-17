import torch
import numpy as np
from torch.utils.data import DataLoader, IterableDataset
import random

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, obs, mean, std, valid_mask):
        valid_mask = valid_mask.squeeze(-1)
        # obs
        self.obs = obs[valid_mask]
        # actions
        self.mu = mean[valid_mask]
        self.std = std[valid_mask]
        # masks
        self.valid_mask = valid_mask

    def __len__(self):
        return len(self.obs)
        
    def __getitem__(self, idx):
        return self.obs[ind], self.mu[ind], self.std[ind], idx
    

if __name__ == "__main__":
    import os
    from torch.utils.data import DataLoader
    data = np.load(f"/data/RL/data/test_trajectory_1000.npz")
    dataset = ExpertDataset(**data)