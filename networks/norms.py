import torch
import torch.nn as nn
import torch.nn.functional as F

class SetNorm(nn.LayerNorm):
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x):
        # standardization
        out = super().forward(x)
        # transform params
        out = F.linear(out, torch.diag_embed(self.weights), self.biases)
        return out
    
class SetBatchNorm(nn.Module):
    def __init__(self, feature_dim=None):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        self.mask = None
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x):   # (B, S, D)
        # Masked Batch Normalization
        alive_mask = (~self.mask).unsqueeze(-1)  # (B, S) -> (B, S, 1)
        valid_counts = alive_mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
        batch_mask = valid_counts > 1   # (B, 1, 1)
        
        x = x * alive_mask  # (B, S, D)
        
        sum_x = x.sum(dim=1, keepdim=True)  # (B, 1, D)
        mean = torch.where(batch_mask, sum_x / valid_counts, sum_x) # (B, 1, D)
        variance = ((x - mean) ** 2 * alive_mask).sum(dim=1, keepdim=True) / valid_counts  # (B, 1, D)
        variance = torch.where(batch_mask, variance, torch.zeros_like(sum_x))
        std = torch.where(batch_mask, torch.sqrt(variance + 1e-6), torch.ones_like(sum_x))

        normalized_x = torch.where(batch_mask, (x - mean) / std, x)
        out = normalized_x * self.weights + self.biases  # (B, S, D)
        return out