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
    def __init__(self, *args, feature_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = nn.Parameter(torch.empty(feature_dim))
        self.biases = nn.Parameter(torch.empty(feature_dim))
        torch.nn.init.constant_(self.weights, 1.)
        torch.nn.init.constant_(self.biases, 0.)

    def forward(self, x, mask=None):
        # standardization
        if mask is not None:
            mask = (1 - mask).unsqueeze(-1)  # (B, S) -> (B, S, 1)
            x = x * mask
            valid_counts = mask.sum(dim=1, keepdim=True)  # (B, 1, 1)
            mean = x.sum(dim=1, keepdim=True) / (valid_counts + self.epsilon)
            variance = ((x - mean) ** 2 * mask).sum(dim=1, keepdim=True) / (valid_counts + self.epsilon)
            std = torch.sqrt(variance + self.epsilon)
        else:
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True)
        normalized_x = (x - mean) / (std + self.epsilon)
        out = normalized_x * self.weights + self.biases  # (B, S, D)
        return out