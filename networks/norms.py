import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.ln = nn.LayerNorm(feature_dim)  # Feature-wise LN

    def forward(self, x):   # (B, S, D)
        # Masked Batch Normalization
        x = self.ln(x)
        alive_mask = (~self.mask).unsqueeze(-1)  # (B, S) -> (B, S, 1)
        valid_counts = alive_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
        batch_mask = valid_counts > 1   # (B, 1, 1)
        
        x = x * alive_mask  # (B, S, D)
        
        sum_x = x.sum(dim=1, keepdim=True)  # (B, 1, D)
        mean = torch.where(batch_mask, sum_x / valid_counts, sum_x) # (B, 1, D)
        variance = ((x - mean) ** 2).sum(dim=1, keepdim=True) / valid_counts  # (B, 1, D)
        variance = torch.where(batch_mask, variance, torch.zeros_like(sum_x))
        std = torch.where(batch_mask, torch.sqrt(variance + 1e-6), torch.ones_like(sum_x))

        normalized_x = (x - mean) / std
        out = F.linear(normalized_x, torch.diag_embed(self.weights), self.biases)
        return out

class CustomBatchNorm(nn.BatchNorm1d):
    def __init__(self, seq_len, feature_dim):
        super().__init__(num_features=feature_dim)
        self.seq_len = seq_len
        
    def forward(self, input):
        x = input.view(-1, self.num_features) # (B * S, D)
        x = super().forward(x)
        x = x.view(-1, self.seq_len, self.num_features)
        return x

class MaskedBatchNorm1d(nn.Module):
    def __init__(self, seq_len, feature_dim, momentum=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.eps = 1e-5

        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_var", torch.ones(feature_dim))

        self.mask = None

    @staticmethod
    def _check_input_dim(x):
        if x.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(x.dim()))

    def forward(self, x):   # x : (B, S, D)
        self._check_input_dim(x)
        
        if self.training:
            if self.mask.all():
                masked_mean = self.running_mean
                masked_var = self.running_var
            else:
                masked_mean = x[~self.mask].mean(dim=0) # (D)
                masked_var = x[~self.mask].var(dim=0)   # (D)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * masked_mean.detach()  # (D)
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * masked_var.detach() # (D)
                
        else:
            masked_mean = self.running_mean # (D)
            masked_var = self.running_var # (D)

        x = (x - masked_mean) / torch.sqrt(masked_var + self.eps)
        x = x.masked_fill(self.mask.unsqueeze(-1), -1e9)
        return x