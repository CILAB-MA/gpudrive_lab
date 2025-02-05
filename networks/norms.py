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

    def forward(self, x):   # (B, S, D)
        # Masked Batch Normalization
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

class CrossSetNorm(nn.Module):
    def __init__(self, feature_dim=None):
        super().__init__()
        self.weights_obj = nn.Parameter(torch.empty(feature_dim))
        self.biases_obj = nn.Parameter(torch.empty(feature_dim))
        self.weights_road = nn.Parameter(torch.empty(feature_dim))
        self.biases_road = nn.Parameter(torch.empty(feature_dim))
        self.mask = None
        self.ro_max = 127
        self.rg_max = 200
        torch.nn.init.constant_(self.weights_obj, 1.)
        torch.nn.init.constant_(self.biases_obj, 0.)
        torch.nn.init.constant_(self.weights_road, 1.)
        torch.nn.init.constant_(self.biases_road, 0.)
        # self.ln = nn.LayerNorm(feature_dim)
    def forward(self, x):   # (B, S, D)
        # Masked Batch Normalization
        # x = self.ln(x)
        object_alive_mask = (~self.mask[:, :self.ro_max + 1]).unsqueeze(-1)
        road_alive_mask = (~self.mask[:, self.ro_max + 1:]).unsqueeze(-1)
        object_valid_counts = object_alive_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
        road_valid_counts = road_alive_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (B, 1, 1)
        object_batch_mask = object_valid_counts > 1   # (B, 1, 1)
        road_batch_mask = road_valid_counts > 1   # (B, 1, 1)

        x_obj = x[:, :self.ro_max + 1]
        x_road = x[:, self.ro_max + 1:]
        x_obj = x_obj * object_alive_mask  # (B, S, D)
        x_road = x_road * road_alive_mask  # (B, S, D)
        
        sum_x_obj = x_obj.sum(dim=1, keepdim=True)  # (B, 1, D)
        sum_x_road = x_road.sum(dim=1, keepdim=True)  # (B, 1, D)

        mean_obj = torch.where(object_batch_mask, sum_x_obj / object_valid_counts, sum_x_obj) # (B, 1, D)
        mean_road = torch.where(road_batch_mask, sum_x_road / road_valid_counts, sum_x_road) # (B, 1, D)

        variance_obj = ((x_obj - mean_obj) ** 2).sum(dim=1, keepdim=True) / object_valid_counts  # (B, 1, D)
        variance_road = ((x_road - mean_road) ** 2).sum(dim=1, keepdim=True) / road_valid_counts  # (B, 1, D)

        variance_obj = torch.where(object_batch_mask, variance_obj, torch.zeros_like(sum_x_obj))
        variance_road = torch.where(road_batch_mask, variance_road, torch.zeros_like(sum_x_road))

        std_obj = torch.where(object_batch_mask, torch.sqrt(variance_obj + 1e-6), torch.ones_like(sum_x_obj))
        std_road = torch.where(road_batch_mask, torch.sqrt(variance_road + 1e-6), torch.ones_like(sum_x_road))

        normalized_x_obj = (x_obj - mean_obj) / std_obj
        out_obj = F.linear(normalized_x_obj, torch.diag_embed(self.weights_obj), self.biases_obj)

        normalized_x_road = (x_road - mean_road) / std_road
        out_road = F.linear(normalized_x_road, torch.diag_embed(self.weights_road), self.biases_road)
        out = torch.cat([out_obj, out_road], dim=1)
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
        
        if self.mask.all() or not self.training:
            masked_mean = self.running_mean # (D)
            masked_var = self.running_var # (D)
        else:
            masked_mean = x[~self.mask].mean(dim=0) # (D)
            masked_var = x[~self.mask].var(dim=0)   # (D)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * masked_mean.detach()  # (D)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * masked_var.detach() # (D)

        x = (x - masked_mean) / torch.sqrt(masked_var + self.eps)
        x = x.masked_fill(self.mask.unsqueeze(-1), -1e9)
        return x