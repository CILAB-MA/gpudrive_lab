from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class LinearProb(nn.Module, ABC):
    def __init__(self, context_dim, other_dim):
        super(LinearProb, self).__init__()
        self.context_dim = context_dim
        self.other_dim = other_dim
    
    @abstractmethod
    def forward(self, context):
        pass
    
    @abstractmethod
    def loss(self, pred_action, expert_action):
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearProbPositionCont(LinearProb):
    def __init__(
        self,
        context_dim: int,
        pos_dim: int = 2,                 # (x,y)
        loss_type: str = "huber",         # "huber" | "l2" | "l1"
        huber_delta: float = 1.0,
    ):
        super().__init__(context_dim, pos_dim)
        self.head = nn.Linear(context_dim, pos_dim)
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # context: [B, context_dim] -> pred: [B, pos_dim]
        return self.head(context)

    @torch.no_grad()
    def predict(self, context: torch.Tensor) -> torch.Tensor:
        return self.forward(context)

    def _loss_vec(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "huber":
            l = F.smooth_l1_loss(pred, target, reduction="none", beta=self.huber_delta)
            return l.mean(dim=-1)
        elif self.loss_type == "l2":
            return ((pred - target) ** 2).mean(dim=-1)
        elif self.loss_type == "l1":
            return (pred - target).abs().mean(dim=-1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

    def loss(
        self,
        pred: torch.Tensor,            # [B, pos_dim]
        target: torch.Tensor,          # [B, pos_dim]
        mask: torch.Tensor | None = None,  # [B] or [B,1], optional
    ):
        assert pred.shape == target.shape
        B = pred.shape[0]

        per_sample_loss = self._loss_vec(pred, target)          # [B]
        mae_per_sample = (pred - target).abs().mean(dim=-1)     # [B]

        if mask is not None:
            m = mask.view(B).float()
            w = m / (m.sum().clamp_min(1.0))
            loss_mean = (per_sample_loss * w).sum()
            mae_mean  = (mae_per_sample * w).sum()
        else:
            loss_mean = per_sample_loss.mean()
            mae_mean  = mae_per_sample.mean()

        return loss_mean, mae_mean.item(), pred

    def loss_no_reduction(
        self,
        pred: torch.Tensor,           # [B, pos_dim]
        target: torch.Tensor,         # [B, pos_dim]
        mask: torch.Tensor | None = None,
    ):
        assert pred.shape == target.shape
        B = pred.shape[0]

        per_sample_loss = self._loss_vec(pred, target)          # [B]
        mae_per_sample  = (pred - target).abs().mean(dim=-1)    # [B]

        if mask is not None:
            m = mask.view(B).float()
            mae_sum = (mae_per_sample * m).sum().item()
            total   = int(m.sum().item())
            per_sample_loss = per_sample_loss * m
        else:
            mae_sum = mae_per_sample.sum().item()
            total   = B

        return per_sample_loss, mae_sum, pred, total


class LinearProbAction(LinearProb):
    def __init__(self, context_dim, other_dim, future_step=None):
        super(LinearProbAction, self).__init__(context_dim, other_dim)
        self.head = nn.Linear(context_dim, other_dim)
        self.future_step = future_step
        
    def forward(self, context):
        logits = self.head(context)
        return logits
    
    def predict(self, context):
        logits = self.forward(context)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        return pred_class
    
    def loss(self, pred_logits, expert_labels):
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_logits, expert_labels)
        
        # compute accuracy
        pred_class = torch.argmax(pred_logits, dim=-1)
        correct = (pred_class == expert_labels).sum().item()
        total = expert_labels.numel()
        accuracy = correct / total
        return loss, accuracy, pred_class

class LinearProbPosition(LinearProb):
    def __init__(self, context_dim, other_dim, future_step=None):
        super(LinearProbPosition, self).__init__(context_dim, other_dim)
        self.head = nn.Linear(context_dim, other_dim)
        self.future_step = future_step
        
    def forward(self, context):
        logits = self.head(context)
        return logits
    
    def predict(self, context):
        logits = self.forward(context)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1)
        return pred_class
    
    def loss(self, pred_logits, expert_labels):
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_logits, expert_labels)
        
        # compute accuracy
        pred_class = torch.argmax(pred_logits, dim=-1)
        correct = (pred_class == expert_labels).sum().item()
        total = expert_labels.numel()
        accuracy = correct / total
        return loss, accuracy, pred_class

    def loss_no_reduction(self, pred_logits, expert_labels):
        # compute loss
        criterion = nn.CrossEntropyLoss(reduction='none')
        loss = criterion(pred_logits, expert_labels)
        
        # compute accuracy
        pred_class = torch.argmax(pred_logits, dim=-1)
        correct = (pred_class == expert_labels).sum().item()
        total = expert_labels.numel()
        return loss, correct, pred_class, total

class LinearProbAngle(LinearProb):
    def __init__(self, context_dim, other_dim, future_step=None):
        super(LinearProbAngle, self).__init__(context_dim, other_dim)
        self.yaw_head = nn.Linear(context_dim, other_dim)
        self.future_step = future_step

    def forward(self, context):
        yaw = self.yaw_head(context)
        return yaw
    
    def loss(self, pred_logits, expert_labels):
        # compute loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_logits, expert_labels)
        
        # compute accuracy
        pred_class = torch.argmax(pred_logits, dim=-1)
        correct = (pred_class == expert_labels).sum().item()
        total = expert_labels.numel()
        accuracy = correct / total
        return loss, accuracy, pred_class
    
