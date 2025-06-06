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
    
