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
    def __init__(self, context_dim, other_dim):
        super(LinearProbAction, self).__init__(context_dim, other_dim)
        self.dx_head = nn.Linear(context_dim, other_dim)
        self.dy_head = nn.Linear(context_dim, other_dim)
        self.dyaw_head = nn.Linear(context_dim, other_dim)
        
    def forward(self, context):
        dx = self.dx_head(context)
        dy = self.dy_head(context)
        dyaw = self.dyaw_head(context)
        action = torch.stack([dx, dy, dyaw], dim=-1)
        return action
    
    def loss(self, pred_action, expert_action):
        criterion = nn.MSELoss()
        loss = criterion(pred_action, expert_action)
        return loss

class LinearProbPosition(LinearProb):
    def __init__(self, context_dim, other_dim):
        super(LinearProbPosition, self).__init__(context_dim, other_dim)
        self.x_head = nn.Linear(context_dim, other_dim)
        self.y_head = nn.Linear(context_dim, other_dim)
        
    def forward(self, context):
        x = self.x_head(context)
        y = self.y_head(context)
        pos = torch.stack([x, y], dim=-1)
        return pos
    
    def loss(self, pred_pos, expert_pos):
        criterion = nn.MSELoss()
        loss = criterion(pred_pos, expert_pos)
        return loss

class LinearProbAngle(LinearProb):
    def __init__(self, context_dim, other_dim):
        super(LinearProbAngle, self).__init__(context_dim, other_dim)
        self.yaw_head = nn.Linear(context_dim, other_dim)
        
    def forward(self, context):
        yaw = self.yaw_head(context)
        return yaw
    
    def loss(self, pred_angle, expert_angle):
        criterion = nn.MSELoss()
        loss = criterion(pred_angle, expert_angle)
        return loss
    
class LinearProbSpeed(LinearProb):
    def __init__(self, context_dim, other_dim):
        super(LinearProbSpeed, self).__init__(context_dim, other_dim)
        self.speed_head = nn.Linear(context_dim, other_dim)
        
    def forward(self, context):
        speed = self.speed_head(context)
        return speed
    
    def loss(self, pred_speed, expert_speed):
        criterion = nn.MSELoss()
        loss = criterion(pred_speed, expert_speed)
        return loss
