# Define network
import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size)])

    def dist(self, obs):
        """Generate action distribution."""
        x_out = self.nn(obs.float())
        return [Categorical(logits=head(x_out)) for head in self.heads]

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        action_dist = self.dist(obs)

        if deterministic:
            actions_idx = action_dist[0].logits.argmax(axis=-1)
        else:
            actions_idx = action_dist.sample()
        return actions_idx

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = pred_action_dist[0].log_prob(expert_actions).mean()
        return log_prob

# Define network
class ContFeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContFeedForward, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(3))
        self.heads = nn.ModuleList([nn.Linear(hidden_size, output_size)])


    def dist(self, obs):
        """Generate action distribution."""
        x_out = self.nn(obs.float())
        return [Normal(head(x_out), torch.exp(std)) for head, std in zip(self.heads, self.log_std)]

    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        action_dist = self.dist(obs)
        if deterministic:
            actions_idx = torch.cat([dist.mean for dist in action_dist], dim=-1)
        else:
            actions_idx = torch.cat([dist.sample() for dist in action_dist], dim=-1)
        return actions_idx

    def _log_prob(self, obs, expert_actions):
        pred_action_dist = self.dist(obs)
        log_prob = torch.cat([dist.log_prob(expert_actions) for dist in pred_action_dist], dim=-1).mean()
        return log_prob

# Define network
class ContFeedForwardMSE(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ContFeedForwardMSE, self).__init__()
        self.nn = nn.Sequential(
            nn.Linear(input_size, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.Tanh(),
        )
        self.heads = nn.Linear(hidden_size[1], 3)


    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        nn = self.nn(obs)
        actions = self.heads(nn)
        return actions
