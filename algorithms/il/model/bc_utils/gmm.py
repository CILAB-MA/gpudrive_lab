import torch
import torch.nn as nn
import torch.distributions as dist


class GMM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, action_dim=3, n_components=10):
        super(GMM, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.residual_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ) for _ in range(3)
        ])
        
        self.head = nn.Linear(hidden_dim, n_components * (2 * action_dim + 1))

        self.n_components = n_components
        self.action_dim = action_dim

    def forward(self, x):
        """
        Forward pass of the model
        """
        x = self.input_layer(x)
        
        for layer in self.residual_block:
            residual = x
            x = layer(x)
            x += residual
        
        params = self.head(x)
        
        means = params[:, :self.n_components * self.action_dim].view(-1, self.n_components, self.action_dim)
        covariances = params[:, self.n_components * self.action_dim:2 * self.n_components * self.action_dim].view(-1, self.n_components, self.action_dim)
        weights = params[:, -self.n_components:]
        
        means = torch.tanh(means)
        covariances = torch.exp(covariances)
        weights = torch.softmax(weights, dim=1)
        
        return means, covariances, weights

    def sample(self, x):
        """
        Sample an action based on the GMM parameters given embedding vector x
        """
        means, covariances, weights = self.forward(x)
        
        # Sample component indices based on weights
        component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        
        sampled_means = means[torch.arange(x.size(0)), component_indices]
        sampled_covariances = covariances[torch.arange(x.size(0)), component_indices]
        
        # Sample actions from the chosen component's Gaussian
        actions = dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        return actions

    def log_prob(self, x, actions):
        """
        Compute the log probability of actions given embedding vector x
        """
        means, covariances, weights = self.forward(x)
        log_probs = []

        for i in range(self.n_components):
            mean = means[:, i, :]
            cov_diag = covariances[:, i, :]
            gaussian = dist.MultivariateNormal(mean, torch.diag_embed(cov_diag))
            log_probs.append(gaussian.log_prob(actions))

        log_probs = torch.stack(log_probs, dim=1)
        weighted_log_probs = log_probs + torch.log(weights + 1e-8)
        return torch.logsumexp(weighted_log_probs, dim=1)
