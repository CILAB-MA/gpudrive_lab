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
            ) for _ in range(4)
        ])
        
        self.head = nn.Linear(hidden_dim, n_components * (2 * action_dim + 1))
        self.n_components = n_components
        self.action_dim = action_dim

    def get_gmm_params(self, x):
        """
        Get the parameters of the Gaussian Mixture Model
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
        
        return means, covariances, weights, self.n_components

    def forward(self, x, deterministic=None):
        """
        Sample actions from the Gaussian Mixture Model
        """
        means, covariances, weights, components = self.get_gmm_params(x)
        # Sample component indices based on weights
        component_indices = torch.multinomial(weights, num_samples=1).squeeze(1)
        
        sampled_means = means[torch.arange(x.size(0)), component_indices]
        sampled_covariances = covariances[torch.arange(x.size(0)), component_indices]
        
        # Sample actions from the chosen component's Gaussian
        actions = dist.MultivariateNormal(sampled_means, torch.diag_embed(sampled_covariances)).sample()
        return actions
