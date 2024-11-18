# Define network
import torch.nn as nn
import torch
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch.nn.functional as F
from pygpudrive.env import constants
from networks.perm_eq_late_fusion import LateFusionNet
from algorithms.il.model.wayformer import MultiHeadAttention, Residual

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
            nn.Linear(input_size, hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[0]),
            nn.Tanh(),
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Tanh(),
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(3))
        self.heads = nn.ModuleList([nn.Linear(hidden_size[1], output_size)])


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
        log_prob = torch.cat([dist.log_prob(expert_actions) for dist in pred_action_dist], dim=-1)
        return log_prob

    def get_std(self):
        return self.log_std
    
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
        self.dx_heads = nn.Linear(hidden_size[1], 1)
        self.dy_heads = nn.Linear(hidden_size[1], 1)
        self.dyaw_heads = nn.Linear(hidden_size[1], 1)


    def forward(self, obs, deterministic=False):
        """Generate an output from tensor input."""
        nn = self.nn(obs)
        dx = self.dx_heads(nn)
        dy = self.dy_heads(nn)
        dyaw = self.dyaw_heads(nn)
        actions = torch.cat([dx, dy, dyaw], dim=-1)
        return actions

class LateFusionBCNet(LateFusionNet):

    def __init__(self,  observation_space, env_config, exp_config,
                 num_stack=5):
        super(LateFusionBCNet, self).__init__(observation_space, env_config, exp_config)
        self.config = env_config
        self.net_config = exp_config
        self.ego_input_dim = constants.EGO_FEAT_DIM
        self.ro_input_dim = constants.PARTNER_FEAT_DIM
        self.rg_input_dim = constants.ROAD_GRAPH_FEAT_DIM
        
        self.ro_max = self.config.max_num_agents_in_scene-1
        self.rg_max = self.config.roadgraph_top_k
        self.arch_ego_state = self.net_config.ego_state_layers
        self.arch_road_objects = self.net_config.road_object_layers
        self.arch_road_graph = self.net_config.road_graph_layers
        self.arch_shared_net = self.net_config.shared_layers
        self.act_func = (
            nn.Tanh() if self.net_config.act_func == "tanh" else nn.ReLU()
        )
        self.dropout = self.net_config.dropout

        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
            net_arch=self.arch_ego_state,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * num_stack,
            net_arch=self.arch_road_objects,
        )
        self.rg_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
            net_arch=self.arch_road_graph,
        )

        self.dx_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )

        self.dy_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )

        self.dyaw_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )
    def _unpack_obs(self, obs_flat, num_stack=1):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_staye, road_objects, stop_signs, road_graph (torch.Tensor).
            ego_stayawe, road_objects, stop_signs, road_graph (torch.Tensor).
            ego_state, rodx, dy, dyawobjects, stop_signs, road_graph (torch.Tensor).
        """
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, num_stack,  ego_size + ro_size + rg_size)
        ego_stack = obs_flat_unstack[..., :ego_size].view(-1, num_stack, self.ego_input_dim).reshape(-1, num_stack * self.ego_input_dim)
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, num_stack, self.ro_max, self.ro_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, ro_max, num_stack, ro_input_dim)
            .reshape(-1, self.ro_max, num_stack * self.ro_input_dim)
         )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, num_stack, self.rg_max, self.rg_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, rg_max, num_stack, rg_input_dim)
            .reshape(-1, self.rg_max, num_stack * self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack
    
    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ):
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)
    
    def forward(self, obss):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss, num_stack=5)
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_graph = self.rg_net(road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects.permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph.permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        dx = self.dx_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        dy = self.dy_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        dyaw = self.dyaw_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        actions = torch.cat([dx, dy, dyaw], dim=-1)
        return actions

class LateFusionAttnBCNet(LateFusionNet):

    def __init__(self,  observation_space, env_config, exp_config,
                 num_stack=5):
        super(LateFusionAttnBCNet, self).__init__(observation_space, env_config, exp_config)
        self.config = env_config
        self.net_config = exp_config
        self.ego_input_dim = constants.EGO_FEAT_DIM
        self.ro_input_dim = constants.PARTNER_FEAT_DIM
        self.rg_input_dim = constants.ROAD_GRAPH_FEAT_DIM
        
        self.ro_max = self.config.max_num_agents_in_scene-1
        self.rg_max = self.config.roadgraph_top_k
        self.arch_ego_state = self.net_config.ego_state_layers
        self.arch_road_objects = self.net_config.road_object_layers
        self.arch_road_graph = self.net_config.road_graph_layers
        self.arch_shared_net = self.net_config.shared_layers
        self.act_func = (
            nn.Tanh() if self.net_config.act_func == "tanh" else nn.ReLU()
        )
        self.dropout = self.net_config.dropout

        self.ego_state_net = self._build_network(
            input_dim=self.ego_input_dim * num_stack,
            net_arch=self.arch_ego_state,
        )
        self.road_object_net = self._build_network(
            input_dim=self.ro_input_dim * num_stack,
            net_arch=self.arch_road_objects,
        )
        self.ro_attn = MultiHeadAttention(
            num_heads=4,
            num_q_input_channels=self.arch_road_objects[-1],
            num_kv_input_channels=self.arch_road_objects[-1],
            )
        self.rg_net = self._build_network(
            input_dim=self.rg_input_dim * num_stack,
            net_arch=self.arch_road_graph,
        )
        self.rg_attn = MultiHeadAttention(
            num_heads=4,
            num_q_input_channels=self.arch_road_graph[-1],
            num_kv_input_channels=self.arch_road_graph[-1],
            )
        self.dx_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )

        self.dy_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )

        self.dyaw_head = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=1,
            net_arch=self.arch_shared_net,
        )
    def _unpack_obs(self, obs_flat, num_stack=1):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_staye, road_objects, stop_signs, road_graph (torch.Tensor).
            ego_stayawe, road_objects, stop_signs, road_graph (torch.Tensor).
            ego_state, rodx, dy, dyawobjects, stop_signs, road_graph (torch.Tensor).
        """
        ego_size = self.ego_input_dim
        ro_size = self.ro_input_dim * self.ro_max
        rg_size = self.rg_input_dim * self.rg_max
        obs_flat_unstack = obs_flat.reshape(-1, num_stack,  ego_size + ro_size + rg_size)
        ego_stack = obs_flat_unstack[..., :ego_size].view(-1, num_stack, self.ego_input_dim).reshape(-1, num_stack * self.ego_input_dim)
        ro_stack = (
            obs_flat_unstack[..., ego_size:ego_size + ro_size]
            .view(-1, num_stack, self.ro_max, self.ro_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, ro_max, num_stack, ro_input_dim)
            .reshape(-1, self.ro_max, num_stack * self.ro_input_dim)
         )

        # rg_stack: Original reshape, then combine num_stack and self.rg_input_dim dimensions
        rg_stack = (
            obs_flat_unstack[..., ego_size + ro_size: ego_size + ro_size + rg_size]
            .view(-1, num_stack, self.rg_max, self.rg_input_dim)
            .permute(0, 2, 1, 3)  # Reorder to (batch, rg_max, num_stack, rg_input_dim)
            .reshape(-1, self.rg_max, num_stack * self.rg_input_dim)
        )

        return ego_stack, ro_stack, rg_stack
    
    def _build_out_network(
        self, input_dim: int, output_dim: int, net_arch: List[int]
    ):
        """Create the output network architecture."""
        layers = []
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim

        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)
    
    def forward(self, obss, attn_weights=False):
        # Unpack observation
        ego_state, road_objects, road_graph = self._unpack_obs(obss, num_stack=5)
        # Embed features
        ego_state = self.ego_state_net(ego_state)
        road_objects = self.road_object_net(road_objects)
        road_objects_attn = self.ro_attn(road_objects, road_objects)
        
        road_graph = self.rg_net(road_graph)
        road_graph_attn = self.ro_attn(road_graph, road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(
            road_objects_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.ro_max
        ).squeeze(-1)
        road_graph = F.max_pool1d(
            road_graph_attn['last_hidden_state'].permute(0, 2, 1), kernel_size=self.rg_max
        ).squeeze(-1)

        dx = self.dx_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        dy = self.dy_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        dyaw = self.dyaw_head(torch.cat((ego_state, road_objects, road_graph), dim=1))
        actions = torch.cat([dx, dy, dyaw], dim=-1)
        if attn_weights:
            return actions, ro_weights
        return actions
    
if __name__ == "__main__":
    from pygpudrive.registration import make
    from pygpudrive.env.config import DynamicsModel, ActionSpace
    from baselines.ippo.config import ExperimentConfig
    from pygpudrive.env.config import EnvConfig, RenderConfig, SceneConfig
    import pyrallis
    # CONFIGURE
    TOTAL_STEPS = 90
    MAX_CONTROLLED_AGENTS = 128
    NUM_WORLDS = 1

    env_config = EnvConfig(dynamics_model="delta_local")
    render_config = RenderConfig()
    scene_config = SceneConfig("data/processed/examples", NUM_WORLDS)

    # MAKE ENVIRONMENT
    kwargs = {
        "config": env_config,
        "scene_config": scene_config,
        "max_cont_agents": MAX_CONTROLLED_AGENTS,
        "device": "cuda", 
        "num_stack": 5
    }
    
    env = make(dynamics_id=DynamicsModel.DELTA_LOCAL, action_space=ActionSpace.CONTINUOUS, kwargs=kwargs)
    obs = env.reset()
    exp_config = pyrallis.parse(config_class=ExperimentConfig)
    bc_policy = LateFusionAttnBCNet(
        observation_space=None,
        exp_config=exp_config,
        env_config=env_config
    ).to("cuda")
    actions = bc_policy(obs.squeeze(0))
    dead_agent_mask = ~env.cont_agent_mask.clone()
    actions = actions.unsqueeze(0)
    print(actions.shape)
    for _ in range(5):
        env.step_dynamics(actions, use_indices=False)
        obs = env.get_obs()


    