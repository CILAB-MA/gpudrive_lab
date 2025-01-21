from dataclasses import dataclass, field
import torch

@dataclass
# Originally it is in pygpudrive/envs/config ->, but for running independent from installing pygpudrive, move it temp
class EnvConfig:
    """Configuration settings for the GPUDrive gym environment.

    This class contains both Python-specific configurations and settings that
    are shared between Python and C++ components of the simulator.

    To modify simulator settings shared with C++, follow these steps:
    1. Navigate to `src/consts.hpp` in the C++ codebase.
    2. Locate and modify the constant (e.g., `kMaxAgentCount`).
    3. Save the changes to `src/consts.hpp`.
    4. Recompile the code to apply changes across both C++ and Python.
    """

    # Python-specific configurations
    # Observation space settings
    ego_state: bool = True  # Include ego vehicle state in observations
    road_map_obs: bool = True  # Include road graph in observations
    partner_obs: bool = True  # Include partner vehicle info in observations
    norm_obs: bool = True  # Normalize observations
    
    # NOTE: If disable_classic_obs is True, ego_state, road_map_obs, 
    # and partner_obs are invalid. This makes the sim 2x faster
    disable_classic_obs: bool = False  # Disable classic observations 
    lidar_obs: bool = False  # Use LiDAR in observations

    # Set the weights for the reward components
    # R = a * collided + b * goal_achieved + c * off_road
    collision_weight: float = 0.0
    goal_achieved_weight: float = 1.0
    off_road_weight: float = 0.0

    # Road observation algorithm settings
    road_obs_algorithm: str = "linear"  # Algorithm for road observations
    obs_radius: float = 100.0  # Radius for road observations
    polyline_reduction_threshold: float = (
        1.0  # Threshold for polyline reduction
    )

    # Dynamics model
    dynamics_model: str = (
        "delta_local"  # Options: "classic", "bicycle", "delta_local", or "state"
    )

    # Action space settings (if discretized)
    # Classic or Invertible Bicycle dynamics model
    steer_actions=torch.round(
        torch.linspace(-0.3, 0.3, 7), decimals=3
    )
    accel_actions=torch.round(
        torch.linspace(-6.0, 6.0, 7), decimals=3
    )
    head_tilt_actions: torch.Tensor = torch.Tensor([0])
    
    # Delta Local dynamics model
    dx=torch.round(
        torch.linspace(-6.0, 6.0, 100), decimals=3
    )
    dy=torch.round(
        torch.linspace(-6.0, 6.0, 100), decimals=3
    )
    dyaw=torch.round(
        torch.linspace(-torch.pi, torch.pi, 100), decimals=3
    )

    # Global action space settings if StateDynamicsModel is used
    x: torch.Tensor = torch.round(
        torch.linspace(-100.0, 100.0, 10), decimals=3
    )
    y: torch.Tensor = torch.round(
        torch.linspace(-100.0, 100.0, 10), decimals=3
    )
    yaw: torch.Tensor = torch.round(
        torch.linspace(-3.14, 3.14, 10), decimals=3
    )
    vx: torch.Tensor = torch.round(torch.linspace(-10.0, 10.0, 10), decimals=3)
    vy: torch.Tensor = torch.round(torch.linspace(-10.0, 10.0, 10), decimals=3)

    # Collision behavior settings
    collision_behavior: str = "remove"  # Options: "remove", "stop", "ignore"

    # Scene configuration
    remove_non_vehicles: bool = True  # Remove non-vehicle entities from scene

    # Reward settings
    reward_type: str = (
        "sparse_on_goal_achieved"  # Alternatively, "weighted_combination"
    )

    dist_to_goal_threshold: float = (
        3.0  # Radius around goal considered as "goal achieved"
    )

    max_num_agents_in_scene: int = 128
    max_num_rg_points: int = 6000
    roadgraph_top_k: int = 200
    episode_len: int = 91
    num_lidar_samples: int = 30

@dataclass
class ExperimentConfig:
    # Hyperparameters
    batch_size: int = 512
    epochs: int = 500
    lr: float = 5e-4
    sample_per_epoch: int = 438763
    
@dataclass
class NetworkConfig:
    # BASE LATEFUSION
    network_dim: int = 128
    network_num_layers: int = 2
    act_func: str = "tanh"
    dropout: float = 0.0
    norm: str = "LN" # LN, BN, SN, SBN, None

@dataclass
class HeadConfig:
    head_dim: int = 128
    head_num_layers: int = 2
    action_dim: int = 3
    n_components: int = 10
    time_dim: int = 91
    clip_value: float = -20.0
