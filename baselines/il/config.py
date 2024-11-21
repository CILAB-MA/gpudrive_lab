from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # Hyperparameters
    batch_size: int = 64
    epochs: int = 1000
    lr: float = 5e-4
    hidden_size: list = field(default_factory=lambda: [1024, 256])
    net_arch: list = field(default_factory=lambda: [64, 128])
    sample_per_epoch: int = 50000
    
    # LATEFUSION NETWORK
    ego_state_layers = [64, 64]
    road_object_layers = [64, 64]
    road_graph_layers = [64, 64]
    shared_layers = [64, 64]
    act_func = "tanh"
    dropout = 0.0
    last_layer_dim_pi = 64
    last_layer_dim_vf = 64
    
    # GMM
    hidden_dim: int = 128
    action_dim: int = 3
    n_components: int = 10
    