from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    # Hyperparameters
    batch_size: int = 256
    epochs: int = 1000
    lr: float = 5e-4
    sample_per_epoch: int = 438763
    
    # BASE LATEFUSION
    ego_state_layers = [64, 64]
    road_object_layers = [64, 64]
    road_graph_layers = [64, 64]
    shared_layers = [64, 64]
    act_func = "tanh"
    dropout = 0.0
    last_layer_dim_pi = 64
    last_layer_dim_vf = 64  

    @dataclass
    class FeedForwardConfig:
        hidden_size: list = field(default_factory=lambda: [1024, 256])
        net_arch: list = field(default_factory=lambda: [64, 128])
    
    @dataclass
    class LatefusionConfig:
        #TODO: latefusion network hyperparameters
        NotImplemented

    @dataclass
    class LatefusionAttnConfig:
        #TODO: latefusion attention network hyperparameters
        NotImplemented
    
    @dataclass
    class WayformerConfig:
        #TODO: wayformer network hyperparameters
        NotImplemented
    
    @dataclass
    class ContHeadConfig:
        #TODO: conthead hyperparameters
        NotImplemented 
    
    @dataclass
    class GmmHeadConfig:
        hidden_dim: int = 128
        action_dim: int = 3
        n_components: int = 10
        time_dim: int = 91
        
    # Sub-configurations
    feedforward: FeedForwardConfig = field(default_factory=FeedForwardConfig)
    latefusion: LatefusionConfig = field(default_factory=LatefusionConfig)
    latefusion_attn: LatefusionAttnConfig = field(default_factory=LatefusionAttnConfig)
    wayformer: WayformerConfig = field(default_factory=WayformerConfig)  
    
    conthead: ContHeadConfig = field(default_factory=ContHeadConfig)
    gmm: GmmHeadConfig = field(default_factory=GmmHeadConfig)
    