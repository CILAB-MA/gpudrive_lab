from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # MODEL
    model_path: str = "/data/model"
    model_name: str = "early_attn_gmm_LN_100_20250208_0357"
    
    # DATA
    data_path: str = "/data/tom_v2"
    train_data: str = "train_trajectory_100.npz"
    test_data: str = "test_trajectory_200.npz"
    rollout_len: int = 5
    pred_len: int = 1
    
    # EXPERIMENT
    batch_size: int = 512
    epochs: int = 500
    lr: float = 0.0005
    sample_per_epoch: int = 438763
    seed: int = 0