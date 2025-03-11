from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # MODEL
    model_path: str = "/data/model/early_attn_500"
    model_name: str = "early_attn_gmm_data_500_seed_3_20250228_1101"
    
    # DATA
    data_path: str = "/data/tom_v5"
    train_data: str = "train_trajectory_100.npz"
    test_data: str = "test_trajectory_200.npz"
    rollout_len: int = 5
    pred_len: int = 1
    
    # EXPERIMENT
    batch_size: int = 512
    epochs: int = 30
    lr: float = 0.0005
    sample_per_epoch: int = 438763
    seed: int = 0