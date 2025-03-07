from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    # MODEL
    model_path: str = "/data/model/tom_only_move"
    model_name: str = "aux_attn_gmm_only_move_infer_20250223_1519"
    
    # DATA
    data_path: str = "/data/tom_v4"
    train_data: str = "train_trajectory_1000.npz"
    test_data: str = "test_trajectory_200.npz"
    rollout_len: int = 5
    pred_len: int = 1
    
    # EXPERIMENT
    batch_size: int = 512
    epochs: int = 30
    lr: float = 0.0005
    sample_per_epoch: int = 438763
    seed: int = 0