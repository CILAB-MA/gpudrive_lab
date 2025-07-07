import sys, os
sys.path.append(os.getcwd())
from gpudrive.integrations.il.model.model import EarlyFusionAttnBCNet
import torch.nn as nn

class EarlyFusionAttnAuxNet(EarlyFusionAttnBCNet):
    def __init__(self, env_config, exp_config, num_stack=5, use_tom=True):
        super().__init__(env_config, exp_config, num_stack=num_stack, use_tom=use_tom)

        if use_tom:
            self.aux_head = nn.Sequential(
                nn.Linear(768, exp_config.network_dim),
                nn.ReLU(),
                nn.Linear(exp_config.network_dim, 384)
            )