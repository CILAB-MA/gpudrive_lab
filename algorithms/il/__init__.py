import torch.nn.functional as F

from algorithms.il.model.bc import *
from algorithms.il.loss import *

MODELS = dict(bc=ContFeedForward, late_fusion=LateFusionBCNet,
              attention=LateFusionAttnBCNet, wayformer=None)

LOSS = dict(l1=F.smooth_l1_loss, mse=F.mse_loss,
            twohot=two_hot_loss, gmm=gmm_loss)

GET_LOSS = dict(
    l1=lambda model, obs, expert_action, dead_mask: (F.smooth_l1_loss(model(obs), expert_action)),
    gmm=lambda model, obs, expert_action, dead_mask: (gmm_loss(model.head.get_gmm_params(model.get_embedded_obs(obs)), expert_action)),
    twohot=None,
    dist=None
)
