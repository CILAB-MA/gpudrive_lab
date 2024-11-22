import torch.nn.functional as F

from algorithms.il.model.bc import *
from algorithms.il.loss import *

MODELS = dict(bc=ContFeedForward, late_fusion=LateFusionBCNet,
              attention=LateFusionAttnBCNet, wayformer=None)

LOSS = dict(l1=F.smooth_l1_loss, mse=F.mse_loss,
            twohot=two_hot_loss, gmm=gmm_loss)