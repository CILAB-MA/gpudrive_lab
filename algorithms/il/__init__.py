import torch.nn.functional as F

from algorithms.il.model.bc import *
from algorithms.il.loss import *
from algorithms.il.model.aux import *

MODELS = dict(bc=ContFeedForward, late_fusion=LateFusionBCNet,
              attention=LateFusionAttnBCNet, wayformer=WayformerEncoder,
              aux_fusion=LateFusionAuxNet, aux_attn=LateFusionAttnAuxNet)

LOSS = dict(
    l1=l1_loss, 
    mse=mse_loss,
    twohot=two_hot_loss,
    nll=nll_loss,
    gmm=gmm_loss,
    new_gmm=new_gmm_loss
)