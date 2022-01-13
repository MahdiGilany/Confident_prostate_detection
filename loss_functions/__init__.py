from torch import nn
from .abstention_loss import DacLossPid
from .active_passive_loss import (
    NCEandRCE, GeneralizedCrossEntropy, NLNL, NFLandRCE, FocalLoss, CrossEntropy
)
from .misc import f_score
from .elr import ELR
from .isomax import IsoMaxLossSecondPart
from .simclr import info_nce_loss
from .evidential_loss import Edl_losses
