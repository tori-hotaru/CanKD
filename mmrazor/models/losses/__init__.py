# Copyright (c) OpenMMLab. All rights reserved.
from .ab_loss import ABLoss
from .at_loss import ATLoss
from .crd_loss import CRDLoss
from .cross_entropy_loss import CrossEntropyLoss
from .cwd import ChannelWiseDivergence
from .dafl_loss import ActivationLoss, InformationEntropyLoss, OnehotLikeLoss
from .decoupled_kd import DKDLoss
from .dist_loss import DISTLoss
from .factor_transfer_loss import FTLoss
from .fbkd_loss import FBKDLoss
from .kd_soft_ce_loss import KDSoftCELoss
from .kl_divergence import KLDivergence
from .l1_loss import L1Loss
from .l2_loss import L2Loss
from .mgd_loss import MGDLoss
from .ofd_loss import OFDLoss
from .pkd_loss import PKDLoss
from .relational_kd import AngleWiseRKD, DistanceWiseRKD
from .weighted_soft_label_distillation import WSLD

from .ct_loss import ContourletTransKD
from .shiftwindow_loss import ShiftWindowLoss
from .shiftwindowmask_loss import ShiftWindowRevLoss
from .fusion_loss import FusionLoss
from .wavelet_att_loss import WaveletAttLoss
from .wavelet_att_loss import WaveletAtt_L2_Loss
from .att_loss import Att_L2_Loss
from .attpro_loss import AttPro_L2_Loss
from .attprov2_loss import AttProv2_L2_Loss
from .attprov3_loss import AttProv3_L2_Loss, AttProv3_PCC_Loss
from .nlkd_loss import NLKD_Loss
from .smooth_l1_loss import SmoothL1Loss
from .nlkd_woskip_loss import NLKD_WOSkip_Loss
from .nlkd_IN_loss import NLKD_IN_Loss
from .dcn_nlkd_loss import DCN_NLKD_Loss
from .nlkd_IN_align_loss import NLKD_IN_align_Loss
from .nlkd_IN_gaussain_loss import NLKD_IN_Gaussian_Loss
from .nlkd_IN_EBgaussain_loss import NLKD_IN_EBGaussian_Loss
from .nlkd_woIN_loss import NLKD_woIN_Loss
from .IN_only_loss import IN_only_loss
from .nlkd_IN_4x4_loss import NLKD_IN_4x4_Loss
from .nlkd_IN_8x8_loss import NLKD_IN_8x8_Loss
from .mhcan_loss import MHCan_Loss
from .dw_loss import DirectionLoss, WeightLoss, FeatureLoss
from .nlkd_IN_1d_loss import NLKD_IN_1d_Loss
from .layer_norm_loss import LayerNorm_loss

__all__ = [
    'ChannelWiseDivergence', 'KLDivergence', 'AngleWiseRKD', 'DistanceWiseRKD',
    'WSLD', 'L2Loss', 'ABLoss', 'DKDLoss', 'KDSoftCELoss', 'ActivationLoss',
    'OnehotLikeLoss', 'InformationEntropyLoss', 'FTLoss', 'ATLoss', 'OFDLoss',
    'L1Loss', 'FBKDLoss', 'CRDLoss', 'CrossEntropyLoss', 'PKDLoss', 'MGDLoss',
    'DISTLoss', 
    'ContourletTransKD',
    'ShiftWindowLoss',
    'ShiftWindowRevLoss',
    'FusionLoss',
    'WaveletAttLoss',
    'WaveletAtt_L2_Loss',
    'AttPro_L2_Loss',
    'AttProv2_L2_Loss',
    'AttProv3_L2_Loss',
    'AttProv3_PCC_Loss',
    'NLKD_Loss',
    'NLKD_WOSkip_Loss',
    'NLKD_IN_Loss',
    'DCN_NLKD_Loss',
    'NLKD_IN_align_Loss',
    'NLKD_IN_Gaussian_Loss',
    'NLKD_IN_EBGaussian_Loss',
    'NLKD_woIN_Loss',
    'IN_only_loss',
    'NLKD_IN_4x4_Loss',
    'NLKD_IN_8x8_Loss',
    'MHCan_Loss',
    'DirectionLoss',
    'WeightLoss',
    'FeatureLoss',
    'NLKD_IN_1d_Loss',
    'LayerNorm_loss',
]
