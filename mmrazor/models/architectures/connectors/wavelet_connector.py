from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import NonLocal2d

from mmrazor.registry import MODELS
from pytorch_wavelets import DWTForward, DWTInverse

try:
    from .base_connector import BaseConnector
except ImportError:
    from mmrazor.models.architectures.connectors.base_connector import BaseConnector

@MODELS.register_module()
class WaveletConnector(BaseConnector):

    def __init__(self, 
                 student_channels: int, 
                 teacher_channels: int, 
                 norm: bool = False,
                 J: int = 3,
                 mode: str = 'zero',
                 wave: str = 'db1',
                 init_cfg: Optional[Dict] = None,
                 ):
        super().__init__(init_cfg)
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels
        self.norm = norm
        if self.student_channels != self.teacher_channels:
            self.align = nn.Conv2d(self.student_channels, self.teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None

        self.dwt = DWTForward(J=J, mode=mode, wave=wave)

    def forward_train(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature.shape
        if self.norm:
            feature = F.layer_norm(feature, (C, H, W))
        
        if self.align is not None:
            feature = self.align(feature)

        feature_l, feature_h = self.dwt(feature) # feature_l: B, C, H/2, W/2; feature_h: B, C, 3, H/2, W/2
        
        return feature_l, feature_h
    

if __name__ == '__main__':
    connector = WaveletConnector(student_channels=16, teacher_channels=16, norm=True, J=3, mode='zero', wave='db1')
    feature = torch.randn(1, 16, 32, 32)
    feature_l, feature_h = connector.forward_train(feature)
    print(feature_l.shape, len(feature_h))
    