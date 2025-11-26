import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

@MODELS.register_module()
class IN_only_loss(nn.Module):
    def __init__(self,  
                 in_channels,
                 loss_weight=1.0):
        super(IN_only_loss, self).__init__()
        self.loss_weight = loss_weight
        self.in_norm = nn.InstanceNorm2d(in_channels, affine=False)
        
    def forward(self, s_feature, t_feature):
        assert s_feature.shape == t_feature.shape
        s_feature = self.in_norm(s_feature)
        t_feature = self.in_norm(t_feature)
        loss = F.mse_loss(s_feature, t_feature) #/ 2 #20250803
        return loss * self.loss_weight
    
if __name__ == '__main__':
    s_feature = torch.rand(2, 256, 167, 100)
    t_feature = torch.rand(2, 256, 167, 100)
    
    in_only_loss = IN_only_loss(in_channels=256)
    loss = in_only_loss(s_feature, t_feature)
    print(loss)