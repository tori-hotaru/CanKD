import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

@MODELS.register_module()
class LayerNorm_loss(nn.Module):
    def __init__(self,
                switch=False,
                loss_weight=1.0):
        super(LayerNorm_loss, self).__init__()
        self.loss_weight = loss_weight
        self.switch = switch
        
    def forward(self, s_feature, t_feature):
        assert s_feature.shape == t_feature.shape
        B, C, H, W = s_feature.shape
        s_feature = nn.LayerNorm(normalized_shape=[C, H, W], elementwise_affine=self.switch)(s_feature)
        t_feature = nn.LayerNorm(normalized_shape=[C, H, W], elementwise_affine=self.switch)(t_feature)
        loss = F.mse_loss(s_feature, t_feature) 
        return loss * self.loss_weight

    
if __name__ == '__main__':
    s_feature = torch.rand(2, 256, 167, 100)
    t_feature = torch.rand(2, 256, 167, 100)
    
    layer_norm_loss = LayerNorm_loss()
    loss = layer_norm_loss(s_feature, t_feature)
    print(loss)