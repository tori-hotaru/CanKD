import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

@MODELS.register_module()
class LayerNorm_weight_loss(nn.Module):
    def __init__(self, switch=False, num_channels=None, loss_weight=1.0): ####20250902
        super().__init__()
        self.loss_weight = loss_weight
        self.switch = switch
        if self.switch:
            assert num_channels is not None
            self.gamma = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, s_feature, t_feature):
        assert s_feature.shape == t_feature.shape
        _, C, H, W = s_feature.shape
        s = F.layer_norm(s_feature, (C, H, W))
        t = F.layer_norm(t_feature, (C, H, W))
        if self.switch:
            s = s * self.gamma.detach() + self.beta.detach()
            t = t * self.gamma + self.beta
        return F.mse_loss(s, t) * self.loss_weight
    
if __name__ == '__main__':
    s_feature = torch.rand(2, 256, 167, 100)
    t_feature = torch.rand(2, 256, 167, 100)
    
    in_only_loss = IN_only_loss(in_channels=256)
    loss = in_only_loss(s_feature, t_feature)
    print(loss)