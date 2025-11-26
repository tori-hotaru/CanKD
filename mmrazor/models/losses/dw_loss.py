import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

@MODELS.register_module()
class DirectionLoss(nn.Module):
    def __init__(self,
                in_channels: int,
                kernel_size: int,
                alpha: float):
        super(DirectionLoss, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.alpha = alpha
    
    def forward(self, s_offset, t_offset):
        '''
        offset is from SDDCNConnector offset output, shape is (B, (o1+o2)*(kernel_size*kernel_size), H, W)
        s_offset and t_offset are student and teacher offset output
        '''
        assert s_offset.shape == t_offset.shape and s_offset.shape[1] == 2*self.kernel_size*self.kernel_size

        batch_size, _, h, w = s_offset.shape
        # reshape offset to (B, 2, (kernel_size*kernel_size), H, W)
        s_offset = s_offset.view(batch_size, 2, self.kernel_size*self.kernel_size, h, w)
        t_offset = t_offset.view(batch_size, 2, self.kernel_size*self.kernel_size, h, w)

        # reshape offset to (B, 2, (kernel_size*kernel_size), H*W)
        s_offset = s_offset.view(batch_size, 2, self.kernel_size*self.kernel_size, -1)
        t_offset = t_offset.view(batch_size, 2, self.kernel_size*self.kernel_size, -1)

        points_direction_loss = 0
        kernel_points_num = self.kernel_size*self.kernel_size
        for i in range(kernel_points_num):
            s_point_offset = s_offset[:, :, i, :]
            t_point_offset = t_offset[:, :, i, :]

            # transform into (B, HW, 2)
            s_point_offset = s_point_offset.permute(0, 2, 1)
            t_point_offset = t_point_offset.permute(0, 2, 1)

            cos_similarity = F.cosine_similarity(s_point_offset, t_point_offset, dim=-1)
            points_direction_loss += (1-cos_similarity).mean()

        direction_loss = self.alpha * (points_direction_loss / kernel_points_num)
        return direction_loss
    
@MODELS.register_module()
class WeightLoss(nn.Module):
    def __init__(self,
                in_channels: int,
                kernel_size: int,
                beta: float,
                loss_type: str):
        super(WeightLoss, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.beta = beta
        self.loss_type = loss_type
    def forward(self, s_mask, t_mask):
        '''
        mask is from SDDCNConnector mask output, shape is (B, (kernel_size*kernel_size), H, W)
        s_mask and t_mask are student and teacher mask output
        '''
        assert s_mask.shape == t_mask.shape and s_mask.shape[1] == self.kernel_size*self.kernel_size
        if self.loss_type == 'mse':
            weight_loss = F.mse_loss(s_mask, t_mask)
        elif self.loss_type == 'l1':
            weight_loss = F.l1_loss(s_mask, t_mask)
        elif self.loss_type == 'smooth_l1_loss':
            weight_loss = F.smooth_l1_loss(s_mask, t_mask)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}")
        weight_loss = self.beta * weight_loss

        return weight_loss
    
@MODELS.register_module()
class FeatureLoss(nn.Module):
    def __init__(self,
                in_channels: int,
                kernel_size: int,
                gamma: float):
        super(FeatureLoss, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.gamma = gamma
    
    def forward(self, s_feature, t_feature):
        '''
        feature is from SDDCNConnector deformable conv feature output, shape is (B, (kernel_size*kernel_size), H, W)
        s_feature and t_feature are student and teacher deformable conv feature output
        '''
        assert s_feature.shape == t_feature.shape
        feature_loss = F.mse_loss(s_feature, t_feature)
        feature_loss = self.gamma * feature_loss
        return feature_loss
        
        
        

if __name__ == '__main__':
    s_offset = torch.randn(4, 18, 28, 28)
    t_offset = torch.randn(4, 18, 28, 28)
    s_mask = torch.randn(4, 9, 28, 28)
    t_mask = torch.randn(4, 9, 28, 28)

    direction_loss = DirectionLoss(in_channels=18, kernel_size=3, alpha=1.0)
    loss = direction_loss(s_offset, t_offset)
    print(loss)

    weight_loss = WeightLoss(in_channels=18, kernel_size=3, beta=1.0, loss_type='smooth_l1_loss')
    loss = weight_loss(s_mask, t_mask)
    print(loss)
