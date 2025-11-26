import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint


class attentionMatrix(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels, 
                 dimension=3, 
                 sub_sample=True):
        super(attentionMatrix, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.phi = nn.Sequential(self.phi, max_pool_layer)
        
    def forward(self, y_s, y_t):
        """
        y_s: student network's feature maps
        y_t: teacher network's feature maps
        """
        
        assert y_s.shape == y_t.shape

        batch_size = y_s.size(0)

        theta_y_s = self.theta(y_s).view(batch_size, self.inter_channels, -1)
        theta_y_s = theta_y_s.permute(0, 2, 1)

        phi_y_t = self.phi(y_t).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_y_s, phi_y_t)
        N = f.size(-1)
        f_div_C = f / N

        return f_div_C


class NonLocalAttention(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels=None, 
                 dimension=3, 
                 sub_sample=True, 
                 bn_layer=True):
        super(NonLocalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.att = attentionMatrix(self.in_channels, self.inter_channels, dimension, sub_sample)

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                            kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)
        
        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
        
    def forward(self, y_s, y_t):
        """
        y_s: student network's feature maps
        y_t: teacher network's feature maps
        """
        assert y_s.shape == y_t.shape
        batch_size = y_s.size(0)

        # g_y_s = self.g(y_s).view(batch_size, self.inter_channels, -1)
        # g_y_s = g_y_s.permute(0, 2, 1)

        # f_div_C = self.att(y_s, y_t)

        # y_s_ = torch.matmul(f_div_C, g_y_s)
        # y_s_ = y_s_.permute(0, 2, 1).contiguous()
        # y_s_ = y_s_.view(batch_size, self.inter_channels, *y_s.size()[2:])
        # W_y_s = self.W(y_s_)
        # z_s = W_y_s + y_s

        # g_y_t = self.g(y_t).view(batch_size, self.inter_channels, -1)
        # g_y_t = g_y_t.permute(0, 2, 1)

        # y_t_ = torch.matmul(f_div_C, g_y_t)
        # y_t_ = y_t_.permute(0, 2, 1).contiguous()
        # y_t_ = y_t_.view(batch_size, self.inter_channels, *y_t.size()[2:])
        # W_y_t = self.W(y_t_)
        # z_t = W_y_t + y_t

        f_div_C = self.att(y_s, y_t)

        g_y_t = self.g(y_t).view(batch_size, self.inter_channels, -1)
        g_y_t = g_y_t.permute(0, 2, 1)

        y_s_ = torch.matmul(f_div_C, g_y_t)
        y_s_ = y_s_.permute(0, 2, 1).contiguous()
        y_s_ = y_s_.view(batch_size, self.inter_channels, *y_s.size()[2:])
        W_y_s = self.W(y_s_)
        
        # # 添加分析代码
        # W_norm = torch.norm(W_y_s)
        # y_norm = torch.norm(y_s)
        # print(f"W_y_s norm: {W_norm.item():.4f}")
        # print(f"y_s norm: {y_norm.item():.4f}")
        # print(f"Ratio (W_y_s/y_s): {(W_norm/y_norm).item():.4f}")
        
        z_s = W_y_s + y_s

        z_t = y_t

        return z_s, z_t

    
@MODELS.register_module()
class NLKD_Loss(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels=None, 
                 dimension=2, 
                 sub_sample=True,
                 bn_layer=True, 
                 tau=1.0, 
                 loss_weight=1.0):
        super(NLKD_Loss, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels == 0: self.inter_channels = 1
        self.tau = tau
        self.loss_weight = loss_weight
        self.non_local_att = NonLocalAttention(self.in_channels, self.inter_channels, dimension, sub_sample, bn_layer)

    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        assert len(feat.shape) == 4
        N, C, H, W = feat.shape
        feat = feat.permute(1, 0, 2, 3).reshape(C, -1)
        mean = feat.mean(dim=-1, keepdim=True)
        std = feat.std(dim=-1, keepdim=True)
        feat = (feat - mean) / (std + 1e-6)
        return feat.reshape(C, N, H, W).permute(1, 0, 2, 3)
    
    def forward(self, s_feature, t_feature):
        assert s_feature.shape == t_feature.shape
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        n_z_s, n_z_t = self.norm(z_s), self.norm(z_t)
        loss = F.mse_loss(n_z_s, n_z_t) / 2
        
        return loss * self.loss_weight
    
if __name__ == '__main__':
    y_s = torch.randn(2, 256, 167, 100)
    y_t = torch.randn(2, 256, 167, 100)

    nlkd_loss = NLKD_Loss(in_channels=256, dimension=2)
    
    # 修改W的初始化权重，不再使用0
    if isinstance(nlkd_loss.non_local_att.W, nn.Sequential):
        nn.init.normal_(nlkd_loss.non_local_att.W[0].weight, mean=0.0, std=0.017)
        nn.init.normal_(nlkd_loss.non_local_att.W[1].weight, mean=0.0, std=0.4)
    else:
        nn.init.normal_(nlkd_loss.non_local_att.W.weight, mean=0.0, std=0.01)
    
    # 计算两种方式的loss
    loss1 = nlkd_loss(y_s, y_t)
    print("Loss with non-local attention:", loss1.item())
    
    ys_norm = nlkd_loss.norm(y_s)
    yt_norm = nlkd_loss.norm(y_t)
    loss2 = F.mse_loss(ys_norm, yt_norm) / 2
    print("Direct norm loss:", loss2.item())
    
