import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

from pytorch_wavelets import DWTForward, DWTInverse

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
            self.max_pool_layer = max_pool_layer


    def forward(self, y_s, y_t):
        """
        y_s: student network's feature maps
        y_t: teacher network's feature maps
        """
        
        assert y_s.shape == y_t.shape

        batch_size = y_s.size(0)

        theta_y_s_ = self.theta(y_s)
        theta_y_s = theta_y_s_.view(batch_size, self.inter_channels, -1)
        theta_y_s = theta_y_s.permute(0, 2, 1)

        phi_y_t_ = self.phi(y_t)

        if self.sub_sample:
            phi_y_t = self.max_pool_layer(phi_y_t_).view(batch_size, self.inter_channels, -1)
        else:
            phi_y_t = phi_y_t_.view(batch_size, self.inter_channels, -1)
            
        # print(theta_y_s.shape)
        # print(phi_y_t.shape)

        f = torch.matmul(theta_y_s, phi_y_t)
        N = f.size(-1)
        f_div_C_student = f / N
        ##########################

        phi_y_t = phi_y_t_.view(batch_size, self.inter_channels, -1).permute(0, 2, 1)

        if self.sub_sample:
            theta_y_s = self.max_pool_layer(theta_y_s_).view(batch_size, self.inter_channels, -1)
        else:
            theta_y_s = theta_y_s_.view(batch_size, self.inter_channels, -1)
        
        f_ = torch.matmul(phi_y_t, theta_y_s)
        N = f_.size(-1)
        f_div_C_teacher = f_ / N

 
        # print(f_div_C_teacher.shape)

        # theta_y_t = self.theta(y_t).view(batch_size, self.inter_channels, -1)
        # theta_y_t = theta_y_t.permute(0, 2, 1)

        # phi_y_s = self.phi(y_s).view(batch_size, self.inter_channels, -1)

        # f = torch.matmul(theta_y_t, phi_y_s)
        # N = f.size(-1)
        # f_div_C_student = f / N

        return f_div_C_teacher, f_div_C_student

class attentionMatrix_2(nn.Module):

    pass

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

        g_y_s = self.g(y_s).view(batch_size, self.inter_channels, -1)
        g_y_s = g_y_s.permute(0, 2, 1)

        f_div_C_teacher, f_div_C_student = self.att(y_s, y_t)

        y_t_ = torch.matmul(f_div_C_teacher, g_y_s)
        y_t_ = y_t_.permute(0, 2, 1).contiguous()
        y_t_ = y_t_.view(batch_size, self.inter_channels, *y_s.size()[2:])
        W_y_t = self.W(y_t_)
        z_t = W_y_t + y_t
        #########################################

        g_y_t = self.g(y_t).view(batch_size, self.inter_channels, -1)
        g_y_t = g_y_t.permute(0, 2, 1)

        y_s_ = torch.matmul(f_div_C_student, g_y_t)
        y_s_ = y_s_.permute(0, 2, 1).contiguous()
        y_s_ = y_s_.view(batch_size, self.inter_channels, *y_t.size()[2:])
        W_y_s = self.W(y_s_)
        z_s = W_y_s + y_s

        return z_s, z_t
    
    
@MODELS.register_module()
class AttProv2Loss(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels=None, 
                 dimension=2, 
                 sub_sample=True,
                 bn_layer=True, 
                 gamma=0.5, 
                 tau=1.0, 
                 loss_weight=1.0):
        super(AttProv2Loss, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels == 0: self.inter_channels = 1
        self.gamma = gamma
        self.tau = tau
        self.loss_weight = loss_weight
        self.non_local_att = NonLocalAttention(self.in_channels, self.inter_channels, dimension, sub_sample, bn_layer)
        self.KL_loss = nn.KLDivLoss(reduction="sum")

    def forward(self, s_feature, t_feature):
        N, C, H, W = s_feature.shape
        assert s_feature.shape == t_feature.shape
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        # print(z_s.shape, z_t.shape)

        s_feature_reshaped = z_s.view(-1, W * H)
        t_feature_reshaped = z_t.view(-1, W * H)

        s_feature_log = F.log_softmax(s_feature_reshaped/self.tau, dim=1)
        t_feature_soft = F.softmax(t_feature_reshaped/self.tau, dim=1)

        loss = self.gamma * self.KL_loss(s_feature_log, t_feature_soft)
        loss = self.loss_weight * loss/(C * N)

        return loss


@MODELS.register_module()
class AttProv2_L2_Loss(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels=None, 
                 dimension=2, 
                 sub_sample=True,
                 bn_layer=True, 
                 tau=1.0, 
                 loss_weight=1.0):
        super(AttProv2_L2_Loss, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels == 0: self.inter_channels = 1
        self.tau = tau
        self.loss_weight = loss_weight
        self.non_local_att = NonLocalAttention(self.in_channels, self.inter_channels, dimension, sub_sample, bn_layer)
        self.l2_loss = nn.MSELoss()

    def forward(self, s_feature, t_feature):
        assert s_feature.shape == t_feature.shape
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        loss = self.l2_loss(z_s, z_t)/self.tau
        loss = self.loss_weight * loss
        return loss


if __name__ == '__main__':
    # y_s = torch.randn(1, 32, 16, 16, 16)
    # y_t = torch.randn(1, 32, 16, 16, 16)
    # non_local_att = NonLocalAttention(in_channels=32, inter_channels=16, dimension=3)
    # z_s, z_t = non_local_att(y_s, y_t)
    # print(z_s.shape, z_t.shape)
    # print(y_s)
    # print(z_s)

    y_s = torch.randn(1, 32, 16, 16)
    y_t = torch.randn(1, 32, 16, 16)
    att_loss = AttProv2Loss(in_channels=32, dimension=2)
    loss = att_loss(y_s, y_t)
    print(loss)

    att_l2_loss = AttProv2_L2_Loss(in_channels=32, dimension=2)
    loss = att_l2_loss(y_s, y_t)
    print(loss)
        
