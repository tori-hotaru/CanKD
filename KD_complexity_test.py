import torch
from thop import profile
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

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

class NLKD_IN_Loss(nn.Module):
    def __init__(self, 
                 in_channels, 
                 inter_channels=None, 
                 dimension=2, 
                 sub_sample=True,
                 bn_layer=True, 
                 tau=1.0, 
                 loss_weight=1.0):
        super(NLKD_IN_Loss, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels else in_channels // 2
        if self.inter_channels == 0: self.inter_channels = 1
        self.tau = tau
        self.loss_weight = loss_weight
        self.non_local_att = NonLocalAttention(self.in_channels, self.inter_channels, dimension, sub_sample, bn_layer)
        
        # 添加Instance Normalization层
        self.in_norm = nn.InstanceNorm2d(in_channels, affine=False)
        # # 初始化IN参数
        # if self.in_norm.weight is not None:
        #     nn.init.constant_(self.in_norm.weight, 1.0)
        # if self.in_norm.bias is not None:
        #     nn.init.constant_(self.in_norm.bias, 0.0)
    
    def norm(self, feat: torch.Tensor) -> torch.Tensor:
        # 使用IN进行标准化
        return self.in_norm(feat)
    
    def forward(self, s_feature, t_feature):
        # print(f"Student feature shape: {s_feature.shape}")
        # print(f"Teacher feature shape: {t_feature.shape}")
        assert s_feature.shape == t_feature.shape
        z_s, z_t = self.non_local_att(s_feature, t_feature)
        n_z_s, n_z_t = self.norm(z_s), self.norm(z_t)
        loss = F.mse_loss(n_z_s, n_z_t) /2
        
        return loss * self.loss_weight



class FeatureLoss(nn.Module):

    """PyTorch version of `Focal and Global Knowledge Distillation for Detectors`
   
    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map. 
        temp (float, optional): Temperature coefficient. Defaults to 0.5.
        name (str): the loss name of the layer
        alpha_fgd (float, optional): Weight of fg_loss. Defaults to 0.001
        beta_fgd (float, optional): Weight of bg_loss. Defaults to 0.0005
        gamma_fgd (float, optional): Weight of mask_loss. Defaults to 0.001
        lambda_fgd (float, optional): Weight of relation_loss. Defaults to 0.000005
    """
    def __init__(self,
                 student_channels,
                 teacher_channels,
                 temp=0.5,
                 alpha_fgd=0.001,
                 beta_fgd=0.0005,
                 gamma_fgd=0.001,
                 lambda_fgd=0.000005,
                 ):
        super(FeatureLoss, self).__init__()
        self.temp = temp
        self.alpha_fgd = alpha_fgd
        self.beta_fgd = beta_fgd
        self.gamma_fgd = gamma_fgd
        self.lambda_fgd = lambda_fgd

        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.align = None
        
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, kernel_size=1)
        self.channel_add_conv_s = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))
        self.channel_add_conv_t = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels//2, kernel_size=1),
            nn.LayerNorm([teacher_channels//2, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(teacher_channels//2, teacher_channels, kernel_size=1))


    def forward(self,
                preds_S,
                preds_T,
                gt_bboxes,
                img_metas):
        """Forward function.
        Args:
            preds_S(Tensor): Bs*C*H*W, student's feature map
            preds_T(Tensor): Bs*C*H*W, teacher's feature map
            gt_bboxes(tuple): Bs*[nt*4], pixel decimal: (tl_x, tl_y, br_x, br_y)
            img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        """
        assert preds_S.shape[-2:] == preds_T.shape[-2:],'the output dim of teacher and student differ'

        if self.align is not None:
            preds_S = self.align(preds_S)
        
        N,C,H,W = preds_S.shape

        S_attention_t, C_attention_t = self.get_attention(preds_T, self.temp)
        S_attention_s, C_attention_s = self.get_attention(preds_S, self.temp)

        Mask_fg = torch.zeros_like(S_attention_t)
        Mask_bg = torch.ones_like(S_attention_t)
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])

        fg_loss, bg_loss = self.get_fea_loss(preds_S, preds_T, Mask_fg, Mask_bg, 
                           C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        mask_loss = self.get_mask_loss(C_attention_s, C_attention_t, S_attention_s, S_attention_t)
        rela_loss = self.get_rela_loss(preds_S, preds_T)


        loss = self.alpha_fgd * fg_loss + self.beta_fgd * bg_loss \
               + self.gamma_fgd * mask_loss + self.lambda_fgd * rela_loss
            
        return loss


    def get_attention(self, preds, temp):
        """ preds: Bs*C*W*H """
        N, C, H, W= preds.shape

        value = torch.abs(preds)
        # Bs*W*H
        fea_map = value.mean(axis=1, keepdim=True)
        S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)

        # Bs*C
        channel_map = value.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False)
        C_attention = C * F.softmax(channel_map/temp, dim=1)

        return S_attention, C_attention


    def get_fea_loss(self, preds_S, preds_T, Mask_fg, Mask_bg, C_s, C_t, S_s, S_t):
        loss_mse = nn.MSELoss(reduction='sum')
        
        Mask_fg = Mask_fg.unsqueeze(dim=1)
        Mask_bg = Mask_bg.unsqueeze(dim=1)

        C_t = C_t.unsqueeze(dim=-1)
        C_t = C_t.unsqueeze(dim=-1)

        S_t = S_t.unsqueeze(dim=1)

        fea_t= torch.mul(preds_T, torch.sqrt(S_t))
        fea_t = torch.mul(fea_t, torch.sqrt(C_t))
        fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
        bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))

        fea_s = torch.mul(preds_S, torch.sqrt(S_t))
        fea_s = torch.mul(fea_s, torch.sqrt(C_t))
        fg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_fg))
        bg_fea_s = torch.mul(fea_s, torch.sqrt(Mask_bg))

        fg_loss = loss_mse(fg_fea_s, fg_fea_t)/len(Mask_fg)
        bg_loss = loss_mse(bg_fea_s, bg_fea_t)/len(Mask_bg)

        return fg_loss, bg_loss


    def get_mask_loss(self, C_s, C_t, S_s, S_t):

        mask_loss = torch.sum(torch.abs((C_s-C_t)))/len(C_s) + torch.sum(torch.abs((S_s-S_t)))/len(S_s)

        return mask_loss
     
    
    def spatial_pool(self, x, in_type):
        batch, channel, width, height = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        if in_type == 0:
            context_mask = self.conv_mask_s(x)
        else:
            context_mask = self.conv_mask_t(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = F.softmax(context_mask, dim=2)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context


    def get_rela_loss(self, preds_S, preds_T):
        loss_mse = nn.MSELoss(reduction='sum')

        context_s = self.spatial_pool(preds_S, 0)
        context_t = self.spatial_pool(preds_T, 1)

        out_s = preds_S
        out_t = preds_T

        channel_add_s = self.channel_add_conv_s(context_s)
        out_s = out_s + channel_add_s

        channel_add_t = self.channel_add_conv_t(context_t)
        out_t = out_t + channel_add_t

        rela_loss = loss_mse(out_s, out_t)/len(out_s)
        
        return rela_loss



class MGDConnector(nn.Module):
    """PyTorch version of `Masked Generative Distillation.

    <https://arxiv.org/abs/2205.01529>`

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        lambda_mgd (float, optional): masked ratio. Defaults to 0.65
        init_cfg (Optional[Dict], optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        lambda_mgd: float = 0.65,
        mask_on_channel: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_mgd = lambda_mgd
        self.mask_on_channel = mask_on_channel
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels,
                teacher_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        else:
            self.align = None

        self.generation = nn.Sequential(
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                teacher_channels, teacher_channels, kernel_size=3, padding=1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        if self.align is not None:
            feature = self.align(feature)

        N, C, H, W = feature.shape

        device = feature.device
        if not self.mask_on_channel:
            mat = torch.rand((N, 1, H, W)).to(device)
        else:
            mat = torch.rand((N, C, 1, 1)).to(device)

        mat = torch.where(mat > 1 - self.lambda_mgd,
                          torch.zeros(1).to(device),
                          torch.ones(1).to(device)).to(device)

        masked_fea = torch.mul(feature, mat)
        new_fea = self.generation(masked_fea)
        return new_fea

from mmcv.cnn import NonLocal2d

class NonLocal2dMaxpoolNstride(NonLocal2d):
    """Nonlocal block for 2-dimension inputs, with a configurable
    maxpool_stride.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971
    Code reference: https://github.com/AlexHex7/Non-local_pytorch

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Defaults to 2.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to `nn.Conv2d`.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to `BN`. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: dot_product.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        maxpool_stride (int): The stride of the maxpooling module.
            Defaults to 2.
        zeros_init (bool): Whether to use zero to initialize weights of
            `conv_out`. Defaults to True.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2,
                 conv_cfg: dict = dict(type='Conv2d'),
                 norm_cfg: dict = dict(type='BN'),
                 mode: str = 'embedded_gaussian',
                 sub_sample: bool = False,
                 maxpool_stride: int = 2,
                 zeros_init: bool = True,
                 **kwargs) -> None:
        """Inits the NonLocal2dMaxpoolNstride module."""
        super().__init__(
            in_channels=in_channels,
            sub_sample=sub_sample,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            reduction=reduction,
            mode=mode,
            zeros_init=zeros_init,
            **kwargs)
        self.norm_cfg = norm_cfg

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(
                kernel_size=(maxpool_stride, maxpool_stride))
            self.g: nn.Sequential = nn.Sequential(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi: nn.Sequential = nn.Sequential(
                    self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer


class FBKDStudentConnector(nn.Module):
    """Improve Object Detection with Feature-based Knowledge Distillation:
    Towards Accurate and Efficient Detectors, ICLR2021.
    https://openreview.net/pdf?id=uKhGRvM8QNH.

    Student connector for FBKD.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Defaults to 2.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to `nn.Conv2d`.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to `BN`. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: dot_product.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        maxpool_stride (int): The stride of the maxpooling module.
            Defaults to 2.
        zeros_init (bool): Whether to use zero to initialize weights of
            `conv_out`. Defaults to True.
        spatial_T (float): Temperature used in spatial-wise pooling.
            Defaults to 0.5.
        channel_T (float): Temperature used in channel-wise pooling.
            Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2,
                 conv_cfg: dict = dict(type='Conv2d'),
                 norm_cfg: dict = dict(type='BN'),
                 mode: str = 'dot_product',
                 sub_sample: bool = False,
                 maxpool_stride: int = 2,
                 zeros_init: bool = True,
                 spatial_T: float = 0.5,
                 channel_T: float = 0.5,
                 **kwargs) -> None:
        """Inits the FBKDStuConnector."""
        super().__init__()
        self.channel_wise_adaptation = nn.Linear(in_channels, in_channels)

        self.spatial_wise_adaptation = nn.Conv2d(
            1, 1, kernel_size=3, stride=1, padding=1)

        self.adaptation_layers = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.student_non_local = NonLocal2dMaxpoolNstride(
            in_channels=in_channels,
            reduction=reduction,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            mode=mode,
            sub_sample=sub_sample,
            maxpool_stride=maxpool_stride,
            zeros_init=zeros_init,
            **kwargs)

        self.non_local_adaptation = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.in_channels = in_channels
        self.spatial_T = spatial_T
        self.channel_T = channel_T

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input student features.

        Returns:
            s_spatial_mask (torch.Tensor): Student spatial-wise mask.
            s_channel_mask (torch.Tensor): Student channel-wise mask.
            s_feat_adapt (torch.Tensor): Adaptative student feature.
            s_channel_pool_adapt (torch.Tensor): Student feature which through
                channel-wise pooling and adaptation_layers.
            s_spatial_pool_adapt (torch.Tensor): Student feature which through
                spatial-wise pooling and adaptation_layers.
            s_relation_adapt (torch.Tensor): Adaptative student relations.
        """
        # Calculate spatial-wise mask.
        s_spatial_mask = torch.mean(torch.abs(x), [1], keepdim=True)
        size = s_spatial_mask.size()
        s_spatial_mask = s_spatial_mask.view(x.size(0), -1)

        # Soften or sharpen the spatial-wise mask by temperature.
        s_spatial_mask = torch.softmax(
            s_spatial_mask / self.spatial_T, dim=1) * size[-1] * size[-2]
        s_spatial_mask = s_spatial_mask.view(size)

        # Calculate channel-wise mask.
        s_channel_mask = torch.mean(torch.abs(x), [2, 3], keepdim=True)
        channel_mask_size = s_channel_mask.size()
        s_channel_mask = s_channel_mask.view(x.size(0), -1)

        # Soften or sharpen the channel-wise mask by temperature.
        s_channel_mask = torch.softmax(
            s_channel_mask / self.channel_T, dim=1) * self.in_channels
        s_channel_mask = s_channel_mask.view(channel_mask_size)

        # Adaptative and pool student feature through channel-wise.
        s_feat_adapt = self.adaptation_layers(x)
        s_channel_pool_adapt = self.channel_wise_adaptation(
            torch.mean(x, [2, 3]))

        # Adaptative and pool student feature through spatial-wise.
        s_spatial_pool = torch.mean(x, [1]).view(
            x.size(0), 1, x.size(2), x.size(3))
        s_spatial_pool_adapt = self.spatial_wise_adaptation(s_spatial_pool)

        # Calculate non_local_adaptation.
        s_relation = self.student_non_local(x)
        s_relation_adapt = self.non_local_adaptation(s_relation)

        return (s_spatial_mask, s_channel_mask, s_channel_pool_adapt,
                s_spatial_pool_adapt, s_relation_adapt, s_feat_adapt)

class FBKDTeacherConnector(nn.Module):
    """Improve Object Detection with Feature-based Knowledge Distillation:
    Towards Accurate and Efficient Detectors, ICLR2021.
    https://openreview.net/pdf?id=uKhGRvM8QNH.

    Teacher connector for FBKD.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Defaults to 2.
        conv_cfg (dict): The config dict for convolution layers.
            Defaults to `nn.Conv2d`.
        norm_cfg (dict): The config dict for normalization layers.
            Defaults to `BN`. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: dot_product.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        maxpool_stride (int): The stride of the maxpooling module.
            Defaults to 2.
        zeros_init (bool): Whether to use zero to initialize weights of
            `conv_out`. Defaults to True.
        spatial_T (float): Temperature used in spatial-wise pooling.
            Defaults to 0.5.
        channel_T (float): Temperature used in channel-wise pooling.
            Defaults to 0.5.
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 conv_cfg: dict = dict(type='Conv2d'),
                 norm_cfg: dict = dict(type='BN'),
                 mode: str = 'dot_product',
                 sub_sample: bool = False,
                 maxpool_stride: int = 2,
                 zeros_init: bool = True,
                 spatial_T: float = 0.5,
                 channel_T: float = 0.5,
                 **kwargs) -> None:
        super().__init__()
        self.teacher_non_local = NonLocal2dMaxpoolNstride(
            in_channels=in_channels,
            reduction=reduction,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            mode=mode,
            sub_sample=sub_sample,
            maxpool_stride=maxpool_stride,
            zeros_init=zeros_init,
            **kwargs)

        self.in_channels = in_channels
        self.spatial_T = spatial_T
        self.channel_T = channel_T

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Frorward function for training.

        Args:
            x (torch.Tensor): Input teacher features.

        Returns:
            t_spatial_mask (torch.Tensor): Teacher spatial-wise mask.
            t_channel_mask (torch.Tensor): Teacher channel-wise mask.
            t_spatial_pool (torch.Tensor): Teacher features which through
                spatial-wise pooling.
            t_relation (torch.Tensor): Teacher relation matrix.
        """
        # Calculate spatial-wise mask.
        t_spatial_mask = torch.mean(torch.abs(x), [1], keepdim=True)
        size = t_spatial_mask.size()
        t_spatial_mask = t_spatial_mask.view(x.size(0), -1)

        # Soften or sharpen the spatial-wise mask by temperature.
        t_spatial_mask = torch.softmax(
            t_spatial_mask / self.spatial_T, dim=1) * size[-1] * size[-2]
        t_spatial_mask = t_spatial_mask.view(size)

        # Calculate channel-wise mask.
        t_channel_mask = torch.mean(torch.abs(x), [2, 3], keepdim=True)
        channel_mask_size = t_channel_mask.size()
        t_channel_mask = t_channel_mask.view(x.size(0), -1)

        # Soften or sharpen the channel-wise mask by temperature.
        t_channel_mask = torch.softmax(
            t_channel_mask / self.channel_T, dim=1) * self.in_channels
        t_channel_mask = t_channel_mask.view(channel_mask_size)

        # Adaptative and pool student feature through spatial-wise.
        t_spatial_pool = torch.mean(x, [1]).view(
            x.size(0), 1, x.size(2), x.size(3))

        # Calculate non_local relation.
        t_relation = self.teacher_non_local(x)

        return (t_spatial_mask, t_channel_mask, t_spatial_pool, t_relation, x)


def get_shapes(item):
    """Recursively get shapes from a nested structure of tensors."""
    if isinstance(item, torch.Tensor):
        return item.shape
    elif isinstance(item, (list, tuple)):
        return [get_shapes(x) for x in item]
    else:
        return type(item)


def calculate_complexity(model, inputs, model_name):
    """Calculates and prints the parameters and FLOPs of a model."""
    model.eval()

    # thop expects inputs as a tuple
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    
    flops, params = profile(model, inputs=inputs, verbose=False)
    
    print(f"--- {model_name} ---")
    input_shapes = get_shapes(inputs)
    print(f"Input shapes: {input_shapes}")
    print(f"Parameters (M): {params / 1e6:.4f}")
    print(f"FLOPs (G): {flops / 1e9:.4f}")
    print("-" * (len(model_name) + 6))
    print()

def main():
    # --- Test NLKD_IN_Loss ---
    in_channels = 256
    s_feature_nlkd = torch.randn(2, in_channels, 64, 64)
    t_feature_nlkd = torch.randn(2, in_channels, 64, 64)
    nlkd_loss = NLKD_IN_Loss(in_channels=in_channels, dimension=2)
    calculate_complexity(nlkd_loss, (s_feature_nlkd, t_feature_nlkd), "NLKD_IN_Loss")

    # --- Test FeatureLoss ---
    preds_S = torch.randn(2, in_channels, 64, 64)
    preds_T = torch.randn(2, in_channels, 64, 64)
    gt_bboxes = (torch.rand(5, 4) * 256, torch.rand(3, 4) * 256)
    img_metas = [{'img_shape': (256, 256, 3)} for _ in range(2)]
    feature_loss = FeatureLoss(student_channels=in_channels, teacher_channels=in_channels)
    calculate_complexity(feature_loss, (preds_S, preds_T, gt_bboxes, img_metas), "FeatureLoss")

    # --- Test MGDConnector ---
    mgd_connector = MGDConnector(student_channels=in_channels, teacher_channels=in_channels)
    calculate_complexity(mgd_connector, preds_S, "MGDConnector")

    # --- Test FBKDStudentConnector ---
    fbkd_student_connector = FBKDStudentConnector(in_channels=in_channels)
    calculate_complexity(fbkd_student_connector, preds_S, "FBKDStudentConnector")

    # --- Test FBKDTeacherConnector ---
    fbkd_teacher_connector = FBKDTeacherConnector(in_channels=in_channels)
    calculate_complexity(fbkd_teacher_connector, preds_T, "FBKDTeacherConnector")

if __name__ == '__main__':
    main()
