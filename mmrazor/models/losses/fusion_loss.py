import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

from pytorch_wavelets import DWTForward, DWTInverse

class ImgFusion(nn.Module):
    def __init__(self, J=2, mode='zero', wave='haar'):
        super(ImgFusion, self).__init__()
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)
    
    def fusion_strategy(self, c1, c2, h1, h2):
        # 低频分量采用平均融合
        c = (c1 + c2) * 0.5
        
        # h1_1 shape: B, C, 3, H/2, W/2
        # h1_2 shape: B, C, 3, H/4, W/4
        h = []
        for i in range(len(h1)):
            # 在通道维度上求和，认为所有通道共同构成特征的重要性
            E1 = torch.sum(torch.abs(h1[i]), dim=1, keepdim=True)  # [B, 1, 3, H, W]
            E2 = torch.sum(torch.abs(h2[i]), dim=1, keepdim=True)
            mask = (E1 >= E2).float()
            h_fused = h1[i] * mask + h2[i] * (1 - mask)
            h.append(h_fused)
        return c, h

    def forward(self, img1, img2):
        # 确保 dwt 和 idwt 在与输入相同的设备上
        if img1.is_cuda and not isinstance(self.dwt, torch.nn.DataParallel):
            self.dwt = self.dwt.cuda(img1.device)
            self.idwt = self.idwt.cuda(img1.device)
            
        # 进行小波分解
        c1, h1 = self.dwt(img1)
        c2, h2 = self.dwt(img2)
        
        # 融合低频和高频系数
        c_fused, h_fused = self.fusion_strategy(c1, c2, h1, h2)
        
        # 重建融合图像
        fused_img = self.idwt((c_fused, h_fused))
        
        return fused_img
    
@MODELS.register_module()
class FusionLoss(nn.Module):
    def __init__(self, gamma=0.5, tau=1.0, loss_weight=1.0):
        super(FusionLoss, self).__init__()
        self.gamma = gamma
        self.tau = tau
        self.loss_weight = loss_weight
        self.mse_loss = nn.MSELoss()
        self.KL_loss = nn.KLDivLoss(reduction="sum")
    
    def forward(self, s_feature, t_feature):
        N, C, H, W = s_feature.shape
        assert s_feature.shape == t_feature.shape
        
        # 进行图像融合
        fused_img = ImgFusion()(s_feature, t_feature)
        # print(fused_img.shape)
        # print(s_feature.shape)
        
        # 重塑维度以匹配输入特征
        s_feature_reshaped = s_feature.view(-1, W * H)
        fused_img_reshaped = fused_img.view(-1, W * H)
        
        # 应用 softmax 和 log_softmax
        s_feature_log = F.log_softmax(s_feature_reshaped/self.tau, dim=1)
        fused_img_soft = F.softmax(fused_img_reshaped/self.tau, dim=1)
        
        # 计算 KL 损失
        loss = self.gamma * self.KL_loss(s_feature_log, fused_img_soft)
        loss = self.loss_weight * loss/(C * N)
        
        return loss






if __name__ == '__main__':
    img1 = torch.randn(1, 3, 256, 256)
    img2 = torch.randn(1, 3, 256, 256)
    fusion_loss = FusionLoss()
    loss = fusion_loss(img1, img2)
    print(loss)
