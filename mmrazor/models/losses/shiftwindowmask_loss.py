import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint

from pytorch_wavelets import DWTForward, DWTInverse


import torch
import torch.nn as nn

class PatchAttentionMask(nn.Module):
    def __init__(self, patch_size, k, channels, stride=None):
        """
        初始化 PatchAttentionMask 类。

        参数:
            patch_size (int): 补丁的边长大小。
            k (int): 要选择的得分最高的补丁数量。
            channels (int): 输入特征图的通道数。
            stride (int, optional): 补丁滑动的步长，默认等于1。
        """
        super(PatchAttentionMask, self).__init__()
        self.patch_size = patch_size
        self.k = k
        self.channels = channels
        self.stride = stride if stride is not None else 1

        # 使用可配置stride的nn.Unfold提取补丁
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=self.stride)
        self.fold = nn.Fold(output_size=(1,1), kernel_size=patch_size, stride=self.stride)
        
        # 添加可学习的空间和通道注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.Linear(channels, channels // 16),
            nn.ReLU(),
            nn.Linear(channels // 16, channels),
            nn.Sigmoid()
        )

    def get_attention(self, preds, temp=0.1, update_S=True, update_C=True):
        N, C, H, W = preds.shape
        
        # 使用更强的特征表示
        value = preds ** 2
        
        # 改进的空间注意力
        fea_map = value.mean(axis=1, keepdim=True)
        max_pool = F.max_pool2d(fea_map, kernel_size=3, stride=1, padding=1)
        fea_map = fea_map + max_pool
        if update_S:
            spatial_weights = self.spatial_attention(fea_map)
            S_attention = (H * W * F.softmax((fea_map * spatial_weights/temp).view(N,-1), dim=1)).view(N, H, W)
        else:
            S_attention = (H * W * F.softmax((fea_map/temp).view(N,-1), dim=1)).view(N, H, W)
        
        
        # 改进的通道注意力
        channel_map = value.mean(axis=2, keepdim=False).mean(axis=2, keepdim=False)
        if update_C:
            channel_weights = self.channel_attention(channel_map)
            C_attention = C * F.softmax(channel_map * channel_weights/temp, dim=1)
        else:
            C_attention = C * F.softmax(channel_map/temp, dim=1)
        
        return S_attention, C_attention

    def score_calculate(self, patches, alpha=0.7):  # 增加空间注意力的权重
        """
        patches(torch.Tensor): (B,L,C*W*H) B 为 batch size, L 为 patch 数量, C*W*H 为 patch 的维度
        """
        B, L, D = patches.shape
        
        # Reshape all patches at once: (B,L,C*W*H) -> (B*L,C,H,W)
        patches = patches.reshape(-1, self.channels, self.patch_size, self.patch_size)
        # patches = patches.view(-1, self.channels, self.patch_size, self.patch_size)
        
        # Get attention for all patches simultaneously
        S_attention, C_attention = self.get_attention(patches)
        
        # Calculate spatial scores: (B*L,H,W) -> (B*L,1)
        spatial_score = S_attention.mean(dim=(1,2)).unsqueeze(1)
        
        # Calculate channel scores: (B*L,C) -> (B*L,1) 
        channel_score = C_attention.mean(dim=1).unsqueeze(1)
        
        # Combine scores and reshape back to (B,L)
        final_score = (alpha * spatial_score + (1 - alpha) * channel_score).view(B, L)
        
        return final_score

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入特征图，形状为 (batch_size, channels, height, width)。

        返回:
            mask (torch.Tensor): 掩码，形状为 (batch_size, 1, height, width)。
        """
        batch_size, channels, height, width = x.size()

        # 检查输入尺寸是否合法
        h_out = (height - self.patch_size) / self.stride + 1
        w_out = (width - self.patch_size) / self.stride + 1
        
        if not (h_out.is_integer() and w_out.is_integer()):
            raise ValueError(
                f"Input size ({height}, {width}) with patch_size={self.patch_size} "
                f"and stride={self.stride} is invalid. "
                f"Height and width should satisfy: (H - P) / S + 1 = integer"
            )
        
        # 使用 unfold 提取所有补丁
        patches = self.unfold(x)  # 形状: (batch_size, channels * patch_size * patch_size, L)
        L = patches.size(-1)  # 补丁数量

        # 转置便于计算得分
        patches = patches.transpose(1, 2)  # 形状: (batch_size, L, channels * patch_size * patch_size)

        # 计算每个补丁的注意力得分
        scores = self.score_calculate(patches)

        # 选择得分最低的 k 个补丁
        bottomk_scores, bottomk_indices = torch.topk(scores, self.k, dim=1, largest=False)  # 使用 largest=False 选择最小值

        # 创建一个 one-hot 编码的选中补丁（初始值全1）
        one_hot = torch.ones(batch_size, L, device=x.device)
        one_hot.scatter_(1, bottomk_indices, 0)  # 将选中的最低分补丁位置设为0

        # 将 one-hot 编码扩展并重复
        one_hot = one_hot.unsqueeze(1)
        one_hot = one_hot.expand(-1, channels * self.patch_size * self.patch_size, -1)

        # 使用 Fold 将选中的补丁映射回空间掩码
        mask = F.fold(one_hot, output_size=(height, width), 
                     kernel_size=self.patch_size, 
                     stride=self.stride)

        # 将重叠区域的处理：只要有一个patch是1，该位置就为1
        mask = (mask > 0).float()

        return mask

class muti_shift_window_mask(nn.Module):
    def __init__(self, patch_sizes, k, channels_list, strides=None):
        """
        patch_sizes: list, 每个元素为int, 表示每个补丁的大小
        k: list, 每个元素为int, 表示每个shift window选择k个最显著的特征
        channels_list: list, 每个元素为int, 表示每个shift window的通道数
        strides: list, 每个元素为int, 表示每个shift window滑动的步长, 默认为None, 即stride=patch_size
        """
        super(muti_shift_window_mask, self).__init__()
        self.patch_sizes = patch_sizes
        self.k = k
        self.channels_list = channels_list
        self.strides = strides if strides is not None else [1] * len(patch_sizes)

        self.pam_list = nn.ModuleList([PatchAttentionMask(patch_size=patch_size, k=k, channels=channels, stride=stride) for patch_size, k, channels, stride in zip(patch_sizes, k, channels_list, strides)])

    def forward(self, x):
        masks = [pam(x) for pam in self.pam_list]
        return masks

@MODELS.register_module()
class ShiftWindowRevLoss(nn.Module):
    def __init__(self, patch_sizes, k, channels_list, strides=None, alpha=0.7, beta=0.3):
        """
        patch_sizes: list, 每个元素为int, 表示每个补丁的大小
        k: list, 每个元素为int, 表示每个shift window选择k个最显著的特征
        channels_list: list, 每个元素为int, 表示每个shift window的通道数
        strides: list, 每个元素为int, 表示每个shift window滑动的步长, 默认为None, 即stride=patch_size
        alpha: float, 对特征图进行shift window后，计算教师网络和学生网络的global loss的权重
        beta: float, 对特征图进行wavelet变换后，计算教师网络和学生网络的frequency loss的权重
        """
        super(ShiftWindowRevLoss, self).__init__()
        self.msm_global = muti_shift_window_mask(patch_sizes, k, channels_list, strides)
        k_frenquency = [ki // 10 for ki in k]
        self.msm_frequency = muti_shift_window_mask(patch_sizes, k_frenquency, channels_list, strides)
        self.alpha = alpha
        self.beta = beta
        self.wavelet = DWTForward(J=2, mode='zero', wave='haar')

    def wavelet_transform(self, x):
        yl, yH = self.wavelet(x)
        return yl, yH

    def mask_apply(self, y_s, y_t):
        """
        y_s: student network的特征图
        y_t: teacher network的特征图
        """
        assert y_s.shape == y_t.shape
        # 直接使用教师网络生成mask，允许参数更新
        masks = self.msm_global(y_t) 
        
        # 将mask应用到学生网络和教师网络的特征图上
        y_s_list = [y_s * mask for mask in masks]
        y_t_list = [y_t * mask for mask in masks]
        
        return y_s_list, y_t_list

    def global_loss(self, y_s_list, y_t_list):
        """
        y_s_list: student network的特征图列表
        y_t_list: teacher network的特征图列表
        """
        assert len(y_s_list) == len(y_t_list)
        y_s_list = torch.stack(y_s_list, dim=1)
        y_t_list = torch.stack(y_t_list, dim=1)

        global_loss = F.mse_loss(y_s_list, y_t_list)
        return global_loss

    def frequency_loss(self, y_s, y_t):
        """
        y_s: student network's feature maps
        y_t: teacher network's feature maps
        """
        # Perform wavelet transforms
        cA_s, cH_s = self.wavelet_transform(y_s)
        cA_t, cH_t = self.wavelet_transform(y_t)
        
        # Initialize total loss
        total_loss = 0
        
        # Iterate over wavelet levels
        num_levels = len(cH_s)  # Correctly get the number of levels
        for j in range(num_levels):
            freq_s = cH_s[j] # B, C, 3, H/2, W/2
            freq_t = cH_t[j]
            freq_s = freq_s.sum(dim=2) # B, C, H/2, W/2
            freq_t = freq_t.sum(dim=2)
            # Generate masks using the teacher's high-frequency components
            masks = self.msm_frequency(freq_t)
            
            # Apply masks to both student and teacher features
            # Using wavelet frequency components do mask instead of the original features
            freq_s_masked = [freq_s * mask for mask in masks]
            freq_t_masked = [freq_t * mask for mask in masks]
            
            # Compute mean squared error loss for the current level
            layer_loss = F.mse_loss(
                torch.stack(freq_s_masked, dim=1),
                torch.stack(freq_t_masked, dim=1)
            )
            total_loss += layer_loss
        
        # Average the loss over all levels
        return total_loss / num_levels  # Return average loss

    def forward(self, y_s, y_t):
        # print(y_s.shape)
        # print(y_t.shape)
        y_s_list, y_t_list = self.mask_apply(y_s, y_t)
        global_loss = self.global_loss(y_s_list, y_t_list)
        frequency_loss = self.frequency_loss(y_s, y_t)
        return self.alpha * global_loss + self.beta * frequency_loss

# 示例用法
if __name__ == "__main__":
    # import os
    # from PIL import Image
    # import torchvision.transforms as transforms

    # # 设置输入输出路径
    # input_dir = "/home/shizhe/New_project/test_image"  # 输入特征图文件夹
    # output_dir = "/home/shizhe/New_project/output_img"   # 输出掩码保存文件夹
    
    # # 创建输出文件夹
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # # 设置参数
    # patch_size = 64
    # k = 500
    # channels = 3

    # # 获取输入文件夹中的第一张图片
    # image_file = os.listdir(input_dir)[0]
    # image_path = os.path.join(input_dir, image_file)
    
    # # 读取并预处理图片
    # image = Image.open(image_path)
    # transform = transforms.ToTensor()  # 添加这行
    # x = transform(image)  # 将PIL图像转换为张量
    # x = x.unsqueeze(0)  # 添加batch维度
    # print(x.shape)
    
    # # 初始化 PatchAttentionMask
    # pam = PatchAttentionMask(patch_size=patch_size, k=k, channels=channels)

    # # 前向传播得到mask
    # mask = pam(x)
    
    # # 将mask应用到原图
    # masked_image = x * mask

    # print(masked_image.shape)
    
    # # 转换为PIL图像并保存
    # to_pil = transforms.ToPILImage()
    # masked_image = to_pil(masked_image.squeeze(0))
    
    # # 保存结果
    # output_path = os.path.join(output_dir, f"masked_{image_file}")
    # masked_image.save(output_path)
    
    # print(f"已处理图像: {image_file}")
    # print(f"掩码后的图像已保存至: {output_path}")

    batch_size = 4
    channels = 128
    height, width = 42, 25
    patch_size = 3
    k = 2

    # 创建随机输入
    x = torch.randn(batch_size, channels, height, width)
    y = torch.randn(batch_size, channels, height, width)

    # 初始化 PatchAttentionMask
    pam = PatchAttentionMask(patch_size=patch_size, k=k, channels=channels, stride=1)

    # 前向传播
    mask = pam(x)

    print("输入特征图形状:", x.shape)
    print("生成的掩码形状:", mask.shape)
    # print("掩码:", mask)

    # msm = muti_shift_window_mask(patch_sizes=[2, 4, 6], k=[100, 100, 100], channels_list=[128, 128, 128], strides=[1, 1, 1])
    # masks = msm(x)
    # print("生成的掩码形状:", masks[0].shape)

    swl = ShiftWindowRevLoss(patch_sizes=[3,5,7], k=[50,50,50], channels_list=[128,128,128], strides=[1,1,1], alpha=0.7, beta=0.3)
    loss = swl(x, y)
    print("计算的loss:", loss)






