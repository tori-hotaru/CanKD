import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS
from mmengine.runner.checkpoint import _load_checkpoint


class TraceLoss(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 loss_weight=1.0):
        super(TraceLoss, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.loss_weight = loss_weight
        self.conv_alpha = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_beta = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, f_s, f_t):
        alpha = self.conv_alpha(f_s)
        beta = self.conv_beta(f_t)
        trace = torch.trace(alpha @ beta)
        normalized_trace = trace / (torch.norm(alpha) * torch.norm(beta))
        loss = -self.loss_weight * normalized_trace
        return loss

