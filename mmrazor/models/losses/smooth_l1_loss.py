from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class SmoothL1Loss(nn.Module):
    """Calculate the smooth L1 loss between the two inputs.

    Args:
        loss_weight (float): Weight of loss. Defaults to 1.0.
        beta (float, optional): Specifies the threshold at which to change
            between L1 and L2 loss. Default: 1.0.
        size_average (bool, optional): Deprecated (see :attr:`reduction`).
            By default, the losses are averaged over each loss element in the
            batch. Note that for some losses, there are multiple elements per
            sample. If the field :attr:`size_average` is set to ``False``,
            the losses are instead summed for each minibatch. Ignored when
            reduce is ``False``. Defaults to True.
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By
            default, the losses are averaged or summed over observations for
            each minibatch depending on :attr:`size_average`. When
            :attr:`reduce` is ``False``, returns a loss per batch element
            instead and ignores :attr:`size_average`. Defaults to True.
        reduction (string, optional): Specifies the reduction to apply to
            the output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no
            reduction will be applied, ``'mean'``: the sum of the output will
            be divided by the number of elements in the output, ``'sum'``: the
            output will be summed. Note: :attr:`size_average` and
            :attr:`reduce` are in the process of being deprecated, and in the
            meantime, specifying either of those two args will override
            :attr:`reduction`. Defaults to 'mean'.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        beta: float = 1.0,
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.beta = beta
        self.size_average = size_average
        self.reduce = reduce

        accept_reduction = {'none', 'sum', 'mean'}
        assert reduction in accept_reduction, (
            f'SmoothL1Loss supports reduction {accept_reduction}, '
            f'but got {reduction}.')
        self.reduction = reduction

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Forward computation.

        Args:
            s_feature (torch.Tensor): The student model feature with shape
                (N, C, H, W) or shape (N, C).
            t_feature (torch.Tensor): The teacher model feature with shape
                (N, C, H, W) or shape (N, C).
        """
        # Check if s_feature and t_feature have different sizes
        if s_feature.shape != t_feature.shape:
            # Transform s_feature to match t_feature size
            if s_feature.dim() == t_feature.dim():
                if s_feature.dim() == 4:
                    # Both are (N, C, H, W)
                    N_s, C_s, H_s, W_s = s_feature.shape
                    N_t, C_t, H_t, W_t = t_feature.shape
                    # Use 1x1 convolution to match channel dimension
                    if C_s != C_t:
                        conv1x1 = nn.Conv2d(C_s, C_t, kernel_size=1).to(s_feature.device)
                        s_feature = conv1x1(s_feature)
                    # Resize spatial dimensions
                    if (H_s != H_t) or (W_s != W_t):
                        s_feature = F.interpolate(
                            s_feature,
                            size=(H_t, W_t),
                            mode='bilinear',
                            align_corners=False)
                elif s_feature.dim() == 2:
                    # Both are (N, C)
                    N_s, C_s = s_feature.shape
                    N_t, C_t = t_feature.shape
                    if C_s != C_t:
                        linear = nn.Linear(C_s, C_t).to(s_feature.device)
                        s_feature = linear(s_feature)
                else:
                    raise ValueError('Unsupported tensor dimensions.')
            else:
                raise ValueError('s_feature and t_feature must have the same number of dimensions.')

        # Compute the smooth L1 loss
        loss = F.smooth_l1_loss(
            s_feature,
            t_feature,
            size_average=self.size_average,
            reduce=self.reduce,
            reduction=self.reduction,
            beta=self.beta)
        return self.loss_weight * loss
