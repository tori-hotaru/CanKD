import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from mmrazor.registry import MODELS


def _validate_patch_grid(patch_grid: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """Normalize the patch grid definition."""
    if isinstance(patch_grid, int):
        sqrt_val = int(math.sqrt(patch_grid))
        if sqrt_val * sqrt_val != patch_grid:
            raise ValueError(
                'When `patch_grid` is an int, it must be a perfect square. '
                f'Got {patch_grid}.')
        return sqrt_val, sqrt_val

    if len(patch_grid) != 2:
        raise ValueError('`patch_grid` must contain two integers when a '
                         f'sequence is provided, but got {patch_grid}.')
    return int(patch_grid[0]), int(patch_grid[1])


def _split_feature_to_patches(feature: torch.Tensor,
                              patch_grid: Tuple[int, int]) -> torch.Tensor:
    """Split a 4D feature map into a grid of patches.

    Args:
        feature (Tensor): Feature map of shape (N, C, H, W).
        patch_grid (Tuple[int, int]): Number of patches along height and width.

    Returns:
        Tensor: Patch tensor of shape (N, P, C, Kh, Kw) where
            P = patch_grid[0] * patch_grid[1].
    """
    if feature.dim() != 4:
        raise ValueError('Expected a 4D tensor to split into patches, '
                         f'but got {feature.dim()}D tensor.')
    n, c, h, w = feature.shape
    gh, gw = patch_grid
    if h % gh != 0 or w % gw != 0:
        raise ValueError(
            'Feature spatial size must be divisible by the patch grid. '
            f'Got feature size {(h, w)} and grid {(gh, gw)}.')

    ph, pw = h // gh, w // gw
    patches = feature.view(n, c, gh, ph, gw, pw)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(n, gh * gw, c, ph, pw)
    return patches


def _expand_as_patches(reference: torch.Tensor,
                       value: Optional[torch.Tensor],
                       device: torch.device) -> torch.Tensor:
    """Expand scalar/batch tensors to match patch shape."""
    if value is None:
        return torch.zeros_like(reference, device=device)

    if value.dim() == 0:
        expanded = value.view(1, 1).to(device)
        expanded = expanded.expand(reference.size(0), reference.size(1))
    elif value.dim() == 1:
        expanded = value.to(device).view(-1, 1)
        expanded = expanded.expand(-1, reference.size(1))
    else:
        expanded = value.to(device)
        if expanded.shape != reference.shape:
            raise ValueError('Unable to broadcast value with shape '
                             f'{expanded.shape} to reference '
                             f'{reference.shape}.')
    return expanded


@MODELS.register_module()
class PatchWeightPolicy(nn.Module):
    """A lightweight policy network that computes per-patch weights."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: Sequence[int] = (64, 32),
                 activation: str = 'relu',
                 normalize: str = 'softmax',
                 temperature: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError('`input_dim` must be positive.')
        self.normalize = normalize.lower()
        self.temperature = temperature
        self.eps = eps

        layers: Sequence[nn.Module] = []
        last_dim = input_dim
        act_layer: nn.Module
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            if activation.lower() == 'relu':
                act_layer = nn.ReLU(inplace=True)
            elif activation.lower() == 'gelu':
                act_layer = nn.GELU()
            else:
                raise ValueError(f'Unsupported activation: {activation}.')
            layers.append(act_layer)
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self,
                patch_diffs: torch.Tensor,
                teacher_grad: Optional[torch.Tensor] = None,
                student_grad: Optional[torch.Tensor] = None,
                teacher_task_loss: Optional[torch.Tensor] = None,
                student_task_loss: Optional[torch.Tensor] = None) -> torch.Tensor:
        if patch_diffs.dim() != 2:
            raise ValueError('`patch_diffs` must be 2D with shape (N, P). '
                             f'Got shape {patch_diffs.shape}.')

        device = patch_diffs.device
        feats = [patch_diffs.unsqueeze(-1)]

        teacher_grad = _expand_as_patches(patch_diffs, teacher_grad, device)
        student_grad = _expand_as_patches(patch_diffs, student_grad, device)
        feats.append(teacher_grad.unsqueeze(-1))
        feats.append(student_grad.unsqueeze(-1))

        teacher_task_loss = _expand_as_patches(patch_diffs, teacher_task_loss,
                                               device)
        student_task_loss = _expand_as_patches(patch_diffs, student_task_loss,
                                               device)
        feats.append(teacher_task_loss.unsqueeze(-1))
        feats.append(student_task_loss.unsqueeze(-1))

        policy_input = torch.cat(feats, dim=-1)
        logits = self.mlp(policy_input).squeeze(-1)

        if self.normalize == 'softmax':
            weights = torch.softmax(logits / self.temperature, dim=-1)
        elif self.normalize == 'sigmoid':
            weights = torch.sigmoid(logits / self.temperature)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + self.eps)
        elif self.normalize == 'none':
            weights = logits
        else:
            raise ValueError(f'Unsupported normalization: {self.normalize}.')
        return weights


@MODELS.register_module()
class PatchWiseDistillationLoss(nn.Module):
    """Patch-level distillation loss with a dynamic weighting policy."""

    def __init__(self,
                 patch_grid: Union[int, Sequence[int]],
                 policy: Optional[Dict] = None,
                 loss_weight: float = 1.0,
                 reduction: str = 'mean',
                 interpolate_mode: str = 'bilinear',
                 align_corners: Optional[bool] = False,
                 detach_teacher: bool = False,
                 store_gradients: bool = True) -> None:
        super().__init__()
        self.patch_grid = _validate_patch_grid(patch_grid)
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.interpolate_mode = interpolate_mode
        self.align_corners = align_corners
        self.detach_teacher = detach_teacher
        self.store_gradients = store_gradients

        self.policy = MODELS.build(policy) if policy is not None else None
        self._cached_teacher_grad: Optional[torch.Tensor] = None
        self._cached_student_grad: Optional[torch.Tensor] = None

    def _maybe_interpolate(self, student_feature: torch.Tensor,
                           teacher_feature: torch.Tensor) -> torch.Tensor:
        if student_feature.shape[-2:] == teacher_feature.shape[-2:]:
            return student_feature
        return F.interpolate(
            student_feature,
            size=teacher_feature.shape[-2:],
            mode=self.interpolate_mode,
            align_corners=self.align_corners)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction in ('none', None):
            return loss
        raise ValueError(f'Unsupported reduction: {self.reduction}.')

    def _prepare_gradients(self, reference: torch.Tensor,
                           external_grad: Optional[torch.Tensor],
                           cached_grad: Optional[torch.Tensor]) -> torch.Tensor:
        if external_grad is not None:
            target = external_grad
        else:
            target = cached_grad

        if target is None:
            return torch.zeros_like(reference.mean(dim=(2, 3, 4)))

        if target.shape != reference.shape:
            raise ValueError('Gradient tensor shape mismatch. Expected '
                             f'{reference.shape}, got {target.shape}.')
        grad_mag = target.pow(2).mean(dim=(2, 3, 4)).sqrt()
        return grad_mag

    def _register_grad_hooks(self, teacher_patches: torch.Tensor,
                             student_patches: torch.Tensor) -> None:
        if not self.store_gradients:
            return

        if teacher_patches.requires_grad:
            teacher_patches.retain_grad()

            def _save_teacher_grad(grad: torch.Tensor) -> None:
                self._cached_teacher_grad = grad.detach()

            teacher_patches.register_hook(_save_teacher_grad)
        else:
            self._cached_teacher_grad = None

        if student_patches.requires_grad:
            student_patches.retain_grad()

            def _save_student_grad(grad: torch.Tensor) -> None:
                self._cached_student_grad = grad.detach()

            student_patches.register_hook(_save_student_grad)
        else:
            self._cached_student_grad = None

    def forward(self,
                student_feature: torch.Tensor,
                teacher_feature: torch.Tensor,
                teacher_task_loss: Optional[torch.Tensor] = None,
                student_task_loss: Optional[torch.Tensor] = None,
                teacher_patch_grad: Optional[torch.Tensor] = None,
                student_patch_grad: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.detach_teacher:
            teacher_feature = teacher_feature.detach()

        student_feature = self._maybe_interpolate(student_feature,
                                                  teacher_feature)

        teacher_patches = _split_feature_to_patches(teacher_feature,
                                                    self.patch_grid)
        student_patches = _split_feature_to_patches(student_feature,
                                                    self.patch_grid)

        self._register_grad_hooks(teacher_patches, student_patches)

        patch_diffs = (student_patches - teacher_patches).pow(2)
        patch_diffs = patch_diffs.mean(dim=(2, 3, 4))

        teacher_grad_feat = self._prepare_gradients(teacher_patches,
                                                    teacher_patch_grad,
                                                    self._cached_teacher_grad)
        student_grad_feat = self._prepare_gradients(student_patches,
                                                    student_patch_grad,
                                                    self._cached_student_grad)

        if self.policy is not None:
            weights = self.policy(
                patch_diffs=patch_diffs,
                teacher_grad=teacher_grad_feat,
                student_grad=student_grad_feat,
                teacher_task_loss=teacher_task_loss,
                student_task_loss=student_task_loss)

            if weights.shape != patch_diffs.shape:
                raise ValueError('Policy must return weights with shape '
                                 f'{patch_diffs.shape}, but got '
                                 f'{weights.shape}.')
        else:
            weights = torch.ones_like(patch_diffs) / patch_diffs.size(1)

        weighted_loss = (weights * patch_diffs).sum(dim=-1)
        loss = self._reduce(weighted_loss) * self.loss_weight
        return loss
