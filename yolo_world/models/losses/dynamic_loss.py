# Copyright (c) Tencent Inc. All rights reserved.
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.models.losses.mse_loss import mse_loss
from mmyolo.registry import MODELS

@MODELS.register_module()
class CoVMSELossWithMargin(nn.Module):
    def __init__(self, lambda_pos=1.0, lambda_neg=1.0, margin=0.5, reduction='mean'):
        super(CoVMSELossWithMargin, self).__init__()
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.margin = margin
        self.reduction = reduction

    def forward(self, pred, target, is_positive):
        mean_pred = torch.mean(pred, dim=0)
        std_pred = torch.std(pred, dim=0)

        cov = std_pred / (mean_pred + 1e-6)

        cov_pos = cov[is_positive]
        cov_neg = cov[~is_positive]

        loss_pos = F.mse_loss(cov_pos, torch.zeros_like(cov_pos), reduction=self.reduction)

        loss_neg = F.mse_loss(cov_neg, torch.full_like(cov_neg, self.margin), reduction=self.reduction)

        loss_total = self.lambda_pos * loss_pos + self.lambda_neg * loss_neg

        return loss_total

@MODELS.register_module()
class CoVMSELoss(nn.Module):

    def __init__(self,
                 dim: int = 0,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self,
                pred: Tensor,
                weight: Optional[Tensor] = None,
                avg_factor: Optional[int] = None,
                reduction_override: Optional[str] = None) -> Tensor:
        """Forward function of loss."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        cov = pred.std(self.dim) / pred.mean(self.dim).clamp(min=self.eps)
        target = torch.zeros_like(cov)
        loss = self.loss_weight * mse_loss(
            cov, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
