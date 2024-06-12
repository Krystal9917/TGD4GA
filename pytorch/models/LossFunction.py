# 定义Focal Loss
import torch
from torch import nn
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits, binary_cross_entropy


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = cross_entropy(inputs, targets, reduction='none')  # 交叉熵
        pt = torch.exp(-ce_loss)  # p_t
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Focal Loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


# 定义带权重参数的Focal Loss
class WeightedFocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2.0, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        assert weight is not None, "weight parameter is required for WeightedFocalLoss"
        self.alpha = alpha
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class WeightedFocalBCELoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2.0, reduction='mean'):
        super(WeightedFocalBCELoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Assume inputs are the logits
        inputs = inputs.type(torch.float32)
        targets = targets.type(torch.float32)
        # bce_loss = binary_cross_entropy_with_logits(inputs, targets, pos_weight=self.weight, reduction='none')
        bce_loss = binary_cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-bce_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
