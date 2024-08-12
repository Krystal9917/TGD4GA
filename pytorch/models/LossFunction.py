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


class WeightedFocalBalanceBCELoss(nn.Module):
    def __init__(self, weight=None, alpha=0.25, gamma=2.0, reduction='mean', max_zero_ratio=1.0, mask=-1):
        super(WeightedFocalBalanceBCELoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.mask = mask
        self.max_zero_ratio = max_zero_ratio

    def forward(self, inputs, targets):
        # Assume inputs are the logits
        inputs = inputs.type(torch.float32)
        targets = targets.type(torch.float32)
        bce_loss = self.weighted_balance_cross_entropy_loss(inputs, targets)
        pt = torch.exp(-bce_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

    def weighted_balance_cross_entropy_loss(self, y_pred, y_true):
        # 对y_true进行随机mask
        y_true = self.mask_target_to_balance_sample(y_true)

        # 确保 y_pred 的值在 (0, 1) 之间，避免 log(0) 的情况
        epsilon = 1e-15
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)

        # 计算交叉熵损失
        loss_matrix = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))

        # 忽略y_true中为-1的部分
        mask = (y_true != self.mask).float()
        loss_matrix = loss_matrix * mask

        # 应用权重
        weighted_loss_matrix = loss_matrix * self.weight

        # 计算每行样本的损失
        sample_loss_row = torch.sum(weighted_loss_matrix, dim=1)
        # 计算每行样本的权重
        sample_loss_row_weight = torch.sum(mask * self.weight, dim=1)
        # 防止除数为0的情况
        sample_loss_row_weight[sample_loss_row_weight == 0] = epsilon

        # 计算每个样本的平均损失
        sample_loss = sample_loss_row / sample_loss_row_weight

        # 计算所有样本的平均损失
        mean_loss = torch.mean(sample_loss)

        return mean_loss

    def mask_target_to_balance_sample(self, target):
        n, m = target.shape
        masked_target = target.clone()

        for col in range(m):
            col_data = masked_target[:, col]
            num_zeros = (col_data == 0).sum().item()
            num_ones = (col_data == 1).sum().item()

            if num_zeros > self.max_zero_ratio * num_ones:
                # 需要掩码的0的数量
                num_to_mask = int(num_zeros - self.max_zero_ratio * num_ones)

                # 获取所有0的索引
                zero_indices = (col_data == 0).nonzero(as_tuple=True)[0]

                # 随机选择需要掩码的0的索引
                mask_indices = torch.randperm(num_zeros)[:num_to_mask]
                selected_indices = zero_indices[mask_indices]

                # 将选中的0替换为-1
                masked_target[selected_indices, col] = self.mask

        return masked_target


class InfoNCELoss(nn.Module):
    def __init__(self, device, temperature=0.5, ignore_labels=None):
        super(InfoNCELoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.ignore_labels = set(ignore_labels) if ignore_labels else set()

    def forward(self, features, labels):
        # 计算余弦相似度
        similarities = self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature

        # Mask自身比较
        batch_size = features.shape[0]
        mask = torch.eye(batch_size).bool().to(self.device)
        similarities.masked_fill_(mask, float('-inf'))

        # 创建标签矩阵
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.float().to(self.device)

        # 忽略特定标签的相同距离计算
        for label in self.ignore_labels:
            label_mask = (labels == label).unsqueeze(1)
            labels_matrix.masked_fill_(label_mask & label_mask.transpose(0, 1), 0)

        # 应用softmax
        exp_similarities = torch.exp(similarities).to(self.device)
        sum_exp_similarities = torch.sum(exp_similarities * labels_matrix, dim=1)

        # 计算损失
        loss = -torch.log(sum_exp_similarities / torch.sum(exp_similarities, dim=1))
        return loss.mean()


if __name__ == '__main__':
    y_true = torch.tensor([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0],
    ], dtype=torch.float32)
    y_pred = torch.tensor([
        [0.1, 0.8],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.4, 0.6]
    ], dtype=torch.float32)
    sample_weights = torch.tensor([1, 200], dtype=torch.float32)
    loss = WeightedFocalBalanceBCELoss(weight=sample_weights, alpha=0.25, gamma=2.0, reduction='mean',
                                       max_zero_ratio=1.0, mask=-1)
    print(loss(y_pred, y_true))
