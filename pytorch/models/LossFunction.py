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
        self.weight = weight.type(torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.type(torch.float32)
        targets = targets.type(torch.long)
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
        targets = targets.type(torch.long)
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
        # print("y_true\n", y_true)

        # 确保 y_pred 的值在 (0, 1) 之间，避免 log(0) 的情况
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        # print("y_pred\n", y_pred)

        # 计算交叉熵损失
        loss_matrix = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        # print("loss_matrix\n", loss_matrix)

        # 忽略y_true中为-1的部分
        mask = (y_true != self.mask).float()
        # print("mask\n", mask)
        loss_matrix = loss_matrix * mask
        # print("loss_matrix\n", loss_matrix)

        # 应用权重
        weighted_loss_matrix = loss_matrix * self.weight
        # print("self.weight\n", self.weight)
        # print("weighted_loss_matrix\n", weighted_loss_matrix)

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
    def __init__(self, temperature=0.5, ignore_labels=None):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.ignore_labels = set(ignore_labels) if ignore_labels else set()

    def forward(self, features, labels):
        device = features.device
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of features and labels must be equal."
                f" features.shape = {features.shape}, labels.shape = {labels.shape}")
        # print("features", features.shape)
        # print("labels", labels.shape)
        # 归一化
        features = nn.functional.normalize(features, dim=1)
        # 计算余弦相似度
        similarities = self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        # print("similarities", similarities.shape)

        # Mask自身比较
        batch_size = features.shape[0]
        mask = torch.eye(batch_size).bool().to(device)
        similarities.masked_fill_(mask, float('-inf'))
        # print("similarities2", similarities.shape)

        # 创建标签矩阵
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.float().to(device)
        # print("labels_matrix", labels_matrix.shape)

        # 忽略特定标签的相同距离计算
        for label in self.ignore_labels:
            label_mask = (labels == label).unsqueeze(1)
            labels_matrix.masked_fill_(label_mask & label_mask.transpose(0, 1), 0)

        # 应用softmax
        epsilon = 1e-15
        exp_similarities = torch.exp(similarities).to(device) + epsilon
        # print("exp_similarities", exp_similarities)
        sum_exp_similarities = torch.sum(exp_similarities * labels_matrix, dim=1) + epsilon
        # print("sum_exp_similarities", sum_exp_similarities)

        # 计算损失
        loss = -torch.log(sum_exp_similarities / (torch.sum(exp_similarities, dim=1)))
        return loss.mean()


class InfoNCELossV2(nn.Module):
    def __init__(self, temperature=0.5, ignore_labels=None):
        super(InfoNCELossV2, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.ignore_labels = set(ignore_labels) if ignore_labels else set()

    def forward(self, features, labels):
        device = features.device
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of features and labels must be equal."
                f" features.shape = {features.shape}, labels.shape = {labels.shape}")
        # print("features", features.shape)
        # print("labels", labels.shape)
        # 归一化
        features = nn.functional.normalize(features, dim=1)
        # 计算余弦相似度
        similarities = (self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0)) + 1.0) / 2.0
        epsilon = 1e-7
        similarities = torch.clamp(similarities, epsilon, 1 - epsilon).to(device)
        # print("similarities\n", similarities)
        # 创建标签矩阵
        labels_matrix = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_matrix = labels_matrix.int().to(device)

        # 计算交叉熵损失
        cross_loss_matrix = - (
                labels_matrix * torch.log(similarities) + (1 - labels_matrix) * torch.log(1 - similarities))

        # print("cross_loss_matrix\n", cross_loss_matrix)

        # 创建掩码矩阵，忽略对角线自身的比较
        mask = torch.ones_like(labels_matrix, dtype=torch.float)
        torch.Tensor.fill_diagonal_(mask, 0)

        # 更新掩码以忽略特定标签
        for label in self.ignore_labels:
            label_positions = (labels == label)
            # 创建一个外积掩码，忽略特定标签的交叉点
            label_mask = label_positions[:, None] & label_positions[None, :]
            mask[label_mask] = 0

        # print("mask\n", mask)

        # 应用掩码
        masked_loss = cross_loss_matrix * mask

        # 计算损失
        loss = torch.sum(masked_loss) / (torch.sum(mask) + epsilon)
        return loss


class InfoNCELossV3(nn.Module):
    def __init__(self, temperature=1):
        super(InfoNCELossV3, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, features, labels):
        device = features.device
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"The number of features and labels must be equal."
                f" features.shape = {features.shape}, labels.shape = {labels.shape}")

        # 计算余弦相似度
        similarity_matrix = self.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0)) / self.temperature
        # print("similarity_matrix\n", similarity_matrix)

        # 获取标签相同的掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 对角线元素设置为0，忽略自身比较
        mask.fill_diagonal_(0)
        # print("mask\n", mask)

        # 对相似度矩阵的每一行应用softmax，然后乘以标签掩码
        exp_similarities = torch.exp(similarity_matrix) * mask
        # print("exp_similarities\n", exp_similarities)
        sum_exp_similarities = torch.sum(exp_similarities, dim=1, keepdim=True)
        # print("sum_exp_similarities\n", sum_exp_similarities)

        # 计算Log-Softmax
        log_prob = similarity_matrix - torch.log(sum_exp_similarities + 1e-15)
        # print("log_prob\n", log_prob)

        # 仅考虑正样本（相同标签）的对数概率
        positive_log_prob = (mask * log_prob).sum(1)
        # print("positive_log_prob\n", positive_log_prob)

        # 计算损失
        loss = -positive_log_prob.mean()

        return loss


class CosineEmbeddingLossModule(nn.Module):
    def __init__(self, negative_positive_ratio=1.0):
        super(CosineEmbeddingLossModule, self).__init__()
        self.negative_positive_ratio = negative_positive_ratio
        self.loss_fn = nn.CosineEmbeddingLoss(margin=0.5)

    def forward(self, features, labels):
        # 归一化特征向量
        normalized_features = nn.functional.normalize(features, p=2, dim=1)
        features1, features2, targets = self.create_pairs(normalized_features, labels)
        features1, features2, targets = self.balance_pairs(features1, features2, targets)
        loss = self.loss_fn(features1, features2, targets)
        return loss

    def create_pairs(self, features, labels):
        n = labels.size(0)
        indices = torch.combinations(torch.arange(n), r=2)
        features1 = features[indices[:, 0]]
        features2 = features[indices[:, 1]]
        labels1 = labels[indices[:, 0]]
        labels2 = labels[indices[:, 1]]
        targets = (labels1 == labels2).long() * 2 - 1  # True/False to 1/-1
        return features1, features2, targets

    def balance_pairs(self, features1, features2, targets):
        positive_indices = targets == 1
        negative_indices = targets == -1

        num_positives = positive_indices.sum()
        num_negatives = negative_indices.sum()

        # 检查是否有足够的正负样本对
        if num_positives == 0 or num_negatives == 0:
            print("Warning: Not enough positive or negative pairs.")
            # 可以选择返回一个特定的损失值，例如0或一个很大的数
            return features1[:0], features2[:0], targets[:0]  # 返回空张量以避免计算损失

        # 限制选择的样本对数量不超过存在的样本对数量
        max_positive_pairs = num_positives
        max_negative_pairs = int(min(num_negatives, self.negative_positive_ratio * max_positive_pairs))

        # 随机选择正负样本对
        positives = torch.randperm(num_positives)[:max_positive_pairs]
        negatives = torch.randperm(num_negatives)[:max_negative_pairs]

        # 获取平衡后的索引
        balanced_positive_indices = positive_indices.nonzero().squeeze(1)[positives]
        balanced_negative_indices = negative_indices.nonzero().squeeze(1)[negatives]
        balanced_indices = torch.cat((balanced_positive_indices, balanced_negative_indices), dim=0)

        return features1[balanced_indices], features2[balanced_indices], targets[balanced_indices]


if __name__ == '__main__':
    features = torch.randn(10, 128)  # 10个样本，每个样本128维
    labels = torch.tensor([1, 1, 2, 2, 1, 3, 3, 3, 2, 1])

    # 创建模型实例
    model = CosineEmbeddingLossModule(negative_positive_ratio=20)

    # 计算损失
    loss = model(features, labels)
    print("Loss:", loss.item())
