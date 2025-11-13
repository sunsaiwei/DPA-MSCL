from __future__ import print_function



from __future__ import print_function
import torch
import torch.nn as nn

class BalSCL(nn.Module):
    def __init__(self, cls_num_list=None, temperature=0.1):
        super(BalSCL, self).__init__()
        self.temperature = temperature
        self.cls_num_list = cls_num_list

    def forward(self, features, targets):  # 移除了centers1参数
        device = features.device
        batch_size = features.shape[0]

        # 仅处理原始目标
        targets = targets.contiguous().view(-1, 1).repeat(2, 1)  # [2B, 1]

        # 类别计数（基于双增强样本）
        batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[targets].sum(dim=0).squeeze()

        # Mask生成（纯样本级）
        mask = torch.eq(targets, targets.T).float()  # [2B, 2B]
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2*batch_size, device=device).view(-1,1),
            0
        )
        mask = mask * logits_mask  # 排除自对比

        # 特征双增强展开
        features = torch.cat(features.chunk(2, dim=1), dim=0)  # [2B, D]

        features = features.view(features.size(0), -1)  # 结果是 (64, 128)

        # 纯样本级对比
        logits = features.mm(features.T) / self.temperature  # [2B, 2B]

        # 数值稳定计算
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 平衡权重计算
        exp_logits = torch.exp(logits) * logits_mask
        class_weights = batch_cls_count[targets.squeeze()].view(1, -1)
        exp_logits = exp_logits / class_weights  # 类平衡加权

        # 损失计算
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        loss = -(mask * log_prob).sum() / mask.sum()

        return loss

