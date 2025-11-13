from __future__ import print_function

import torch
import torch.nn as nn


class SCLLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SCLLoss, self).__init__()
        self.temperature = temperature

    def forward(self, centers1, features, targets):
        device = features.device
        batch_size = features.shape[0]

        # 拼接原始样本和类中心
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        features = torch.cat([features, centers1], dim=0)

        # 构造对比目标标签
        targets = targets.contiguous().view(-1, 1)
        targets_centers = torch.arange(len(centers1), device=device).view(-1, 1)
        extended_targets = torch.cat([targets.repeat(2, 1), targets_centers], dim=0)

        # 计算相似度矩阵
        logits = torch.div(
            features[:2*batch_size].mm(features.T),
            self.temperature
        )

        # 构建正样本掩码
        mask = torch.eq(extended_targets[:2*batch_size], extended_targets.T).float()
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(2*batch_size).view(-1,1).to(device),
            0
        )
        mask *= logits_mask

        # 数值稳定性处理
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # 计算对比损失
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob.mean()

        return loss
