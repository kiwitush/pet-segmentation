"""
手写 Dice Loss、 Cross-Entropy Loss和组合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss = 1 - 2 * |X ∩ Y| / (|X| + |Y|)

    对多分类问题逐类计算 Dice 后取平均
    """

    def __init__(self, smooth: float = 1e-6, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        # 构建 one-hot 编码的 mask，忽略 ignore_index 像素
        mask = (target != self.ignore_index).unsqueeze(1).float()  # (N, 1, H, W)
        target_one_hot = F.one_hot(
            target.clamp(min=0), num_classes=n_classes
        ).permute(0, 3, 1, 2).float()   # (N, C, H, W)
        target_one_hot = target_one_hot * mask

        # 逐类计算 Dice 系数
        dice_per_class = []
        for c in range(n_classes):
            pred_c = probs[:, c:c+1, :, :] * mask      
            tgt_c = target_one_hot[:, c:c+1, :, :]     
            
            intersection = (pred_c * tgt_c).sum()
            union = pred_c.sum() + tgt_c.sum()

            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(dice)

        dice_mean = torch.stack(dice_per_class).mean()
        return 1.0 - dice_mean


class CombinedLoss(nn.Module):
    """
    Cross-Entropy Loss + Dice Loss 组合损失

    Loss = alpha * CE_Loss + (1 - alpha) * Dice_Loss
    """

    def __init__(
        self,
        alpha: float = 0.5,
        smooth: float = 1e-6,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(smooth=smooth, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ce(logits, target)
        loss_dice = self.dice(logits, target)
        return self.alpha * loss_ce + (1.0 - self.alpha) * loss_dice
