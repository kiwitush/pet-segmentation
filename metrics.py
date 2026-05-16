"""
辅助函数：训练完一个 epoch 后，在验证集上算模型的 mIoU 分数
"""

import torch


@torch.no_grad()
def evaluate_model(model, val_loader, device, n_classes=3, ignore_index=255):
    """
    在验证集上评估模型，累加各类 intersection/union，返回 mIoU 和逐类 IoU
    """
    model.eval()
    # 初始化每个类别累加: intersection 和 union
    intersections = torch.zeros(n_classes, device=device)
    unions = torch.zeros(n_classes, device=device)

    # 遍历验证集
    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)  # 每个像素取概率最大的类别

        valid_mask = (targets != ignore_index)
        
        # 逐类累加 intersection 和 union
        for c in range(n_classes):
            pred_c = (preds == c) & valid_mask
            target_c = (targets == c) & valid_mask
            intersections[c] += (pred_c & target_c).sum()  
            unions[c] += (pred_c | target_c).sum()

    intersections = intersections.cpu()
    unions = unions.cpu()

    ious = []
    for c in range(n_classes):
        if unions[c] == 0:
            ious.append(float("nan"))
        else:
            ious.append((intersections[c] / unions[c]).item())   # 全局交集/全局并集

    # 平均有效类别的 IoU 作为 mIoU
    valid_ious = [v for v in ious if v == v] 
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0  

    return {"per_class_iou": ious, "miou": miou}
