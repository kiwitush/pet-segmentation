"""
加载训练好的模型，在验证集上评估所有指标。
"""
import argparse
import os
import sys

import torch
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from dataset import PetSegDataset
from unet import UNet

CLASS_NAMES = ["Pet", "Background", "Boundary"]


@torch.no_grad()
def evaluate(model, val_loader, device, n_classes=3, ignore_index=255):
    """
    在验证集上全局累积计算各项指标，避免 batch 平均带来的偏差。
    """
    model.eval()
    intersections = torch.zeros(n_classes, device=device)
    unions = torch.zeros(n_classes, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = torch.tensor(0, device=device, dtype=torch.long)
    confusion = torch.zeros(n_classes, n_classes, device=device, dtype=torch.long)

    for images, targets in val_loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        valid_mask = targets != ignore_index
        correct += (preds[valid_mask] == targets[valid_mask]).sum()
        total += valid_mask.sum()

        for c in range(n_classes):
            pred_c = (preds == c) & valid_mask
            target_c = (targets == c) & valid_mask
            intersections[c] += (pred_c & target_c).sum()      # TP
            unions[c] += (pred_c | target_c).sum()              # TP + FP + FN
            for k in range(n_classes):
                confusion[c, k] += ((preds == c) & (targets == k) & valid_mask).sum()

    intersections = intersections.cpu()
    unions = unions.cpu()
    confusion = confusion.cpu()

    # 逐类 IoU 和 Dice（Dice = 2*TP / (2*TP + FP + FN)）
    ious = []
    dices = []
    for c in range(n_classes):
        iou = (intersections[c] / unions[c]).item() if unions[c] > 0 else float("nan")
        dice = (2 * intersections[c] / (unions[c] + intersections[c])).item() if unions[c] + intersections[c] > 0 else float("nan")
        ious.append(iou)
        dices.append(dice)

    # mIoU 仅对有效类别取平均
    valid_ious = [v for v in ious if v == v]  # NaN is not equal to itself
    miou = sum(valid_ious) / len(valid_ious) if valid_ious else 0.0
    acc = (correct / total).item()

    return {
        "miou": miou,
        "pixel_acc": acc,
        "per_class_iou": ious,
        "per_class_dice": dices,
        "confusion": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description="验证集评估")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=os.path.join(_PROJECT_ROOT, "data"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    model = UNet(n_channels=3, n_classes=3)
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model] {args.checkpoint}")

    data = PetSegDataset(root=args.data_root, batch_size=1, num_workers=0)
    val_loader = data.get_val_loader()
    print(f"[Data] Val samples: {len(data.val_set)}")

    result = evaluate(model, val_loader, device)

    print(f"\n{'='*45}")
    print(f"  Pixel Accuracy : {result['pixel_acc']:.4f}")
    print(f"  mIoU           : {result['miou']:.4f}")
    print(f"  Mean Dice      : {np.nanmean(result['per_class_dice']):.4f}")
    print(f"{'='*45}")
    print(f"{'Class':<14} {'IoU':>8}  {'Dice':>8}")
    print(f"{'-'*34}")
    for i, name in enumerate(CLASS_NAMES):
        iou_str = f"{result['per_class_iou'][i]:.4f}" if result['per_class_iou'][i] == result['per_class_iou'][i] else "   N/A"
        dice_str = f"{result['per_class_dice'][i]:.4f}" if result['per_class_dice'][i] == result['per_class_dice'][i] else "   N/A"
        print(f"  {name:<12} {iou_str:>8}  {dice_str:>8}")

    print(f"\n{'='*45}")
    print("Confusion Matrix (rows=Pred, cols=GT):")
    print(f"{'':>10} ", end="")
    for name in CLASS_NAMES:
        print(f"{name:>10}", end="  ")
    print()
    cm = result["confusion"].numpy()
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:>10} ", end="")
        for j in range(len(CLASS_NAMES)):
            print(f"{cm[i, j]:>10}", end="  ")
        print()


if __name__ == "__main__":
    main()
