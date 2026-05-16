"""
加载训练好的模型，对验证集随机几张图片展示三分类分割结果。
"""
import argparse
import os
import sys
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from dataset import PetSegDataset
from unet import UNet

# 三类配色: 前景、背景、边界
CLASS_COLORS = [
    [0.95, 0.80, 0.30],   # 黄色
    [0.87, 0.72, 0.53],   # 米色
    [0.88, 0.35, 0.35],   # 红色
]

CLASS_COLORS_255 = [[int(c * 255) for c in color] for color in CLASS_COLORS]


def label_to_color(mask: torch.Tensor):
    """
    (H, W) 类别索引 → (H, W, 3) RGB 
    """
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for c, rgb in enumerate(CLASS_COLORS_255):
        color[mask.numpy() == c] = rgb
    return color


def main():
    parser = argparse.ArgumentParser(description="预测可视化")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--data-root", type=str, default=os.path.join(_PROJECT_ROOT, "data"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None, help="保存路径，默认从 checkpoint 名自动生成")
    args = parser.parse_args()

    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.checkpoint))[0]  # unet_ce_best
        label = stem.replace("unet_", "").replace("_best", "")          # ce
        args.output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures", f"predict_{label}.png")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    model = UNet(n_channels=3, n_classes=3)
    model.to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[Model] Loaded from {args.checkpoint}")

    data = PetSegDataset(root=args.data_root, batch_size=1, num_workers=0)
    val_set = data.val_set

    # 随机抽 n 张图，不超过验证集总数
    n_samples = min(args.num_samples, len(val_set))
    indices = random.sample(range(len(val_set)), n_samples)

    images_raw = []
    targets_raw = []
    for idx in indices:
        img, tgt = val_set[idx]
        images_raw.append(img)
        targets_raw.append(PetSegDataset.transform_target(tgt))

    images = torch.stack(images_raw).to(device)
    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()     # 每个像素取最大概率类别

    # 把图像恢复到 [0,1] 范围以便显示
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])

    _, axes = plt.subplots(n_samples, 3, figsize=(10, 3.5 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]                                

    for i in range(n_samples):
        img_np = images_raw[i].permute(1, 2, 0).numpy()
        img_np = img_np * std + mean                        
        img_np = np.clip(img_np, 0, 1)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label_to_color(targets_raw[i]))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(label_to_color(preds[i]))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"[Saved] {args.output}")


if __name__ == "__main__":
    main()
