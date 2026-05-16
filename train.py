"""
训练入口：支持切换 CE / Dice / CE+Dice 三种损失函数
集成 SwanLab 记录 loss 和 mIoU 曲线
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import PetSegDataset
from unet import UNet
from loss import DiceLoss, CombinedLoss
from metrics import evaluate_model

import swanlab


def get_loss_fn(loss_type: str, ignore_index: int = 255, alpha: float = 0.5) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "dice":
        return DiceLoss(ignore_index=ignore_index)
    elif loss_type == "combined":
        return CombinedLoss(alpha=alpha, ignore_index=ignore_index)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def val_loss_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = loss_fn(logits, targets)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def run_training(
    loss_type: str = "combined",
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-3,
    weight_decay: float = 5e-3,
    alpha: float = 0.5,
    data_root: str = "./data",
    device_str: str = "cuda",
    checkpoint_dir: str = "./checkpoints",
    num_workers: int = 0,
    resume: str | None = None,
) -> float:
    """
    执行一次完整训练过程，包含数据加载、模型构建、训练循环、评估和模型保存。

    Args:
        loss_type: "ce" | "dice" | "combined"
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 初始学习率
        data_root: 数据集根目录
        device_str: 训练设备
        checkpoint_dir: 模型保存目录
        num_workers: DataLoader 进程数

    Returns:
        best_miou: 最佳 mIoU 值
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    print("[Data] Loading Oxford-IIIT Pet dataset ...")
    data = PetSegDataset(
        root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader = data.get_train_loader()
    val_loader = data.get_val_loader()
    print(f"[Data] Train: {len(data.train_set)}, Val: {len(data.val_set)}")

    print("[Model] Building U-Net ...")
    model = UNet(n_channels=3, n_classes=3)
    model.to(device)

    loss_fn = get_loss_fn(loss_type, ignore_index=255, alpha=alpha)
    print(f"[Loss] Using {loss_type} loss")

    # adamw 优化器
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # 余弦衰减
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0
    best_miou = 0.0

    if resume is not None:
        print(f"[Resume] Loading checkpoint from {resume}")
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"]
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"[Resume] Resuming from epoch {start_epoch}, best_mIoU={best_miou:.4f}")

    swanlab.init(
        project="unet-pet-segmentation",
        experiment_name=f"unet_{loss_type}_loss",
        config={
            "model": "U-Net (from scratch)",
            "loss": loss_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "dataset": "Oxford-IIIT Pet",
            "n_classes": 3,
            "input_size": "256x256",
            "alpha": alpha,
        },
    )

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch + 1, epochs + 1):
        train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = val_loss_epoch(model, val_loader, loss_fn, device)
        eval_result = evaluate_model(model, val_loader, device, n_classes=3, ignore_index=255)
        miou = eval_result["miou"]

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        swanlab.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/mIoU": miou,
            "lr": current_lr,
        }, step=epoch)

        print(
            f"[Epoch {epoch:3d}/{epochs}] "
            f"train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"mIoU={miou:.4f}  lr={current_lr:.2e}"
        )

        if miou > best_miou:
            best_miou = miou
            ckpt_path = os.path.join(checkpoint_dir, f"unet_{loss_type}_best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [*] Best model saved to {ckpt_path} (mIoU={best_miou:.4f})")

        # 每轮保存可续训 checkpoint
        ckpt_path = os.path.join(checkpoint_dir, f"unet_{loss_type}_last.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_miou": best_miou,
        }, ckpt_path)

    print(f"\n[Done] Best mIoU={best_miou:.4f}")
    swanlab.finish()
    return best_miou


def main():
    parser = argparse.ArgumentParser(description="U-Net 图像分割训练")
    parser.add_argument("--loss", type=str, default="combined",
                        choices=["ce", "dice", "combined"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="Combined loss 中 CE 的权重 (1-alpha 为 Dice 权重)")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="续训 checkpoint 路径")
    args = parser.parse_args()

    run_training(
        loss_type=args.loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        data_root=args.data_root,
        device_str=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
