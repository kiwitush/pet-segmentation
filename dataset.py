"""
Oxford-IIIT Pet Dataset 加载与预处理
三分类分割：前景(pet)、背景、边界
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


class PetSegDataset:
    """
    Oxford-IIIT Pet 三分类分割数据集
    Trimap 原始标注:
        - 1: 前景 (pet)
        - 2: 背景
        - 3: 边界
    映射为 0-indexed 类别: {1: 0, 2: 1, 3: 2}
    """

    TARGET_SIZE = (256, 256)
    # 将 trimap 像素值映射为类别索引 0,1,2
    LABEL_MAP = {1: 0, 2: 1, 3: 2}
    # 未匹配像素的默认填充值，训练/评估时对应的标签将被忽略
    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str = "./data",
        val_split: float = 0.2,
        batch_size: int = 8,
        num_workers: int = 0,
    ):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize(self.TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 标注预处理: 最近邻插值，保持离散标签
        self.target_transform = transforms.Compose([
            transforms.Resize(
                self.TARGET_SIZE, interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.PILToTensor(),
        ])

        # 加载完整训练集
        full_dataset = OxfordIIITPet(
            root=self.root,
            split="trainval",
            target_types="segmentation",
            download=True,
            transform=self.image_transform,
            target_transform=self.target_transform,
        )

        # 划分 train/val
        n_total = len(full_dataset)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(42)
        self.train_set, self.val_set = random_split(
            full_dataset, [n_train, n_val], generator=gen
        )

    @staticmethod
    def transform_target(target_tensor: torch.Tensor) -> torch.Tensor:
        """
        将原始 trimap 像素值映射为 0/1/2 类别索引。
        输入: (1, H, W) PILToTensor，像素值为 1,2,3
        输出: (H, W) LongTensor，像素值为 0,1,2
        """
        target = target_tensor.squeeze(0).long()
        mapped = torch.full_like(target, PetSegDataset.IGNORE_INDEX)
        for raw_val, class_idx in PetSegDataset.LABEL_MAP.items():
            mapped[target == raw_val] = class_idx
        return mapped

    def get_train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    def get_val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = torch.stack([
            PetSegDataset.transform_target(item[1]) for item in batch
        ])
        return images, targets
