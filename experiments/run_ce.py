"""
实验1: 仅使用标准交叉熵损失
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from train import run_training


if __name__ == "__main__":
    run_training(
        loss_type="ce",
        epochs=80,
        batch_size=8,
        lr=5e-4,
        weight_decay=5e-3,
        data_root=os.path.join(_PROJECT_ROOT, "data"),
        device_str="cuda",
        checkpoint_dir=os.path.join(_PROJECT_ROOT, "checkpoints"),
    )
